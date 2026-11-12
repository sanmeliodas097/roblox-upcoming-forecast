#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (Monte Carlo rank diffusion).

Reads /input/history.ndjson (hourly-ish snapshots of Roblox's Up-and-Coming
chart: rank, universe_id, name, playing per game) and /input/task.json, and
writes /output/forecast.json: for each of the top 10 places, a probability
for each game named being at that place, plus the guessed top 10 in order.

Model
-----
The chart is a fast-moving rank order (its #1 changed 13 times in 45 reads;
the top 10 changed in 42 of 44 consecutive-read pairs). A persistence
forecast scores terribly (-43.5 log score), so this program simulates the
chart forward instead:

* Each game keeps a continuous rank position. Every simulated hour its
  position moves by a bucket drift + a momentum term (its recent rank
  velocity, decaying over the window) + Gaussian noise whose scale is
  estimated from the history's own consecutive reads, per rank bucket,
  scaled by sqrt(dt). Top ranks are sticky (small sigma), deep ranks
  diffuse fast.
* Rare breakout jumps (a game suddenly climbing several places) and fade
  jumps (a game dropping) are drawn with small hourly probabilities,
  because ~1.3 new games enter the top 10 per read step and most come
  from ranks 11-20.
* 2000 seeded Monte Carlo runs x 72 hourly slots are averaged: the
  forecast is the time-averaged probability that each game holds each
  place over the scored window. Probabilities are then shrunk toward a
  prior centred on the game's current rank (honest, calibrated
  probabilities win log score; overconfidence is punished at -9.21 per
  missed place), and a share of every place is left unnamed for games
  the simulation never names (true newcomers).

Deterministic: fixed seed, no network, standard library only. Two runs
on the same history write byte-identical output.
"""
import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
N_SIMS = int(os.environ.get("MF_N_SIMS", "4000"))
HOURS = 72
TOP_CANDIDATES = 50          # games from the last chart simulated in full
MOM_W = float(os.environ.get("MF_MOM_W", "0.25"))  # weight on recent rank velocity (places/hour)
MOM_DECAY = float(os.environ.get("MF_MOM_DECAY", "0.97"))  # per-hour decay of the momentum term
SIGMA_SCALE = float(os.environ.get("MF_SIGMA_SCALE", "0.8"))  # multiplier on history-estimated noise
P_BREAK = float(os.environ.get("MF_P_BREAK", "0"))  # per game per hour: sudden climb of 3-8 places
P_FADE = float(os.environ.get("MF_P_FADE", "0.001"))  # per game per hour: sudden drop of 5-15 places
SHRINK = float(os.environ.get("MF_SHRINK", "0"))  # weight of the rank-centred prior in final p
PRIOR_SIGMA = float(os.environ.get("MF_PRIOR_SIGMA", "6.0"))  # spread (places) of the rank-centred prior
UNNAMED_SCALE = float(os.environ.get("MF_UNNAMED_SCALE", "1.5"))
UNNAMED = {1: 0.05, 2: 0.08, 3: 0.12, 4: 0.12, 5: 0.12,
           6: 0.18, 7: 0.18, 8: 0.18, 9: 0.18, 10: 0.18}
MIN_P = 0.002                # rows below this are dropped
# Age-out: games leave the chart as they age, but `created` is NOT a hard
# 28-day clock (games created 2026-10-06 still charted 2026-11-12; over
# 13 days only 4.5% of >28d-old games were still charting vs 32% of
# younger ones). So: a gentle extra downward drift past 28 days, not a
# cliff — 0.02 places/hour per day past 28.
AGE_DRIFT = 0.02
AGE_DAYS = 28.0

BUCKETS = [(1, 3), (4, 10), (11, 20), (21, 50), (51, 100000)]


def bucket_of(rank):
    for lo, hi in BUCKETS:
        if lo <= rank <= hi:
            return (lo, hi)
    return BUCKETS[-1]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 3600.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    hours = int(task.get("window", {}).get("slots", HOURS))
    out_name = task.get("output", "forecast.json")

    reads = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])

    # ---- per-game series: (time_hours, rank, playing, name) ----
    series = {}
    for r in reads:
        t = parse_time(r["at"])
        for g in r.get("up_and_coming") or []:
            series.setdefault(g["universe_id"], []).append(
                (t, g["rank"], g.get("playing"), g.get("name")))
    last = reads[-1]
    last_chart = sorted(last.get("up_and_coming") or [], key=lambda g: g["rank"])
    last_rank = {g["universe_id"]: g["rank"] for g in last_chart}
    names = {g["universe_id"]: g.get("name", "") for g in last_chart}

    # ---- estimate drift + noise per rank bucket from consecutive reads ----
    acc = {b: {"n": 0, "d": 0.0, "d2": 0.0, "dt": 0.0} for b in BUCKETS}
    for r1, r2 in zip(reads, reads[1:]):
        dt = parse_time(r2["at"]) - parse_time(r1["at"])
        if dt <= 0 or dt > 48:
            continue
        c1 = {g["universe_id"]: g["rank"] for g in r1.get("up_and_coming") or []}
        c2 = {g["universe_id"]: g["rank"] for g in r2.get("up_and_coming") or []}
        for uid, rk1 in c1.items():
            if uid in c2:
                d = c2[uid] - rk1
                a = acc[bucket_of(rk1)]
                a["n"] += 1
                a["d"] += d / math.sqrt(dt)
                a["d2"] += (d / math.sqrt(dt)) ** 2
                a["dt"] += dt
    drift, sigma = {}, {}
    for b in BUCKETS:
        a = acc[b]
        if a["n"] >= 10:
            mean = a["d"] / a["n"]
            var = max(a["d2"] / a["n"] - mean * mean, 0.04)
            drift[b] = mean / math.sqrt(max(a["dt"] / a["n"], 1.0))
            sigma[b] = math.sqrt(var) * SIGMA_SCALE
        else:
            drift[b] = 0.1
            sigma[b] = 1.0
    # fallbacks keep top ranks stickier even with thin data
    for b, s in [((1, 3), 0.25), ((4, 10), 0.45), ((11, 20), 0.85),
                 ((21, 50), 1.5), ((51, 100000), 2.0)]:
        sigma.setdefault(b, s)

    # ---- recent rank velocity per game (places/hour, capped) ----
    velocity = {}
    for uid, pts in series.items():
        pts = pts[-5:]
        if len(pts) >= 2:
            t0, r0 = pts[0][0], pts[0][1]
            t1, r1 = pts[-1][0], pts[-1][1]
            dt = t1 - t0
            if dt > 0:
                v = (r1 - r0) / dt
                velocity[uid] = max(-1.0, min(1.0, v))
    # age-out: games whose details say they were created ~28 days ago
    # leave the chart; use it only when details exist (final history).
    created = {}
    for r in reads:
        for uid, det in (r.get("details") or {}).items():
            if det and det.get("created"):
                try:
                    created[int(uid)] = parse_time(det["created"])
                except (TypeError, ValueError):
                    pass
    last_t = parse_time(last["at"])

    cand = [g["universe_id"] for g in last_chart[:TOP_CANDIDATES]]
    cand_set = set(cand)
    # any game top-10 in the last 5 reads is a candidate too
    for r in reads[-5:]:
        for g in r.get("up_and_coming") or []:
            if g["rank"] <= 10 and g["universe_id"] in last_rank \
                    and g["universe_id"] not in cand_set:
                cand.append(g["universe_id"])
                cand_set.add(g["universe_id"])
    n_chart = len(last_chart)

    rng = random.Random(SEED)
    counts = {uid: [0] * (places + 1) for uid in cand}
    total_slots = N_SIMS * hours
    for _ in range(N_SIMS):
        pos = {uid: float(last_rank[uid]) for uid in cand}
        vel = {uid: velocity.get(uid, 0.0) for uid in cand}
        for h in range(1, hours + 1):
            decay = MOM_DECAY ** h
            for uid in cand:
                b = bucket_of(max(1, int(round(pos[uid]))))
                step = drift[b] + MOM_W * vel[uid] * decay
                step += rng.gauss(0.0, sigma[b])
                if rng.random() < P_BREAK and pos[uid] > 11:
                    step -= rng.uniform(3.0, 8.0)
                elif rng.random() < P_FADE:
                    step += rng.uniform(5.0, 15.0)
                # age-out: gentle extra downward drift past 28 days
                ct = created.get(uid)
                if ct is not None:
                    age_days = (last_t + h - ct) / 24.0
                    if age_days > AGE_DAYS:
                        step += (age_days - AGE_DAYS) * AGE_DRIFT
                pos[uid] = min(max(pos[uid] + step, 0.5), n_chart + 0.5)
            order = sorted(cand, key=lambda u: (pos[u], u))
            for place in range(1, places + 1):
                counts[order[place - 1]][place] += 1

    # ---- aggregate + shrink toward a rank-centred prior ----
    probs = []
    for uid in cand:
        cur = last_rank[uid]
        for place in range(1, places + 1):
            p_sim = counts[uid][place] / total_slots
            # prior: discretised normal around the game's current rank
            z = (place - cur) / PRIOR_SIGMA
            p_prior = math.exp(-0.5 * z * z)
            p = (1.0 - SHRINK) * p_sim + SHRINK * p_prior
            probs.append((uid, place, p))
    # normalise each place to 1 - UNNAMED, then drop tiny rows
    by_place = {}
    for uid, place, p in probs:
        by_place.setdefault(place, []).append([uid, p])
    rows_out = []
    for place in range(1, places + 1):
        lst = by_place.get(place, [])
        s = sum(p for _, p in lst)
        if s <= 0:
            continue
        target = 1.0 - UNNAMED.get(place, 0.15) * UNNAMED_SCALE
        for uid, p in lst:
            q = p / s * target
            if q >= MIN_P:
                rows_out.append({"universe_id": uid, "place": place,
                                 "p": round(q, 6)})
    rows_out.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # ---- top10 by expected place ----
    exp_rank = {}
    for uid in cand:
        e = 0.0
        mass = 0.0
        for place in range(1, places + 1):
            p = counts[uid][place] / total_slots
            e += place * p
            mass += p
        exp_rank[uid] = e + (places + 1) * (1.0 - mass)
    top10 = sorted(cand, key=lambda u: (exp_rank[u], last_rank[u]))[:places]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps({"probabilities": rows_out, "top10": top10}) + "\n")


if __name__ == "__main__":
    main()
