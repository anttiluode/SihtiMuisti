"""Gates 1-2: equal-budget memory race.

Gate 1: python -m experiments.run_gate1
Gate 2: python -m experiments.run_gate1 --seeds 6,7,8,9,10,11,12,13,14,15 --primary rd --out results/gate2.json
"""
import argparse
import json
import time

import numpy as np

from sihtimuisti import race
from sihtimuisti.memory import POLICIES
from sihtimuisti.world import make_world

SEEDS = [1, 2, 3, 4, 5]
BUDGETS = [8, 32]


def acc(rows, **filt):
    sel = [r for r in rows if all(r[k] == v for k, v in filt.items())]
    return (sum(r["correct"] for r in sel), len(sel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gate1.json")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--primary", default="fade_recall", choices=["fade_recall", "rd"])
    args = ap.parse_args()
    SEEDS[:] = [int(x) for x in args.seeds.split(",")]
    P = args.primary
    results = {}
    t0 = time.time()
    for seed in SEEDS:
        world = make_world(seed, 600)
        sched = race.build_schedule(world, seed)
        frng = np.random.default_rng(seed * 1000 + 1)
        frames = [world.frame(t, frng) for t in range(world.T)]
        for budget in BUDGETS:
            for pol in POLICIES:
                if pol == "full" and budget != BUDGETS[0]:
                    continue
                b = None if pol == "full" else budget
                r = race.run(seed, b, pol, world=world, schedule=sched, frames=frames)
                r.pop("memory")
                key = f"{pol}@{'all' if b is None else b}"
                results.setdefault(key, []).append(r)
                a = np.mean([f["correct"] for f in r["final"]])
                print(f"seed {seed} {key:22s} final {a:.3f} items {r['n_items']:4d} "
                      f"levels {r['levels']} ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for key, runs in results.items():
        rows = [f for r in runs for f in r["final"]]
        onl = [dict(kind=k, hot=h, correct=c) for r in runs for (k, h, c) in r["online"]]
        s = {
            "final_all": acc(rows),
            "final_hot": acc(rows, hot=True),
            "final_cold": acc(rows, hot=False),
            "final_moment": acc(rows, kind="moment"),
            "final_object": acc(rows, kind="object"),
            "online_all": acc(onl),
        }
        for p in (2, 4, 8):
            s[f"object_period_{p}"] = acc(rows, kind="object", period=p)
        s["used_scalars_mean"] = float(np.mean([r["used"] for r in runs]))
        s["items_mean"] = float(np.mean([r["n_items"] for r in runs]))
        s["levels_total"] = [int(x) for x in np.sum([r["levels"] for r in runs], 0)]
        summary[key] = s

    full = summary["full@all"]
    fo = full["final_object"][0] / full["final_object"][1]
    fm = full["final_moment"][0] / full["final_moment"][1]
    valid = fo >= 0.95 and fm >= 0.95
    verdict = {}
    for b in BUDGETS:
        frac = lambda k: summary[f"{k}@{b}"]["final_all"][0] / summary[f"{k}@{b}"]["final_all"][1]
        thumbs = {f"thumbnails_L{L}": frac(f"thumbnails_L{L}") for L in range(1, 6)}
        best_t = max(thumbs, key=thumbs.get)
        verdict[b] = {P: frac(P), "keyframes": frac("keyframes"),
                      "best_thumbnail": best_t, "best_thumbnail_acc": thumbs[best_t],
                      "beats_both": frac(P) > max(frac("keyframes"), thumbs[best_t])}
    wins = [verdict[b]["beats_both"] for b in BUDGETS]
    if all(wins):
        cls = f"PASS_{P.upper()}_BEATS_KEYFRAMES_AND_BEST_THUMBNAIL"
    elif not any(verdict[b][P] > verdict[b]["best_thumbnail_acc"] for b in BUDGETS):
        cls = f"KILL_BEST_THUMBNAIL_MATCHES_OR_BEATS_{P.upper()}"
    else:
        cls = "MIXED"
    if not valid:
        cls = "INVALID_FULL_REFERENCE_BELOW_95"
    out = dict(gate="gate1_memory_race" if P == "fade_recall" else "gate2_rd_confirmation",
               primary=P, seeds=SEEDS, budgets_frames=BUDGETS,
               validity=dict(full_object=fo, full_moment=fm, valid=valid),
               classification=cls, verdict=verdict, summary=summary,
               prereg="docs/prereg/gate1-memory-race.md" if P == "fade_recall"
               else "docs/prereg/gate2-rd-confirmation.md")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(cls)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
