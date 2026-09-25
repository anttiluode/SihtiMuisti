"""The equal-budget race: stream a world through a memory, ask questions
during the stream (recalls), then ask the final questions at the end."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .memory import make_policy
from .scorer import Background, Scorer
from .world import World, make_world

FRAME_SCALARS = 64 * 64
ONLINE_RATE = 0.15
HOT_SHARE = 0.7
N_HOT_MOMENTS = 8
N_COLD_MOMENTS = 40
HOT_OBJECTS = [0, 7, 9, 12]  # X of pair 0, Y of pair 3, Y of pair 4, X of pair 6


@dataclass
class Query:
    kind: str  # "moment" | "object"
    target: int  # time or object id
    image: np.ndarray | None
    at: int  # stream time the query is asked
    hot: bool


def build_schedule(world: World, seed: int):
    rng = np.random.default_rng(seed * 7919 + 17)
    T = world.T
    hot_moments = sorted(int(t) for t in rng.choice(np.arange(int(0.05 * T), int(0.5 * T)),
                                                    N_HOT_MOMENTS, replace=False))
    hot_cfgs = {world.configs[t] for t in hot_moments}

    def render(t):
        return world.frame(t, rng)

    online = []
    for t in range(1, T):
        if rng.random() >= ONLINE_RATE:
            continue
        hot = rng.random() < HOT_SHARE
        if rng.random() < 0.5:
            pool = [h for h in hot_moments if h < t] if hot else None
            if hot and pool:
                tq = int(rng.choice(pool))
            else:
                hot, tq = False, int(rng.integers(0, t))
            online.append(Query("moment", tq, render(tq), t, hot))
        else:
            seen = [o for o in range(16) if world.first_seen(o) < t]
            if not seen:
                continue
            pool = [o for o in HOT_OBJECTS if o in seen] if hot else None
            if hot and pool:
                o = int(rng.choice(pool))
            else:
                hot, o = False, int(rng.choice(seen))
            online.append(Query("object", o, world.templates[o], t, hot))

    final = [Query("moment", t, render(t), T, True) for t in hot_moments]
    final += [Query("object", o, world.templates[o], T, True) for o in HOT_OBJECTS]
    cold = []
    while len(cold) < N_COLD_MOMENTS:
        t = int(rng.integers(0, T))
        if world.configs[t] not in hot_cfgs and t not in cold:
            cold.append(t)
    final += [Query("moment", t, render(t), T, False) for t in cold]
    final += [Query("object", o, world.templates[o], T, False)
              for o in range(16) if o not in HOT_OBJECTS]
    return online, final


def judge(world: World, q: Query, t_hat: int) -> bool:
    t_hat = int(np.clip(t_hat, 0, world.T - 1))
    if q.kind == "moment":
        return world.configs[t_hat] == world.configs[q.target]
    return world.present(q.target, t_hat)


def answer(scorer, mem, prior, q):
    if q.kind == "moment":
        s = scorer.moment_scores(q.image, mem.items, prior)
    else:
        s = scorer.object_scores(q.image, mem.items, prior)
    k = int(np.argmax(s))
    return mem.items[k], int(round(mem.items[k].time))


def run(seed: int, budget_frames: int | None, policy: str, T: int = 600, world=None,
        schedule=None, frames=None):
    world = world or make_world(seed, T)
    online, final = schedule or build_schedule(world, seed)
    if frames is None:
        frng = np.random.default_rng(seed * 1000 + 1)
        frames = [world.frame(t, frng) for t in range(world.T)]
    budget = None if budget_frames is None else budget_frames * FRAME_SCALARS
    mem = make_policy(policy, budget)
    bg, scorer = Background(), Scorer()
    by_time = {}
    for q in online:
        by_time.setdefault(q.at, []).append(q)
    online_log = []
    for t in range(world.T):
        bg.update(frames[t])
        mem.ingest(frames[t], t)
        if t in by_time:
            prior = bg.snapshot()
            for q in by_time[t]:
                it, t_hat = answer(scorer, mem, prior, q)
                online_log.append((q.kind, q.hot, judge(world, q, t_hat)))
                mem.recall(it)
    prior = bg.snapshot()
    final_log = []
    for q in final:
        it, t_hat = answer(scorer, mem, prior, q)
        period = world.periods[q.target] if q.kind == "object" else None
        final_log.append(dict(kind=q.kind, hot=q.hot, target=q.target, t_hat=t_hat,
                              correct=judge(world, q, t_hat), period=period,
                              item_level=it.level))
    return dict(seed=seed, budget_frames=budget_frames, policy=policy,
                used=mem.used(), n_items=len(mem.items), levels=mem.level_histogram(),
                online=online_log, final=final_log, memory=mem)
