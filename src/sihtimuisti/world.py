"""A frozen synthetic 'room camera' stream where detail provably matters.

64x64 grayscale frames, T frames long. A fixed multi-octave background texture.
16 square objects (16x16 px) in 8 *twin pairs*. Twins share brightness and a
coarse quadrant pattern, and differ ONLY in stripe orientation (horizontal vs
vertical) at a fixed period of 2, 4 or 8 px. Objects sit on an 8-px grid so the
stripes land exactly in one Haar octave:

    period 2 px  -> level-1 detail band   (lost by any 1-octave coarsening)
    period 4 px  -> level-2 detail band
    period 8 px  -> level-3 detail band

So "is this X or its twin?" can be answered from a memory only if that memory
still holds the right octave for that moment. Objects visit, sometimes move to a
new place mid-visit (the position test), and leave. Every frame gets fresh
camera noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIZE = 64
OBJ = 16
GRID = list(range(0, SIZE - OBJ + 1, 8))  # 0..48
PERIODS = [2, 2, 2, 4, 4, 4, 8, 8]  # one per twin pair
NOISE_SIGMA = 0.03


@dataclass
class World:
    seed: int
    T: int
    background: np.ndarray
    templates: list  # 16 clean 16x16 object images
    periods: list  # period per object
    configs: list  # per frame: frozenset of (obj, (r, c))
    visits: list = field(default_factory=list)

    def clean_frame(self, t: int) -> np.ndarray:
        img = self.background.copy()
        for obj, (r, c) in self.configs[t]:
            img[r:r + OBJ, c:c + OBJ] = self.templates[obj]
        return img

    def frame(self, t: int, rng: np.random.Generator) -> np.ndarray:
        return self.clean_frame(t) + rng.normal(0.0, NOISE_SIGMA, (SIZE, SIZE))

    def present(self, obj: int, t: int) -> bool:
        return any(o == obj for o, _ in self.configs[t])

    def first_seen(self, obj: int) -> int:
        for t, cfg in enumerate(self.configs):
            if any(o == obj for o, _ in cfg):
                return t
        return self.T


def _background(rng: np.random.Generator) -> np.ndarray:
    img = np.full((SIZE, SIZE), 0.5)
    for octave, amp in [(32, 0.06), (16, 0.04), (8, 0.03), (4, 0.02), (2, 0.015)]:
        n = SIZE // octave
        coarse = rng.normal(0.0, 1.0, (n, n))
        img += amp * np.kron(coarse, np.ones((octave, octave)))
    return img


def _templates(rng: np.random.Generator):
    temps, periods = [], []
    base_levels = np.linspace(0.18, 0.82, 8)
    rng.shuffle(base_levels)
    for pair in range(8):
        base = base_levels[pair]
        quad = rng.choice([-1.0, 1.0], size=(2, 2)) * 0.10
        coarse = base + np.kron(quad, np.ones((8, 8)))
        p = PERIODS[pair]
        stripe = np.where((np.arange(OBJ) // (p // 2)) % 2 == 0, 1.0, -1.0) * 0.12
        horiz = coarse + stripe[:, None]  # rows alternate -> horizontal stripes
        vert = coarse + stripe[None, :]
        temps += [horiz, vert]
        periods += [p, p]
    return temps, periods


def _overlap(a, b) -> bool:
    return abs(a[0] - b[0]) < OBJ and abs(a[1] - b[1]) < OBJ


def make_world(seed: int, T: int = 600) -> World:
    rng = np.random.default_rng(seed)
    bg = _background(rng)
    temps, periods = _templates(rng)
    occupancy = [dict() for _ in range(T)]  # t -> {obj: pos}
    visits = []
    for rep in range(2):
        for obj in rng.permutation(16):
            for _attempt in range(200):
                dur = int(rng.integers(25, 71))
                start = int(rng.integers(0, T - dur))
                move_at = start + int(rng.integers(8, dur - 8)) if rng.random() < 0.5 else None
                p1 = (int(rng.choice(GRID)), int(rng.choice(GRID)))
                p2 = (int(rng.choice(GRID)), int(rng.choice(GRID))) if move_at else p1
                ok = True
                for t in range(start, start + dur):
                    pos = p2 if (move_at and t >= move_at) else p1
                    occ = occupancy[t]
                    if obj in occ or any(_overlap(pos, q) for q in occ.values()) or len(occ) >= 3:
                        ok = False
                        break
                if ok:
                    for t in range(start, start + dur):
                        occupancy[t][obj] = p2 if (move_at and t >= move_at) else p1
                    visits.append((int(obj), start, start + dur, p1, p2, move_at))
                    break
    configs = [frozenset(occ.items()) for occ in occupancy]
    return World(seed=seed, T=T, background=bg, templates=temps, periods=periods,
                 configs=configs, visits=visits)
