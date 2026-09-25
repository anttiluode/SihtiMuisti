"""Bounded visual memories, all charged in the same currency.

Every memory stores *items*. An item is one frame, or several merged frames,
held as an orthonormal Haar approximation at some level (0 = full 64x64,
1 = 32x32, ... 5 = 2x2). Cost of an item = its coefficient count + 2 scalars of
metadata (mass and mass-weighted time sum). A budget is a number of scalars.

Policies
--------
full            unbounded reference
keyframes       full-resolution frames, thinned to keep even coverage in time
thumbnails(L)   every frame at a fixed level L, thinned the same way
merge_any       full resolution, merge the pair whose merge loses least (Ward),
                any two items  -- the TransformerToX Gate 5 winner, for pictures
merge_adjacent  the same but only neighbours in time  -- online event segments
fade            new frames arrive sharp; the item whose level is most below
                log2(1 + age) loses its finest octave (Sihti residue dropped)
fade_recall     as fade, but each doubling of an item's recall count buys it
                one octave of protection
rd, rd_recall   no age rule: repeatedly do whichever of {drop one octave of one
                item, merge two time-neighbours} adds the least squared error;
                rd_recall multiplies that loss by (1 + recalls)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import haar

SIZE = 64
LMAX = 5
META = 2


def item_cost(level: int) -> int:
    return haar.coeff_count(SIZE, level) + META


_ids = iter(range(10**12))


@dataclass
class Item:
    A: np.ndarray
    level: int
    mass: float
    tsum: float
    tmin: int
    tmax: int
    recalls: int = 0
    id: int = 0
    version: int = 0

    @property
    def time(self) -> float:
        return self.tsum / self.mass

    @property
    def cost(self) -> int:
        return item_cost(self.level)

    def touch(self):
        self.version += 1

    def coarsen(self) -> float:
        """Drop the finest remaining octave. Returns the squared error added
        (per represented frame, times mass)."""
        ap, (h, v, d) = haar.analyze1(self.A)
        lost = float((h * h).sum() + (v * v).sum() + (d * d).sum()) * self.mass
        self.A = ap
        self.level += 1
        self.touch()
        return lost

    def view(self) -> np.ndarray:
        return haar.upsample_view(self.A, self.level)


def new_item(frame: np.ndarray, t: int, level: int = 0) -> Item:
    return Item(A=haar.approx(frame, level), level=level, mass=1.0, tsum=float(t),
                tmin=t, tmax=t, id=next(_ids))


def merge_items(a: Item, b: Item) -> Item:
    L = max(a.level, b.level)
    Aa = haar.approx(a.A, L - a.level)
    Ab = haar.approx(b.A, L - b.level)
    m = a.mass + b.mass
    A = (a.mass * Aa + b.mass * Ab) / m
    return Item(A=A, level=L, mass=m, tsum=a.tsum + b.tsum, tmin=min(a.tmin, b.tmin),
                tmax=max(a.tmax, b.tmax), recalls=a.recalls + b.recalls, id=next(_ids))


def ward_cost(a: Item, b: Item) -> float:
    """Squared error added by merging a and b (at their common level),
    including the octaves one of them must drop first."""
    L = max(a.level, b.level)
    loss = 0.0
    Aa, Ab = a.A, b.A
    for lvl in range(a.level, L):
        Aa, (h, v, d) = haar.analyze1(Aa)
        loss += float((h * h).sum() + (v * v).sum() + (d * d).sum()) * a.mass
    for lvl in range(b.level, L):
        Ab, (h, v, d) = haar.analyze1(Ab)
        loss += float((h * h).sum() + (v * v).sum() + (d * d).sum()) * b.mass
    diff = Aa - Ab
    loss += a.mass * b.mass / (a.mass + b.mass) * float((diff * diff).sum())
    return loss


class Memory:
    name = "base"

    def __init__(self, budget: int | None):
        self.budget = budget
        self.items: list[Item] = []
        self.t = 0

    def used(self) -> int:
        return sum(it.cost for it in self.items)

    def ingest(self, frame: np.ndarray, t: int):
        self.t = t
        self.items.append(self.make(frame, t))
        if self.budget is not None:
            self.enforce()
        assert self.budget is None or self.used() <= self.budget, (self.name, self.used())

    def make(self, frame, t):
        return new_item(frame, t, 0)

    def enforce(self):
        raise NotImplementedError

    def recall(self, item: Item):
        item.recalls += 1

    def level_histogram(self):
        h = [0] * (LMAX + 1)
        for it in self.items:
            h[it.level] += 1
        return h


class Full(Memory):
    name = "full"

    def __init__(self, budget=None):
        super().__init__(None)


class Thinned(Memory):
    """Fixed level; when over budget drop the interior item whose removal
    leaves the smallest gap, which keeps coverage close to uniform."""

    def __init__(self, budget, level=0):
        super().__init__(budget)
        self.level = level
        self.name = "keyframes" if level == 0 else f"thumbnails_L{level}"

    def make(self, frame, t):
        return new_item(frame, t, self.level)

    def enforce(self):
        while self.used() > self.budget:
            ts = np.array([it.time for it in self.items])
            if len(ts) <= 2:
                self.items.pop(0)
                continue
            gaps = ts[2:] - ts[:-2]
            i = int(np.argmin(gaps)) + 1
            self.items.pop(i)


class MergeAny(Memory):
    name = "merge_any"

    def enforce(self):
        while self.used() > self.budget:
            X = np.stack([it.A.ravel() for it in self.items])
            m = np.array([it.mass for it in self.items])
            sq = (X * X).sum(1)
            D = sq[:, None] + sq[None, :] - 2 * X @ X.T
            W = (m[:, None] * m[None, :]) / (m[:, None] + m[None, :]) * np.maximum(D, 0)
            np.fill_diagonal(W, np.inf)
            i, j = np.unravel_index(int(np.argmin(W)), W.shape)
            i, j = min(i, j), max(i, j)
            merged = merge_items(self.items[i], self.items[j])
            self.items.pop(j)
            self.items[i] = merged


class MergeAdjacent(Memory):
    name = "merge_adjacent"

    def enforce(self):
        while self.used() > self.budget:
            costs = [ward_cost(self.items[k], self.items[k + 1]) for k in range(len(self.items) - 1)]
            k = int(np.argmin(costs))
            self.items[k] = merge_items(self.items[k], self.items[k + 1])
            self.items.pop(k + 1)


class Fade(Memory):
    def __init__(self, budget, recall_weight=0.0):
        super().__init__(budget)
        self.recall_weight = recall_weight
        self.name = "fade_recall" if recall_weight else "fade"

    def enforce(self):
        while self.used() > self.budget:
            ages = np.array([self.t - it.time for it in self.items])
            lev = np.array([it.level for it in self.items], dtype=float)
            rec = np.array([it.recalls for it in self.items], dtype=float)
            prio = np.log2(1 + ages) - lev - self.recall_weight * np.log2(1 + rec)
            prio[lev >= LMAX] = -np.inf
            if not np.isfinite(prio).any():
                # everything is at 2x2; forget the oldest least-recalled item
                self.items.pop(int(np.argmax(ages - 1e6 * rec)))
                continue
            i = int(np.argmax(prio + 1e-9 * ages))  # ties -> older
            self.items[i].coarsen()


class RateDistortion(Memory):
    def __init__(self, budget, use_recall=False):
        super().__init__(budget)
        self.use_recall = use_recall
        self.name = "rd_recall" if use_recall else "rd"
        self._cc = {}  # (id, version) -> coarsen loss
        self._mc = {}  # (id, v, id, v) -> merge loss

    def _coarsen_loss(self, it: Item) -> float:
        key = (it.id, it.version)
        if key not in self._cc:
            if it.level >= LMAX:
                self._cc[key] = np.inf
            else:
                _, (h, v, d) = haar.analyze1(it.A)
                self._cc[key] = float((h * h).sum() + (v * v).sum() + (d * d).sum()) * it.mass
        return self._cc[key]

    def _merge_loss(self, a: Item, b: Item) -> float:
        key = (a.id, a.version, b.id, b.version)
        if key not in self._mc:
            self._mc[key] = ward_cost(a, b)
        return self._mc[key]

    def enforce(self):
        while self.used() > self.budget:
            best, op = np.inf, None
            for k, it in enumerate(self.items):
                w = (1 + it.recalls) if self.use_recall else 1
                c = self._coarsen_loss(it) * w
                if c < best:
                    best, op = c, ("c", k)
            for k in range(len(self.items) - 1):
                a, b = self.items[k], self.items[k + 1]
                w = (1 + a.recalls + b.recalls) if self.use_recall else 1
                c = self._merge_loss(a, b) * w
                if c < best:
                    best, op = c, ("m", k)
            if op[0] == "c":
                self.items[op[1]].coarsen()
            else:
                k = op[1]
                self.items[k] = merge_items(self.items[k], self.items[k + 1])
                self.items.pop(k + 1)
        if len(self._cc) > 20000:
            self._cc.clear()
            self._mc.clear()


def make_policy(name: str, budget: int) -> Memory:
    if name == "full":
        return Full()
    if name == "keyframes":
        return Thinned(budget, 0)
    if name.startswith("thumbnails_L"):
        return Thinned(budget, int(name[-1]))
    if name == "merge_any":
        return MergeAny(budget)
    if name == "merge_adjacent":
        return MergeAdjacent(budget)
    if name == "fade":
        return Fade(budget, 0.0)
    if name == "fade_recall":
        return Fade(budget, 1.0)
    if name == "rd":
        return RateDistortion(budget, False)
    if name == "rd_recall":
        return RateDistortion(budget, True)
    raise ValueError(name)


POLICIES = ["full", "keyframes", "thumbnails_L1", "thumbnails_L2", "thumbnails_L3",
            "thumbnails_L4", "thumbnails_L5", "merge_any", "merge_adjacent", "fade",
            "fade_recall", "rd", "rd_recall"]
