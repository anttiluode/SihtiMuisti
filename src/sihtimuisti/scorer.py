"""One retrieval rule for every memory: evidence per octave.

The same scorer is used for every policy, so the race only compares what the
memories kept.

Shared background model (outside every budget, identical for all policies):
the running mean frame, the per-octave variance of frames around it, and a
camera-noise estimate (median absolute deviation of the finest diagonal band).
That is one mean frame and one variance frame per octave (about 11k scalars) plus 7 numbers.

Moment query ("show me when it looked like this"): log-likelihood ratio that
the query is a noisy copy of the item rather than an arbitrary frame, summed
only over the octaves the item still holds. An octave an item has dropped gives
no evidence either way, so a blurred item can be found by its coarse shape but
cannot out-argue a sharp item on detail.

Object query ("when was this object here?"): slide the clean object template
over the item at the item's own resolution and take the best evidence that the
stored window holds the template rather than whatever typically appears at
that place (per-coefficient mean and variance of all frames seen). Searching every grid
position makes it position-invariant: the object is found wherever it stood.
"""
from __future__ import annotations

import numpy as np

from . import haar
from .memory import LMAX, Item

SIZE = 64
J = LMAX
OBJ = 16


def mad_sigma(frame: np.ndarray) -> float:
    _, (_, _, d) = haar.analyze1(np.asarray(frame, float))
    return float(np.median(np.abs(d)) / 0.6745)


class Background:
    def __init__(self):
        self.n = 0
        self.sum = np.zeros((SIZE, SIZE))
        self.approx_sum = {lvl: np.zeros((SIZE >> lvl, SIZE >> lvl)) for lvl in range(LMAX + 1)}
        self.approx_sq = {lvl: np.zeros((SIZE >> lvl, SIZE >> lvl)) for lvl in range(LMAX + 1)}
        self.band_sum = None
        self.band_sq = None
        self.sigmas = []

    def update(self, frame: np.ndarray):
        a, det = haar.decompose(frame, J)
        if self.band_sum is None:
            self.band_sum = {j: np.zeros_like(det[j]) for j in det}
            self.band_sum["a"] = np.zeros_like(a)
            self.band_sq = {k: np.zeros_like(v) for k, v in self.band_sum.items()}
        for j in det:
            self.band_sum[j] += det[j]
            self.band_sq[j] += det[j] ** 2
        self.band_sum["a"] += a
        self.band_sq["a"] += a ** 2
        self.sum += frame
        ap = np.asarray(frame, float)
        for lvl in range(LMAX + 1):
            if lvl:
                ap, _ = haar.analyze1(ap)
            self.approx_sum[lvl] += ap
            self.approx_sq[lvl] += ap ** 2
        self.n += 1
        self.sigmas.append(mad_sigma(frame))

    def snapshot(self) -> "Prior":
        n = max(self.n, 1)
        mu = {k: v / n for k, v in self.band_sum.items()}
        var = {k: float(np.mean(self.band_sq[k] / n - mu[k] ** 2)) for k in mu}
        sigma = float(np.median(self.sigmas))
        mean_approx = {lvl: self.approx_sum[lvl] / n for lvl in self.approx_sum}
        var_approx = {lvl: np.maximum(self.approx_sq[lvl] / n - mean_approx[lvl] ** 2, 0.0)
                      for lvl in self.approx_sum}
        return Prior(mu=mu, var=var, sigma=sigma, mean_approx=mean_approx, var_approx=var_approx)


class Prior:
    def __init__(self, mu, var, sigma, mean_approx, var_approx):
        self.mu, self.var, self.sigma = mu, var, sigma
        self.mean_approx, self.var_approx = mean_approx, var_approx


class Scorer:
    def __init__(self):
        self._dec = {}

    def _decomp(self, it: Item):
        key = (it.id, it.version)
        d = self._dec.get(key)
        if d is None:
            a, det = haar.decompose(it.A, J - it.level)
            # re-index detail levels to absolute octave numbers
            d = (a, {j + it.level: det[j] for j in det})
            if len(self._dec) > 50000:
                self._dec.clear()
            self._dec[key] = d
        return d

    def moment_scores(self, query: np.ndarray, items: list, prior: Prior) -> np.ndarray:
        qa, qdet = haar.decompose(query, J)
        sq = mad_sigma(query) ** 2
        out = np.empty(len(items))
        for i, it in enumerate(items):
            s2 = sq + prior.sigma ** 2 / it.mass + 1e-8
            ma, mdet = self._decomp(it)
            llr = 0.0
            bands = [(qa, ma, prior.mu["a"], prior.var["a"])]
            bands += [(qdet[j], mdet[j], prior.mu[j], prior.var[j]) for j in mdet]
            for q, m, mu, v in bands:
                vn = v + sq + 1e-8
                llr += 0.5 * (float(((q - mu) ** 2).sum()) / vn - float(((q - m) ** 2).sum()) / s2
                              + q.size * np.log(vn / s2))
            out[i] = llr
        return out

    def object_scores(self, template: np.ndarray, items: list, prior: Prior) -> np.ndarray:
        out = np.zeros(len(items))
        tcache = {}
        for i, it in enumerate(items):
            L = it.level
            if L > 4:
                continue  # a 16-px object is smaller than one coefficient
            if L not in tcache:
                tcache[L] = haar.approx(template, L)
            t = tcache[L]
            s = OBJ >> L
            step = max(8 >> L, 1) if L <= 3 else 1
            if L == 4:
                step = 1  # 16-px blocks; positions multiple of 16 only
            W = np.lib.stride_tricks.sliding_window_view(it.A, (s, s))[::step, ::step]
            M = np.lib.stride_tricks.sliding_window_view(prior.mean_approx[L], (s, s))[::step, ::step]
            V = np.lib.stride_tricks.sliding_window_view(prior.var_approx[L], (s, s))[::step, ::step]
            s2 = prior.sigma ** 2 / it.mass + 1e-8
            vn = V + s2
            # evidence that the stored window is the template (+ noise) rather than
            # a typical frame at that place (mean M, per-coefficient variance V)
            e = 0.5 * (((W - M) ** 2 / vn).sum((-1, -2)) - ((W - t) ** 2).sum((-1, -2)) / s2
                       + np.log(vn / s2).sum((-1, -2)))
            out[i] = float(e.max())
        return out
