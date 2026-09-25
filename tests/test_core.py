import numpy as np
import pytest

from sihtimuisti import haar
from sihtimuisti.memory import (LMAX, POLICIES, Item, item_cost, make_policy, merge_items,
                                new_item, ward_cost)
from sihtimuisti.scorer import Background, Scorer
from sihtimuisti.world import OBJ, make_world


def test_haar_is_orthonormal_and_lossless():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 64))
    a, det = haar.decompose(x, 5)
    energy = (a ** 2).sum() + sum((d ** 2).sum() for d in det.values())
    assert np.isclose(energy, (x ** 2).sum())
    assert np.allclose(haar.reconstruct(a, det, 5), x, atol=1e-12)
    assert a.size + sum(d.size for d in det.values()) == 4096


def test_coarsen_loss_equals_reconstruction_error():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 64))
    it = new_item(x, 0)
    lost = it.coarsen()
    assert np.isclose(lost, ((it.view() - x) ** 2).sum())
    assert it.cost == 32 * 32 + 2


def test_ward_cost_matches_merged_error():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(2, 64, 64))
    a, b = new_item(x, 0), new_item(y, 1)
    m = merge_items(a, b)
    err = ((m.view() - x) ** 2).sum() + ((m.view() - y) ** 2).sum()
    assert np.isclose(ward_cost(a, b), err)


def test_mixed_level_ward_includes_coarsening():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(2, 64, 64))
    a, b = new_item(x, 0), new_item(y, 1, level=1)
    m = merge_items(a, b)
    err = ((m.view() - x) ** 2).sum() + ((m.view() - b.view()) ** 2).sum()
    assert m.level == 1
    assert np.isclose(ward_cost(a, b), err)


@pytest.mark.parametrize("name", [p for p in POLICIES if p != "full"])
def test_every_policy_respects_budget(name):
    rng = np.random.default_rng(4)
    mem = make_policy(name, 3 * 4096)
    for t in range(80):
        mem.ingest(rng.normal(size=(64, 64)), t)
        assert mem.used() <= 3 * 4096
    total_mass = sum(it.mass for it in mem.items)
    if name.startswith(("merge", "fade", "rd")):
        assert total_mass == 80  # nothing silently dropped


def test_twin_stripes_live_in_one_octave():
    w = make_world(1, T=200)
    for obj in range(16):
        t = w.templates[obj]
        p = w.periods[obj]
        _, det = haar.decompose(t - haar.upsample_view(haar.approx(t, 4), 4), 4)
        energies = {j: float((det[j] ** 2).sum()) for j in det}
        target = {2: 1, 4: 2, 8: 3}[p]
        assert max(energies, key=energies.get) == target
        # twins differ only in that octave
        twin = w.templates[obj ^ 1]
        _, dt = haar.decompose(t - twin, 4)
        other = sum(float((dt[j] ** 2).sum()) for j in dt if j != target)
        assert other < 1e-9


def test_object_scorer_finds_object_anywhere():
    w = make_world(1, T=200)
    bg = Background()
    rng = np.random.default_rng(5)
    for _ in range(20):
        bg.update(w.background + rng.normal(0, 0.03, (64, 64)))
    prior = bg.snapshot()
    items = []
    for pos in [(0, 0), (48, 40), (24, 8)]:
        img = w.background.copy()
        img[pos[0]:pos[0] + OBJ, pos[1]:pos[1] + OBJ] = w.templates[3]
        items.append(new_item(img, 0))
    empty = new_item(w.background.copy(), 0)
    twin = w.background.copy()
    twin[16:32, 16:32] = w.templates[2]
    s = Scorer().object_scores(w.templates[3], items + [empty, new_item(twin, 0)], prior)
    assert s[:3].min() > s[3] and s[:3].min() > s[4]


def test_demo_runs_headless(tmp_path):
    pytest.importorskip("cv2")
    import subprocess
    import sys
    out = tmp_path / "shot.png"
    r = subprocess.run([sys.executable, "demo/live_memory.py", "--synthetic", "1", "--headless", "120",
                        "--ask", "50", "--save", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
