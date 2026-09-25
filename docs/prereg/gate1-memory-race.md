# Gate 1 pre-registration — does "blur with age, stay sharp when recalled" beat storing everything small?

Written 2026-09-25, before the scored run.

## Disclosure of what was seen before writing this

One smoke run: world seed 1, budget 8 frames, five policies (keyframes 0.20,
thumbnails_L3 0.86, merge_any 0.11, fade_recall 0.83, rd 0.78 final accuracy).
Seed 1 is still in the scored set below. Nothing was tuned after the smoke run:
every constant below was already in the code when it ran.

## Frozen setup

- Worlds: `make_world(seed, T=600)`, seeds **1, 2, 3, 4, 5**. 64x64 frames, 16 objects in
  8 twin pairs, twins differing only in stripe orientation at period 2 / 4 / 8 px
  (3 / 3 / 2 pairs), camera noise sigma 0.03.
- Budgets: **8 and 32 frame-equivalents** (32,768 and 131,072 scalars), i.e. 1.3% and
  5.3% of storing all 600 frames. Every item costs its Haar coefficients + 2 metadata scalars.
- Shared scorer for every policy (`scorer.py`), shared background model outside the budget.
- Online questions: each step asks with probability 0.15; 70% go to the hot set
  (8 hot moments, 4 hot objects `[0, 7, 9, 12]`). A retrieved item counts as recalled.
- Final questions at t = 600: hot = 8 moments + 4 objects; cold = 40 moments whose scene
  configuration differs from every hot moment + the 12 non-hot objects. 64 per world, 320 total.
- A moment answer is correct if the returned item's (mass-weighted) time has the same scene
  configuration as the target. An object answer is correct if the object is visible then.
- Frozen constants: recall weight 1 (each doubling of recalls = one octave of protection),
  LMAX = 5, fade priority `log2(1+age) - level - w*log2(1+recalls)`.

## Policies

`keyframes`, `thumbnails_L1..L5`, `merge_any`, `merge_adjacent`, `fade`, `fade_recall`, `rd`,
`rd_recall`, plus `full` (unbounded reference). "Best thumbnail" is chosen **after** seeing
the scores, per budget, which makes it a stronger baseline than any real user would have.

## Primary

**P:** at *both* budgets, pooled final accuracy of `fade_recall` is higher than both
`keyframes` and the best thumbnail level.

- PASS: both budgets.
- KILL: best thumbnail ≥ `fade_recall` at both budgets → blurring with age buys nothing
  over storing every frame small.
- Anything else: MIXED, reported as such.

## Secondary (reported, not pass/fail)

1. Recall: `fade_recall` vs `fade`, on hot and on cold final questions separately.
   Protection should help hot and may cost cold.
2. Merge vs octaves: `merge_any` / `merge_adjacent` vs `rd` / `fade`.
3. Object accuracy by stripe period (2 / 4 / 8 px), i.e. which octave each memory kept.
4. Moment vs object questions.

## Amendment 1 (2026-09-25, after run 1, before run 2): scorer bug and a validity check

Run 1 (`results/gate1_run1_INVALID_scorer_bug.json`) classified KILL, but its unbounded
`full` reference answered only 67.5% of object questions — a complete sharp memory should
answer almost all of them. The object scorer compared the *template* with the background
mean instead of asking whether the *stored window* looks more like the template or like
what normally appears at that place. It picked unrelated objects, not twins. Run 1 is
kept but invalid.

The fix uses the stored window as the observation and a per-coefficient background mean
and variance per octave as the null (`scorer.py`). Nothing else changed: same worlds, seeds,
schedules, budgets, policies, constants, primary and kill rule.

Added validity condition, checked before reading any policy result: the `full` reference
must answer ≥ 95% of final object questions and ≥ 95% of final moment questions, pooled.
Otherwise the run is INVALID.
