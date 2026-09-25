# Gate 2 pre-registration — confirm the unplanned Gate 1 winner on fresh worlds

Written 2026-09-25, after Gate 1 run 2, before any Gate 2 run.

## Why

Gate 1's primary (`fade_recall` beats keyframes and the best thumbnail at both budgets)
came out MIXED. The strongest row was a secondary policy, `rd`: drop one octave of one
item or merge two time-neighbours, whichever adds the least squared error. It was
chosen by looking at Gate 1, so Gate 1 cannot confirm it. Gate 2 asks the same question
of `rd` on worlds nobody has looked at.

## Frozen

- Code exactly as at Gate 1 run 2 (scorer fix included). No constants change.
- Fresh worlds: seeds **6..15** (10 worlds, 640 final questions per policy per budget).
- Budgets 8 and 32 frame-equivalents. Same schedule rules, validity condition (full ≥ 95%
  on both question kinds) and hindsight-best thumbnail (chosen per budget on these seeds).

## Primary

**P2:** at both budgets, pooled final accuracy of `rd` is higher than the best thumbnail
and than keyframes.

- PASS: both budgets. KILL: `rd` ≤ best thumbnail at both budgets. Otherwise MIXED.

## Secondary

`rd_recall` vs `rd` (hot / cold); `fade_recall` replication; object accuracy by period;
moment vs object split.
