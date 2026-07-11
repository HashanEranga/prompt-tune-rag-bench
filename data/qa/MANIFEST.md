# Phase A dataset manifest

Frozen split of the human-verified pool.

- Seed: `42`
- Pool: 433 pairs; near-duplicates dropped: 33; working set: 400
- **Train: 300** → `train.jsonl`  ·  **Test: 100** → `test.jsonl`
- sha256(train.jsonl): `2504f86dd91f754cf41783323a7221004f80d9c6deb820165bbd17ec8411e323`
- sha256(test.jsonl):  `3e96de251ffb5a5b796971b68ba0633f038a86c1519694e24de80ccad636d3c1`

## Test-set coverage by document

| doc_id | test | train |
|---|---|---|
| SGH-ER-001 | 6 | 17 |
| SGH-FAQ-002 | 5 | 14 |
| SGH-FAQ-003 | 6 | 19 |
| SGH-FAQ-004 | 5 | 17 |
| SGH-FAQ-005 | 6 | 17 |
| SGH-FAQ-006 | 6 | 18 |
| SGH-FIN-001 | 8 | 23 |
| SGH-IP-001 | 7 | 22 |
| SGH-LAB-001 | 13 | 38 |
| SGH-OPD-001 | 10 | 32 |
| SGH-PH-001 | 6 | 18 |
| SGH-PI-001 | 10 | 28 |
| SGH-SU-001 | 12 | 37 |

## Test-set answer-type mix

| answer_type | count |
|---|---|
| factual | 39 |
| numeric | 37 |
| procedural | 24 |
