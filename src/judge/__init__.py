"""Phase C — the judge.

A separate AI (Claude, a different family than every Phase B answerer) scores all 1,000
answers against the **source document**, on one frozen rubric, blind to which producer
wrote what. Its output is results/scores.csv, which aggregates into the master table.
"""
