# Corpus Unicode audit

Before training, inspect the actual character distribution of each corpus split.
The audit reports Unicode categories, broad script-name prefixes, frequent
characters, non-Latin letters, and invisible/control-like characters.

It is intentionally **read-only**. It does not remove non-Latin text: names,
quotes, technical terms, and source material may legitimately contain it.
Review the report and representative source passages before deciding whether
a filtering policy is appropriate.

Run on the local training split:

```bash
PYTHONPATH=python python -m simple_ai.experiments.audit_corpus \
  --input data/processed/indonesian_wikipedia/train.txt \
  --output artifacts/evaluation/corpus_unicode_audit.json
```

You can audit validation and test files separately by changing `--input`.
Keep the three reports separate so data-cleaning decisions do not accidentally
use the held-out test split.

A large non-Latin count is a signal to inspect, not proof of bad data. The
audit uses Unicode character names as a lightweight heuristic, not a complete
language or script detector.
