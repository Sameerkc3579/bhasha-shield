"""Phase 2 of docs/v2-plan.md: the guard's training data, kept apart from the frozen test set.

schema (GuardItem, Exclusions) -> carriers (where a payload hides) -> augment (adaptive rewrites)
-> dedup (near-duplicates) -> split (train/validation by goal and family) -> leakage (TF-IDF audit
against the test set). `build` runs them in that order on generated drafts.
"""
