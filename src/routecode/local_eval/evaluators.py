from __future__ import annotations

from routecode.local_eval.parsers import normalize_answer


def score_exact(parsed_answer: str, gold_answer: str) -> float:
    return 1.0 if normalize_answer(parsed_answer) == normalize_answer(gold_answer) else 0.0
