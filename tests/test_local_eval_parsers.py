from __future__ import annotations

from routecode.local_eval.evaluators import score_exact
from routecode.local_eval.parsers import normalize_answer, parse_math_answer, parse_multiple_choice_answer


def test_math_parser_prefers_final_answer_then_boxed_then_last_number():
    assert parse_math_answer("Reasoning...\nFinal answer: 42") == "42"
    assert parse_math_answer("We get \\boxed{3/4}.") == "3/4"
    assert parse_math_answer("Try 1, then 2, therefore 17.") == "17"


def test_multiple_choice_parser_extracts_first_valid_letter():
    assert parse_multiple_choice_answer("B") == "B"
    assert parse_multiple_choice_answer("The answer is (d).") == "D"
    assert parse_multiple_choice_answer("Final answer: E") == "E"


def test_exact_scorer_normalizes_numbers_and_text():
    assert normalize_answer("  42.0 ") == "42"
    assert score_exact("42.0", "42") == 1.0
    assert score_exact("B", "b") == 1.0
    assert score_exact("forty two", "42") == 0.0
