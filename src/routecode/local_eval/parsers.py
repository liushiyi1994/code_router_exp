from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


LETTER_RE = re.compile(r"\b([A-E])\b", flags=re.IGNORECASE)
FINAL_RE = re.compile(r"final\s+answer\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def parse_math_answer(text: str) -> str:
    raw = str(text or "")
    final = FINAL_RE.search(raw)
    if final:
        return _strip_answer(final.group(1).splitlines()[0])
    boxed = BOXED_RE.findall(raw)
    if boxed:
        return _strip_answer(boxed[-1])
    numbers = NUMBER_RE.findall(raw)
    if numbers:
        return _strip_answer(numbers[-1])
    return _strip_answer(raw)


def parse_multiple_choice_answer(text: str) -> str:
    match = LETTER_RE.search(str(text or ""))
    return match.group(1).upper() if match else ""


def normalize_answer(answer: str) -> str:
    text = _strip_answer(answer).lower()
    text = re.sub(r"^(final answer:|answer:)\s*", "", text).strip()
    text = text.replace(",", "")
    text = text.rstrip(".")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return re.sub(r"\s+", " ", text)
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def _strip_answer(answer: str) -> str:
    text = str(answer or "").strip()
    text = text.strip("`")
    text = text.strip()
    return text
