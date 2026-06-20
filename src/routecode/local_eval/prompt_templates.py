from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    text: str

    def render(self, **kwargs: object) -> str:
        return self.text.format(**kwargs)


MATH_ANSWER = PromptTemplate(
    template_id="math_answer_v1",
    text=(
        "Solve the problem privately. Do not show reasoning. Respond with exactly one line in this format:\n"
        "Final answer: <answer>\n\n"
        "Problem:\n{query_text}"
    ),
)

MULTIPLE_CHOICE_LETTER = PromptTemplate(
    template_id="multiple_choice_letter_v1",
    text=(
        "Answer the following multiple-choice question. Respond with only the letter A, B, C, D, or E.\n\n"
        "Question:\n{query_text}\n\n"
        "Choices:\n{choices}"
    ),
)


def prompt_for_task(task_type: str) -> PromptTemplate:
    normalized = str(task_type).lower()
    if normalized == "math":
        return MATH_ANSWER
    if normalized == "multiple_choice":
        return MULTIPLE_CHOICE_LETTER
    raise ValueError(f"Unsupported local-eval task type without sandboxed evaluator: {task_type}")
