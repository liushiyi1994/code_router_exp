from __future__ import annotations

import json

import pandas as pd

from routecode.local_eval.generation_runner import (
    LOCAL_OUTCOME_COLUMNS,
    DryRunLocalClient,
    LocalEvalTask,
    OpenAICompatibleLocalClient,
    TransformersLocalClient,
    run_generation_matrix,
)


def test_dry_run_generation_matrix_logs_schema_scores_and_raw_outputs():
    tasks = [
        LocalEvalTask(
            query_id="math_0",
            query_text="What is 20 + 22?",
            dataset="gsm8k_smoke",
            domain="math",
            task_type="math",
            gold_answer="42",
        ),
        LocalEvalTask(
            query_id="mc_0",
            query_text="Which option is correct?",
            dataset="mmlu_smoke",
            domain="broad_knowledge",
            task_type="multiple_choice",
            gold_answer="C",
            choices=["A. no", "B. no", "C. yes", "D. no"],
        ),
    ]

    frame, raw_logs, errors = run_generation_matrix(
        tasks=tasks,
        model_ids=["dry_run_model"],
        client=DryRunLocalClient(),
        generation_params={"temperature": 0.0, "max_tokens": 16},
        model_revision="dry-run",
    )

    assert list(frame.columns) == LOCAL_OUTCOME_COLUMNS
    assert len(frame) == 2
    assert not errors
    assert len(raw_logs) == 2
    assert frame["quality"].tolist() == [1.0, 1.0]
    assert frame["model_id"].tolist() == ["dry_run_model", "dry_run_model"]
    assert frame["prompt_template"].tolist() == ["math_answer_v1", "multiple_choice_letter_v1"]
    assert frame["error_type"].fillna("").tolist() == ["", ""]
    assert frame["tokens_input"].min() > 0
    assert frame["tokens_output"].min() > 0
    assert json.loads(frame.loc[0, "generation_params_json"])["temperature"] == 0.0
    assert raw_logs[0]["raw_output"] == frame.loc[0, "raw_output"]


def test_generation_matrix_records_client_errors_without_dropping_rows():
    class FailingClient:
        def generate(self, **_kwargs):
            raise RuntimeError("server unavailable")

    task = LocalEvalTask(
        query_id="math_0",
        query_text="What is 1 + 1?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="2",
    )

    frame, raw_logs, errors = run_generation_matrix(
        tasks=[task],
        model_ids=["local_model"],
        client=FailingClient(),
        generation_params={"temperature": 0.0},
        model_revision="unknown",
    )

    assert len(frame) == 1
    assert len(raw_logs) == 1
    assert len(errors) == 1
    assert frame.loc[0, "quality"] == 0.0
    assert frame.loc[0, "error_type"] == "RuntimeError"
    assert "server unavailable" in frame.loc[0, "error_message"]


def test_transformers_local_client_can_use_injected_model_and_tokenizer():
    class FakeBatch(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            assert tokenize is False
            assert add_generation_prompt is True
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"] + "\nAssistant:"

        def __call__(self, text, return_tensors="pt"):
            import torch

            assert return_tensors == "pt"
            assert "Assistant:" in text
            return FakeBatch({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])})

        def decode(self, tokens, skip_special_tokens=True):
            assert skip_special_tokens is True
            assert len(tokens) == 2
            return "Final answer: 42"

    class FakeModel:
        device = "cpu"

        def eval(self):
            return self

        def generate(self, **kwargs):
            import torch

            assert kwargs["max_new_tokens"] == 8
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 4]])

    task = LocalEvalTask(
        query_id="math_0",
        query_text="What is 20 + 22?",
        dataset="gsm8k_smoke",
        domain="math",
        task_type="math",
        gold_answer="42",
    )
    client = TransformersLocalClient(
        model_id_or_path="fake-model",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    result = client.generate(
        model_id="fake-model",
        prompt="What is 20 + 22?",
        generation_params={"temperature": 0.0, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}},
        task=task,
    )

    assert result.raw_output == "Final answer: 42"
    assert result.tokens_input == 2
    assert result.tokens_output == 2


def test_openai_compatible_local_client_lists_served_model_ids(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"data": [{"id": "served-qwen"}, {"id": "served-coder"}]}'

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["authorization"] = req.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr("routecode.local_eval.generation_runner.request.urlopen", fake_urlopen)
    client = OpenAICompatibleLocalClient(
        base_url="http://localhost:8001/v1/",
        api_key="local-routecode",
        timeout_sec=7.0,
    )

    assert client.list_models() == ["served-qwen", "served-coder"]
    assert captured == {
        "url": "http://localhost:8001/v1/models",
        "timeout": 7.0,
        "authorization": "Bearer local-routecode",
    }
