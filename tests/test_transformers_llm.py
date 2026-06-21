import pytest

from rag_ai import TransformersLLM


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages[0]["role"] == "user"
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT: {messages[0]['content']}"

    def __call__(self, prompt, *, return_tensors):
        assert prompt.startswith("CHAT:")
        assert return_tensors == "pt"
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def decode(self, tokens, *, skip_special_tokens):
        assert tokens == [4, 5]
        assert skip_special_tokens is True
        return " local answer "


class FakeModel:
    def generate(self, **kwargs):
        assert kwargs["input_ids"] == [[1, 2, 3]]
        assert kwargs["max_new_tokens"] == 32
        assert kwargs["do_sample"] is False
        return [[1, 2, 3, 4, 5]]


def test_transformers_generation_and_metrics() -> None:
    llm = TransformersLLM(
        model_name_or_path="local-model",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        max_new_tokens=32,
    )

    result = llm.generate_detailed("question")

    assert result.text == "local answer"
    assert result.metrics.input_tokens == 3
    assert result.metrics.output_tokens == 2
    assert result.metrics.elapsed_ms >= 0
    assert llm.generate("question") == "local answer"


def test_transformers_requires_tokenizer_and_model_together() -> None:
    with pytest.raises(ValueError, match="provided together"):
        TransformersLLM(tokenizer=FakeTokenizer())


def test_transformers_rejects_invalid_generation_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        TransformersLLM(tokenizer=FakeTokenizer(), model=FakeModel(), max_new_tokens=0)


def test_transformers_wraps_generation_errors() -> None:
    class BrokenModel:
        def generate(self, **kwargs):
            raise RuntimeError("out of memory")

    llm = TransformersLLM(tokenizer=FakeTokenizer(), model=BrokenModel())

    with pytest.raises(RuntimeError, match="generation failed"):
        llm.generate("question")
