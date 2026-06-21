from __future__ import annotations

from contextlib import nullcontext
from os import getenv
from time import perf_counter
from typing import Any

from rag_ai.embeddings import TOKEN_RE
from rag_ai.models import GenerationMetrics, GenerationResult, RetrievedChunk
from rag_ai.prompts import INSUFFICIENT_CONTEXT


def generate_with_metrics(
    llm: object,
    prompt: str,
    *,
    context: list[RetrievedChunk] | None = None,
) -> GenerationResult:
    detailed = getattr(llm, "generate_detailed", None)
    if callable(detailed):
        return detailed(prompt, context=context)
    generate = getattr(llm, "generate", None)
    if not callable(generate):
        raise TypeError("LLM must provide generate() or generate_detailed()")
    started = perf_counter()
    text = generate(prompt, context=context)
    return GenerationResult(
        text=str(text),
        metrics=GenerationMetrics(elapsed_ms=(perf_counter() - started) * 1000),
    )


class StaticLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        return self.response


class ExtractiveLLM:
    """Local fallback LLM that extracts relevant source snippets instead of calling a model."""

    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        if not context:
            return INSUFFICIENT_CONTEXT

        question_terms = {token.lower() for token in TOKEN_RE.findall(prompt.split("Question:")[-1])}
        best_index = 0
        best_overlap = -1
        for index, item in enumerate(context):
            terms = {token.lower() for token in TOKEN_RE.findall(item.chunk.text)}
            overlap = len(question_terms & terms)
            if overlap > best_overlap:
                best_index = index
                best_overlap = overlap

        if best_overlap <= 0:
            return INSUFFICIENT_CONTEXT

        source = context[best_index]
        snippet = " ".join(source.chunk.text.split())
        if len(snippet) > 500:
            snippet = snippet[:497].rstrip() + "..."
        return f"{snippet} [{best_index + 1}]"


class OpenAILLM:
    def __init__(self, *, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI LLM requires the optional 'openai' extra: pip install -e .[openai]") from exc
        self.client = OpenAI(api_key=api_key or getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text


class TransformersLLM:
    """Run a Transformers causal language model entirely on the local machine."""

    def __init__(
        self,
        *,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 256,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be provided together")

        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self._torch: Any | None = None
        if tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Local generation requires the 'demo' extra: pip install -e .[demo]"
                ) from exc
            self._torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=False,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=False,
                )
                model.to(self.device)
                model.eval()
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not load Transformers model {model_name_or_path!r}. Check the model "
                    "ID or local path and verify that the machine has enough memory."
                ) from exc
        else:
            self.device = device or "cpu"

        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str, *, context: list[RetrievedChunk] | None = None) -> str:
        return self.generate_detailed(prompt, context=context).text

    def generate_detailed(
        self,
        prompt: str,
        *,
        context: list[RetrievedChunk] | None = None,
    ) -> GenerationResult:
        started = perf_counter()
        rendered_prompt = self._render_prompt(prompt)
        try:
            encoded = self.tokenizer(rendered_prompt, return_tensors="pt")
            model_inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_tokens = _last_dimension(model_inputs["input_ids"])
            inference_context = (
                self._torch.inference_mode() if self._torch is not None else nullcontext()
            )
            with inference_context:
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            sequences = getattr(output, "sequences", output)
            generated_tokens = sequences[0][input_tokens:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("Local Transformers generation failed.") from exc
        return GenerationResult(
            text=text,
            metrics=GenerationMetrics(
                elapsed_ms=(perf_counter() - started) * 1000,
                input_tokens=input_tokens,
                output_tokens=len(generated_tokens),
            ),
        )

    def _render_prompt(self, prompt: str) -> str:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return prompt
        try:
            return apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            return prompt


def _last_dimension(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(value[0])
