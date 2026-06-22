"""Local LLM wrapper around HuggingFace transformers.

Runs a small instruct model fully locally (CPU or GPU if available). The model
is loaded lazily and reused. Both the RAG and non-RAG pipelines call the *same*
``LocalLLM`` instance so any quality difference is attributable to retrieval,
not to a different model.
"""

from __future__ import annotations


class LocalLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 320,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (imported for side effects / availability)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
        )
        self._model.eval()

    def generate(self, messages: list[dict]) -> str:
        """Generate a completion for a list of chat messages.

        ``messages`` follows the OpenAI-style role/content convention and is
        rendered with the model's own chat template so it works across most
        instruct models.
        """
        self._ensure_loaded()
        import torch

        tokenizer = self._tokenizer
        model = self._model

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        do_sample = self.temperature and self.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=bool(do_sample),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)

        # Strip the prompt tokens so we return only the newly generated text.
        generated = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
