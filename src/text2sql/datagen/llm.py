"""Minimal OpenAI-compatible JSON chat client for the local model (standard library only)."""
import json
import urllib.error
import urllib.request

from text2sql.config import LLAMA_CPP_API_BASE, LLAMA_CPP_API_KEY, LLAMA_CPP_MODEL


class LLMFormatError(ValueError):
    """The model answered, but not with a JSON object."""


class LLMTransportError(RuntimeError):
    """The model endpoint could not be reached or returned an HTTP error."""


class LocalJSONModel:
    def __init__(self, api_base=LLAMA_CPP_API_BASE, model=LLAMA_CPP_MODEL,
                 api_key=LLAMA_CPP_API_KEY, timeout=300):
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.model = model.split("/", 1)[1] if model.startswith("openai/") else model
        self.api_key = api_key
        self.timeout = timeout
        self.calls = 0

    def __call__(self, messages, *, temperature=0.0, max_tokens=600) -> dict:
        body = json.dumps({
            "model": self.model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        request = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMTransportError(f"model request failed: {exc}") from exc
        content = payload["choices"][0]["message"].get("content") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMFormatError(f"not JSON: {content[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMFormatError(f"JSON is not an object: {content[:200]!r}")
        return data
