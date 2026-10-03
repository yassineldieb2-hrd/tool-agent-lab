"""LLM backends. An LLM here is anything with `chat(messages, tools) -> dict` returning an
assistant message: {"role": "assistant", "content": str, "tool_calls": [...]?}."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict: ...


class OllamaChat:
    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.2",
                 timeout: float = 120.0, temperature: float = 0.0):
        self.url, self.model, self.timeout, self.temperature = url.rstrip("/"), model, timeout, temperature

    def chat(self, messages, tools=None):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": self.temperature}}
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise LLMError(f"Ollama returned HTTP {e.code}: {e.read()[:200]!r}. "
                           f"Does model '{self.model}' exist and support tools?") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"Could not reach Ollama at {self.url} ({e}). Is it running?") from e
        if "message" not in data:
            raise LLMError(f"unexpected Ollama response: {str(data)[:200]}")
        return data["message"]


class ScriptedLLM:
    """Replays a fixed list of assistant messages. Used by tests and by `--llm demo`.
    It is NOT a model: it cannot adapt to the prompt."""

    def __init__(self, script: list[dict]):
        self.script, self.calls = list(script), []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            raise LLMError("ScriptedLLM ran out of scripted messages")
        return self.script.pop(0)


def call(name: str, **arguments) -> dict:
    return {"function": {"name": name, "arguments": arguments}}
