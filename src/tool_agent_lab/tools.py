"""Tools: a Python function + a JSON Schema derived from its signature + argument validation.

Upstream (agentic-patterns-course) validated arguments against a hand-written signature string
and crashed on unknown tools. Here the schema is generated from type hints and docstrings, and
every failure is turned into a message the model can read and recover from.
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


class ToolError(Exception):
    """Raised for bad arguments or tool failures; the message is shown to the model."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON Schema object
    fn: Callable[..., Any]

    def schema(self) -> dict:
        """OpenAI/Ollama-style tool definition."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def validate(self, args: Any) -> dict:
        if isinstance(args, str):  # some models send arguments as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                raise ToolError(f"arguments for '{self.name}' are not valid JSON") from None
        if not isinstance(args, dict):
            raise ToolError(f"arguments for '{self.name}' must be an object") from None
        props, required = self.parameters["properties"], self.parameters["required"]
        missing = [r for r in required if r not in args]
        if missing:
            raise ToolError(f"missing required argument(s) for '{self.name}': {', '.join(missing)}") from None
        unknown = [a for a in args if a not in props]
        if unknown:
            raise ToolError(f"unknown argument(s) for '{self.name}': {', '.join(unknown)}") from None
        clean = {}
        for k, v in args.items():
            want = props[k]["type"]
            if want == "number" and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            ok = {"string": isinstance(v, str), "integer": isinstance(v, int) and not isinstance(v, bool),
                  "number": isinstance(v, float), "boolean": isinstance(v, bool)}[want]
            if not ok:
                raise ToolError(f"argument '{k}' of '{self.name}' must be {want}, got {type(v).__name__}") from None
            clean[k] = v
        return clean

    def run(self, args: Any) -> str:
        """Validate, execute, and always return a string (errors included)."""
        try:
            result = self.fn(**self.validate(args))
        except ToolError as e:
            return f"ERROR: {e}"
        except Exception as e:  # tool bugs must not kill the agent loop
            return f"ERROR: tool '{self.name}' failed: {type(e).__name__}: {e}"
        return result if isinstance(result, str) else json.dumps(result)


def _arg_docs(doc: str) -> dict[str, str]:
    """Parse 'Args:' lines of the form 'name: description' from a docstring."""
    out, in_args = {}, False
    for line in doc.splitlines():
        s = line.strip()
        if s.lower() == "args:":
            in_args = True
        elif in_args and ":" in s:
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def tool(fn: Callable) -> Tool:
    """Decorator: build a Tool from a typed function with a docstring."""
    doc = inspect.getdoc(fn) or ""
    if not doc:
        raise ValueError(f"{fn.__name__} needs a docstring: the model reads it")
    hints = get_type_hints(fn)
    arg_docs = _arg_docs(doc)
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        t = hints.get(name)
        if t not in _JSON_TYPES:
            raise ValueError(f"{fn.__name__}.{name}: unsupported type {t}; use str/int/float/bool")
        props[name] = {"type": _JSON_TYPES[t]}
        if name in arg_docs:
            props[name]["description"] = arg_docs[name]
        if p.default is inspect.Parameter.empty:
            required.append(name)
    description = re.split(r"\n\s*Args:", doc)[0].strip()
    return Tool(fn.__name__, description,
                {"type": "object", "properties": props, "required": required}, fn)
