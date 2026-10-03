"""CLI: `tool-agent ask "..."`, `tool-agent reflect "..."`, `tool-agent tools`."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import ToolAgent
from .builtin_tools import ALL_TOOLS
from .llm import LLMError, OllamaChat, ScriptedLLM, call
from .reflection import reflect

DEMO_QUESTION = "What is 15% of 240 plus 7, and how many miles is 10 km?"


def demo_llm() -> ScriptedLLM:
    """Scripted replay for the demo question: shows the loop without needing a model."""
    return ScriptedLLM([
        {"role": "assistant", "content": "", "tool_calls": [
            call("calculator", expression="0.15 * 240 + 7"),
            call("convert_units", value=10, from_unit="km", to_unit="mi")]},
        {"role": "assistant", "content": "15% of 240 plus 7 is 43, and 10 km is about 6.21371 miles."},
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tool-agent", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer a question using tools")
    a.add_argument("question", nargs="?", default=DEMO_QUESTION)
    a.add_argument("--llm", choices=["ollama", "demo"], default="ollama",
                   help="'demo' replays a scripted conversation (no model needed)")
    a.add_argument("--max-steps", type=int, default=6)
    a.add_argument("--json", action="store_true", help="print the full trace as JSON")
    r = sub.add_parser("reflect", help="draft -> critique -> revise")
    r.add_argument("task")
    r.add_argument("--rounds", type=int, default=2)
    sub.add_parser("tools", help="print the tool schemas sent to the model")
    args = p.parse_args(argv)

    if args.cmd == "tools":
        print(json.dumps([t.schema() for t in ALL_TOOLS], indent=2))
        return 0
    llm = OllamaChat(os.getenv("OLLAMA_URL", "http://localhost:11434"), os.getenv("LLM_MODEL", "llama3.2"))
    try:
        if args.cmd == "reflect":
            res = reflect(llm, args.task, args.rounds)
            print(res.final)
            print(f"\n[{len(res.drafts)} draft(s), approved={res.approved}]", file=sys.stderr)
            return 0
        if args.llm == "demo":
            if args.question != DEMO_QUESTION:
                sys.exit("--llm demo only replays the built-in question; omit the question or use --llm ollama")
            llm = demo_llm()
        res = ToolAgent(llm, ALL_TOOLS, max_steps=args.max_steps).run(args.question)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"answer": res.answer, "stopped": res.stopped, "llm_calls": res.llm_calls,
                          "steps": [vars(s) for s in res.steps]}, indent=2))
    else:
        for i, s in enumerate(res.steps, 1):
            print(f"step {i}: {s.tool}({json.dumps(s.arguments)}) -> {s.observation}")
        print(f"\n{res.answer}\n[{res.llm_calls} model call(s), stopped: {res.stopped}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
