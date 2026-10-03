# tool-agent-lab

A tool-calling **agent loop built from scratch** (no LangChain/CrewAI) for local LLMs via [Ollama](https://ollama.com),
plus a small **reflection** (draft → critique → revise) loop. The focus is on the unglamorous parts that make agents
usable: schema generation, argument validation, error recovery, loop guards and testing without a real model.

Based in part on [neural-maze/agentic-patterns-course](https://github.com/neural-maze/agentic-patterns-course) (MIT). See [SOURCES.md](SOURCES.md).

## Why I built it
To understand what an "agent" actually is at the code level - a loop around a model, a set of tools, and a message history - and what can go wrong.

## Features
- `@tool` decorator: JSON Schema built from type hints + docstring `Args:`
- Strict validation (missing/unknown/mistyped args, JSON-string arguments); errors returned to the model as `ERROR: ...` so it can self-correct
- Multi-step loop with `max_steps`, repeated-call detection, truncated observations
- Unknown tools and crashing tools never crash the agent
- Calculator uses `ast`, not `eval` (model output is untrusted)
- Reflection loop with approval token and bounded rounds
- Ollama client with friendly errors; `ScriptedLLM` for offline demos/tests
- 39 pytest tests

## Architecture
```
user question
     |
 ToolAgent.run ──► LLM.chat(messages, tool schemas) ──► assistant message
     ^                                                      |
     |                                       has tool_calls? |
     |                                    no ──► final answer (return)
     |                                    yes
     |   observations as role="tool"          |
     └────────── Tool.run(args) ◄─────────────┘
              validate → execute → str (errors included)
 guards: max_steps · same call 3x → stop · observation ≤ 2000 chars · unknown tool → error message
```

## Tech stack
Python 3.10+, standard library only at runtime (`urllib`, `ast`, `zoneinfo`), pytest.

## How to run
```bash
pip install -e ".[dev]"
python -m pytest                                  # no model needed
python -m tool_agent_lab tools                    # show generated tool schemas
python -m tool_agent_lab ask --llm demo           # scripted replay, no model needed
# with a real local model:
ollama pull llama3.2
python -m tool_agent_lab ask "What is 15% of 240 plus 7, and how many miles is 10 km?"
python -m tool_agent_lab reflect "Write a two-sentence product description for a reusable bottle"
```

## Example
`--llm demo` is a **scripted replay** (it proves the loop, not a model's ability). Output in [examples/demo_output.txt](examples/demo_output.txt):
```
step 1: calculator({"expression": "0.15 * 240 + 7"}) -> 43
step 2: convert_units({"value": 10, "from_unit": "km", "to_unit": "mi"}) -> 6.21371 mi

15% of 240 plus 7 is 43, and 10 km is about 6.21371 miles.
[2 model call(s), stopped: answered]
```
I have **not** run this against a real Ollama model in my build environment; the Ollama client is tested against a local fake HTTP server. Whether a given small model emits good tool calls is for you to try.

## Adding a tool
```python
from tool_agent_lab.tools import tool

@tool
def word_count(text: str) -> str:
    """Count words in a text.

    Args:
        text: the text to count
    """
    return str(len(text.split()))
```
Pass it in the list given to `ToolAgent`.

## What I changed / added vs. upstream
See [SOURCES.md](SOURCES.md).

## Key Technical Concepts
- **Agent = loop**: call the model; if it asks for tools, run them and append results; repeat until it answers.
- **Function/tool calling**: the model is given JSON Schemas and returns structured call requests (name + arguments); *your code* executes them.
- **Message roles**: system, user, assistant (may include `tool_calls`), tool (observation).
- **Tool descriptions matter**: the model chooses tools from the name and description text.
- **Validation and error feedback**: models make malformed calls; returning a clear error often lets them retry correctly.
- **Guardrails**: step limits, loop detection, output truncation, never `eval` model output.
- **Prompt injection via tool output**: tool results are data; the system prompt says so, but this is mitigation only.
- **Reflection pattern**: a second pass critiques the first; costs extra calls and only helps when the critic can actually find problems.
- **Testing non-deterministic systems**: replace the model with a scripted fake and assert on the loop's behaviour.

## Limitations
- Sequential tool calls only; no parallelism, streaming or memory across runs
- Only str/int/float/bool parameters; no nested objects or enums
- Loop detection is exact-match (a model varying its arguments slightly can still loop until `max_steps`)
- Tool-calling quality depends on the model; small models can ignore tools
- Reflection uses the same model as critic, which can approve its own mistakes
- ReAct/planning and multi-agent patterns are not implemented

## Future improvements
Enum/optional types in schemas, tool-call retries with a repair prompt, token/time budgets, a small evaluation set of questions with expected tool traces, a planning step, structured logging.
