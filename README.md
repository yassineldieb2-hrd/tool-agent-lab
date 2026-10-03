# tool-agent-lab

A tool-calling **agent loop built from scratch** (no LangChain or CrewAI) for local LLMs via [Ollama](https://ollama.com),
plus a **reflection** loop (draft, critique, revise). The focus is the engineering that makes agents dependable:
schema generation, argument validation, error recovery, loop guards, and testing without a real model.

## The problem
Agent frameworks hide the loop that matters: the model requests a tool, your code runs it, and the result goes back into the
conversation. Models also produce malformed calls, repeat themselves and call tools that do not exist. This project implements
that loop explicitly and makes each failure mode observable and testable.

## Features
- `@tool` decorator: JSON Schema built from type hints and the docstring `Args:` section
- Strict argument validation (missing, unknown and mistyped arguments, arguments passed as JSON strings); errors go back to the model as `ERROR: ...` so it can self-correct
- Multi-step loop with `max_steps`, repeated-call detection and truncated observations
- Unknown tools and crashing tools never crash the agent
- Safe calculator built on `ast`, never `eval`
- Reflection loop with an approval token and bounded rounds
- Ollama client with clear errors; `ScriptedLLM` for deterministic runs and tests
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
Runtime dependencies: Python 3.10+ standard library only (`urllib`, `ast`, `zoneinfo`). Tests: pytest.

## Key technical decisions
- **Native tool calling** through Ollama's `/api/chat` with `tools`, so no API key, no paid service and no regex parsing of model text.
- **All failures become observations.** Bad JSON, bad arguments, unknown tools and tool exceptions are returned to the model as `ERROR:` messages instead of exceptions.
- **Generated schemas.** Tool definitions come from type hints and docstrings, so the schema the model sees cannot drift from the function.
- **Guardrails.** Step limit, identical-call detection, observation truncation, and `ast`-based arithmetic because model output is untrusted.
- **Tool output is data.** The system prompt says so; this is a mitigation, not a guarantee.
- **Model-free testing.** A scripted LLM and a fake Ollama HTTP server make loop behaviour deterministic.

## How to run
```bash
pip install -e ".[dev]"
python -m pytest                                  # no model needed
python -m tool_agent_lab tools                    # show generated tool schemas
python -m tool_agent_lab ask --llm demo           # scripted model, no Ollama needed
# with a real local model:
ollama pull llama3.2
python -m tool_agent_lab ask "What is 15% of 240 plus 7, and how many miles is 10 km?"
python -m tool_agent_lab reflect "Write a two-sentence product description for a reusable bottle"
```
Configuration (`OLLAMA_URL`, `LLM_MODEL`) is described in `.env.example`; export the variables in your shell.

## Example
`--llm demo` runs a scripted model, so it exercises the loop, tools and guards rather than a real model's ability
([examples/demo_output.txt](examples/demo_output.txt)):
```
step 1: calculator({"expression": "0.15 * 240 + 7"}) -> 43
step 2: convert_units({"value": 10, "from_unit": "km", "to_unit": "mi"}) -> 6.21371 mi

15% of 240 plus 7 is 43, and 10 km is about 6.21371 miles.
[2 model call(s), stopped: answered]
```

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

## Testing
39 tests cover schema generation, argument validation, the loop and its guards (step limit, repeated calls, truncation, unknown and
crashing tools), the safe calculator, the reflection loop, and the Ollama client against a local fake HTTP server.

## Limitations
- Sequential tool calls only; no parallelism, streaming or memory across runs
- Only `str`, `int`, `float` and `bool` parameters; no nested objects or enums
- Loop detection is exact-match, so a model that varies its arguments slightly can still loop until `max_steps`
- Tool-calling quality depends on the model, and small models can ignore tools; behaviour against a real Ollama model has not been benchmarked
- Reflection uses the same model as critic, which can approve its own mistakes
- ReAct/planning and multi-agent patterns are not implemented

## License
MIT, see [LICENSE](LICENSE). Third-party attributions: [SOURCES.md](SOURCES.md).
