# Interview notes - tool-agent-lab

Answers describe what this repository actually does.

## 10 likely questions
1. **What is an AI agent, in your implementation?** A loop in `ToolAgent.run`: send messages + tool schemas to the model; if the reply contains `tool_calls`, execute them, append the results as `tool` messages, and call the model again; stop when it replies without tool calls.
2. **How does the model know what tools exist?** The `@tool` decorator builds a JSON Schema from the function's type hints and the `Args:` section of its docstring; the agent sends these in the `tools` field of Ollama's `/api/chat`.
3. **What happens if the model calls a tool wrongly?** `Tool.validate` checks for JSON-string arguments, missing/unknown arguments and wrong types, and returns `ERROR: ...`. That string goes back as the observation, so the model can fix the call. A test (`test_model_recovers_from_bad_arguments`) covers this.
4. **How do you stop infinite loops?** `max_steps` caps model calls; and if the exact same call (name + sorted JSON args) occurs a third time the agent stops with `loop_detected`. Limitation: slightly varied arguments evade the second guard.
5. **Why not `eval` in the calculator?** The expression comes from a model, which can be manipulated. `safe_eval` parses with `ast` and only allows numeric constants and arithmetic operators, with limits on length and exponent. Tests include `__import__` and `open` attempts.
6. **What is prompt injection here and how do you reduce it?** Tool output (or user text) could contain instructions. The system prompt marks tool output as data, outputs are truncated to 2000 chars, and tools are low-privilege. It's mitigation, not prevention - a tool that reads the web or files would need more care.
7. **How did you test an agent without a model?** `ScriptedLLM` replays a fixed list of assistant messages; tests assert on the history sent in later calls, steps, stop reasons. The Ollama client is tested against a local fake HTTP server (payload shape, error messages).
8. **What did you take from the upstream repo and what is different?** The tool and reflection patterns and the no-framework approach. Different: Ollama native tool-calling instead of Groq + regex-parsed XML, a multi-step loop (upstream does one round), generated schemas, error-as-observation, guards, tests. ReAct and multi-agent are not implemented.
9. **What is the reflection pattern and when does it help?** Generate, critique, revise. It helps when the critic can identify concrete defects; with the same model as critic it can rubber-stamp. Mine stops on an explicit `APPROVED` or after `max_rounds`.
10. **How would you evaluate this agent?** I haven't built an evaluation yet. I would write questions with expected tool traces and answers, run a real model several times at temperature 0, and measure correct-tool rate, argument validity rate, and success rate.

## 5 areas to study further
- How function calling is trained/formatted in models, and why small models fail at it
- ReAct, planning and multi-agent orchestration trade-offs
- Evaluating agents (trajectory-level evals, flakiness, cost/latency)
- Prompt injection and tool-permission design (least privilege, confirmations)
- Context management for long runs (summarising, truncation, memory)

## Design decisions
- **Errors as observations:** keeps the loop alive and gives the model a chance to repair.
- **Standard library only:** fewer moving parts; easier to explain every line.
- **Schemas from type hints:** one source of truth; unsupported types fail loudly at decoration time.
- **Scripted LLM:** deterministic tests of control flow; honest that it says nothing about model quality.
- **Small tool set:** enough to show multi-step use without pretending to be a product.
