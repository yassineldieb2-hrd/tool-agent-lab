# Sources and attribution

## neural-maze/agentic-patterns-course (MIT)
Repository: https://github.com/neural-maze/agentic-patterns-course - Copyright (c) 2024 The Neural Maze (studied at commit `9c4b70a`).
It implements Andrew Ng's four agentic patterns (reflection, tool use, planning/ReAct, multi-agent) against the Groq API without frameworks.

**What is based on it (ideas and structure, re-implemented):**
- The "no frameworks, plain API calls" approach and the **Tool pattern**: describe functions to the model, let it request calls, run them, feed back observations.
- The **Reflection pattern** (generate → critique → revise).

**What is different (my own code):**
- Local models via Ollama's native tool-calling API (`/api/chat` with `tools`) instead of Groq with XML `<tool_call>` tags parsed by regex; no API key or paid service.
- A real loop (multiple tool rounds until a final answer) with `max_steps`, loop detection, and observation truncation. Upstream's `ToolAgent.run` does one tool round followed by one summarising call.
- Tool schemas generated from type hints and docstrings (upstream uses a hand-built signature string); strict argument validation; **all** failures (bad JSON, missing/unknown args, unknown tool, tool crash) become `ERROR:` observations the model can read, instead of exceptions.
- A safe arithmetic evaluator based on `ast` (never `eval`), unit/time tools.
- Reflection with an explicit approval token, bounded rounds and draft history.
- A scripted LLM + fake Ollama HTTP server for deterministic tests; CLI; documentation.
- **Not implemented:** the ReAct/planning and multi-agent patterns from upstream.

The upstream notebooks and images are not used. The MIT license text is reproduced in `LICENSE` with the upstream copyright notice.

Concept credit: the four patterns were popularised by Andrew Ng's DeepLearning.AI blog series (referenced by the upstream README); no text from it is used.
