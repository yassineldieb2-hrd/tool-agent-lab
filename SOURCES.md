# Attribution

## neural-maze/agentic-patterns-course (MIT)
https://github.com/neural-maze/agentic-patterns-course - Copyright (c) 2024 The Neural Maze (reviewed at commit `9c4b70a`).
The upstream project implements the tool-use and reflection agentic patterns without frameworks. This repository follows the same
overall approach (plain API calls; describe functions to the model, run the calls it requests, feed back observations; generate,
critique and revise) as a separate implementation, which differs as follows:
- local models through Ollama's native tool-calling API instead of Groq with XML `<tool_call>` tags parsed by regex
- a multi-round loop with `max_steps`, loop detection and observation truncation (upstream does one tool round followed by one summarising call)
- schemas generated from type hints and docstrings, strict argument validation, and all failures returned as `ERROR:` observations
- an `ast`-based safe arithmetic evaluator, unit and time tools
- reflection with an approval token, bounded rounds and draft history
- a scripted LLM and fake Ollama HTTP server for deterministic tests, a CLI and documentation
The ReAct/planning and multi-agent patterns of the upstream course are not implemented. The upstream notebooks and images are not used.
The MIT license text with the upstream copyright notice is reproduced in `LICENSE`.
