"""Reflection pattern: draft -> critique -> revise, stopping when the critic approves.

Idea from the Reflection pattern in neural-maze/agentic-patterns-course (MIT); this is a
re-implementation with a bounded number of rounds, an explicit stop token, and a history of drafts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM

GENERATE = "You are a precise writer. Complete the task. Output only the result."
CRITIQUE = ("You are a strict reviewer. Review the draft for the task. If it fully satisfies the task, "
            "reply with exactly: APPROVED. Otherwise list the concrete problems, one per line.")


@dataclass
class ReflectionResult:
    final: str
    drafts: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    approved: bool = False


def reflect(llm: LLM, task: str, max_rounds: int = 2) -> ReflectionResult:
    if max_rounds < 0:
        raise ValueError("max_rounds must be >= 0")
    draft = llm.chat([{"role": "system", "content": GENERATE}, {"role": "user", "content": task}])["content"].strip()
    res = ReflectionResult(final=draft, drafts=[draft])
    for _ in range(max_rounds):
        critique = llm.chat([{"role": "system", "content": CRITIQUE},
                             {"role": "user", "content": f"Task: {task}\n\nDraft:\n{draft}"}])["content"].strip()
        res.critiques.append(critique)
        if critique.upper().rstrip(".") == "APPROVED":
            res.approved = True
            break
        draft = llm.chat([{"role": "system", "content": GENERATE},
                          {"role": "user", "content": f"Task: {task}\n\nPrevious draft:\n{draft}\n\n"
                                                      f"Reviewer feedback:\n{critique}\n\nWrite an improved version."}]
                         )["content"].strip()
        res.drafts.append(draft)
        res.final = draft
    return res
