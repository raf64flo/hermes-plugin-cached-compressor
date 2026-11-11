"""Quality probe: ask about the material, answer from the summary only.

Containment measures whether a string survived. This measures whether the information is retrievable: the
questions are built mechanically from the material, the answer options come from the material as well, and a
small independent model answers from the summary alone. A summary that paraphrased the fact can still answer;
one that dropped it cannot.

The answerer is never the model that wrote the summary, so a model cannot be credited for recognising its own
phrasing. Questions only use paths and numbers: those have an unambiguous answer key. Decisions are excluded
because their paraphrase cannot be graded without a judge.
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request

NUMBER = re.compile(r"\b(\d+(?:[.,]\d+)?)\s?(ms|s|MB|GB|KB|%|USD|tokens?)\b")
PATH = re.compile(r"\b([A-Za-z0-9_./-]+\.(?:py|ts|tsx|rs|js|md|yaml|yml|json))(?::(\d+))?")

PROMPT = """You read a summary of a work session and answer questions about it.
Answer with the letter of the correct option and nothing else. If the summary does not say, answer X.

Summary:
{summary}

{questions}"""


def build(material: list[dict], wanted: list[dict], count: int = 6, seed: int = 11) -> list[dict]:
    """Questions with one true option and three distractors drawn from the same material."""
    text = "\n".join(message.get("content") or "" for message in material)
    pairs = [(m.group(1), m.group(2)) for m in PATH.finditer(text) if m.group(2)]
    numbers = [m.group(0) for m in NUMBER.finditer(text)]
    all_lines = list(dict.fromkeys(line for _, line in pairs))
    all_numbers = list(dict.fromkeys(numbers))
    rng = random.Random(seed)
    pool: list[dict] = []

    for file, line in dict.fromkeys(pairs):
        distractors = [other for other in all_lines if other != line]
        if len(distractors) < 3:
            break
        options = [line, *rng.sample(distractors, 3)]
        rng.shuffle(options)
        pool.append({"q": f"Which line number of {file} does the summary mention?",
                     "options": options, "answer": options.index(line)})

    for value in all_numbers:
        distractors = [other for other in all_numbers if other != value]
        if len(distractors) < 3:
            break
        options = [value, *rng.sample(distractors, 3)]
        rng.shuffle(options)
        pool.append({"q": "Which of these values does the summary mention?",
                     "options": options, "answer": options.index(value)})

    rng.shuffle(pool)
    return pool[:count]


def render(questions: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(questions):
        letters = "ABCD"
        options = " ".join(f"{letters[i]}) {value}" for i, value in enumerate(item["options"]))
        blocks.append(f"{index + 1}. {item['q']}\n   {options}")
    return "\n".join(blocks)


def ask(summary: str, questions: list[dict], transport, model: str) -> dict:
    """One call, letters parsed, no judge: the answer key is known."""
    if not questions:
        return {"asked": 0, "correct": 0, "answers": []}
    prompt = PROMPT.format(summary=summary[:6000], questions=render(questions))
    response = transport(model=model, messages=[{"role": "user", "content": prompt}],
                         max_tokens=24, temperature=0)
    text = (response.choices[0].message.content or "").upper()
    letters = re.findall(r"\b([ABCDX])\b", text)
    answers = []
    correct = 0
    for index, item in enumerate(questions):
        given = letters[index] if index < len(letters) else "?"
        right = "ABCD"[item["answer"]]
        answers.append({"given": given, "expected": right})
        correct += int(given == right)
    return {"asked": len(questions), "correct": correct, "answers": answers}
