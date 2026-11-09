"""Pinned rule parsers, loaded without upstream API clients or global RNG changes.

Upstream function bodies are compiled unchanged. MMMU's random fallback is
intercepted and returns a missing extraction; no randomly selected answer exists.
See THIRD_PARTY.md for commits, licenses and deliberate policy differences.
"""
from __future__ import annotations

import ast
import copy
import logging
import re
import string
from pathlib import Path
from types import SimpleNamespace

VENDOR = Path(__file__).parent / "vendor"


class NoExtraction(Exception):
    pass


def no_random_choice(_choices):
    raise NoExtraction("MMMU reached its upstream random-fallback branch")


def load_functions(filename, names, namespace, constants=()):
    tree = ast.parse((VENDOR / filename).read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != set(names):
        raise ValueError(f"Missing pinned functions in {filename}")
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
                   len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in constants]
    if {n.targets[0].id for n in assignments} != set(constants):
        raise ValueError(f"Missing pinned constants in {filename}")
    tree = ast.Module(body=assignments + functions, type_ignores=[])
    exec(compile(tree, str(VENDOR / filename), "exec"), namespace)
    return namespace


QWEN = load_functions(
    "qwen_eval_utils.py",
    ["can_infer_option", "can_infer_text", "can_infer", "build_prompt"],
    {"copy": copy, "string": string},
)
VLMEVAL = load_functions(
    "vlmevalkit_matching.py", ["can_infer_option", "can_infer_text", "can_infer"],
    {"cp": copy, "os": SimpleNamespace(environ={}), "string": string,
     "re": re, "logger": logging.getLogger(__name__)}, constants=("_VERBOSE_ANSWER_RE",),
)
MMMU = load_functions(
    "mmmu_eval.py",
    ["parse_multi_choice_response", "check_is_number", "normalize_str",
     "extract_numbers", "parse_open_response", "eval_open", "eval_multi_choice"],
    {"re": re, "random": SimpleNamespace(choice=no_random_choice),
     "np": SimpleNamespace(argmax=lambda xs: max(range(len(xs)), key=xs.__getitem__))},
)

# Exact legacy rule from our 2048/9048 experiments. It is a candidate under audit,
# NOT an official MMMU rule and NOT proof that the model committed to this answer.
FINAL_ANSWER_RE = re.compile(
    r"(?i)\b(?:the\s+)?final\s+answer\s*(?:is\s*)?(?::|=|-)?\s*\*{0,2}\s*\(?([A-I])\)?\b"
)


def marker(text, choices):
    matches = [m for m in FINAL_ANSWER_RE.finditer(text) if m.group(1).upper() in choices]
    labels = [m.group(1).upper() for m in matches]
    trailing = len(text) - matches[-1].end() if matches else None
    accepted = bool(labels and len(set(labels)) == 1 and trailing <= 100)
    return {"answer": labels[-1] if accepted else None,
            "labels": labels, "trailing_characters": trailing,
            "conflicting_markers": len(set(labels)) > 1}


def qwen_parse(text, choices):
    answer = QWEN["can_infer"](text, copy.deepcopy(choices))
    return answer if answer in choices else None


def vlmeval_parse(text, choices):
    answer = VLMEVAL["can_infer"](text, copy.deepcopy(choices))
    return answer if answer in choices else None


def mmmu_mc_parse(text, choices):
    try:
        return MMMU["parse_multi_choice_response"](text, list(choices), dict(choices)), False
    except NoExtraction:
        return None, True


def open_candidates(text):
    return sorted(MMMU["parse_open_response"](text), key=lambda x: (type(x).__name__, str(x)))


def judge_prompt(question, choices, raw_text):
    options = "There are several options: \n" + "".join(f"{k}. {v}\n" for k, v in choices.items())
    return QWEN["build_prompt"](question, options, raw_text)


PRIMARY_PARSERS = ("qwen", "mmmu_no_random", "vlmevalkit")
PARSERS = (*PRIMARY_PARSERS, "hybrid100")
GATES = ("none", "always_judge", "marker_exception")


def analyze(row):
    is_mc = row["question_type"] == "multiple-choice"
    choices = row["options"] if is_mc else {"A": str(row["answer"]), "B": "Other Answers"}
    qwen = qwen_parse(row["raw_text"], choices)
    vlmeval = vlmeval_parse(row["raw_text"], choices)
    final = marker(row["raw_text"], choices) if is_mc else {
        "answer": None, "labels": [], "trailing_characters": None, "conflicting_markers": False}
    conflict = bool(qwen and final["answer"] and qwen != final["answer"])
    hybrid = None if conflict else (qwen or final["answer"])
    if is_mc:
        mmmu, random_blocked = mmmu_mc_parse(row["raw_text"], choices)
    else:
        mmmu, random_blocked = open_candidates(row["raw_text"]), False
    outputs = {"qwen": qwen, "mmmu_no_random": mmmu, "vlmevalkit": vlmeval, "hybrid100": hybrid}
    policies = {}
    for parser in PARSERS:
        parsed = outputs[parser]
        success = parsed is not None and parsed != []
        for gate in GATES:
            route = "auto" if success else "judge_rule_failure"
            if parser == "hybrid100" and conflict:
                route = "judge_conflict"
            if row["finish_reason"] == "length" and gate != "none":
                # Apply the SAME exception criterion across parsers for a paired
                # gate experiment. Parser/marker agreement is mandatory.
                exception = (gate == "marker_exception" and is_mc and
                             final["answer"] is not None and not conflict and parsed == final["answer"])
                if not exception:
                    route = "judge_length"
            correct = False
            if route == "auto":
                if parser == "mmmu_no_random" and not is_mc:
                    correct = bool(MMMU["eval_open"](row["answer"], parsed))
                else:
                    correct = parsed == (row["answer"] if is_mc else "A")
            policies[f"{parser}__{gate}"] = {
                "route": route, "parsed": parsed,
                "auto_correct": correct, "rule_success_before_gate": success,
            }
    return {"qwen": qwen, "mmmu_no_random": mmmu, "vlmevalkit": vlmeval, "hybrid100": hybrid,
            "marker": final, "qwen_marker_conflict": conflict,
            "mmmu_random_fallback_blocked": random_blocked, "policies": policies}
