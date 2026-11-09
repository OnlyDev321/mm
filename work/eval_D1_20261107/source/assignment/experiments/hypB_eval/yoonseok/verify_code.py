"""Verify frozen main evaluation sources without modifying them."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def verify():
    freeze = json.loads((HERE / "code_freeze.json").read_text(encoding="utf-8"))
    for relative, expected in freeze["sha256_lf"].items():
        data = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("frozen evaluation source changed: " + relative)
    return freeze["main_commit"]


if __name__ == "__main__":
    print("Frozen main evaluation code verified", verify())
