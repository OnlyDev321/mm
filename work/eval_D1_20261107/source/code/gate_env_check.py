"""Apply the fatal environment gate used by the one-shot runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FATAL_SECTIONS = {"vllm_import", "gpu", "qwen_vl_utils", "model"}
FATAL_PACKAGE_CHECKS = {"expected_version", "requirements_pins"}


def failures(report: dict) -> list[str]:
    result = []
    for section, checks in report.get("sections", {}).items():
        for item in checks:
            if item.get("status") != "FAIL":
                continue
            name = item.get("name", "unknown")
            if section in FATAL_SECTIONS or (section == "packages" and name in FATAL_PACKAGE_CHECKS):
                result.append(f"{section}.{name}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_check", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.env_check.read_text(encoding="utf-8"))
    bad = failures(report)
    if bad:
        for item in bad:
            print(f"FAIL: {item}")
        return 1
    print("Environment gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
