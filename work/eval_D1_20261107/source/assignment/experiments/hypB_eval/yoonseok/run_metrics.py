"""Summarize completed generation metadata without responses."""
import argparse
import hashlib
import json
from pathlib import Path

from aggregate import raw_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--started", required=True)
    parser.add_argument("--finished", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows, digest = raw_rows(args.run / "raw.jsonl")
    metadata = json.loads((args.run / "run_metadata.json").read_text())
    env = json.loads((args.run / "env.json").read_text())
    require_sha = metadata["run"]["raw_file"]["sha256"]
    if require_sha != digest:
        raise ValueError("generation metadata raw hash differs")
    tokens = sum(r["output_tokens"] for r in rows)
    if tokens != metadata["run"]["token_stats"]["total"] or tokens != sum(x["output_tokens_total"] for x in env["invocations"]):
        raise ValueError("generation token totals differ")
    vram = [x.get("vram") for x in env["invocations"]]
    if not vram or any(not x or x["nvidia_smi"]["samples"] <= 0 or
                       not isinstance(x["nvidia_smi"]["max_used_mib"], (int, float)) for x in vram):
        raise ValueError("VRAM samples are missing")
    result = {"raw_sha256": digest, "rows": len(rows), "raw_bytes": (args.run / "raw.jsonl").stat().st_size,
              "started": args.started, "finished": args.finished,
              "generation_seconds": metadata["run"]["duration_seconds"],
              "output_tokens": tokens,
              "length_count": sum(r["finish_reason"] == "length" for r in rows),
              "max_vram_mib": max(x["nvidia_smi"]["max_used_mib"] for x in vram), "invocation_vram": vram,
              "generation_config_sha256": metadata["code_and_reproduction"]["config"]["sha256"],
              "generation_git": metadata["code_and_reproduction"]["git"],
              "current_code_config_sha256_lf": hashlib.sha256((Path(__file__).resolve().parents[4] / "code/config.yaml").read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
