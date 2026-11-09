"""Merge a final PEFT checkpoint into the pinned bf16 base on the GPU host."""
import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration
import torch
from train import CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    base = Path(snapshot_download(CONFIG["model"], revision=CONFIG["model_revision"]))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu")
    adapted = PeftModel.from_pretrained(model, args.checkpoint)
    merged = adapted.merge_and_unload(safe_merge=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        result = Path(tmp) / "merged"
        merged.save_pretrained(result, safe_serialization=True)
        # Copy original processor/tokenizer files byte-for-byte, including chat template.
        for file in base.iterdir():
            if file.suffix in {".json", ".jinja", ".txt"} and file.name not in {
                    "config.json", "model.safetensors.index.json"}:
                shutil.copyfile(file, result / file.name)
        result.rename(args.output)


if __name__ == "__main__":
    main()
