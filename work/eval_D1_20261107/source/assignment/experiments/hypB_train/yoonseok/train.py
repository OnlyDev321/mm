"""Run on one GPU: python train.py E1 pilot DATA_DIR NEW_OUTPUT_DIR."""
import argparse
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "train_config.json").read_text())


def image_env():
    # Qwen3-VL uses a 16px patch and a 2x2 spatial merge.
    os.environ.update(MIN_PIXELS="1003520", MAX_PIXELS="4014080",
                      IMAGE_MIN_TOKEN_NUM="980", IMAGE_MAX_TOKEN_NUM="3920",
                      SPATIAL_MERGE_SIZE="2")


def ordered_indices(size):
    order = list(range(size))
    random.Random(CONFIG["data_seed"]).shuffle(order)
    # Cover the longest input by step 20, with the same order in all four runs.
    longest = order.index(1161)
    boundary = 20 * CONFIG["gradient_accumulation_steps"] - 1
    if longest > boundary:
        order[boundary], order[longest] = order[longest], order[boundary]
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["E1", "E2"])
    parser.add_argument("stage", choices=["pilot", "full"])
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if output.exists():
        parser.error("use a new output directory")
    os.chdir(data)  # Both swift JSONLs use images/ relative to the data root.
    rows = Path(f"{args.run}_train_swift.jsonl").read_text(encoding="utf-8").splitlines()
    holdout = data / "holdout_300_swift.jsonl"
    if len(rows) != 3000 or len(holdout.read_text(encoding="utf-8").splitlines()) != 299:
        parser.error("expected 3000 train and 299 holdout examples")
    image_env()
    import torch
    from swift.llm import sft_main, TrainArguments
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("expose exactly one GPU for effective batch size 16")
    output.mkdir(parents=True)
    ordered = output / "ordered_train.jsonl"
    order = ordered_indices(len(rows))
    ordered.write_text("\n".join(rows[i] for i in order) + "\n", encoding="utf-8")
    print("row 1161 optimizer step:", order.index(1161) // 16 + 1, flush=True)
    pilot = args.stage == "pilot"
    sft_main(TrainArguments(**CONFIG, dataset=[str(ordered)], val_dataset=[str(holdout)],
                            max_steps=20 if pilot else -1, eval_steps=10 if pilot else 60,
                            output_dir=str(output)))


if __name__ == "__main__":
    main()
