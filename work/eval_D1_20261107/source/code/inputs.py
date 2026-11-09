"""Prepare the pinned Qwen3-VL inputs used for counting and generation.

Source: https://github.com/QwenLM/Qwen3-VL/blob/f8dca99056bb6352cf6ab36d4ea1848a09c54b5b/evaluation/mmmu/run_mmmu.py
License: Apache-2.0
Changes to prepare_inputs_for_vllm: retain resized image sizes; omit an empty
image key; return the original video kwargs; resolve a pinned, weight-free
snapshot for local checks. The chat-template and vision calls retain their
source order and arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

if __package__:
    from .generate import TokenCounter
else:
    from generate import TokenCounter


def resolve_model_dir(model_path: str, revision: str, weights: bool = False) -> Path:
    local = Path(model_path).expanduser()
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    ignored = None if weights else ["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.onnx"]
    return Path(snapshot_download(repo_id=model_path, revision=revision,
                                  ignore_patterns=ignored)).resolve()


def load_processor(model_dir: Path, cfg: dict) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    actual = processor.image_processor.patch_size
    expected = cfg["image"]["image_patch_size"]
    if actual != expected:
        raise ValueError(f"image patch size mismatch: processor={actual}, config={expected}")
    actual_merge = processor.image_processor.merge_size
    expected_merge = cfg["image"]["spatial_merge_size"]
    if actual_merge != expected_merge:
        raise ValueError(f"spatial merge size mismatch: processor={actual_merge}, config={expected_merge}")
    return processor


def prepare_inputs(messages: list[dict], processor: Any) -> tuple[dict, list[tuple[int, int]]]:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    prepared = {"prompt": text, "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs}
    sizes = [tuple(image.size) for image in image_inputs] if image_inputs is not None else []
    return prepared, sizes


class HFTokenCounter(TokenCounter):
    """Count the exact prepared prompt and retain it for the generation engine."""

    def __init__(self, model_dir: Path, cfg: dict, processor: Any = None):
        self.processor = processor if processor is not None else load_processor(model_dir, cfg)
        self.prepared: dict[str, dict] = {}

    def count(self, sample: Any, prompt: str, messages: list[dict],
              ordinal: int, cfg: dict) -> int:
        prepared, _ = prepare_inputs(messages, self.processor)
        images = prepared["multi_modal_data"].get("image")
        encoded = self.processor(text=[prepared["prompt"]], images=images,
                                 do_resize=False)
        self.prepared[sample.id] = prepared
        return len(encoded["input_ids"][0])

    def pop_prepared(self, sample_id: str) -> dict:
        return self.prepared.pop(sample_id)
