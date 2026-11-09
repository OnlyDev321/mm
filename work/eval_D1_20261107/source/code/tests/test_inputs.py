import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

import common
import inputs
import preflight


@pytest.fixture
def cfg():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())


class Grid:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class Processor:
    def __init__(self, patch_size):
        self.image_processor = SimpleNamespace(patch_size=patch_size, merge_size=2)
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(("template", messages, kwargs))
        return "rendered"

    def __call__(self, **kwargs):
        self.calls.append(("encode", kwargs))
        images = kwargs["images"] or []
        return {"input_ids": [[1, 2, 3]],
                "image_grid_thw": Grid([[1, 2, 2] for _ in images])}


def sample(images=None):
    return common.Sample(id="test", subject="Accounting", subfield=None,
                         topic_difficulty=None, img_type=None, question_type="open",
                         question="Q", options=[], options_raw="[]", answer="A",
                         image_indices=list(images or []))


def test_prepare_inputs_and_counter_cache(monkeypatch, cfg):
    observed = []
    picture = SimpleNamespace(size=(64, 64))

    def process(messages, **kwargs):
        observed.append((messages, kwargs))
        return [picture], None, {"fps": [2]}

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=process))
    processor = Processor(cfg["image"]["image_patch_size"])
    item = sample([1])
    messages = common.build_messages(item, cfg, {1: picture})
    prepared, sizes = inputs.prepare_inputs(messages, processor)
    assert sizes == [(64, 64)]
    assert prepared == {"prompt": "rendered", "multi_modal_data": {"image": [picture]},
                        "mm_processor_kwargs": {"fps": [2]}}
    assert observed[0][1] == {"image_patch_size": cfg["image"]["image_patch_size"],
                              "return_video_kwargs": True, "return_video_metadata": True}
    assert processor.calls[0][2] == {"tokenize": False, "add_generation_prompt": True}
    counter = inputs.HFTokenCounter(Path("unused"), cfg, processor=processor)
    assert counter.count(item, "Q", messages, 0, cfg) == 3
    assert processor.calls[-1][1]["do_resize"] is False
    assert counter.pop_prepared(item.id) == prepared
    with pytest.raises(KeyError):
        counter.pop_prepared(item.id)


def test_no_image_and_patch_mismatch(monkeypatch, cfg):
    def process(messages, **kwargs):
        return None, None, {}

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=process))
    processor = Processor(cfg["image"]["image_patch_size"])
    item = sample()
    messages = common.build_messages(item, cfg, {})
    prepared, sizes = inputs.prepare_inputs(messages, processor)
    assert prepared["multi_modal_data"] == {}
    assert sizes == []
    counter = inputs.HFTokenCounter(Path("unused"), cfg, processor=processor)
    assert counter.count(item, "Q", messages, 0, cfg) == 3
    assert processor.calls[-1][1]["images"] is None
    row, errors, video_kwargs = preflight.inspect_sample(item, cfg, processor)
    assert not errors
    assert row["images"] == [] and row["total_tokens"] == 3
    assert video_kwargs == {}

    bad_processor = Processor(cfg["image"]["image_patch_size"] + 1)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda _: bad_processor)))
    with pytest.raises(ValueError, match="image patch size mismatch"):
        inputs.load_processor(Path("unused"), cfg)


def test_resolve_model_dir_uses_pinned_weight_free_snapshot(monkeypatch, tmp_path):
    assert inputs.resolve_model_dir(str(tmp_path), "revision") == tmp_path.resolve()
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    assert inputs.resolve_model_dir("repo/name", "pinned") == tmp_path.resolve()
    assert calls == [{"repo_id": "repo/name", "revision": "pinned",
                      "ignore_patterns": ["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.onnx"]}]


@pytest.mark.slow
def test_pinned_processor_and_accounting_sample():
    from preflight import inspect_sample

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    model_dir = inputs.resolve_model_dir(cfg["model"]["repo_id"], cfg["model"]["revision"])
    processor = inputs.load_processor(model_dir, cfg)
    from common import load_samples

    item = load_samples(cfg, os.getenv("PREFLIGHT_DATA_ROOT"), ["Accounting"])[0]
    row, errors, _ = inspect_sample(item, cfg, processor)
    assert not errors, (row["id"], errors)
    counter = inputs.HFTokenCounter(model_dir, cfg, processor=processor)
    messages = common.build_messages(item, cfg, item.images())
    assert counter.count(item, common.build_prompt(item, cfg), messages, 0, cfg) == row["total_tokens"]
    assert counter.pop_prepared(item.id)["multi_modal_data"]["image"]
