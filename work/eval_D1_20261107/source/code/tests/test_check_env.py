import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
from packaging.requirements import Requirement


import check_env


def test_parse_requirements_comments_options_markers(tmp_path):
    requirements = tmp_path / "pins.txt"
    requirements.write_text(
        "# heading\n--extra-index-url https://example.invalid/simple\n"
        "torch==2.8.0 # compatible local build\n"
        "other==1.0; python_version < '2'\n"
        "tokenizers==0.22.2; python_version >= '3'\n",
        encoding="utf-8",
    )
    pins = check_env.parse_requirements(requirements)
    assert [pin["requirement"].name for pin in pins] == ["torch", "tokenizers"]


def test_version_local_tag_and_exact_comparison():
    assert check_env.version_satisfies(Requirement("torch==2.8.0"), "2.8.0+cu128")
    assert not check_env.version_satisfies(Requirement("torch==2.8.0+cu128"), "2.8.0+cpu")
    assert not check_env.version_satisfies(Requirement("torch==2.8.0"), None)


def test_options_and_markers_in_fake_dataset():
    dataset = Dataset.from_dict({
        "id": ["one", "two", "three"],
        "question": ["Look <image 1>", "Look <image 2>", "No marker"],
        "options": ["['A', '<image 2>']", "not a literal", "['A']"],
        "question_type": ["multiple-choice"] * 3,
        "answer": ["A"] * 3,
        "image_1": ["a", None, None],
        "image_2": ["b", None, "c"],
    })
    details = check_env.inspect_subject(dataset, "Example")
    assert details["options_literal_eval"] == {"success": 2, "failure": 1}
    assert details["parsed_option_counts"] == {2: 1, 1: 1}
    assert details["marker_comparison"] == {
        "match": 1, "marker_without_image": 1, "image_without_marker": 1
    }
    assert details["options_with_marker"] == 1
    categories, _, markers, images = check_env.marker_comparison(
        {"question": "<image 1>", "options": "[]", "image_1": None, "image_2": "present"},
        ["image_1", "image_2"],
    )
    assert categories == ["marker_without_image", "image_without_marker"]
    assert markers == {1} and images == {2}


def test_section_exception_does_not_stop_following_section():
    def broken(config, args):
        raise RuntimeError("intentional")

    def healthy(config, args):
        return [check_env.check("PASS", "healthy")]

    args = SimpleNamespace(sections=["broken", "healthy"])
    result = check_env.run_sections({}, args, {"broken": broken, "healthy": healthy})
    assert result["broken"][0]["status"] == "FAIL"
    assert "RuntimeError: intentional" in result["broken"][0]["traceback"]
    assert result["healthy"][0]["status"] == "PASS"


def test_qwen_smart_resize_fallback_and_missing(monkeypatch):
    def process_vision_info(messages, image_patch_size=None):
        return messages

    def smart_resize(height, width):
        return height, width

    module = SimpleNamespace(process_vision_info=process_vision_info)
    vision = SimpleNamespace(smart_resize=smart_resize, IMAGE_TOKEN_SIZE=42)
    monkeypatch.setattr(check_env, "installed_version", lambda name: "present")
    monkeypatch.setattr(check_env.importlib, "import_module", lambda name: vision if name.endswith("vision_process") else module)
    item = check_env.qwen_vl_utils_section({}, None)[0]
    assert item["status"] == "PASS"
    assert item["smart_resize"] == "(height, width)"
    assert item["constants"]["IMAGE_TOKEN_SIZE"] == 42
    del vision.smart_resize
    item = check_env.qwen_vl_utils_section({}, None)[0]
    assert item["status"] == "PASS" and item["smart_resize"] == "unknown"


def test_snapshot_dtype_falls_back_to_text_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"dtype": "bfloat16"}}))
    items = check_env.snapshot_details(tmp_path, {"model": {"dtype": "bfloat16"}})
    assert items[0]["status"] == "PASS"
    assert items[0]["dtype_source"] == "text_config.dtype"
