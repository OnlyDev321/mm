import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

import envinfo
import generate
import common
import inputs
import vllm_engine


SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return yaml.safe_load((SOURCE / "config.yaml").read_text())


def test_kwargs_and_fake_vllm_import_order(monkeypatch, cfg):
    for key in cfg["environment"]["env_vars"]:
        monkeypatch.setenv(key, "wrong")
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "fake_vllm"))
    sys.modules.pop("vllm", None)
    engine = vllm_engine.VLLMEngine(cfg, Path("model"))
    fake = sys.modules["vllm"]
    assert fake.ENV_AT_IMPORT == cfg["environment"]["env_vars"]
    assert fake.LAST_LLM_KWARGS == vllm_engine.llm_kwargs(cfg, Path("model"))
    assert fake.LAST_SAMPLING_KWARGS == vllm_engine.sampling_kwargs(cfg)
    assert "mm_processor_kwargs" not in fake.LAST_LLM_KWARGS
    assert all(value != "unknown" for value in engine.describe()["logs"].values())
    completion = lambda reason: SimpleNamespace(text=" raw  ", token_ids=(2, 3),
                                                 finish_reason=reason, stop_reason=None)
    engine.llm.generate = lambda inputs, sampling, use_tqdm: [
        SimpleNamespace(prompt=entry["prompt"], prompt_token_ids=[1, 2, 3],
                        outputs=[completion("stop" if index == 0 else "abort")])
        for index, entry in enumerate(inputs)]
    requests = [{"prepared": {"prompt": f"p{i}", "multi_modal_data": {},
                              "mm_processor_kwargs": {}}} for i in range(2)]
    outputs = engine.generate(requests)
    assert outputs[0] == {"raw_text": " raw  ", "output_token_ids": [2, 3],
                          "finish_reason": "stop", "stop_reason": None, "num_prompt_tokens": 3}
    assert isinstance(outputs[1], RuntimeError)
    greedy = copy.deepcopy(cfg)
    greedy["sampling"]["do_sample"] = False
    assert vllm_engine.sampling_kwargs(greedy)["temperature"] == 0
    sys.modules.pop("vllm", None)


def test_overrides_and_chunks(cfg, tmp_path):
    assert generate.chunk_plan(list(range(10)), 3, 4) == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
    assert generate.parse_engine_overrides(["max_num_seqs=4", "enforce_eager=true"]) == {
        "max_num_seqs": 4, "enforce_eager": True}
    for value in ("unknown=1", "max_num_seqs=oops", "max_num_seqs=1=max"):
        with pytest.raises(ValueError):
            generate.parse_engine_overrides([value])
    kwargs = vllm_engine.llm_kwargs(cfg, tmp_path, {"enforce_eager": True})
    assert kwargs["enforce_eager"] is True and "revision" not in kwargs


def test_sampler_and_snapshot(tmp_path):
    class Output:
        stdout = "12, 40\n"
    sampler = envinfo.NvidiaSmiSampler(1, run=lambda *a, **k: Output())
    sampler._sample()
    assert sampler.report() == {"max_used_mib": 12, "total_mib": 40, "samples": 1}
    def fail(*a, **k):
        raise OSError("missing")
    sampler.run = fail
    sampler._sample()
    assert sampler.report()["samples"] == 1
    blob = tmp_path / "blob"
    blob.write_bytes(b"data")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").symlink_to(blob)
    info = envinfo.model_snapshot(snapshot)
    assert info["files"]["config.json"]["blob_id"] == "blob"
    assert info["files"]["config.json"]["sha256"] == envinfo.sha256_file(blob)


def test_dry_imports_are_lazy():
    script = "import generate, vllm_engine; import sys; assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], cwd=SOURCE, check=True)


def test_real_path_resolves_once_and_shares_snapshot(monkeypatch, cfg, tmp_path):
    resolved = []
    received = []
    def resolve(model, revision, weights=False):
        resolved.append((model, revision, weights))
        return tmp_path
    class Counter:
        def __init__(self, path, configuration):
            received.append(("counter", path))
            self.prepared = {}
        def count(self, sample, prompt, messages, ordinal, configuration):
            self.prepared[sample.id] = {"prompt": prompt, "multi_modal_data": {},
                                        "mm_processor_kwargs": {}}
            return 1
        def pop_prepared(self, sample_id):
            return self.prepared.pop(sample_id)
    class Engine:
        def __init__(self, configuration, path, overrides):
            received.append(("engine", path))
        def generate(self, requests):
            return [{"raw_text": "x", "output_token_ids": [1], "finish_reason": "stop",
                     "stop_reason": None, "num_prompt_tokens": 1} for _ in requests]
        def describe(self):
            return {"logs": {}}
        def memory_report(self):
            return {}
    class Sampler:
        def __init__(self, interval):
            pass
        def start(self):
            pass
        def stop(self):
            return {}
    monkeypatch.setattr(inputs, "resolve_model_dir", resolve)
    monkeypatch.setattr(inputs, "HFTokenCounter", Counter)
    monkeypatch.setattr(vllm_engine, "VLLMEngine", Engine)
    monkeypatch.setattr(envinfo, "NvidiaSmiSampler", Sampler)
    monkeypatch.setattr(envinfo, "environment_info", lambda *a: {"packages": {}})
    item = common.Sample(id="one", subject="Test", subfield=None, topic_difficulty=None,
                         img_type=None, question_type="open", question="Q", options=[],
                         options_raw="[]", answer="A", image_indices=[])
    assert generate.run_pipeline(cfg, b"config", tmp_path / "out", [item], "repo", "rev", None,
                                 False) == 0
    assert resolved == [("repo", "rev", True)]
    assert received == [("counter", tmp_path), ("engine", tmp_path)]


def test_override_resume_rejected_and_mismatch_details(cfg, tmp_path):
    item = common.Sample(id="one", subject="Test", subfield=None, topic_difficulty=None,
                         img_type=None, question_type="open", question="Q", options=[],
                         options_raw="[]", answer="A", image_indices=[])
    out = tmp_path / "out"
    assert generate.run_pipeline(cfg, b"config", out, [item], "repo", "rev", None, True,
                                 engine_overrides={"max_num_seqs": 2}) == 0
    assert json.loads((out / cfg["run"]["env_filename"]).read_text())["engine_overrides"] == {
        "max_num_seqs": 2}
    with pytest.raises(ValueError, match="engine_overrides mismatch"):
        generate.run_pipeline(cfg, b"config", out, [item], "repo", "rev", None, True)
    item2 = common.Sample(id="two", subject="Test", subfield=None, topic_difficulty=None,
                          img_type=None, question_type="open", question="Q", options=[],
                          options_raw="[]", answer="A", image_indices=[])
    with pytest.raises(AssertionError, match=r"one.*difference min/max=1/1"):
        generate.run_pipeline(cfg, b"config", tmp_path / "mismatch", [item, item2],
                              "repo", "rev", None, True,
                              engine=generate.FakeEngine(mismatch_ordinal=0))
    assert (tmp_path / "mismatch" / cfg["run"]["raw_filename"]).read_text() == ""


def test_non_dry_setup_failures_finalize_invocation(monkeypatch, cfg, tmp_path):
    item = common.Sample("one", "Test", None, None, None, "open", "Q", [], "[]", "A", [])
    import inputs
    def resolve_failure(*args, **kwargs):
        raise RuntimeError("resolve failed")
    monkeypatch.setattr(inputs, "resolve_model_dir", resolve_failure)
    with pytest.raises(RuntimeError, match="resolve failed"):
        generate.run_pipeline(cfg, b"resolve", tmp_path / "resolve", [item], "repo", "rev", None, False)
    env = json.loads((tmp_path / "resolve" / cfg["run"]["env_filename"]).read_text())
    assert env["invocations"][0]["finished_at"] is not None
    def resolve_ok(*args, **kwargs):
        return tmp_path
    monkeypatch.setattr(inputs, "resolve_model_dir", resolve_ok)
    import vllm_engine
    class FailingEngine:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("engine failed")
    monkeypatch.setattr(vllm_engine, "VLLMEngine", FailingEngine)
    class Counter:
        pass
    with pytest.raises(RuntimeError, match="engine failed"):
        generate.run_pipeline(cfg, b"engine", tmp_path / "engine", [item], "repo", "rev", None, False,
                              counter=Counter())
    env = json.loads((tmp_path / "engine" / cfg["run"]["env_filename"]).read_text())
    assert env["invocations"][0]["finished_at"] is not None


class _ErrorEngine:
    def __init__(self, error_calls, error_call=1):
        self.calls = 0
        self.error_calls = set(error_calls)
        self.error_call = error_call
    def generate(self, requests):
        self.calls += 1
        result = []
        for index, request in enumerate(requests):
            if self.calls == self.error_call and index in self.error_calls:
                result.append(RuntimeError(f"synthetic error {self.calls}:{index}"))
            else:
                result.append({"raw_text": "x", "output_token_ids": [1], "finish_reason": "stop",
                               "stop_reason": None, "num_prompt_tokens": request["precomputed"]})
        return result


def test_chunk_error_gates(cfg, tmp_path):
    items = [common.Sample(str(i), "Test", None, None, None, "open", "Q", [], "[]", "A", [])
             for i in range(4)]
    cfg["run"]["first_chunk_size"] = 2
    cfg["run"]["chunk_size"] = 2
    with pytest.raises(AssertionError, match="error gate"):
        generate.run_pipeline(cfg, b"first", tmp_path / "first", items, "model", "rev", None, True,
                              engine=_ErrorEngine({0}), counter=generate.FakeTokenCounter())
    assert not (tmp_path / "first" / cfg["run"]["raw_filename"]).read_text()

    cfg["run"]["abort_error_fraction"] = 0.1
    cfg["run"]["first_chunk_size"] = 1
    with pytest.raises(AssertionError, match="error gate"):
        generate.run_pipeline(cfg, b"later", tmp_path / "later", items, "model", "rev", None, True,
                              engine=_ErrorEngine({0}, error_call=2), counter=generate.FakeTokenCounter())
    assert len((tmp_path / "later" / cfg["run"]["raw_filename"]).read_text().splitlines()) == 1

    cfg["run"]["abort_error_fraction"] = 0.5
    assert generate.run_pipeline(cfg, b"under", tmp_path / "under", items, "model", "rev", None, True,
                                 engine=_ErrorEngine({0}, error_call=2), counter=generate.FakeTokenCounter()) == 2


@pytest.mark.slow
def test_vllm_source_contract(cfg):
    root_text = os.environ.get("VLLM_SRC")
    if not root_text:
        pytest.skip("VLLM_SRC unset")
    root = Path(root_text) / "vllm"
    llm = ast.parse((root / "entrypoints/llm.py").read_text())
    args = ast.parse((root / "engine/arg_utils.py").read_text())
    sampling = ast.parse((root / "sampling_params.py").read_text())
    def cls(tree, name):
        return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    llm_class = cls(llm, "LLM")
    init = next(node for node in llm_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    generate_fn = next(node for node in llm_class.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    llm_keys = {arg.arg for arg in init.args.args + init.args.kwonlyargs}
    engine_fields = {node.target.id for node in cls(args, "EngineArgs").body
                     if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert vllm_engine.llm_kwargs(cfg, "model").keys() <= llm_keys | engine_fields
    assert "use_tqdm" in {arg.arg for arg in generate_fn.args.args + generate_fn.args.kwonlyargs}
    sampling_fields = {node.target.id for node in cls(sampling, "SamplingParams").body
                       if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert vllm_engine.sampling_kwargs(cfg).keys() <= sampling_fields
    source = "\n".join(path.read_text(errors="replace") for path in root.rglob("*.py"))
    assert all(marker in source for marker in vllm_engine.LOG_MARKERS)
