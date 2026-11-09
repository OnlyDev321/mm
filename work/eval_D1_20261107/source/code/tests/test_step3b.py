import json
import sys
import types
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import common
import check_env
from gate_env_check import failures
from validate_raw import validate


SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return yaml.safe_load((SOURCE / "config.yaml").read_text())


def test_gate_fatal_sections_and_package_checks():
    report = {"sections": {
        "vllm_import": [{"name": "import", "status": "FAIL"}],
        "packages": [{"name": "pip_check", "status": "FAIL"},
                     {"name": "expected_version", "status": "FAIL"}],
        "gpu": [{"name": "available", "status": "SKIP"}],
    }}
    assert failures(report) == ["vllm_import.import", "packages.expected_version"]


def test_gate_cli(tmp_path):
    path = tmp_path / "env_check.json"
    path.write_text(json.dumps({"sections": {"model": [{"name": "x", "status": "FAIL"}]}}))
    result = subprocess.run([sys.executable, str(SOURCE / "gate_env_check.py"), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "model.x" in result.stdout


def record(identifier, cfg, subject="Accounting", status="ok"):
    sample = common.Sample(identifier, subject, None, None, None, "open", "Q", [], "[]", "A", [])
    return common.make_record(sample, cfg, synthetic=True,
                              status=status, reason="x" if status == "skip" else None,
                              error="bad" if status == "error" else None,
                              precomputed_prompt_tokens=1 if status == "ok" else None,
                              num_prompt_tokens=1 if status == "ok" else None,
                              raw_text="x" if status == "ok" else None,
                              output_token_ids=[1] if status == "ok" else None,
                              output_tokens=1 if status == "ok" else None,
                              finish_reason="stop" if status == "ok" else None)


def test_validate_raw_status_and_damage(tmp_path):
    cfg = yaml.safe_load((SOURCE / "config.yaml").read_text())
    cfg["dataset"]["expected_total_rows"] = 2
    cfg["dataset"]["subjects"] = ["Accounting"]
    cfg["dataset"]["expected_rows_per_subject"] = 2
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in [record("a", cfg), record("b", cfg, status="skip")]) + "\n")
    code, summary = validate(path, cfg, 2)
    assert code == 2 and summary["skip"] == 1
    path.write_text(path.read_text() + "{broken")
    code, summary = validate(path, cfg, 2)
    assert code == 1 and summary["errors"]


def test_runner_failure_records_stage(tmp_path):
    result = subprocess.run([str(SOURCE / "run_generate.sh"), "--out", str(tmp_path / "bad"),
                             "--skip_env_gate", "--model_path", str(tmp_path / "missing")],
                            cwd=SOURCE, capture_output=True, text=True)
    assert result.returncode != 0
    failed = tmp_path / "bad" / "logs" / "FAILED"
    assert failed.exists() and "stage=" in failed.read_text()


def test_runner_missing_hf_exits_four(tmp_path):
    result = subprocess.run([str(SOURCE / "run_generate.sh"), "--out", str(tmp_path / "no-hf")],
                            env={key: value for key, value in __import__("os").environ.items() if key != "HF_HOME"},
                            capture_output=True, text=True)
    assert result.returncode == 4
    assert "export HF_HOME=..." in result.stdout
    assert "stage=environment export" in (tmp_path / "no-hf/logs/FAILED").read_text()


def test_runner_missing_hf_directory_is_created(tmp_path):
    cache = Path("/tmp/preflight-cache")
    if not cache.exists():
        pytest.skip("preflight cache unavailable")
    hf = tmp_path / "new-hf"
    out = tmp_path / "new-out"
    result = subprocess.run([str(SOURCE / "run_generate.sh"), "--skip_env_gate", "--model_path",
                             str(tmp_path / "missing-model"), "--data_root", str(cache), "--out", str(out),
                             "--subjects", "Math", "--limit", "1"],
                            env={**__import__("os").environ, "HF_HOME": str(hf),
                                 "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
                            capture_output=True, text=True)
    assert result.returncode != 0 and hf.is_dir()
    assert "stage=generation" in (out / "logs/FAILED").read_text()


def test_runner_disk_gate_exit_five(tmp_path):
    config = yaml.safe_load((SOURCE / "config.yaml").read_text())
    config["run"]["min_free_disk_gb"] = 999999
    config_path = tmp_path / "high.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    hf = tmp_path / "hf"
    result = subprocess.run([str(SOURCE / "run_generate.sh"), "--skip_env_gate", "--config",
                             str(config_path), "--out", str(tmp_path / "disk")],
                            env={**__import__("os").environ, "HF_HOME": str(hf)},
                            capture_output=True, text=True)
    assert result.returncode == 5 and "insufficient free disk space" in result.stdout


def test_empty_requirements_install_fails(tmp_path):
    # The repository requirements file is nonempty; exercise the same branch with
    # an isolated temporary script directory and a copied empty requirements file.
    script = tmp_path / "run.sh"
    script.write_text((SOURCE / "run_generate.sh").read_text().replace(
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        f'SCRIPT_DIR="{tmp_path}"'))
    script.chmod(0o755)
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "config.yaml").write_text((SOURCE / "config.yaml").read_text())
    result = subprocess.run([str(script), "--out", str(tmp_path / "out"), "--install"],
                            capture_output=True, text=True)
    assert result.returncode == 3
    assert "requirements.txt is empty" in (tmp_path / "out/logs/01_install.log").read_text()


def test_runner_hardening_contract():
    source = (SOURCE / "run_generate.sh").read_text()
    assert source.index('if (( INSTALL )); then') < source.index('STAGE="environment export"')
    assert 'exit 4' in source and 'export HF_HOME=...' in source
    assert 'externally-managed-environment' in source and '--break-system-packages' in source
    assert 'import hf_transfer' in source and 'HF_HUB_ENABLE_HF_TRANSFER=1' in source
    assert 'min_free_disk_gb' in source and 'exit 5' in source
    assert 'stop command: runpodctl stop pod' in source


def test_gpu_section_driver_parse_and_gate(monkeypatch, cfg):
    def command(argv, timeout, env=None):
        if any(arg.startswith("--query-gpu") for arg in argv):
            return {"returncode": 0, "stdout": "NVIDIA GeForce RTX 4090, 24564 MiB, 1 MiB, 570.211.01\n",
                    "stderr": "", "elapsed_seconds": 0}
        return {"returncode": 0, "stdout": "CUDA Version: 12.8\n", "stderr": "", "elapsed_seconds": 0}
    monkeypatch.setattr(check_env, "run_command", command)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda _: "fake", is_bf16_supported=lambda: True),
        version=types.SimpleNamespace(cuda="12.8")))
    checks = check_env.gpu_section(cfg, types.SimpleNamespace())
    driver = next(item for item in checks if item["name"] == "gpu_driver")
    assert driver["status"] == "PASS" and driver["gpu_name"].startswith("NVIDIA")
    assert not __import__("gate_env_check").failures({"sections": {"gpu": checks}})

    def old_driver(argv, timeout, env=None):
        result = command(argv, timeout, env)
        if any(arg.startswith("--query-gpu") for arg in argv):
            result["stdout"] = "NVIDIA GeForce RTX 4090, 24564 MiB, 1 MiB, 560.10.01\n"
        return result
    monkeypatch.setattr(check_env, "run_command", old_driver)
    checks = check_env.gpu_section(cfg, types.SimpleNamespace())
    assert any(item["name"] == "gpu_driver" and item["status"] == "FAIL" for item in checks)
    assert "gpu.gpu_driver" in __import__("gate_env_check").failures({"sections": {"gpu": checks}})

    def malformed(argv, timeout, env=None):
        result = command(argv, timeout, env)
        if any(arg.startswith("--query-gpu") for arg in argv):
            result["stdout"] = "not-a-gpu-row\n"
        return result
    monkeypatch.setattr(check_env, "run_command", malformed)
    broken_torch = types.ModuleType("torch")
    broken_torch.cuda = types.SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("no torch")))
    monkeypatch.setitem(sys.modules, "torch", broken_torch)
    checks = check_env.gpu_section(cfg, types.SimpleNamespace())
    assert any(item["name"] == "gpu_query_parse" and item["status"] == "WARN" for item in checks)
    assert any(item["name"] == "torch_cuda" and item["status"] == "WARN" for item in checks)


def test_envinfo_gpu_info_injected_commands():
    calls = []
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
    def run(argv, **kwargs):
        calls.append(argv)
        if "--query-gpu=name,memory.total,driver_version" in argv:
            return Result("NVIDIA GeForce RTX 4090, 24564 MiB, 570.211.01\n")
        return Result("CUDA Version: 12.8\n")
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda _: "NVIDIA GeForce RTX 4090",
        is_bf16_supported=lambda: True))
    result = __import__("envinfo").gpu_info(run=run, torch_module=torch)
    assert result["name"] == "NVIDIA GeForce RTX 4090"
    assert result["cuda_header"] == "12.8" and result["bf16_supported"] is True


def _runner_env(tmp_path, extra=None):
    return {**__import__("os").environ, "HF_HOME": str(tmp_path / "hf"),
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", **(extra or {})}


def test_runner_pip_retry_behavior(tmp_path):
    fake = tmp_path / "bin"; fake.mkdir()
    wrapper = fake / "python"
    wrapper.write_text("""#!/usr/bin/env bash
if [[ \"$1\" == \"-m\" && \"$2\" == \"pip\" && \"$3\" == \"install\" && \" $* \" != *\"--break-system-packages\"* ]]; then echo externally-managed-environment; exit 1; fi
if [[ \"$1\" == \"-m\" && \"$2\" == \"pip\" && \"$3\" == \"install\" && \" $* \" == *\"--break-system-packages\"* ]]; then echo fake-install; exit 0; fi
exec /tmp/preflight-venv/bin/python \"$@\"
""")
    wrapper.chmod(0o755)
    out = tmp_path / "out"
    result = __import__("subprocess").run([str(SOURCE / "run_generate.sh"), "--skip_env_gate", "--install",
                                           "--data_root", "/tmp/preflight-cache", "--out", str(out),
                                           "--subjects", "Math", "--limit", "0"],
                                          env={**_runner_env(tmp_path), "PATH": f"{fake}:/tmp/preflight-venv/bin:" + __import__("os").environ["PATH"]},
                                          capture_output=True, text=True)
    log = (out / "logs/01_install.log").read_text()
    assert result.returncode == 0 and "retrying with --break-system-packages" in log


def test_runner_stop_pod_on_failure_preserves_code(tmp_path):
    fake = tmp_path / "bin"; fake.mkdir()
    ctl = fake / "runpodctl"
    ctl.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$RUNPOD_LOG\"\n")
    ctl.chmod(0o755)
    out = tmp_path / "out"; runpod_log = tmp_path / "runpod.args"
    result = __import__("subprocess").run([str(SOURCE / "run_generate.sh"), "--skip_env_gate",
                                           "--stop_pod_when_done", "--model_path", str(tmp_path / "missing"),
                                           "--data_root", "/tmp/preflight-cache", "--out", str(out),
                                           "--subjects", "Math", "--limit", "1"],
                                          env={**_runner_env(tmp_path), "RUNPOD_POD_ID": "pod-test",
                                               "RUNPOD_LOG": str(runpod_log),
                                               "PATH": f"{fake}:/tmp/preflight-venv/bin:" + __import__("os").environ["PATH"]},
                                          capture_output=True, text=True)
    assert result.returncode != 0 and runpod_log.read_text().strip() == "stop pod pod-test"
    assert "stop command: runpodctl stop pod pod-test" in (out / "logs/stop_pod.log").read_text()


@pytest.mark.parametrize("available", [True, False])
def test_runner_hf_transfer_export_log(tmp_path, available):
    fake = tmp_path / "bin"; fake.mkdir()
    wrapper = fake / "python"
    probe = "exit 0" if available else "exit 1"
    wrapper.write_text(f"""#!/usr/bin/env bash
if [[ \"$1\" == \"-c\" && \"$2\" == *\"import hf_transfer\"* ]]; then {probe}; fi
exec /tmp/preflight-venv/bin/python \"$@\"
""")
    wrapper.chmod(0o755)
    out = tmp_path / "out"
    __import__("subprocess").run([str(SOURCE / "run_generate.sh"), "--skip_env_gate", "--data_root",
                                  "/tmp/preflight-cache", "--out", str(out), "--subjects", "Math", "--limit", "0"],
                                 env={**_runner_env(tmp_path), "PATH": f"{fake}:/tmp/preflight-venv/bin:" + __import__("os").environ["PATH"]},
                                 capture_output=True, text=True)
    log = (out / "logs/00_system.log").read_text()
    assert ("HF transfer: enabled" in log) is available
