"""Read-only runtime and snapshot metadata for generation records."""

from __future__ import annotations

from importlib import metadata
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from threading import Event, Thread


PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "qwen-vl-utils", "openai",
            "flashinfer-python", "xformers", "datasets", "huggingface_hub", "pillow",
            "numpy", "pyyaml")
HASH_NAMES = {"tokenizer.json", "tokenizer_config.json", "config.json",
              "generation_config.json", "preprocessor_config.json", "chat_template.json",
              "chat_template.jinja", "vocab.json", "merges.txt", "vocab.txt"}


def package_versions(version=metadata.version) -> dict:
    result = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except metadata.PackageNotFoundError:
            result[name] = "unknown"
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_snapshot(model_dir: Path) -> dict:
    files = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        files[relative] = {"bytes": path.stat().st_size,
                           "blob_id": path.resolve().name if path.is_symlink() else None}
        if path.name in HASH_NAMES:
            files[relative]["sha256"] = sha256_file(path)
    return {"directory_name": model_dir.name, "files": files}


def gpu_info(run=subprocess.run, torch_module=None) -> dict:
    result = {"name": "unknown", "memory_total": "unknown", "driver": "unknown",
              "cuda_header": "unknown", "torch_device_name": "unknown", "bf16_supported": "unknown"}
    try:
        output = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True)
        first = output.stdout.splitlines()[0].split(", ")
        if len(first) >= 3:
            result.update(zip(("name", "memory_total", "driver"), first[:3]))
        header = run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True).stdout
        match = re.search(r"CUDA Version:\s*([\d.]+)", header)
        if match:
            result["cuda_header"] = match.group(1)
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    try:
        if torch_module is None:
            import torch as torch_module
        if torch_module.cuda.is_available():
            result["torch_device_name"] = torch_module.cuda.get_device_name(0)
            result["bf16_supported"] = torch_module.cuda.is_bf16_supported()
    except Exception:
        pass
    return result


def environment_info(cfg: dict, model_dir: Path, run=subprocess.run,
                     version=metadata.version, torch_module=None) -> dict:
    keys = [*cfg["environment"]["env_vars"], "HF_HOME", "HF_HUB_CACHE", "CUDA_VISIBLE_DEVICES"]
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": package_versions(version), "gpu": gpu_info(run, torch_module),
            "env_vars": {key: os.environ.get(key, "unset") for key in keys},
            "model_snapshot": model_snapshot(model_dir)}


class NvidiaSmiSampler:
    def __init__(self, interval: float, run=subprocess.run):
        self.interval = interval
        self.run = run
        self.stop_event = Event()
        self.thread = None
        self.samples = 0
        self.max_used_mib = None
        self.total_mib = None

    def _sample(self):
        try:
            output = self.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                               "--format=csv,noheader,nounits", "-i", "0"],
                              capture_output=True, text=True, timeout=10, check=True)
            used, total = (int(part.strip()) for part in output.stdout.splitlines()[0].split(",")[:2])
            self.samples += 1
            self.max_used_mib = used if self.max_used_mib is None else max(self.max_used_mib, used)
            self.total_mib = total
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass

    def _loop(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=11)
        return self.report()

    def report(self) -> dict:
        return {"max_used_mib": self.max_used_mib if self.max_used_mib is not None else "unknown",
                "total_mib": self.total_mib if self.total_mib is not None else "unknown",
                "samples": self.samples}
