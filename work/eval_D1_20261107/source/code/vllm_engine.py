"""Lazy vLLM adapter for pinned, locally resolved model snapshots."""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any


OVERRIDE_KEYS = frozenset({"gpu_memory_utilization", "max_num_batched_tokens",
                           "max_num_seqs", "mm_processor_kwargs", "enforce_eager"})
LOG_MARKERS = ("Model loading took", "Available KV cache memory",
               "GPU KV cache size", "Maximum concurrency for")


def sampling_kwargs(cfg: dict) -> dict:
    sampling = cfg["sampling"]
    return {"temperature": sampling["temperature"] if sampling["do_sample"] else 0,
            "top_p": sampling["top_p"], "top_k": sampling["top_k"],
            "repetition_penalty": sampling["repetition_penalty"],
            "presence_penalty": sampling["presence_penalty"],
            "seed": sampling["seed"], "max_tokens": cfg["budget"]["max_new_tokens"],
            "skip_special_tokens": sampling["skip_special_tokens"],
            "stop_token_ids": sampling["stop_token_ids"]}


def llm_kwargs(cfg: dict, model_dir, overrides: dict | None = None) -> dict:
    engine = cfg["engine"]
    result = {"model": str(model_dir), "dtype": cfg["model"]["dtype"],
              "seed": cfg["sampling"]["engine_seed"],
              "gpu_memory_utilization": engine["gpu_memory_utilization"],
              "max_model_len": cfg["budget"]["max_model_len"],
              "limit_mm_per_prompt": engine["limit_mm_per_prompt"],
              "trust_remote_code": engine["trust_remote_code"],
              "max_num_batched_tokens": engine["max_num_batched_tokens"]}
    for key in ("max_num_seqs", "mm_processor_kwargs"):
        if engine.get(key) is not None:
            result[key] = engine[key]
    result.update(overrides or {})
    if result.get("mm_processor_kwargs") is None:
        result.pop("mm_processor_kwargs", None)
    return result


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = {key: "unknown" for key in LOG_MARKERS}

    def emit(self, record):
        message = record.getMessage()
        for key in LOG_MARKERS:
            if key in message:
                self.lines[key] = message


class VLLMEngine:
    def __init__(self, cfg: dict, model_dir, overrides: dict | None = None):
        self.cfg = cfg
        self.kwargs = llm_kwargs(cfg, model_dir, overrides)
        for key, value in cfg["environment"]["env_vars"].items():
            old = os.environ.get(key)
            if old is not None and old != value:
                print(f"WARNING: replacing {key}={old!r} with {value!r}")
            os.environ[key] = value
        self.capture = _Capture()
        logger = logging.getLogger("vllm")
        old_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(self.capture)
        try:
            from vllm import LLM, SamplingParams
            self.sampling_params = SamplingParams(**sampling_kwargs(cfg))
            started = time.monotonic()
            self.llm = LLM(**self.kwargs)
            self.init_seconds = round(time.monotonic() - started, 3)
        finally:
            logger.removeHandler(self.capture)
            logger.setLevel(old_level)

    def generate(self, requests: list[dict]) -> list[dict | Exception]:
        if not requests:
            return []
        inputs = [request["prepared"] for request in requests]
        outputs = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
        if len(outputs) != len(requests):
            raise AssertionError(f"vLLM returned {len(outputs)} outputs for {len(requests)} requests")
        mapped = []
        for position, output in enumerate(outputs):
            try:
                completions = getattr(output, "outputs", None)
                if completions is None:
                    raise RuntimeError(f"missing outputs at position {position}")
                if len(completions) != 1:
                    raise RuntimeError(f"expected one completion at position {position}, got {len(completions)}")
                completion = completions[0]
                if completion.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"unexpected finish_reason {completion.finish_reason!r} at position {position}; "
                                       f"outputs={len(completions)}")
                mapped.append({"raw_text": completion.text,
                               "output_token_ids": list(completion.token_ids),
                               "finish_reason": completion.finish_reason,
                               "stop_reason": completion.stop_reason,
                               "num_prompt_tokens": len(output.prompt_token_ids)})
            except Exception as exc:
                exc.vllm_traceback = traceback.format_exc()
                mapped.append(exc)
        return mapped

    def describe(self) -> dict:
        try:
            config_repr = repr(self.llm.llm_engine.vllm_config)
        except Exception:
            config_repr = "unknown"
        return {"llm_kwargs": self.kwargs, "sampling_params": repr(self.sampling_params),
                "logs": self.capture.lines, "init_seconds": self.init_seconds,
                "vllm_config": config_repr}

    def memory_report(self) -> dict:
        try:
            import torch
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            free, total = torch.cuda.mem_get_info()
            return {"max_allocated_gib": allocated, "max_reserved_gib": reserved,
                    "free_gib": free / 2**30, "total_gib": total / 2**30}
        except Exception:
            return {"max_allocated_gib": "unknown", "max_reserved_gib": "unknown",
                    "free_gib": "unknown", "total_gib": "unknown"}
