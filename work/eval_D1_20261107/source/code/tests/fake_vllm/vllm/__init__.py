"""CPU-only vLLM shape used by unit tests and the processor rehearsal."""

import hashlib
import logging
import os
from types import SimpleNamespace


ENV_AT_IMPORT = {key: os.environ.get(key) for key in (
    "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_WORKER_MULTIPROC_METHOD",
    "TOKENIZERS_PARALLELISM", "VLLM_USE_FLASHINFER_SAMPLER")}
LAST_LLM_KWARGS = None
LAST_SAMPLING_KWARGS = None


class SamplingParams:
    def __init__(self, **kwargs):
        global LAST_SAMPLING_KWARGS
        LAST_SAMPLING_KWARGS = kwargs
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"FakeSamplingParams({self.__dict__!r})"


class LLM:
    def __init__(self, **kwargs):
        global LAST_LLM_KWARGS
        LAST_LLM_KWARGS = kwargs
        self.kwargs = kwargs
        self.llm_engine = SimpleNamespace(vllm_config={"fake": True})
        self.processor = None
        logging.getLogger("vllm").info("Model loading took 0.0000 GiB and 0.000001 seconds")
        logging.getLogger("vllm").info("Available KV cache memory: 0.00 GiB")
        logging.getLogger("vllm").info("GPU KV cache size: 0 tokens")
        logging.getLogger("vllm").info("Maximum concurrency for 16384 tokens per request: 0.00x")

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.kwargs["model"])
        results = []
        for entry in prompts:
            images = entry["multi_modal_data"].get("image")
            encoded = self.processor(text=[entry["prompt"]], images=images, do_resize=False)
            ids = encoded["input_ids"][0]
            digest = hashlib.sha256(entry["prompt"].encode()).digest()
            length = digest[0] % 7 == 0
            tokens = [int(digest[1])] * sampling_params.max_tokens if length else [int(digest[1])]
            completion = SimpleNamespace(text=f"fake:{digest.hex()[:12]}  ", token_ids=tokens,
                                         finish_reason="length" if length else "stop", stop_reason=None)
            results.append(SimpleNamespace(prompt=f"<expanded>{entry['prompt']}</expanded>", prompt_token_ids=ids,
                                           outputs=[completion]))
        return results
