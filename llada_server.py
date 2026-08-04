#!/usr/bin/env python3
"""Minimal LAN server for LLaDA masked infilling.

Run on the GPU desktop, not on the unredact/evaluator machine:

  python llada_server.py --model GSAI-ML/LLaDA-8B-Base --host 0.0.0.0 --port 8000

The server deliberately accepts only visible context and generation settings.
It never receives ground_truth, paper_type, IDs, or evaluator metadata.

This version intentionally has no API-key authentication. Bind it only to a
trusted private LAN and do not port-forward port 8000 to the internet.

Endpoint summary:
  GET  /health
  POST /generate

POST /generate request:
  {
    "before": "visible text immediately before the gap",
    "after": "visible text immediately after the gap",
    "min_chars": 10,
    "max_chars": 18,
    "candidates": 4,
    "steps": 64,
    "temperature": 0.0,
    "remasking": "low_confidence"
  }

Response:
  {
    "candidates": ["...", "..."],
    "model": "GSAI-ML/LLaDA-8B-Base",
    "device": "cuda",
    "attempts": 4,
    "returned_candidates": 2,
    "in_range_candidates": 1,
    "gen_length": 12,
    "block_length": 12,
    "steps": 64,
    "elapsed_ms": 1234.5
  }

This is intentionally a small, explicit API rather than an OpenAI-compatible
server: LLaDA's native generation is iterative masked diffusion, not ordinary
left-to-right completion.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Base"


class GenerateRequest(BaseModel):
    before: str = Field(min_length=1, max_length=12000)
    after: str = Field(min_length=1, max_length=12000)
    min_chars: int = Field(ge=1, le=2000)
    max_chars: int = Field(ge=1, le=2000)
    candidates: int = Field(default=4, ge=1, le=16)
    steps: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    remasking: str = Field(default="low_confidence")
    gen_length: int | None = Field(default=None, ge=1, le=256)
    block_length: int | None = Field(default=None, ge=1, le=256)

    @field_validator("max_chars")
    @classmethod
    def max_at_least_min(cls, value: int, info: Any) -> int:
        minimum = info.data.get("min_chars")
        if minimum is not None and value < minimum:
            raise ValueError("max_chars must be >= min_chars")
        return value

    @field_validator("remasking")
    @classmethod
    def valid_remasking(cls, value: str) -> str:
        if value not in {"low_confidence", "random"}:
            raise ValueError("remasking must be 'low_confidence' or 'random'")
        return value


class Server:
    def __init__(self, model_id: str, load_in_4bit: bool = False):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_4bit = load_in_4bit
        self.generation_lock = threading.Lock()
        print(f"Loading {model_id} on {self.device} ...", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        if load_in_4bit:
            # Do not pass torch_dtype alongside this custom quantized load.
            # Older/custom LLaDA loading paths can make Transformers call
            # model.to(dtype) inside from_pretrained, which bitsandbytes rejects.
            load_kwargs.pop("torch_dtype", None)
            # Experimental on native Windows: this requires a compatible
            # bitsandbytes build. The explicit config is preferred by current
            # Transformers releases over the older load_in_4bit kwarg.
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
        # Quantized bitsandbytes models are already dispatched and cast by
        # Transformers/Accelerate. Calling .to(cuda) on them raises:
        # "`.to` is not supported for 4-bit and 8-bit bitsandbytes models".
        # Detect both our flag and the model's own marker for safety.
        model_is_quantized = bool(
            load_in_4bit
            or getattr(self.model, "is_quantized", False)
            or getattr(self.model, "is_loaded_in_4bit", False)
            or getattr(self.model, "is_loaded_in_8bit", False)
        )
        if not model_is_quantized:
            self.model = self.model.to(self.device)
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        print("Model ready.", flush=True)

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return logits
        # Match the official LLaDA sampler's higher precision noise path.
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel

    @staticmethod
    def transfer_schedule(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        schedule = torch.zeros(
            mask_num.size(0), steps,
            device=mask_index.device,
            dtype=torch.int64,
        ) + base
        for row in range(mask_num.size(0)):
            schedule[row, :remainder[row]] += 1
        return schedule

    @torch.no_grad()
    def sample_once(
        self,
        before: str,
        after: str,
        gen_length: int,
        steps: int,
        block_length: int,
        temperature: float,
        remasking: str,
    ) -> str:
        # The server masks exactly the generated middle region. The visible
        # prefix/suffix remain fixed context; no ground truth enters here.
        # Tokenize each side without special tokens, then add one BOS token if
        # the checkpoint defines one. This avoids duplicate BOS/EOS tokens at
        # the artificial prefix/suffix join.
        prefix_ids = self.tokenizer(
            before, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        suffix_ids = self.tokenizer(
            after, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is not None:
            bos = torch.tensor([[bos_id]], dtype=torch.long)
            prefix_ids = torch.cat([bos, prefix_ids], dim=1)
        prefix_ids = prefix_ids.to(self.model_device)
        suffix_ids = suffix_ids.to(self.model_device)

        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None and "llada" in self.model_id.lower():
            # GSAI-ML's official LLaDA sampler passes this checkpoint-specific
            # mask ID explicitly; the custom tokenizer does not always expose
            # it through the generic Hugging Face mask_token_id property.
            mask_id = 126336
        if mask_id is None:
            raise RuntimeError(
                "This tokenizer does not expose mask_token_id and no known "
                "checkpoint-specific mask ID is available"
            )

        x = torch.full(
            (1, prefix_ids.shape[1] + gen_length + suffix_ids.shape[1],),
            mask_id,
            dtype=torch.long,
            device=self.model_device,
        )
        x[:, :prefix_ids.shape[1]] = prefix_ids
        suffix_start = prefix_ids.shape[1] + gen_length
        x[:, suffix_start:] = suffix_ids

        attention_mask = torch.ones_like(x)
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and x.shape[1] > context_limit:
            raise ValueError(
                f"tokenized context ({x.shape[1]}) exceeds model limit "
                f"({context_limit}); send shorter before/after text"
            )

        if gen_length % block_length != 0:
            block_length = gen_length
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            steps = num_blocks * max(1, steps // num_blocks)
        steps_per_block = max(1, steps // num_blocks)
        effective_steps = steps_per_block * num_blocks

        for block in range(num_blocks):
            block_start = prefix_ids.shape[1] + block * block_length
            block_end = block_start + block_length
            block_mask = x[:, block_start:block_end] == mask_id
            schedule = self.transfer_schedule(block_mask, steps_per_block)

            for step in range(steps_per_block):
                mask_index = x == mask_id
                candidate_mask = mask_index.clone()
                candidate_mask[:, :block_start] = False
                candidate_mask[:, block_end:] = False

                logits = self.model(x, attention_mask=attention_mask).logits
                # Never select special structural tokens into the generated
                # span. The exact IDs vary by tokenizer, so suppress the IDs
                # exposed by this checkpoint rather than assuming EOS=1.
                logits[:, :, mask_id] = -torch.inf
                eos_ids = self.tokenizer.eos_token_id
                if eos_ids is not None:
                    if not isinstance(eos_ids, (list, tuple)):
                        eos_ids = [eos_ids]
                    for eos_id in eos_ids:
                        logits[:, :, eos_id] = -torch.inf
                logits_with_noise = self.add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                x0 = torch.where(mask_index, x0, x)

                if remasking == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    confidence = torch.gather(
                        probs, dim=-1, index=x0.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    confidence = torch.rand_like(x0, dtype=torch.float32)

                confidence[~candidate_mask] = -torch.inf
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for row in range(x.shape[0]):
                    count = int(schedule[row, step].item())
                    available = int(candidate_mask[row].sum().item())
                    count = min(count, available)
                    if count:
                        _, selected = torch.topk(confidence[row], k=count)
                        transfer_index[row, selected] = True
                x[transfer_index] = x0[transfer_index]

        middle = x[0, prefix_ids.shape[1]:suffix_start]
        return self.tokenizer.decode(middle, skip_special_tokens=True).strip()

    def generate(self, req: GenerateRequest) -> tuple[list[str], dict[str, Any]]:
        # Character bounds become a conservative token-length estimate. We
        # over-allocate slightly, then the evaluator performs authoritative
        # character verification on the returned strings.
        gen_length = req.gen_length or max(
            4, min(128, (req.max_chars + 2) // 3 + 4)
        )
        block_length = req.block_length or gen_length
        effective_steps = req.steps
        started = time.perf_counter()
        candidates: list[str] = []
        attempts = 0
        with self.generation_lock:
            for _ in range(req.candidates):
                attempts += 1
                text = self.sample_once(
                    req.before,
                    req.after,
                    gen_length,
                    req.steps,
                    block_length,
                    req.temperature,
                    req.remasking,
                )
                if text and text not in candidates:
                    candidates.append(text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return candidates, {
            "model": self.model_id,
            "device": self.device,
            "attempts": attempts,
            "returned_candidates": len(candidates),
            "in_range_candidates": sum(
                req.min_chars <= len(c) <= req.max_chars for c in candidates
            ),
            "gen_length": gen_length,
            "block_length": block_length,
            "steps": req.steps,
            "elapsed_ms": round(elapsed_ms, 1),
            "effective_steps": effective_steps,
            "note": "returned candidates still require client-side character-range verification",
        }


def make_app(server: Server) -> FastAPI:
    app = FastAPI(title="LLaDA masked infill server", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model": server.model_id,
            "device": server.device,
        }

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict[str, Any]:
        try:
            candidates, metadata = server.generate(req)
            return {"candidates": candidates, **metadata}
        except (RuntimeError, ValueError) as exc:
            # Surface CUDA OOM and model/runtime failures as a clean HTTP error
            # rather than leaving the client waiting for a malformed response.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    print(
        "WARNING: API-key authentication is disabled. Keep port "
        f"{args.port} on a trusted private LAN; do not port-forward it.",
        flush=True,
    )
    server = Server(args.model, load_in_4bit=args.load_in_4bit)
    app = make_app(server)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
