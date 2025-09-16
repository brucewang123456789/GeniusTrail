# Skotergy — Latency-First LLM Inference Engine
> **Creator:** Libo Wang — solo, independent R&D
Single-device decode closure · Request-level multi-GPU routing · Triton-accelerated kernels · Model-agnostic packaging

[![PyPI - Version](https://img.shields.io/pypi/v/skotergy.svg)](https://pypi.org/project/skotergy/)
**PyPI (1.0.1)**: https://pypi.org/project/skotergy/1.0.1/

## Install
```bash
pip install skotergy

## Abstract
Skotergy is an inference engine engineered to minimize end-to-end latency without sacrificing scale-out throughput. A request executes entirely on a single device across prefill and decode—KV cache, logits, and fused ops are colocated—so step-internal collectives are removed from the critical path. Throughput scales by dispatching independent requests to per-device replicas (request-level multi-GPU), which preserves the single-request latency envelope. Kernel patches in Triton fuse normalization and strictly enforce device-resident, contiguous tensors. The system ships without model weights and remains agnostic to the user’s model distribution and tooling.

---

## Measured Outcomes (same host · same model · same dtype)
| Scenario | Metric | Baseline (CUDA/Transformers) | Skotergy | Delta |
|---|---:|---:|---:|---:|
| Single GPU · Llama-3.1-8B-Instruct (bf16) | p50 latency | 6093.2 ms | **4689.4 ms** | **−23.04%** |
|  | p95 latency | 11218.1 ms | **8580.4 ms** | **−23.51%** |
| Two GPUs · request-level parallel | 16 × 64 new tokens TOTAL | — | **≈ 3.02–3.12 s** | — |
|  | Aggregate throughput | — | **≈ 330–340 tok/s** | — |
|  | Errors | 0 | **0** | — |

Interpretation: single-device closure lowers median and tail; routing across replicas lifts tokens/s while keeping single-request latency single-GPU-like.

---

## System Architecture
### Decode Closure (single device)
• Prefill and token-by-token decode remain on one accelerator; KV cache is persistent and local.  
• The engine avoids per-step cross-device collectives (e.g., attention/MLP all-reduce), eliminating synchronization spikes and allocator churn.  
• Generation enforces `use_cache=True`; attention is explicit with left-padding and a guaranteed `pad_token_id` to stabilize shapes.  
• Model loading prefers the modern `dtype` keyword and automatically falls back to `torch_dtype` on older stacks.  
• Embeddings are resized only when a pad token must be introduced, maintaining tokenizer-model consistency.

### Request-Level Multi-GPU Routing
• One replica per visible device; each replica holds its own queue and inflight cap.  
• Length-aware dispatch chooses the replica whose recent average prompt length is closest to the incoming request, reducing over-padding and graph churn.  
• An optional micro-batch window consolidates arrivals into denser launches; default is disabled to keep p50 tight.  
• Per-replica concurrency defaults to 1 to avoid stream contention that inflates tail latency.

### Triton Kernel Patches (CUDA path)
• Fused RMSNorm implemented in Triton; launches require contiguous, device-resident pointers and reject CPU tensors at dispatch.  
• Grid/block sizing derives from hidden dimension to avoid under- or over-occupancy; mixed precision is explicit at the API boundary.  
• Patches auto-disable on non-CUDA backends; framework ops are used as safe fallbacks with identical semantics.

### Stability & Determinism Choices
• TF32 is enabled for residual FP32 matmuls to free headroom without observable decode-path accuracy loss in typical use.  
• cuDNN autotune is disabled to keep latency stable under variable shapes.  
• Float32 matmul precision is pinned to “high” to avoid unintentional regression on some builds.

---

## Why It Beats the Baseline
• Eliminates step-internal communication: the token loop does not cross devices, so there are no per-step collectives; queues, caches, and fused ops remain local, shrinking p50 and calming p95.  
• Scales by isolation, not synchronization: independent replicas absorb concurrency, so aggregate tokens/s rise while single-request latency stays bounded by one device.  
• Makes shapes predictable: left-padding, explicit masks, and optional length-bucket warmup cut graph thrash and allocator noise.  
• Protects the kernel boundary: Triton launches operate only on well-formed, contiguous, device pointers; dtype transitions are explicit, preventing silent slow paths.

---

## Compatibility Model
• Accelerators: CUDA first-class; MPS/XPU/CPU supported with graceful feature fallbacks.  
• Models: decoder-style causal LMs compatible with the framework stack; weights are never bundled.  
• Ecosystem: PyTorch is user-supplied to match drivers/runtimes; version constraints are intentionally minimal to maximize interoperability with external tooling.

---

## Public Interface Surface
### CLI Entrypoints
• `skote.runtime.generate:main` — single-device, latency-first text generation.  
• `skote.runtime.router:main` — multi-device request router with length-aware policy.  
• `skote.eval.runner:main` — optional benchmark harness used to obtain the measurements above.

### Python Signatures (illustrative)
    def generate(local_dir: str, prompt: str, *, max_new_tokens: int, greedy: bool) -> str: ...
    class Router:
        def __init__(self, local_dir: str, devices: list[str] | None = None,
                     dtype: str | None = None, policy: str | None = None,
                     max_concurrency_per_device: int | None = None): ...
        def submit(self, text: str, *, max_new_tokens: int, greedy: bool) -> str: ...
        def submit_many(self, texts: list[str], *, max_new_tokens: int, greedy: bool) -> list[str]: ...

---

## Guarantees & Scope
• Focused on inference for decoder-style LLMs; not a training framework; no model distribution.  
• Performance wins were observed under the reported configuration; results may vary with sequence length, driver/runtime stacks, and kernel availability; users should reproduce on their own systems.  
• Router defaults target latency stability; knobs exist to trade a small wait window for higher packing when desired.

---

## Roadmap
• Optional remote service entry (HTTP/gRPC) behind an extras flag, leaving local CLI untouched.  
• Opt-in decode tensor parallel with captured communication graphs for scenarios willing to trade per-request latency for device utilization.  
• Extended fused kernels (paged KV, attention variants) with strict fallbacks on non-CUDA backends.

---

## License
• Code under Apache-2.0.  
• Models and weights remain external; users are responsible for their model licenses and distribution policies.
