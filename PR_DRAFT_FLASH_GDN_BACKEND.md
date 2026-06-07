# Draft PR: Add Opt-In FlashGDN Backend for Gated Delta Rule

## Summary

Adds an opt-in FlashQLA-derived TileLang forward backend for
`chunk_gated_delta_rule`, enabled with `FLA_FLASH_GDN=1`.

The backend is inference-only in this first PR and falls back to the existing
FLA implementation for unsupported calls. It is intended to provide a
maintained FLA-native home for the fast GDN forward kernels without adding
FlashQLA as a runtime dependency.

Current upstream-readiness blocker: the FlashQLA kernels were authored against
the TileLang stack that exposes `tilelang.language.gemm_v1`. FLA currently
declares `tilelang>=0.1.9`, whose public package no longer exposes that helper
and fails these kernels during CUDA compilation if `gemm` is substituted
directly. This branch therefore keeps the backend disabled unless `gemm_v1` is
available. Before opening the PR as ready-for-review, either port the kernels to
current TileLang GEMM lowering or agree on a compatible TileLang dependency
strategy with FLA maintainers.

## Motivation

GDN forward dominates some long-context inference workloads. A
FlashQLA-derived fused forward path is materially faster for the
`B=64,T=16384,H=12,K=64,V=128,bf16` shape on B200, while preserving close
op-level numerical parity with FLA.

## Implementation

- Adds `fla.ops.gated_delta_rule.backends.flashgdn`.
- Ports the FlashQLA-derived TileLang pieces needed for the forward path:
  - chunk-local cumsum,
  - KKT solve,
  - fused GDR forward.
- Adds backend selection through `FLA_FLASH_GDN=1`.
- Keeps the backend disabled by default.
- Uses FLA `l2norm_fwd` for `use_qk_l2norm_in_kernel=True`.
- Preserves FLA fallback behavior through a verifier.

Supported first-pass surface:

- CUDA inference mode only.
- `float16` / `bfloat16` q/k/v.
- `K in {64, 128}`.
- `V == 128`.
- Precomputed log-space `g`.
- Equal-length packed input.

Unsupported and intentionally falling back:

- training/autograd,
- context parallel,
- varlen,
- `state_v_first=True`,
- `use_gate_in_kernel=True`,
- `use_beta_sigmoid_in_kernel=True`,
- `allow_neg_eigval=True`,
- unsupported shapes/dtypes.

## Attribution

The TileLang kernels under `fla.ops.gated_delta_rule.backends.flashgdn` are
adapted from QwenLM/FlashQLA, licensed under the MIT License.

Files that include adapted code carry:

```text
Portions Copyright (c) 2026 The Qwen team, Alibaba Group.
```

## Correctness

Focused local test on B200:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/stevenson/worktrees/fla-flashqla-gdn-backend \
TILELANG_EXECUTION_BACKEND=cython \
TILELANG_DISABLE_CACHE=1 \
FLA_FLASH_GDN_DISABLE_WGMMA=1 \
CUDA_ROOT=/root/stevenson/worktrees/inference-optimization-experiments/.venv/lib/python3.11/site-packages/nvidia/cu13 \
CUDA_HOME=/root/stevenson/worktrees/inference-optimization-experiments/.venv/lib/python3.11/site-packages/nvidia/cu13 \
PATH=/root/stevenson/worktrees/inference-optimization-experiments/.venv/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH \
CPATH=/root/stevenson/worktrees/inference-optimization-experiments/.venv/lib/python3.11/site-packages/nvidia/cu13/include/cccl/libcudacxx/include:/root/stevenson/worktrees/inference-optimization-experiments/.venv/lib/python3.11/site-packages/nvidia/cu13/include/cccl:/root/stevenson/worktrees/inference-optimization-experiments/.venv/lib/python3.11/site-packages/nvidia/cu13/include:${CPATH:-} \
/root/stevenson/worktrees/inference-optimization-experiments/.venv/bin/python \
  -m pytest tests/ops/test_gdn.py::test_chunk_flashgdn_backend_prefill -q -rs
```

Result:

```text
1 passed in 58.00s
o diff: 0.000488
ht diff: 0.005440
```

Clean FLA optional-extra check:

```bash
uv run --extra tilelang --extra test pytest \
  tests/ops/test_gdn.py::test_chunk_flashgdn_backend_prefill -q -rs
```

Result:

```text
Fails with current TileLang package: the FlashQLA kernels require gemm_v1, and
directly substituting current TileLang gemm reaches an unsupported f32 MMA
configuration during CUDA compilation.
```

Lint:

```bash
uv run --with ruff ruff check \
  fla/ops/gated_delta_rule/chunk.py \
  fla/ops/gated_delta_rule/backends \
  tests/ops/test_gdn.py
```

Result:

```text
All checks passed.
```

## Performance

Local B200 op-level benchmark:

Shape:

```text
B=64,T=16384,H=12,K=64,V=128,bfloat16
```

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/stevenson/worktrees/fla-flashqla-gdn-backend \
TILELANG_EXECUTION_BACKEND=cython \
TILELANG_DISABLE_CACHE=1 \
FLA_FLASH_GDN_DISABLE_WGMMA=1 \
uv run python /tmp/inf021_compare_fla_op.py \
  --batch-size 64 \
  --seq-len 16384 \
  --warmup 1 \
  --steps 3
```

Results:

| Backend | Seconds / step | Speedup |
| --- | ---: | ---: |
| default FLA | `0.0145999823` | reference |
| `FLA_FLASH_GDN=1` | `0.0092630226` | `1.576x` |

Numerics:

| Metric | Value |
| --- | ---: |
| output max abs delta | `0.001953125` |
| output mean abs delta | `0.0000165258` |
| final-state max abs delta | `0.0080233216` |
| final-state mean abs delta | `0.0001408357` |

Local BN-checkpoint wrapper smoke preserving old GDN semantics:

| Variant | Rows/s median | Score mean | Memory |
| --- | ---: | ---: | ---: |
| old semantics, default FLA | `32.8072` | `0.1484375` | `47.50 GiB` |
| old semantics, `FLA_FLASH_GDN=1` | `34.6348` | `0.1484375` | `38.33 GiB` |

Wrapper-level result:

```text
1.056x speedup, synthetic score delta 0.0
```

## Notes for Reviewers

- This PR deliberately starts with a conservative opt-in inference backend.
- It does not change default behavior.
- It does not attempt to replace the existing training/backward path.
- The high-level `GatedDeltaNet.forward` gate semantics are left untouched by
  this PR; this backend targets the op surface.
- Broader varlen/CP/training support can be added after the forward inference
  backend is reviewed and stabilized.
