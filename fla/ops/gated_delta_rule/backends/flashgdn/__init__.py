# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Portions Copyright (c) 2026 The Qwen team, Alibaba Group.

"""FlashQLA-derived TileLang forward backend for Gated Delta Rule.

The kernel implementation under this package is adapted from QwenLM/FlashQLA,
licensed under the MIT License. This backend is inference-only and opt-in via
``FLA_FLASH_GDN=1`` while coverage is expanded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class FlashGDNBackend(BaseBackend):
    """FlashQLA-derived fused forward backend for chunk_gated_delta_rule.

    Enabled with ``FLA_FLASH_GDN=1``. Unsupported calls fall back to FLA's
    default Triton implementation through the backend verifier.
    """

    backend_type = "flashgdn"
    package_name = "tilelang"
    env_var = "FLA_FLASH_GDN"
    default_enable = False
    priority = 3

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tilelang.language as T

            return hasattr(T, "gemm_v1")
        except ImportError:
            return False

    def chunk_gated_delta_rule_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if torch.is_grad_enabled():
            return False, "FlashGDN only supports inference mode"
        if not q.is_cuda:
            return False, "FlashGDN requires CUDA tensors"
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False, f"FlashGDN requires float16/bfloat16 inputs, got {q.dtype}"
        if q.dtype != k.dtype or q.dtype != v.dtype:
            return False, "FlashGDN requires q, k, and v to share dtype"
        if q.shape[2] != k.shape[2]:
            return False, f"FlashGDN requires q/k heads to match, got {q.shape[2]} and {k.shape[2]}"
        if v.shape[2] % q.shape[2] != 0:
            return False, f"FlashGDN requires v heads to be a multiple of q heads, got {v.shape[2]} and {q.shape[2]}"
        if q.shape[-1] not in (64, 128):
            return False, f"FlashGDN supports K in {{64, 128}}, got {q.shape[-1]}"
        if v.shape[-1] != 128:
            return False, f"FlashGDN requires V=128, got {v.shape[-1]}"
        if use_beta_sigmoid_in_kernel:
            return False, "FlashGDN currently expects beta to be pre-sigmoid-applied"
        if allow_neg_eigval:
            return False, "FlashGDN does not currently support allow_neg_eigval"
        if kwargs.get("use_gate_in_kernel", False):
            return False, "FlashGDN currently expects precomputed log-space decay g"
        if state_v_first:
            return False, "FlashGDN does not currently support state_v_first=True"
        if cp_context is not None:
            return False, "FlashGDN does not currently support context parallel"
        if cu_seqlens is not None:
            return False, "FlashGDN varlen support is not enabled in this backend yet"
        return True, None

    def chunk_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ):
        from fla.ops.gated_delta_rule.backends.flashgdn.chunk import (
            chunk_gated_delta_rule_flashgdn,
        )

        return chunk_gated_delta_rule_flashgdn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


__all__ = ["FlashGDNBackend"]
