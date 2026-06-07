# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Portions Copyright (c) 2026 The Qwen team, Alibaba Group.

"""FlashQLA-derived forward path for chunk_gated_delta_rule."""

from __future__ import annotations

import torch

from fla.modules.l2norm import l2norm_fwd
from fla.ops.gated_delta_rule.backends.flashgdn.cumsum import chunk_local_cumsum
from fla.ops.gated_delta_rule.backends.flashgdn.fused_fwd import fused_gdr_fwd
from fla.ops.gated_delta_rule.backends.flashgdn.kkt_solve import kkt_solve


def chunk_gated_delta_rule_flashgdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    g = chunk_local_cumsum(g, chunk_size=64)
    a = kkt_solve(k=k, b=beta)
    o, _, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        output_h=False,
        output_o=True,
    )
    return o.to(q.dtype), final_state
