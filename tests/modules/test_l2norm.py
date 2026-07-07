# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2_norm, l2norm_fwd, l2norm_fwd_pair
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
            (4, 1024, 5, 1024, torch.float16),
            (4, 1024, 5, 1024, torch.bfloat16),
            (5, 1024, 6, 2048, torch.float16),
            (5, 1024, 6, 2048, torch.bfloat16),
        ]
    ],
)
def test_l2norm(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref = F.normalize(x, dim=-1, p=2)
    tri = l2_norm(x)
    ref_dx = torch.autograd.grad(ref.sum(), x)[0]
    tri_dx = torch.autograd.grad(tri.sum(), x)[0]

    assert_close('y', ref, tri, 0.005)
    assert_close('dx', ref_dx, tri_dx, 0.005)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, torch.float),
            (2, 513, 3, 128, torch.float16),
            (2, 513, 3, 256, torch.bfloat16),
            (2, 513, 3, 512, torch.bfloat16),
        ]
    ],
)
def test_l2norm_fwd_pair(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype).to(device)
    k = torch.randn(B, T, H, D, dtype=dtype).to(device)

    q_ref, q_rstd_ref = l2norm_fwd(q)
    k_ref, k_rstd_ref = l2norm_fwd(k)
    q_tri, q_rstd_tri, k_tri, k_rstd_tri = l2norm_fwd_pair(q, k)

    assert_close('q', q_ref, q_tri, 0.0)
    assert_close('q_rstd', q_rstd_ref, q_rstd_tri, 0.0)
    assert_close('k', k_ref, k_tri, 0.0)
    assert_close('k_rstd', k_rstd_ref, k_rstd_tri, 0.0)
