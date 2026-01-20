import pytest
param = pytest.mark.parametrize

import torch

@param('partial', (False, True))
def test_pope(
    partial
):
    from PoPE_pytorch.PoPE import PoPE

    pope = PoPE(32 if partial else 64, heads = 8)

    pos_embed = pope(1024)

    q = torch.randn(1, 8, 1024, 64)
    k = torch.randn(1, 8, 1024, 64)

    rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q, k)

    final_dim = 128 if not partial else (32 * 2 + 32)

    assert rotated_q.shape == (1, 8, 1024, final_dim)
    assert rotated_k.shape == (1, 8, 1024, final_dim)

    rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q[..., -1:, :], k)

    assert rotated_q.shape == (1, 8, 1, final_dim)
    assert rotated_k.shape == (1, 8, 1024, final_dim)
