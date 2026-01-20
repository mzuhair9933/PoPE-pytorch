import pytest
import torch

def test_pope():
    from PoPE_pytorch.PoPE import PoPE

    pope = PoPE(64, heads = 8)

    pos_embed = pope(1024)

    q = torch.randn(1, 8, 1024, 64)
    k = torch.randn(1, 8, 1024, 64)

    rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q, k)

    assert rotated_q.shape == (1, 8, 1024, 128)
    assert rotated_k.shape == (1, 8, 1024, 128)

    rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q[..., -1:, :], k)

    assert rotated_q.shape == (1, 8, 1, 128)
    assert rotated_k.shape == (1, 8, 1024, 128)
