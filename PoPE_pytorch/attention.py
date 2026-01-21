import torch
import torch.nn.functional as F
from einops import einsum
from .pope import apply_pope_to_qk

try:
    from .triton_pope import triton_compute_qk_similarity
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

def compute_attn_similarity_non_fused(q, k, pope):
    q, k = apply_pope_to_qk(pope, q, k, to_magnitude = F.softplus)
    return einsum(q, k, 'b h i d, b h j d -> b h i j')

def compute_attn_similarity(q, k, pope, allow_tf32 = True):
    freqs, bias = pope
    head_dim = q.shape[-1]
    assert head_dim in {32, 48, 64, 128, 256}, f"head_dim {head_dim} not in common sizes {32, 48, 64, 128, 256}"
    
    is_cuda = q.is_cuda and k.is_cuda and freqs.is_cuda and bias.is_cuda
    
    if TRITON_AVAILABLE and is_cuda:
        rotate_dim = freqs.shape[-1]
        return triton_compute_qk_similarity(q, k, freqs, bias, rotate_dim, allow_tf32 = allow_tf32)

    return compute_attn_similarity_non_fused(q, k, pope)
