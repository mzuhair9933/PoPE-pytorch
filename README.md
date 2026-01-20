<img src="./pope.png" width="400px"></img>

## PoPE-pytorch

Efficient implementation (and explorations) into [polar coordinate positional embedding (PoPE)](https://arxiv.org/abs/2509.10534) - from [Gopalakrishnan](https://agopal42.github.io/) et al. under Schmidhuber

## Install

```shell
$ pip install PoPE-pytorch
```

## Usage

```python
import torch
from PoPE_pytorch import PoPE

# define pope

pope = PoPE(64, heads = 8)

# pass in sequence length

pos_embed = pope(1024)

# queries and keys in attention

q = torch.randn(1, 8, 1024, 64)
k = torch.randn(1, 8, 1024, 64)

# training

rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q, k)

# inference

rotated_q, rotated_k = pope.apply_pope_to_qk(pos_embed, q[..., -1:, :], k)
```

## Citations

```bibtex
@misc{gopalakrishnan2025decouplingwhatwherepolar,
    title   = {Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings}, 
    author  = {Anand Gopalakrishnan and Robert Csordás and Jürgen Schmidhuber and Michael C. Mozer},
    year    = {2025},
    eprint  = {2509.10534},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2509.10534}, 
}
```
