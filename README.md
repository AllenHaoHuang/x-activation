# Expanded Gating Ranges Improve Activation Functions

## About

This repository contains implementations of the proposed activation functions and gated linear units proposed in the
paper [Expanded Gating Ranges Improve Activation Functions](https://arxiv.org/abs/2405.20768).

Key Results

- For **standard MLP**, xATLU > xGELU / xSiLU > GELU / SiLU
- For **first-order GLU**, xATGLU > xGEGLU / xSwiGLU > ATGLU / GEGLU / SwiGLU
- For **second-order GLU**, expanded gating ranges does not improve performance, just use ATGLU / GEGLU / SwiGLU
- First-order GLU with expanded gating ranges appears to match the performance of second-order GLU

## Installation

The code is arranged as a `x_gate` package. To install the `x_gate` package, run:

```
pip install -e .
```

## Usage

Code snippet for standard MLP using xATLU / xGELU / xSiLU

```python
import torch.nn as nn
from x_gate import XATLU, XGELU, XSiLU


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = XATLU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

Code snippet for gated MLP using xATGLU / xGEGLU / xSwiGLU

```python
import torch.nn as nn
from x_gate import XATGLU, XGEGLU, XSwiGLU


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, int(2 * 8 / 3 * config.n_embd), bias=config.bias)
        self.c_proj = nn.Linear(int(8 / 3 * config.n_embd), config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = XATGLU(order=1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

Code snippet for gated MLP directly using the expanded gating function xAT / xGE / xS

```python
import torch.nn as nn
from x_gate import XAT, XGE, XS


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, int(2 * 8 / 3 * config.n_embd), bias=config.bias)
        self.c_proj = nn.Linear(int(8 / 3 * config.n_embd), config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gate = XAT()

    def forward(self, x):
        x = self.c_fc(x)
        x, y = x.chunk(2, dim=-1)
        # first order GLU
        x = self.gate(x) * y
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

## Additional Experiments / TODO

- Experimental changes on [nanoRWKV](https://github.com/BlinkDL/nanoRWKV) using arctan based gating
  for [12 layers](https://api.wandb.ai/links/saesara/w8cny2aj), [24 layers](https://api.wandb.ai/links/saesara/f7s881y2)
- Running larger scale experiments using per channel weights instead of a scalar
- Second order GLU appears to marginally improve (15.57 ppl -> 15.50 ppl) by scaling the range of the gating
  function to (α, β) and will be added if the improvements are consistent

## Citation

```
@misc{huang2024expanded,
      title={Expanded Gating Ranges Improve Activation Functions}, 
      author={Allen Hao Huang},
      journal={arXiv preprint arXiv:2405.20768}
      year={2024}
}
```
