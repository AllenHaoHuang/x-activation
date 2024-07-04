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
from x_activation import XATLU, XGELU, XSiLU


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
from x_activation import XATGLU, XGEGLU, XSwiGLU


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
from x_activation import XAT, XGE, XSig


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
- ~~Running larger scale experiments using per channel weights instead of a scalar~~ Per channel weights do not appear
  to improve performance
- Second order GLU appears to consistently improve by around 0.05-0.06 ppl for 24 layer experiments by introducing the
  following scalar affine transformation (g(x) * α + β) * x * y, where α is initialised to 1 and β is initialised to 0

## Citation

```
@misc{huang2024expanded,
      title={Expanded Gating Ranges Improve Activation Functions}, 
      author={Allen Hao Huang},
      journal={arXiv preprint arXiv:2405.20768}
      year={2024}
}
```
