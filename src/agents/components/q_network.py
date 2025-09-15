# networks/q_network.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    """
    Reusable Q-network.
    - Supports classic and dueling architectures.
    - Supports multi-head outputs (n_heads) for ensembles/uncertainty.
    Output shape:
        classic: [B, n_actions]             if n_heads == 1
                 [B, n_heads, n_actions]    if n_heads > 1
        dueling: same shapes, but computed as V + A - mean(A)
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dueling: bool = False,
        n_heads: int = 1,
    ):
        super().__init__()
        self.dueling = dueling
        self.n_heads = n_heads
        in_dim = int(torch.prod(torch.tensor(obs_shape)).item())

        self.encoder = MLP(in_dim, hidden_sizes)

        last_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else in_dim

        if not dueling:
            # One linear per head -> [B, n_heads, n_actions]
            self.q_heads = nn.ModuleList(
                [nn.Linear(last_dim, n_actions) for _ in range(n_heads)]
            )
        else:
            # Dueling: one value head per head, one advantage per head
            self.v_heads = nn.ModuleList(
                [nn.Linear(last_dim, 1) for _ in range(n_heads)]
            )
            self.a_heads = nn.ModuleList(
                [nn.Linear(last_dim, n_actions) for _ in range(n_heads)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        z = self.encoder(x)

        if not self.dueling:
            qs = [head(z) for head in self.q_heads]
            out = torch.stack(qs, dim=1)  # [B, n_heads, n_actions]
        else:
            vs = [v(z) for v in self.v_heads]
            As = [a(z) for a in self.a_heads]
            qs = []
            for v, A in zip(vs, As):
                A_centered = A - A.mean(dim=1, keepdim=True)
                qs.append(v + A_centered)  # [B, n_actions]
            out = torch.stack(qs, dim=1)  # [B, n_heads, n_actions]

        # if self.n_heads == 1:
        #     out = out[:, 0]  # [B, n_actions]
        return out


class IndependentQNetwork(nn.Module):
    """
    Q-network Ensemble ohne Parameter-Sharing.
    Jeder Head ist ein komplett eigenes MLP (+ Output Layer).
    - Supports classic und dueling architectures.
    - Output shape:
        classic: [B, n_actions]             if n_heads == 1
                 [B, n_heads, n_actions]    if n_heads > 1
        dueling: gleiche Shapes, aber mit V + A - mean(A)
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dueling: bool = False,
        n_heads: int = 1,
    ):
        super().__init__()
        self.dueling = dueling
        self.n_heads = n_heads
        in_dim = int(torch.prod(torch.tensor(obs_shape)).item())

        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            encoder = MLP(in_dim, hidden_sizes)
            last_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else in_dim

            if not dueling:
                q_head = nn.Sequential(encoder, nn.Linear(last_dim, n_actions))
                self.heads.append(q_head)
            else:
                v_head = nn.Sequential(encoder, nn.Linear(last_dim, 1))
                a_head = nn.Sequential(encoder, nn.Linear(last_dim, n_actions))
                self.heads.append(nn.ModuleDict({"v": v_head, "a": a_head}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        qs = []
        for head in self.heads:
            if not self.dueling:
                q = head(x)  # [B, n_actions]
            else:
                v = head["v"](x)  # [B, 1]
                A = head["a"](x)  # [B, n_actions]
                A_centered = A - A.mean(dim=1, keepdim=True)
                q = v + A_centered
            qs.append(q)

        out = torch.stack(qs, dim=1)  # [B, n_heads, n_actions]
        if self.n_heads == 1:
            out = out[:, 0]  # [B, n_actions]
        return out
