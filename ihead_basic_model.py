from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import torch
import sys

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple


@dataclass
class ModelArgs:
    vocab_size: int = -1  # defined later
    dim: int = 64
    max_length: int = 256
    final_ffn: bool = False
    first_ffn: bool = False
    linear_final_ffn: bool = True
    linear_first_ffn: bool = True
    freeze_embeddings: bool = False
    freeze_output: bool = False
    tie_output: bool = False
    use_rope: bool = False
    sqrtd_embeddings: bool = False
    no_sqrtd: bool = False
    sin_cos: bool = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 use_rope: bool = False,
                 no_sqrtd: bool = False,
                 freeze_wk: bool = False,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False):
        super().__init__()
        self.dim = dim
        self.use_rope = use_rope
        self.no_sqrtd = no_sqrtd

        self.wq = nn.Identity()

        self.wk = nn.Linear(dim, dim, bias=False)
        if freeze_wk:
            self.wk.weight.requires_grad_(False)

        self.wv = nn.Linear(dim, dim, bias=False)
        if freeze_wv:
            self.wv.weight.requires_grad_(False)

        self.wo = nn.Linear(dim, dim, bias=False)
        if freeze_wo:
            self.wo.weight.requires_grad_(False)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None):
        bs, slen, _ = x.shape
        assert mask is not None

        xq = self.wq(x).view(bs, slen, 1, self.dim)
        xk = self.wk(x).view(bs, slen, 1, self.dim)
        xv = self.wv(x).view(bs, slen, 1, self.dim)

        if self.use_rope:
            assert freqs_cis is not None
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # change to (bs, n_heads, slen, head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        if self.no_sqrtd:
            scores = torch.matmul(xq, xk.transpose(2, 3))
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.dim)
        scores = scores + mask  # (bs, n_heads, slen, slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        output = torch.matmul(scores, xv)  # (bs, n_heads, slen, head_dim)
        output = output.transpose(1, 2)  # (bs, slen, n_heads, head_dim)

        output = output.reshape(bs, slen, -1)
        return self.wo(output), scores


class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        h = self.w1(x)
        h = F.relu(h.float()).type_as(x)
        return self.w2(h)


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 use_rope: bool = False,
                 no_sqrtd: bool = False,
                 no_ffn: bool = False,
                 linear_ffn: bool = False,
                 parallel: bool = False,
                 freeze_wk: bool = False,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False,
                 freeze_ffn: bool = False,
                ):
        super().__init__()
        self.attention = Attention(
                dim=dim,
                use_rope=use_rope,
                no_sqrtd=no_sqrtd,
                freeze_wk=freeze_wk,
                freeze_wv=freeze_wv,
                freeze_wo=freeze_wo)
        self.no_ffn = no_ffn
        self.parallel = parallel
        if not no_ffn:
            if linear_ffn:
                self.ff = nn.Linear(dim, dim, bias=False)
            else:
                self.ff = FeedForward(dim=dim, hidden_dim=4*dim)
            if freeze_ffn:
                for p in self.ff.parameters():
                    p.requires_grad_(False)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                return_scores: bool = False,
                no_ffn: bool = False):
        no_ffn = no_ffn or self.no_ffn

        h, scores = self.attention(x, mask, freqs_cis=freqs_cis)

        if return_scores:
            return scores
        if no_ffn:
            return x + h
        else:
            if self.parallel:
                return x + h + self.ff(x)
            else:
                h = x + h
                return h + self.ff(h)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.tie_output = args.tie_output
        self.dim = args.dim
        self.use_rope = args.use_rope
        self.sin_cos = args.sin_cos

        # embeddings
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if args.sqrtd_embeddings:
            self.tok_embeddings.weight.data.normal_(std=1./math.sqrt(args.dim))
        if args.freeze_embeddings:
            self.tok_embeddings.weight.requires_grad_(False)

        if self.sin_cos:
            # sin/cos position embeddings
            position = torch.arange(args.max_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, args.dim, 2) * (-math.log(10000.0) / args.dim))
            pe = torch.zeros(args.max_length, args.dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # random absolute positional embeddings
            pe = torch.randn(args.max_length, args.dim)
        if args.sqrtd_embeddings:
            pe *= 1. / math.sqrt(args.dim)

        self.register_buffer('pe', pe)

        freqs_cis = precompute_freqs_cis(
            self.dim // 1, args.max_length
        )
        self.register_buffer('freqs_cis', freqs_cis)

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=args.dim,
                use_rope=args.use_rope,
                no_sqrtd=args.no_sqrtd,
                no_ffn=not args.first_ffn,
                linear_ffn=args.linear_first_ffn,
                freeze_wk=False,
                freeze_wv=True,
                freeze_wo=True,
                ),
            TransformerBlock( 
                dim=args.dim,
                use_rope=False,  # args.use_rope,
                no_sqrtd=args.no_sqrtd,
                no_ffn=not args.final_ffn,
                linear_ffn=args.linear_final_ffn,
                freeze_wk=False,
                freeze_wv=True,
                freeze_wo=False,
                )
            ])

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.freeze_output:
            self.output.weight.requires_grad_(False)
        if args.tie_output:
            if args.freeze_output:
                self.output.weight.data = self.tok_embeddings.weight.data / math.sqrt(args.dim)
            else:
                self.output.weight = self.tok_embeddings.weight / math.sqrt(args.dim)

    def forward(self, tokens: torch.Tensor, return_layer: Optional[int] = None, before_ffn: bool = False):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if not self.use_rope:
            h = h + self.pe.unsqueeze(0)

        if return_layer == 0:
            return h

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if return_layer == i + 1:
                return layer(h, mask, freqs_cis=self.freqs_cis, no_ffn=before_ffn)
            h = layer(h, mask, freqs_cis=self.freqs_cis)

        # output layer
        output = self.output(h)
        if self.tie_output:
            output /= math.sqrt(self.dim)
        return output.float()

    def forward_ff_only(self, tokens: torch.Tensor):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if not self.use_rope:
            h = h + self.pe.unsqueeze(0)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            h = h + layer.ff(h)

        # output layer
        output = self.output(h)
        if self.tie_output:
            output /= math.sqrt(self.dim)
        return output.float()

    def get_layer_scores(self, tokens: torch.Tensor, n: int = 0):
        assert n < len(self.layers)
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        h = h + self.pe.unsqueeze(0)

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if i == n:
                return layer(h, mask, freqs_cis=self.freqs_cis, return_scores=True)
            else:
                h = layer(h, mask, freqs_cis=self.freqs_cis)

    def get_top_preds(self, tokens: torch.Tensor, n: int = 4):
        squeeze = False
        if len(tokens.shape) == 1:
            squeeze = True
            tokens = tokens.unsqueeze(0)
        with torch.no_grad():
            preds = self(tokens).detach()
        vals, idxs = preds.sort(-1, descending=True)
        vals = vals[:,:,:n]
        idxs = idxs[:,:,:n]
        if squeeze:
            return vals.squeeze(0), idxs.squeeze(0)
        return vals, idxs

