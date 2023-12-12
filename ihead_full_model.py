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
    n_layers: int = 2
    dim: int = 128
    n_heads: int = 4
    max_length: int = 256
    pre_norm: bool = True
    no_ffn: bool = False
    no_norm: bool = False
    linear_ffn: bool = False
    no_first_layer_ffn: bool = False
    freeze_embeddings: bool = False
    freeze_output: bool = False
    tie_output: bool = False
    freeze_wv: bool = False
    freeze_wo: bool = False
    no_wo: bool = False
    no_wv: bool = False
    sin_cos: bool = False


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 head_dim: int,
                 n_heads: int,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False,
                 no_wv: bool = False,
                 no_wo: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads

        self.wq = nn.Linear(dim, n_heads*head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads*head_dim, bias=False)
        if no_wv:
            self.wv = nn.Identity()
        else:
            self.wv = nn.Linear(dim, n_heads*head_dim, bias=False)
            if freeze_wv:
                self.wv.weight.requires_grad_(False)

        if no_wo:
            self.wo = nn.Identity()
        else:
            self.wo = nn.Linear(n_heads*head_dim, dim, bias=False)
            if freeze_wo:
                self.wo.weight.requires_grad_(False)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor):
        bs, slen, _ = x.shape
        assert mask is not None

        xq = self.wq(x).view(bs, slen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bs, slen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bs, slen, self.n_heads, self.head_dim)

        # change to (bs, n_heads, slen, head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
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
                 hidden_dim: int,
                 n_heads: int,
                 pre_norm: bool,
                 no_norm: bool = False,
                 no_ffn: bool = False,
                 linear_ffn: bool = False,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False,
                 no_wv: bool = False,
                 no_wo: bool = False,
                ):
        super().__init__()
        assert dim % n_heads == 0
        head_dim = dim // n_heads
        self.attention = Attention(
                dim=dim,
                head_dim=head_dim,
                n_heads=n_heads,
                freeze_wv=freeze_wv,
                freeze_wo=freeze_wo,
                no_wv=no_wv,
                no_wo=no_wo)
        if not no_ffn:
            if linear_ffn:
                self.ff = nn.Linear(dim, dim, bias=False)
            else:
                self.ff = FeedForward(dim=dim, hidden_dim=hidden_dim)
        if no_norm:
            self.attention_norm = nn.Identity()
            self.ff_norm = nn.Identity()
        else:
            self.attention_norm = nn.LayerNorm(dim, eps=1e-5)
            self.ff_norm = nn.LayerNorm(dim, eps=1e-5)
        self.pre_norm = pre_norm
        self.no_ffn = no_ffn

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                return_scores: bool = False,
                no_ffn: bool = False):
        no_ffn = no_ffn or self.no_ffn
        if self.pre_norm:
            h, scores = self.attention(self.attention_norm(x), mask)
            if return_scores:
                return scores
            h = x + h
            if no_ffn:
                return h
            else:
                return h + self.ff(self.ff_norm(x))
        else:
            h, scores = self.attention(x, mask)
            if return_scores:
                return scores
            h = self.attention_norm(x + h)
            if no_ffn:
                return h
            else:
                return self.ff_norm(h + self.ff(h))


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tie_output = args.tie_output
        self.dim = args.dim
        self.sin_cos = args.sin_cos

        # embeddings
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_length, args.dim)
        if args.freeze_embeddings:
            self.tok_embeddings.weight.requires_grad_(False)
            self.pos_embeddings.weight.requires_grad_(False)

        # sin/cos position embeddings
        if self.sin_cos:
            position = torch.arange(args.max_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, args.dim, 2) * (-math.log(10000.0) / args.dim))
            pe = torch.zeros(args.max_length, args.dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        self.layers = nn.ModuleList([TransformerBlock(
            dim=args.dim,
            hidden_dim=4*args.dim,
            n_heads=args.n_heads,
            pre_norm=args.pre_norm,
            no_norm=args.no_norm,
            no_ffn=args.no_ffn or (i == 0 and args.no_first_layer_ffn),
            linear_ffn=args.linear_ffn,
            freeze_wv=args.freeze_wv,
            freeze_wo=args.freeze_wo,
            no_wv=args.no_wv,
            no_wo=args.no_wo,
            ) for i in range(args.n_layers)])

        # final normalization layer (only needed for pre-norm)
        self.norm: Optional[nn.Module] = None
        if args.pre_norm:
            if args.no_norm:
                self.norm = nn.Identity()
            else:
                self.norm = nn.LayerNorm(args.dim, eps=1e-5)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.freeze_output:
            self.output.weight.requires_grad_(False)
        elif args.tie_output:
            # self.tok_embeddings.weight.data /= math.sqrt(args.dim)
            self.output.weight = self.tok_embeddings.weight

    def forward(self, tokens: torch.Tensor, return_layer: Optional[int] = None, before_ffn: bool = False):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if self.sin_cos:
            h = h + self.pe.unsqueeze(0)
        else:
            h = h + self.pos_embeddings(torch.arange(N, device=tokens.device).view(1, N))

        if return_layer == 0:
            return h

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if return_layer == i + 1:
                return layer(h, mask, no_ffn=before_ffn)
            h = layer(h, mask)

        # output layer
        if self.norm is not None:
            h = self.norm(h)
        output = self.output(h)
        return output.float()

    def forward_ff_only(self, tokens: torch.Tensor):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if self.sin_cos:
            h = h + self.pe.unsqueeze(0)
        else:
            h = h + self.pos_embeddings(torch.arange(N, device=tokens.device).view(1, N))

        # transformer blocks
        for i, layer in enumerate(self.layers):
            h = h + layer.ff(h)

        # output layer
        if self.norm is not None:
            h = self.norm(h)
        output = self.output(h)
        if self.tie_output:
            output /= math.sqrt(self.dim)
        return output.float()

    def get_layer_scores(self, tokens: torch.Tensor, n: int = 0):
        assert n < len(self.layers)
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if self.sin_cos:
            h = h + self.pe.unsqueeze(0)
        else:
            h = h + self.pos_embeddings(torch.arange(N, device=tokens.device).view(1, N))

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if i == n:
                return layer(h, mask, return_scores=True)
            else:
                h = layer(h, mask)

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
