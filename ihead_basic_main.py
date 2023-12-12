from collections import defaultdict
from dataclasses import dataclass
import itertools
import logging
import random
import json
import math
import numpy as np
import time
import torch
import sys

from omegaconf import OmegaConf
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
from pathlib import Path

from ihead_data import DataArgs, Dataset, iterate_batches
from ihead_basic_model import ModelArgs, Transformer

logging.getLogger().setLevel(logging.INFO)


@dataclass
class OptimArgs:
    learning_rate: float = 0.2  # for SGD
    weight_decay: float = 1e-4  # for SGD
    momentum: float = 0.9  # for SGD
    batch_size: int = 512
    use_sgd: bool = True  # otherwise use AdamW
    ff_lr_scaling: Optional[float] = None


@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    max_iters: Optional[int] = None
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    freeze_until: str = ''
    loss_head_only: bool = True
    bigram_outs_train: bool = False
    bigram_outs_test: bool = False
    num_data_workers: int = 60
    seed: int = 42
    save_dir: Optional[str] = None
    root_dir: str = ''


if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs()
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())

    ds = Dataset(cfg.data_args, train_test=None, bigram_outs=cfg.bigram_outs_train)
    ds_test = Dataset(cfg.data_args, train_test=None, bigram_outs=cfg.bigram_outs_test)
    ds_test.idxs = ds.idxs
    cfg.model_args.vocab_size = ds.num_tokens

    if cfg.save_dir is not None:
        outdir = Path(cfg.root_dir) / Path(cfg.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        # save params
        with open(outdir / 'params.json', 'w') as f:
                json.dump(dict(cfg), f, sort_keys=True, indent=4)
        outfile = open(outdir / 'res.jsonl', 'w')

    model = Transformer(cfg.model_args)
    model.cuda()

    # attn probes
    attn_features = None
    attn_features2 = None
    attn_input_features = None
    attn_scores = None
    attn_scores2 = None
    def attn0_hook(_, inp, outp):
        global attn_features, attn_input_features, attn_scores
        attn_input_features = inp[0].detach()
        attn_features = outp[0].detach()
        attn_scores = outp[1].detach()
    model.layers[0].attention.register_forward_hook(attn0_hook)
    def attn1_hook(_, inp, outp):
        global attn_scores2, attn_features2
        attn_features2 = outp[0].detach()
        attn_scores2 = outp[1].detach()
    model.layers[1].attention.register_forward_hook(attn1_hook)

    # memory probes
    range_toks = torch.from_numpy(np.arange(ds.n_train_toks)).cuda()
    def test_wo1():
        toks = model.tok_embeddings(range_toks)
        toks = model.layers[1].attention.wv(toks)
        toks = model.layers[1].attention.wo(toks)
        toks = model.output(toks)
        return (toks.argmax(-1) == range_toks).float().mean().item()

    full_range_toks = torch.from_numpy(np.arange(ds.num_tokens)).cuda()
    conds = torch.from_numpy(np.array(ds.cond)).cuda()
    used_idxs = np.arange(ds.num_tokens)
    if cfg.data_args.fixed_special_toks:
        used_idxs = np.setdiff1d(used_idxs, ds.idxs)
    def test_ff1():
        toks = model.tok_embeddings(full_range_toks[used_idxs])
        toks = model.layers[1].ff(toks)
        toks = model.output(toks)
        return F.kl_div(F.log_softmax(toks, dim=1), conds[used_idxs], reduction='batchmean').item()

    range_pos_toks = torch.from_numpy(np.arange(cfg.model_args.max_length)).cuda()
    def test_wk0(cutoff=None):
        pe = model.pe[:cutoff,:]
        k = model.layers[0].attention.wk(pe[:-1])
        q = model.layers[0].attention.wq(pe[1:])
        return ((q @ k.t()).argmax(-1) == range_pos_toks[:pe.shape[0]-1]).float().mean().item()

    wk1_range_toks = full_range_toks.clone()
    if cfg.data_args.fixed_special_toks:
        wk1_range_toks = wk1_range_toks[ds.idxs]
    def test_wk1():
        toksk = model.tok_embeddings(wk1_range_toks)
        toksk = model.layers[0].attention.wv(toksk)
        toksk = model.layers[0].attention.wo(toksk)
        toksk = model.layers[1].attention.wk(toksk)

        toksq = model.tok_embeddings(wk1_range_toks)
        toksq = model.layers[1].attention.wq(toksq)
        return ((toksq @ toksk.t()).argmax(-1) == range_toks[:wk1_range_toks.shape[0]]).float().mean().item()


    # initial param freezing
    freeze_until = defaultdict(list)
    to_freeze = []
    if cfg.freeze_until:
        for kv in cfg.freeze_until.split(','):
            k, v = kv.split(':')
            k = int(k)
            to_freeze.append(v)
            freeze_until[k].append(v)

        for name, p in model.named_parameters():
            if name in to_freeze:
                p.requires_grad_(False)

    # optim
    if cfg.optim_args.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                momentum=cfg.optim_args.momentum)
    else:
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8)

    # a test batch for experimentation
    x_exp, out_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]

    # OOD test data
    x_test, out_test = ds_test.gen_batch(np.random.default_rng(0), 512)
    x_t = torch.from_numpy(x_test[:,:ds.seq_length]).cuda()
    y_t = torch.from_numpy(x_test[:,1:ds.seq_length + 1]).cuda()
    outs_t = torch.from_numpy(out_test[:,:ds.seq_length]).cuda()

    t = time.time()
    t0 = t
    res = []
    for i, (x, y, outs) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size,
                                     num_workers=cfg.num_data_workers, seed=cfg.seed)):
        dt_data = time.time() - t
        if cfg.max_iters is not None and i >= cfg.max_iters:
            if cfg.save_dir is not None:
                outfile.close()
            sys.exit(0)

        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        outs = torch.from_numpy(outs).cuda()

        if i in freeze_until:  # unfreeze params
            for name, p in model.named_parameters():
                if name in freeze_until[i]:
                    p.requires_grad_(True)

        optimizer.zero_grad()
        pred = model(x)

        if cfg.loss_head_only:
            loss = F.cross_entropy(pred[outs >= 2], y[outs >= 2])
        else:
            loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))

        loss.backward()

        optimizer.step()
        dt = time.time() - t
        t = time.time()

        if i % cfg.eval_delta == 0:
            if cfg.data_args.k > 0:
                acc_tot = (pred.argmax(-1)[outs >= 1] == y[outs >= 1]).float().mean().item()
                sl = 10
                acc_start = (pred[:,:sl].argmax(-1)[outs[:,:sl] >= 1] == y[:,:sl][outs[:,:sl] >= 1]).float().mean().item()
                el = 500
                acc_end = (pred[:,-el:].argmax(-1)[outs[:,-el:] >= 2] == y[:,-el:][outs[:,-el:] >= 2]).float().mean().item()
                loss_bigram = F.cross_entropy(pred[outs == 0,:], y[outs == 0]).item()
                loss_head = F.cross_entropy(pred[outs >= 2,:], y[outs >= 2]).item()

                # first layer attn scores probe
                i1, i2 = torch.where(outs[:,:-1] >= 1)
                i1_start, i2_start = torch.where(outs[:,:-1] == 1)
                amax = attn_scores[:,0,:,:].argmax(-1)
                score_acc = (amax[i1, i2 + 1] == i2).float().mean().item()
                score_start_acc = (amax[i1_start, i2_start + 1] == i2_start).float().mean().item()

                # second layer attn scores probe (check that attended token's prev token has correct condition)
                i1, i2 = torch.where(outs >= 2)
                amax2 = attn_scores2.squeeze(1)[i1,i2,:].argmax(-1)
                score2_next_acc = (x[i1, amax2] == y[i1, i2]).float().mean().item()
                pred_attended_acc = (x[i1, amax2] == pred[i1,i2].argmax(-1)).float().mean().item()

                bad = (amax2 == 0).float().sum()
                tot = amax2.shape[0]
                i1 = i1[amax2 >= 1]
                i2 = i2[amax2 >= 1]
                amax2 = amax2[amax2 >= 1]
                score2_acc = (x[i1, amax2 - 1] == x[i1, i2]).float().sum().item() / tot

                # first layer attn score probe conditioned on locations attended by second layer
                score_cond_acc = (amax[i1, amax2] == amax2 - 1).float().mean().item()

                # second layer attn score probe conditioned on repeated tokens
                i1, i2 = torch.where((outs >= 2) & (x == y))
                amax1 = attn_scores.squeeze(1)[i1,i2,:].argmax(-1)
                score1_repeat_val_acc = (x[i1, amax1] == y[i1, i2]).float().mean().item()
                amax2 = attn_scores2.squeeze(1)[i1,i2,:].argmax(-1)
                score2_repeat_val_acc = (x[i1, amax2] == y[i1, i2]).float().mean().item()
                # score2_repeat_prev_acc = (amax2 == i2 - 1).float().mean().item()

                if True:  # cfg.log_probes:
                    wo1_acc = test_wo1()
                    if cfg.model_args.final_ffn:
                        ff1_loss = test_ff1()
                    else:
                        ff1_loss = -1
                    wk0_acc = test_wk0()
                    wk0_64_acc = test_wk0(cutoff=64)
                    wk1_acc = test_wk1()

                repeat_frac = (x[outs >= 1] == y[outs >= 1]).float().mean().item()

                # OOD test (NOTE: do this after the probes sinces it messes hooks!)
                with torch.no_grad():
                    pred_t = model(x_t)
                acc_end_test = (pred_t[:,-el:].argmax(-1)[outs_t[:,-el:] >= 2] == y_t[:,-el:][outs_t[:,-el:] >= 2]).float().mean().item()

                logging.info(
                        f'''{i} ({dt_data:.2f}, {dt:.2f}, {t - t0:.2f}): loss: {loss.item():.4f} ({loss_bigram:.4f}, {loss_head:.4f}), \
acc: {acc_tot:.4f} ({acc_end:.4f} / {acc_end_test:.4f}) \
probes: {score_start_acc:.4f} / {score2_acc:.4f} / {score_cond_acc:.4f} / {pred_attended_acc:.4f} ({repeat_frac:.4f})'''
)
                if cfg.log_probes:
                    logging.info(f'memory probes wk0: {wk0_acc:.4f} ({wk0_64_acc:.4f}), wk1: {wk1_acc:.4f}, wo1: {wo1_acc:.4f}, ff1: {ff1_loss:.4f}')

                curr_res = {'iter': i, 'loss': loss.item(), 'loss_bigram': loss_bigram, 'loss_head': loss_head,
                            'acc_tot': acc_tot, 'acc_start': acc_start, 'acc_end': acc_end, 'acc_end_test': acc_end_test,
                            'score_acc': score_acc, 'score_start_acc': score_start_acc, 'score2_acc': score2_acc,
                            'score_cond_acc': score_cond_acc,
                            'pred_attended_acc': pred_attended_acc, 'repeat_frac': repeat_frac,
                            'wk0_acc': wk0_acc, 'wk0_64_acc': wk0_64_acc, 'wk1_acc': wk1_acc, 'wo1_acc': wo1_acc, 'ff1_loss': ff1_loss}

                for name, p in model.named_parameters():
                    if p.requires_grad:
                        curr_res['norm_' + name] = p.norm().item()
                        curr_res['gradnorm_' + name] = p.grad.norm().item()

                if cfg.log_norms:
                    param_norms = {
                            'wk': [layer.attention.wk.weight.norm().item() for layer in model.layers],
                            'wv': [layer.attention.wv.weight.norm().item() for layer in model.layers],
                            'wo': [layer.attention.wo.weight.norm().item() for layer in model.layers],
                            }
                    grad_norms = {
                            'wk': [layer.attention.wk.weight.grad.norm().item() for layer in model.layers if layer.attention.wk.weight.requires_grad],
                            'wv': [layer.attention.wv.weight.grad.norm().item() for layer in model.layers if layer.attention.wv.weight.requires_grad],
                            'wo': [layer.attention.wo.weight.grad.norm().item() for layer in model.layers if layer.attention.wo.weight.requires_grad],
                            }
                    logging.info(repr(param_norms))
                    logging.info(repr(grad_norms))

                if cfg.save_dir is not None:
                    print(json.dumps(curr_res), file=outfile, flush=True)
                res.append(curr_res)
            else:
                logging.info(f'{i} ({dt_data:.2f}, {dt:.2f}, {t - t0:.2f}): {loss.item():.4f}')
                res.append({'loss': loss.item()})
