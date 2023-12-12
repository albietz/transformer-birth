from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import json
import pickle
import time
import torch
import sys

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple

from ihead_data import DataArgs, Dataset, iterate_batches
from ihead_full_model import ModelArgs, Transformer

logging.getLogger().setLevel(logging.INFO)


@dataclass
class OptimArgs:
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    momentum: float = 0.9  # for SGD
    batch_size: int = 64
    use_sgd: bool = False  # otherwise use AdamW


@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    max_iters: Optional[int] = None
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    num_data_workers: int = 60
    save_dir: Optional[str] = None
    root_dir: str = ''


if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs()
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())

    ds = Dataset(cfg.data_args, train_test=None)
    ds_test = Dataset(cfg.data_args, train_test=None)
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

    # memory probes
    range_toks = torch.from_numpy(np.arange(ds.n_train_toks)).cuda()
    def test_wo1():
        toks = model.tok_embeddings(range_toks)
        toks = model.layers[1].attention.wv(toks)
        toks = model.layers[1].attention.wo(toks)
        toks = model.output(toks)
        return (toks.argmax(-1) == range_toks).float().mean().item()

    range_pos_toks = torch.from_numpy(np.arange(cfg.model_args.max_length)).cuda()
    def test_wk0(cutoff=None):
        if cfg.model_args.sin_cos:
            pe = model.pe[:cutoff,:]
        else:
            pe = model.pos_embeddings.weight[:cutoff,:]
        k = model.layers[0].attention.wk(pe[:-1])
        q = model.layers[0].attention.wq(pe[1:])
        return ((q @ k.t()).argmax(-1) == range_pos_toks[:pe.shape[0]-1]).float().mean().item()

    full_range_toks = torch.from_numpy(np.arange(ds.num_tokens)).cuda()
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
                betas=(0.9, 0.99),
                eps=1e-8)

    def predict(text):
        toks = text.split()
        x = torch.from_numpy(np.array(tokenizer.encode(toks))).cuda().unsqueeze(0)
        return tokenizer.decode([model(x)[0,-1].argmax(-1)])[0]

    # a test batch for experimentation
    x_exp, out_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]

    x_test, out_test = ds_test.gen_batch(np.random.default_rng(0), 512)
    x_t = torch.from_numpy(x_test[:,:ds.seq_length]).cuda()
    y_t = torch.from_numpy(x_test[:,1:ds.seq_length + 1]).cuda()
    outs_t = torch.from_numpy(out_test[:,:ds.seq_length]).cuda()

    t = time.time()
    t0 = t
    res = []
    for i, (x, y, outs) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size,
                                                     num_workers=cfg.num_data_workers)):
        dt_data = time.time() - t
        if cfg.max_iters is not None and i >= cfg.max_iters:
            if cfg.save_dir is not None:
                outfile.close()
                xt = torch.from_numpy(x_exp).cuda()
                outt = torch.from_numpy(out_exp).cuda()
                scores = []
                for layer in range(model.n_layers):
                    scores.append(model.get_layer_scores(xt, n=layer).detach().cpu().numpy())
                preds = model(xt).detach().cpu().numpy()
                pickle.dump({'x': x_exp, 'out': out_exp, 'scores': scores, 'preds': preds}, open(outdir / 'exp.pkl', 'wb'))
            sys.exit(0)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        outs = torch.from_numpy(outs).cuda()

        optimizer.zero_grad()
        pred = model(x)

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

                avg_outs = (outs == 1).to(torch.float32).mean().item()

                if cfg.model_args.n_layers >= 2:
                    if cfg.model_args.no_ffn:
                        ff1_loss = -1
                    else:
                        ff1_loss = test_ff1()
                    wk0_acc = test_wk0()
                    wk0_64_acc = test_wk0(cutoff=64)
                    wk1_acc = test_wk1()
                    wo1_acc = test_wo1()
                else:
                    ff1_loss = -1
                    wk0_acc, wk0_64_acc, wk1_acc, wo1_acc = -1, -1, -1, -1

                with torch.no_grad():
                    pred_t = model(x_t)
                acc_end_test = (pred_t[:,-el:].argmax(-1)[outs_t[:,-el:] >= 1] == y_t[:,-el:][outs_t[:,-el:] >= 1]).float().mean().item()

                logging.info(f'{i} ({dt_data:.2f}, {dt:.2f}, {t - t0:.2f}): {loss.item():.4f} ({loss_bigram:.4f} / {loss_head:.4f}), {acc_tot:.4f} ({acc_start:.4f} / {acc_end:.4f} / {acc_end_test:.4f}, {avg_outs:.2f})')
                if cfg.log_probes:
                    logging.info(f'accs wk0: {wk0_acc:.4f} ({wk0_64_acc:.4f}), wk1: {wk1_acc:.4f}, wo1: {wo1_acc:.4f}, ff1: {ff1_loss:.4f}')

                curr_res = {'iter': i, 'loss': loss.item(), 'loss_bigram': loss_bigram, 'loss_head': loss_head,
                            'acc_tot': acc_tot, 'acc_start': acc_start, 'acc_end': acc_end,
                            'wk0_acc': wk0_acc, 'wk0_64_acc': wk0_64_acc, 'wk1_acc': wk1_acc,
                            'wo1_acc': wo1_acc, 'ff1_loss': ff1_loss}

                res.append(curr_res)

                if cfg.log_norms:
                    param_norms = {
                            'wk': [layer.attention.wk.weight.norm().item() for layer in model.layers],
                            'wq': [layer.attention.wq.weight.norm().item() for layer in model.layers],
                            'wo': [layer.attention.wo.weight.norm().item() for layer in model.layers],
                            }
                    grad_norms = {
                            'wk': [layer.attention.wk.weight.grad.norm().item() for layer in model.layers],
                            'wq': [layer.attention.wq.weight.grad.norm().item() for layer in model.layers],
                            'wo': [layer.attention.wo.weight.grad.norm().item() for layer in model.layers],
                            }
                    if not cfg.model_args.freeze_embeddings:
                        grad_norms['emb'] = model.tok_embeddings.weight.grad.norm().item()
                    if not cfg.model_args.freeze_output:
                        grad_norms['unemb'] = model.output.weight.grad.norm().item()
                    logging.info(repr(param_norms))
                    logging.info(repr(grad_norms))

                if cfg.save_dir is not None:
                    print(json.dumps(curr_res), file=outfile, flush=True)
            else:
                logging.info(f'{i} ({dt_data:.2f}, {dt:.2f}, {t - t0:.2f}): {loss.item():.4f}')
                res.append({'loss': loss.item()})

