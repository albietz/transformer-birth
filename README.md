# Birth of a Transformer: a Memory Viewpoint

This repository contains the code for the experiments in the NeurIPS 2023 paper [Birth of a Transformer: a Memory Viewpoint](https://arxiv.org/abs/2306.00802).

# Install

The code is written in PyTorch. The following requirements should be sufficient:

```
pip install torch numpy omegaconf
```

# Usage examples

There are two main scripts, which both train on synthetic data from our bigram task:
* `ihead_full_main.py` for training a generic Transformer, with arbitrary number of layers, including MLP feed-forward layers and layer-normalization
* `ihead_basic_main.py` for training our simplified Transformer architecture (see Section 4.2 in the paper)

The arguments can be provided in the command line as in the following example:
```
python ihead_basic_main.py max_iters=1000 log_probes=True eval_delta=5 loss_head_only=True \
        data_args.k=5 data_args.fixed_special_toks=True \
        optim_args.use_sgd=True optim_args.learning_rate=0.03 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
        model_args.final_ffn=False model_args.freeze_embeddings=True model_args.freeze_output=True model_args.dim=256
```

Some comments on the above command line arguments:
- `log_probes=True` logs the various *memory recall probes* after each `eval_delta` iterations
- `loss_head_only=True` specifies that the loss should only be computed on the *output* tokens starting at their second occurrence (as in the experiments of Figure 3)
- `data_args.k` is the number of *triggers*
- `data_args.fixed_special_toks=True` indicates fixed triggers, chosen as the most frequent tokens (vs random triggers sampled from the unigram distribution for `False`)
- `model_args.final_ffn=False` drops the second feed-forward layer (note that the first feed-forward layer is always removed in the simplified architecture)
- `model_args.freeze_embeddings=True` and `model_args.freeze_output=True` freezes input/output embeddings at random initialization

For more command line arguments, you can take a look at the classes `TrainerArgs` (arguments with no prefix), `OptimArgs`, `DataArgs` (in `ihead_data.py`) and `ModelArgs` (in `ihead_basic_model.py` or `ihead_full_model.py`).
