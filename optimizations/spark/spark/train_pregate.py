#!/usr/bin/env python3
"""(Research) Train a per-layer pre-gate predictor from a routing trace.

A pre-gate predicts a layer's selected experts from the *block input* hidden,
i.e. one step early, which is what you need to prefetch experts and fuse the
attention+FFN graphs (Pre-gated MoE, arXiv:2308.12066).

Trains a predictor per layer from the trace captured by the engine
(`DFLASH_LAGUNA_PREGATE_TRACE`, fixed-size records: int16 layer, int16 n_sel,
int32[8] selected, float32[n_embd] hidden) and reports recall@K: of the experts
a layer actually selects, how many are in the predicted top-K. High recall@K at
small K means prefetch+fusion is viable.

Result on Laguna-XS.2 (RTX 3090, Claude Code traces): linear ~50% / MLP ~53%
recall@8 -- a richer model barely helps, because the pre-gate sees the
pre-attention hidden while the router decides on the post-attention hidden. That
information gap caps a fitted predictor; reaching high recall needs fine-tuning
the model's gate, not a bigger predictor. See RESULTS.md.

    python -m spark.train_pregate --trace pregate_trace.bin --model mlp
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--n-embd", type=int, default=2048)
    ap.add_argument("--n-expert", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=40)
    ap.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--max-per-layer", type=int, default=5000)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    H, E = args.n_embd, args.n_expert
    dt = np.dtype([("layer", "<i2"), ("nsel", "<i2"), ("sel", "<i4", (8,)), ("hid", "<f4", (H,))])
    arr = np.fromfile(args.trace, dtype=dt)
    print(f"records={len(arr)}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    agg = {8: [], 16: [], 24: []}
    for L in range(1, args.n_layer):
        idx = np.where(arr["layer"] == L)[0][:args.max_per_layer]
        if len(idx) < 400:
            continue
        X = torch.tensor(np.ascontiguousarray(arr["hid"][idx]), device=dev)
        s = arr["sel"][idx]
        Y = torch.zeros(len(idx), E, device=dev)
        for j in range(8):
            c = torch.tensor(s[:, j].astype(np.int64), device=dev)
            m = c >= 0
            Y[torch.arange(len(idx), device=dev)[m], c[m]] = 1.0
        ntr = int(len(idx) * 0.8)
        Xtr, Xte, Ytr, Yte = X[:ntr], X[ntr:], Y[:ntr], Y[ntr:]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        if args.model == "linear":
            net = nn.Linear(H, E).to(dev)
        else:
            net = nn.Sequential(nn.Linear(H, 512), nn.GELU(), nn.Linear(512, E)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(120):
            opt.zero_grad()
            lf(net(Xtr), Ytr).backward()
            opt.step()
        with torch.no_grad():
            order = torch.argsort(net(Xte), dim=1, descending=True)
            den = Yte.sum(1).clamp(min=1)
            for k in agg:
                agg[k].append((100 * (Yte.gather(1, order[:, :k]).sum(1) / den).mean()).item())

    print(f"=== {args.model} pre-gate recall@K (held-out, mean over layers) ===")
    for k in sorted(agg):
        if agg[k]:
            print(f"  recall@{k:2d}: {np.mean(agg[k]):5.1f}%  (worst layer {min(agg[k]):.0f}%)")


if __name__ == "__main__":
    main()
