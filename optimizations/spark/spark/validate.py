#!/usr/bin/env python3
"""Step 3: validate a placement profile on HELD-OUT chunks.

Drives the daemon over held-out corpus chunks and reports mean decode tok/s and
the cold-hit rate from the per-token profiler. Compare runs with/without
`--hotness` (uniform vs calibrated) and with/without the expert cache to see the
ladder in RESULTS.md.

    # all-GPU baseline
    python -m spark.validate --bin ... --gguf ... --tok ... \
        --corpus corpus/test.jsonl --budget-pct 0

    # calibrated 60% + bounded expert cache
    python -m spark.validate --bin ... --gguf ... --tok ... \
        --corpus corpus/test.jsonl --budget-pct 60 \
        --hotness spark_profile.csv --cache-slots 32
"""
import argparse
import json
import os
import re
import statistics
import struct
import subprocess
import threading
import time
from pathlib import Path

from tokenizers import Tokenizer


def write_counted_i32(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack("<%di" % len(ids), *(int(t) for t in ids)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--corpus", required=True, help="test.jsonl (held-out)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--budget-pct", type=int, default=60, help="0 = all-GPU")
    ap.add_argument("--hotness", default="", help="placement CSV (empty = uniform)")
    ap.add_argument("--cache-slots", type=int, default=0, help="bounded expert cache slots/layer")
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--n-chunks", type=int, default=30)
    ap.add_argument("--max-tok", type=int, default=1024)
    ap.add_argument("--natural", action="store_true", help="stop at EOS (no forced length)")
    args = ap.parse_args()

    tk = Tokenizer.from_file(args.tok)
    chunks = [json.loads(l) for l in open(args.corpus)][:args.n_chunks]

    env = dict(os.environ)
    env["DFLASH_LAGUNA_PROFILE"] = "1"
    if not args.natural:
        env["DFLASH_IGNORE_EOS"] = "1"
    if args.budget_pct > 0:
        env["DFLASH_EXPERT_BUDGET_PCT"] = str(args.budget_pct)
    if args.hotness:
        env["DFLASH_LAGUNA_HOTNESS"] = args.hotness
    if args.cache_slots > 0:
        env["DFLASH_LAGUNA_GPU_REMAP"] = "1"
        env["DFLASH_LAGUNA_EXPERT_CACHE"] = "1"
        env["DFLASH_LAGUNA_CACHE_SLOTS"] = str(args.cache_slots)

    proc = subprocess.Popen([args.bin, args.gguf, "--max-ctx", str(args.max_ctx)],
                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    prof, placement = [], []

    def drain():
        for ln in proc.stderr:
            if "lag-prof" in ln:
                prof.append(ln.rstrip())
    threading.Thread(target=drain, daemon=True).start()

    t0 = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("daemon died before ready")
        if "storage ready" in line or "placement result" in line:
            placement.append(line.rstrip())
        if "-daemon] ready" in line:
            break
        if time.time() - t0 > 600:
            proc.kill()
            raise SystemExit("ready timeout")

    pp = Path("/tmp/spark_val_chunk.bin")
    op = Path("/tmp/spark_val_out.bin")
    toks = []
    for c in chunks:
        ids = tk.encode(c).ids[:args.max_tok]
        if len(ids) < 8:
            continue
        write_counted_i32(pp, ids)
        proc.stdin.write(f"generate {pp} {args.n_gen} {op}\n")
        proc.stdin.flush()
        ts = time.time()
        okl = None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("ok ") or line.startswith("err "):
                okl = line
                break
            if time.time() - ts > 120:
                break
        if okl and okl.startswith("ok "):
            m = re.search(r"decode_tok_s=([0-9.]+)", okl)
            if m:
                toks.append(float(m.group(1)))

    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()

    for p in placement:
        print("PLACEMENT:", p.strip())
    label = f"budget={args.budget_pct} hotness={'yes' if args.hotness else 'no'} cache={args.cache_slots}"
    print(f"\n===== VALIDATION  {label} =====")
    if toks:
        print(f"decode tok/s: mean={statistics.mean(toks):.1f} median={statistics.median(toks):.1f} "
              f"over {len(toks)} chunks")
    colds = [float(m.group(1)) for l in prof
             for m in [re.search(r"cold_experts/tok=([0-9.]+)", l)] if m]
    if colds:
        print(f"cold_experts/tok: mean={statistics.mean(colds):.2f} max={max(colds):.2f}")


if __name__ == "__main__":
    main()
