#!/usr/bin/env python3
"""Step 2 of Spark calibration: corpus -> hot/cold placement profile.

Feeds each corpus chunk through the dflash daemon in hybrid mode with
`DFLASH_LAGUNA_NEXT_PLACEMENT_OUT` set, so per-(layer, expert) routing
frequencies accumulate across the whole corpus. The resulting CSV is the
placement profile: load it at serve time with `DFLASH_LAGUNA_HOTNESS=<csv>` and
the greedy knapsack keeps the most-frequent experts resident.

Routing is placement-independent, so calibrate at a high budget (fast, little
cold) regardless of the budget you deploy at.

    python -m spark.calibrate \
        --bin  ../../server/build/test_dflash \
        --gguf laguna-xs2-Q4_K_M.gguf \
        --tok  laguna_tok.json \
        --corpus corpus/train.jsonl --out-profile spark_profile.csv
"""
import argparse
import json
import os
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


def wait_ready(proc, timeout=600):
    t0 = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("daemon died before ready banner")
        if "-daemon] ready" in line:
            return
        if time.time() - t0 > timeout:
            proc.kill()
            raise SystemExit("timeout waiting for ready banner")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="path to test_dflash")
    ap.add_argument("--gguf", required=True, help="laguna GGUF")
    ap.add_argument("--tok", required=True, help="tokenizer.json (see spark.tokenizer)")
    ap.add_argument("--corpus", required=True, help="train.jsonl from extract_sessions")
    ap.add_argument("--out-profile", required=True, help="output placement CSV")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--budget-pct", type=int, default=90,
                    help="high = fast calibration; routing is placement-independent")
    ap.add_argument("--n-gen", type=int, default=8, help="decode tokens per chunk")
    ap.add_argument("--max-tok", type=int, default=2048, help="cap per-chunk prompt tokens")
    args = ap.parse_args()

    tk = Tokenizer.from_file(args.tok)
    chunks = [json.loads(l) for l in open(args.corpus)]
    print(f"[calib] {len(chunks)} chunks", flush=True)

    env = dict(os.environ)
    env["DFLASH_IGNORE_EOS"] = "1"
    env["DFLASH_EXPERT_BUDGET_PCT"] = str(args.budget_pct)
    env["DFLASH_LAGUNA_NEXT_PLACEMENT_OUT"] = str(Path(args.out_profile))
    proc = subprocess.Popen([args.bin, args.gguf, "--max-ctx", str(args.max_ctx)],
                            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=1, universal_newlines=True)
    wait_ready(proc)

    pp = Path(args.out_profile).with_suffix(".chunk.bin")
    op = Path(args.out_profile).with_suffix(".out.bin")
    fed = toks = 0
    t0 = time.time()
    for i, c in enumerate(chunks):
        ids = tk.encode(c).ids[:args.max_tok]
        if len(ids) < 8:
            continue
        write_counted_i32(pp, ids)
        toks += len(ids)
        try:
            proc.stdin.write(f"generate {pp} {args.n_gen} {op}\n")
            proc.stdin.flush()
        except BrokenPipeError:
            print(f"[calib] daemon closed at chunk {i}", flush=True)
            break
        ts = time.time()
        ok = False
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("ok ") or line.startswith("err "):
                ok = True
                break
            if time.time() - ts > 120:
                break
        if not ok:
            break
        fed += 1
        if fed % 50 == 0:
            el = time.time() - t0
            print(f"[calib] {fed}/{len(chunks)} chunks, {toks} tok, {el:.0f}s ({toks/el:.0f} tok/s)",
                  flush=True)

    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    pr = Path(args.out_profile)
    print(f"[calib] done: fed={fed} tok={toks} -> "
          f"{'SAVED ' + str(pr.stat().st_size) + 'B' if pr.exists() else 'PROFILE MISSING'}: {pr}",
          flush=True)


if __name__ == "__main__":
    main()
