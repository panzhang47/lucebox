#!/usr/bin/env python3
"""Reproduce the Luce Spark decode-throughput numbers.

Drives the dflash daemon (server/build/test_dflash) over a fixed prompt and
reports steady-state decode tok/s for three configs:

  1. all-GPU (full residency)              -> the speed ceiling
  2. single-graph @ 100% residency         -> exercises the hybrid graph with
                                              every expert resident; output must
                                              match all-GPU token-for-token
  3. Spark offload (single-graph hybrid)   -> pinned `--budget-pct` of experts +
                                              a bounded GPU cache, the rest on CPU

`decode_tok_s` is generated_tokens / decode_seconds (prefill excluded), the same
metric the daemon prints. The single-graph hybrid lives in LagunaBackend, which
also powers `dflash_server --spark`, so these decode numbers carry over to the
server.

    python -m spark.bench \
        --bin ../../server/build/test_dflash \
        --gguf laguna-xs2-Q4_K_M.gguf --tok laguna_tok.json \
        --hotness laguna-xs2-Q4_K_M.gguf.spark.csv \
        --budget-pct 48 --cache-slots 32
"""
import argparse
import os
import re
import struct
import sys

from tokenizers import Tokenizer

try:
    from ._daemon import Daemon
except ImportError:  # allow `python bench.py`
    from _daemon import Daemon

IN_BIN = "/tmp/spark_bench_in.bin"
OUT_BIN = "/tmp/spark_bench_out.bin"


def write_counted_i32(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack("<%di" % len(ids), *(int(t) for t in ids)))


def read_i32(path):
    raw = open(path, "rb").read()
    a = list(struct.unpack("<%di" % (len(raw) // 4), raw))
    return a[1:] if a and a[0] == len(a) - 1 else a  # tolerate a count prefix


def measure(bin_path, gguf, env, n_gen, ready_timeout, gen_timeout, warmup=1):
    """Launch a daemon, run warmup+1 generations, return (decode_tok_s, out_ids)."""
    d = Daemon([bin_path, gguf, "--max-ctx", "4096"], env, capture_stderr=True)
    d.wait_ready(banner="daemon] ready", timeout=ready_timeout)
    reply = None
    for _ in range(warmup + 1):
        reply = d.request(f"generate {IN_BIN} {n_gen} {OUT_BIN}", timeout=gen_timeout)
        if reply is None:
            d.kill()
            raise SystemExit("daemon stalled or died during generate")
    d.quit()
    m = re.search(r"decode_tok_s=([0-9.]+)", reply)
    return (float(m.group(1)) if m else 0.0), read_i32(OUT_BIN)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="path to test_dflash")
    ap.add_argument("--gguf", required=True, help="target GGUF")
    ap.add_argument("--tok", required=True, help="tokenizer.json")
    ap.add_argument("--hotness", default="", help="placement CSV (spark.calibrate output)")
    ap.add_argument("--budget-pct", type=int, default=48,
                    help="pinned-hot experts (%%). total residency = budget-pct + cache-slots/n_expert")
    ap.add_argument("--cache-slots", type=int, default=32, help="bounded GPU expert cache slots/layer")
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-expert", type=int, default=256, help="experts/layer (for the residency print)")
    ap.add_argument("--bos", type=int, default=2, help="BOS token id to prepend (-1 = none)")
    ap.add_argument("--prompt", default="def fibonacci(n):\n")
    ap.add_argument("--ready-timeout", type=int, default=600)
    ap.add_argument("--gen-timeout", type=int, default=240)
    args = ap.parse_args()

    tk = Tokenizer.from_file(args.tok)
    ids = tk.encode(args.prompt, add_special_tokens=False).ids
    if args.bos >= 0:
        ids = [args.bos] + ids
    write_counted_i32(IN_BIN, ids)

    base = dict(os.environ)
    base["DFLASH_IGNORE_EOS"] = "1"  # fixed-length decode for a stable tok/s

    def cfg(**kw):
        e = dict(base)
        e.update(kw)
        return e

    offload = dict(DFLASH_LAGUNA_GPU_REMAP="1", DFLASH_LAGUNA_EXPERT_CACHE="1",
                   DFLASH_LAGUNA_CACHE_SLOTS=str(args.cache_slots))
    if args.hotness:
        offload["DFLASH_LAGUNA_HOTNESS"] = args.hotness

    total_pct = args.budget_pct + 100.0 * args.cache_slots / args.n_expert
    print(f"prompt={args.prompt!r}  n_gen={args.n_gen}  bos={args.bos}")
    print(f"offload: pinned {args.budget_pct}% + cache {args.cache_slots}/{args.n_expert} "
          f"=> ~{total_pct:.0f}% total residency\n")

    tps_all, ref = measure(args.bin, args.gguf, cfg(),
                           args.n_gen, args.ready_timeout, args.gen_timeout)
    print(f"  all-GPU (full residency)            {tps_all:6.1f} tok/s")

    tps_100, out_100 = measure(args.bin, args.gguf,
                               cfg(DFLASH_EXPERT_BUDGET_PCT="100", **offload),
                               args.n_gen, args.ready_timeout, args.gen_timeout)
    n = min(len(out_100), len(ref))
    match = sum(1 for x, y in zip(out_100, ref) if x == y)
    exact = "EXACT" if match == n and n > 0 else "DIVERGED"
    print(f"  single-graph @100% (vs all-GPU)     {tps_100:6.1f} tok/s   {match}/{n} {exact}")

    tps_off, _ = measure(args.bin, args.gguf,
                         cfg(DFLASH_EXPERT_BUDGET_PCT=str(args.budget_pct), **offload),
                         args.n_gen, args.ready_timeout, args.gen_timeout)
    pct = 100.0 * tps_off / max(1e-9, tps_all)
    print(f"  Spark offload (single-graph)        {tps_off:6.1f} tok/s   {pct:.0f}% of all-GPU")

    ok = (match == n and n > 0)
    print(f"\nresult: {'PASS' if ok else 'FAIL'} (single-graph @100% must match all-GPU token-for-token)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
