#!/usr/bin/env python3
"""Extract a usable HF tokenizer from a GGUF (gpt2 byte-level BPE).

The Spark tools tokenize corpus text with the *target model's own* tokenizer.
GGUFs embed the vocab + merges, so we can rebuild a `tokenizers` BPE without
needing the original HF repo.

    python -m spark.tokenizer --gguf laguna-xs2-Q4_K_M.gguf --out laguna_tok.json
"""
import argparse

import numpy as np  # noqa: F401  (gguf reader pulls it in)
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


def extract(gguf_path: str, out_path: str):
    import gguf
    rd = gguf.GGUFReader(gguf_path)
    f = rd.fields

    def list_str(fl):
        return [bytes(fl.parts[d]).decode("utf-8", errors="replace") for d in fl.data]

    model = bytes(f["tokenizer.ggml.model"].parts[f["tokenizer.ggml.model"].data[0]]).decode()
    if model != "gpt2":
        raise SystemExit(f"only gpt2 byte-level BPE supported, gguf says model={model!r}")

    tokens = list_str(f["tokenizer.ggml.tokens"])
    merges = list_str(f["tokenizer.ggml.merges"]) if "tokenizer.ggml.merges" in f else []
    vocab = {t: i for i, t in enumerate(tokens)}
    bpe = models.BPE(vocab=vocab,
                     merges=[tuple(m.split(" ", 1)) for m in merges if " " in m],
                     fuse_unk=False)
    tk = Tokenizer(bpe)
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()

    # sanity: round-trip
    probe = "def f(n):\n    return n + 1\n"
    if tk.decode(tk.encode(probe).ids).strip() != probe.strip():
        print("warning: tokenizer round-trip mismatch (continuing)")

    tk.save(out_path)
    print(f"{out_path}: vocab={len(tokens)} merges={len(merges)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default="tokenizer.json")
    extract(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
