#!/usr/bin/env python3
"""Step 1 of Spark calibration: extract a text corpus from agent session logs.

Spark calibrates expert placement on *representative traffic*, so the corpus
should be whatever the model actually serves. This reads Claude Code session
transcripts (JSONL) and pulls the conversational text, thinking, tool inputs and
tool results. It splits train/held-out by session hash so validation never sees
calibration data.

    python -m spark.extract_sessions \
        --sessions-dir ~/.claude/projects \
        --out-dir ./corpus

Produces corpus/train.jsonl (calibration) and corpus/test.jsonl (held-out),
one JSON-encoded text chunk per line.
"""
import argparse
import hashlib
import json
from pathlib import Path


def text_from_content(content):
    """Flatten a message's content (str | list of blocks) into plain text."""
    out = []
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and b.get("text"):
                out.append(b["text"])
            elif t == "thinking" and b.get("thinking"):
                out.append(b["thinking"])
            elif t == "tool_use" and b.get("input") is not None:
                out.append(json.dumps(b["input"])[:4000])
            elif t == "tool_result":
                cc = b.get("content")
                if isinstance(cc, str):
                    out.append(cc)
                elif isinstance(cc, list):
                    for x in cc:
                        if isinstance(x, dict) and x.get("type") == "text":
                            out.append(x.get("text", ""))
    return "\n".join(s for s in out if s)


def chunks_from_text(txt, size, min_size):
    txt = txt.strip()
    for i in range(0, len(txt), size):
        c = txt[i:i + size]
        if len(c) >= min_size:
            yield c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-dir", default=str(Path.home() / ".claude" / "projects"),
                    help="dir with <project>/<session>.jsonl transcripts")
    ap.add_argument("--out-dir", default="./corpus")
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--per-session", type=int, default=4, help="chunks sampled per session")
    ap.add_argument("--test-frac", type=int, default=5, help="1/N sessions held out (5 = 20%%)")
    ap.add_argument("--train-cap", type=int, default=500)
    ap.add_argument("--test-cap", type=int, default=60)
    args = ap.parse_args()

    files = sorted(Path(args.sessions_dir).glob("*/*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl transcripts under {args.sessions_dir}")

    train, test = [], []
    for f in files:
        # split by session-path hash so a whole session is train xor test
        bucket = test if int(hashlib.md5(str(f).encode()).hexdigest(), 16) % args.test_frac == 0 else train
        sess = []
        try:
            for ln in f.open():
                try:
                    o = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if o.get("type") in ("user", "assistant") and isinstance(o.get("message"), dict):
                    t = text_from_content(o["message"].get("content"))
                    if t:
                        sess.extend(chunks_from_text(t, args.chunk_chars, args.min_chars))
        except OSError:
            continue
        if sess:
            step = max(1, len(sess) // args.per_session)
            bucket.extend(sess[::step][:args.per_session])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, items, cap):
        items = items[:cap]
        p = out / name
        with p.open("w") as fo:
            for c in items:
                fo.write(json.dumps(c) + "\n")
        nchar = sum(len(c) for c in items)
        print(f"{p}: {len(items)} chunks, ~{nchar // 1000}K chars (~{nchar // 4000}K tok est)")

    write("train.jsonl", train, args.train_cap)
    write("test.jsonl", test, args.test_cap)
    print(f"split by session ({len(files)} sessions, 1/{args.test_frac} held out)")


if __name__ == "__main__":
    main()
