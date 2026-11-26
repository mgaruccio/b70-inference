"""Build reduced DSpark draft vocabularies from Inkling-generated token IDs.

Requests are split 80/20 by index. Frequencies come from the 80% split; coverage
(share of generated tokens inside the kept set) is reported on the held-out 20%,
overall and per prompt source. Every added/special token from tokenizer.json is
always kept (chat, reasoning and tool-call tags).

Usage: python3 -I build_vocab.py TOKENS_JSONL TOKENIZER_JSON OUT_DIR [--sizes 16384 32768 65536]
"""
import argparse
import collections
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokens")
    ap.add_argument("tokenizer")
    ap.add_argument("out", type=Path)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16384, 32768, 65536])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(a.tokens)]
    train = [r for r in rows if r["i"] % 5 != 0]
    held = [r for r in rows if r["i"] % 5 == 0]
    freq = collections.Counter(t for r in train for t in r["token_ids"])
    tok = json.load(open(a.tokenizer))
    special = sorted({t["id"] for t in tok.get("added_tokens", [])})
    ranked = [t for t, _ in freq.most_common()]
    report = dict(requests=len(rows), train_tokens=sum(freq.values()),
                  heldout_tokens=sum(len(r["token_ids"]) for r in held),
                  distinct_train_tokens=len(freq), forced_special=len(special), sizes={})
    sources = sorted({r["source"] for r in rows})
    for k in a.sizes:
        keep = set(special)
        for t in ranked:
            if len(keep) >= k:
                break
            keep.add(t)
        def cov(rs):
            n = sum(len(r["token_ids"]) for r in rs)
            return sum(t in keep for r in rs for t in r["token_ids"]) / n if n else None
        report["sizes"][k] = dict(kept=len(keep), heldout_coverage=cov(held),
                                  by_source={s: cov([r for r in held if r["source"] == s]) for s in sources})
        (a.out / f"draft-vocab-{k}.json").write_text(json.dumps(dict(
            token_ids=sorted(keep), source="Inkling-2-Mini generated tokens, results/20261126-inkling2-draft-vocab",
            train_requests=len(train))) + "\n")
    (a.out / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
