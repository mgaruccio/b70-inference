"""Rebuild prompts.json (the frequency-collection prompt mix) from public datasets.

Seeded sample via the Hugging Face datasets-server rows API: UltraChat 200k test_sft,
Magicoder-Evol-Instruct-110K, GSM8K train, and non-English Aya rows. The original
file had 2,679 prompts; its sha256 is in README.md.

Usage: python3 -I fetch_prompts.py prompts.json
"""
import json
import random
import sys
import urllib.parse
import urllib.request


def rows(ds, config, split, offset, length=100):
    q = urllib.parse.urlencode(dict(dataset=ds, config=config, split=split, offset=offset, length=length))
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def main():
    rng = random.Random(20261125)
    out = []

    def take(ds, config, split, n, total, get, tag):
        for o in sorted(rng.sample(range(0, total - 100, 100), n // 100)):
            for r in rows(ds, config, split, o):
                p = get(r)
                if p and 10 < len(p) < 6000:
                    out.append(dict(source=tag, prompt=p))

    take("HuggingFaceH4/ultrachat_200k", "default", "test_sft", 1200, 23000, lambda r: r["prompt"], "ultrachat")
    take("ise-uiuc/Magicoder-Evol-Instruct-110K", "default", "train", 800, 110000, lambda r: r["instruction"], "magicoder")
    take("openai/gsm8k", "main", "train", 300, 7400, lambda r: r["question"], "gsm8k")
    take("CohereLabs/aya_dataset", "default", "train", 400, 200000,
         lambda r: r["inputs"] if r.get("language") != "English" else None, "aya")
    rng.shuffle(out)
    json.dump(out, open(sys.argv[1], "w"))
    print(len(out))


if __name__ == "__main__":
    main()
