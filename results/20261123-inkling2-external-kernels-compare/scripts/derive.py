import json, glob, sys
def toks(run, dep, i): return json.load(open(f"{run}/decode-{dep}-{i}.json"))["tokens"]
def fd(x, y): return next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))
R = {}
for d in glob.glob("2026112*-r[12]"):
    cfg, s, r = d.split("-")[2:5]; R[(cfg, s, r)] = d
for cfg in "ABC":
    for dep, n in ((0, 6), (6000, 3), (30000, 3)):
        sp = [fd(toks(R[(cfg, "s0", "r1")], dep, i), toks(R[(cfg, "s1", "r1")], dep, i)) for i in range(n)]
        rr0 = [fd(toks(R[(cfg, "s0", "r1")], dep, i), toks(R[(cfg, "s0", "r2")], dep, i)) for i in range(n)]
        rr1 = [fd(toks(R[(cfg, "s1", "r1")], dep, i), toks(R[(cfg, "s1", "r2")], dep, i)) for i in range(n)]
        print(cfg, dep, "spec-vs-nospec:", sp, "| s0 r1/r2:", rr0, "| s1 r1/r2:", rr1)
for cfg in "ABC":
    for r in ("r1", "r2"):
        s = json.load(open(f"{R[(cfg, 's1', r)]}/decode-0-summary.json"))
        print(cfg, r, "s1 per_case", s["per_case"], s["spec"])
