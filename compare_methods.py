"""Compare two runs of run_colab.py on the same questions (e.g. T2RAG vs HippoRAG2).

Example (from the repo root):
    python compare_methods.py outputs/results/musique_trag_<llm>_200q.json \
                              outputs/results/musique_hippo_<llm>_200q.json

It answers one question: how many questions does method A get wrong that method B gets right?
That count is the most a "use B when A is stuck" combination (Huong 2) could gain.
Results are printed per hop count (MuSiQue) and the questions where only B is right
are saved to a CSV file for manual inspection.
"""
import argparse
import csv
import json
import os
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    name = data["summary"]["method"]
    records = {r.get("qid", r["id"]): r for r in data["records"]}
    return name, data["summary"], records


def pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "   - "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="Result JSON of method A (e.g. T2RAG)")
    ap.add_argument("run_b", help="Result JSON of method B (e.g. HippoRAG2)")
    ap.add_argument("--f1_threshold", type=float, default=0.5,
                    help="A question also counts as correct when F1 >= this value (default 0.5)")
    ap.add_argument("--out_csv", default=None, help="Where to save the questions only B gets right")
    args = ap.parse_args()

    name_a, sum_a, rec_a = load(args.run_a)
    name_b, sum_b, rec_b = load(args.run_b)
    common = [q for q in rec_a if q in rec_b]
    if not common:
        raise SystemExit("The two runs share no questions. Run both with the same --dataset and --limit.")
    if len(common) < max(len(rec_a), len(rec_b)):
        print(f"Note: only {len(common)} questions are in both runs; comparing those.")

    def correct(r):
        return r["em"] >= 1.0 or r["f1"] >= args.f1_threshold

    groups = defaultdict(lambda: {"n": 0, "both": 0, "only_a": 0, "only_b": 0, "none": 0,
                                  "em_a": 0.0, "em_b": 0.0, "f1_a": 0.0, "f1_b": 0.0,
                                  "em_oracle": 0.0, "f1_oracle": 0.0})
    only_b_rows = []
    for q in common:
        a, b = rec_a[q], rec_b[q]
        hop = a.get("hops") or b.get("hops")
        for key in ("all", f"{hop} buoc" if hop else None):
            if key is None:
                continue
            g = groups[key]
            g["n"] += 1
            ca, cb = correct(a), correct(b)
            g["both"] += ca and cb
            g["only_a"] += ca and not cb
            g["only_b"] += cb and not ca
            g["none"] += not ca and not cb
            g["em_a"] += a["em"]
            g["em_b"] += b["em"]
            g["f1_a"] += a["f1"]
            g["f1_b"] += b["f1"]
            g["em_oracle"] += max(a["em"], b["em"])
            g["f1_oracle"] += max(a["f1"], b["f1"])
        if correct(b) and not correct(a):
            only_b_rows.append({"qid": q, "hops": hop, "question": a["question"],
                                "gold": " | ".join(a["gold_answers"]),
                                f"{name_a}_answer": a["prediction"], f"{name_b}_answer": b["prediction"],
                                f"{name_a}_f1": a["f1"], f"{name_b}_f1": b["f1"]})

    print(f"\nA = {name_a} ({sum_a['llm']}, {sum_a['embedding']})")
    print(f"B = {name_b} ({sum_b['llm']}, {sum_b['embedding']})")
    print(f"Correct = EM 1 or F1 >= {args.f1_threshold}\n")

    header = (f"{'Nhom':8s} {'So cau':>6s} | {'EM A':>6s} {'EM B':>6s} {'EM tran':>7s} | "
              f"{'F1 A':>6s} {'F1 B':>6s} {'F1 tran':>7s} | {'Ca hai dung':>11s} {'Chi A dung':>10s} "
              f"{'Chi B dung':>10s} {'Ca hai sai':>10s}")
    print(header)
    print("-" * len(header))
    order = ["all"] + sorted(k for k in groups if k != "all")
    for key in order:
        g = groups[key]
        n = g["n"]
        print(f"{key:8s} {n:6d} | {100*g['em_a']/n:6.1f} {100*g['em_b']/n:6.1f} {100*g['em_oracle']/n:7.1f} | "
              f"{100*g['f1_a']/n:6.1f} {100*g['f1_b']/n:6.1f} {100*g['f1_oracle']/n:7.1f} | "
              f"{g['both']:5d} {pct(g['both'], n)} {g['only_a']:4d} {pct(g['only_a'], n)} "
              f"{g['only_b']:4d} {pct(g['only_b'], n)} {g['none']:4d} {pct(g['none'], n)}")

    g = groups["all"]
    print(f"\n'EM tran' / 'F1 tran' = diem neu luon chon dung phuong phap tot hon cho tung cau "
          f"(muc tran ma Huong 2 co the dat).")
    print(f"So cau {name_a} sai nhung {name_b} dung: {g['only_b']} / {g['n']} "
          f"-> Huong 2 co the tang toi da khoang {100*(g['em_oracle']-g['em_a'])/g['n']:.1f} diem EM "
          f"so voi {name_a}.")

    out_csv = args.out_csv or os.path.join(os.path.dirname(args.run_a) or ".",
                                           f"only_{name_b}_correct_vs_{name_a}.csv")
    if only_b_rows:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(only_b_rows[0].keys()))
            writer.writeheader()
            writer.writerows(only_b_rows)
        print(f"Danh sach {len(only_b_rows)} cau chi {name_b} dung: {out_csv}")


if __name__ == "__main__":
    main()
