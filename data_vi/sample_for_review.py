"""M5: sample 50 chunks for manual review of the extracted triples.

    python data_vi/sample_for_review.py                # writes data_vi/work/openie_review_50.csv

Chunks are stratified over 5 article groups (10 each, seed 2026) derived from the section
headers of data_vi/work/seed_titles.txt (aligned line by line with data_vi/articles.txt). Chunks used in the OpenIE prompt pilot
(data_vi/work/openie_pilot_ids.txt) are excluded because the prompt was tuned on them.
15 of the 50 chunks are marked for double labeling (agreement check).
Output: one row per triple; the chunk text is repeated only on the chunk's first row.
"""
import argparse
import csv
import json
import os
import random
import re

WORK = os.path.join("data_vi", "work")

# seed_titles.txt section header -> review group
GROUPS = {
    "Dynasties, states, periods": "G1 dynasties, places, monuments",
    "Capitals, places, monuments": "G1 dynasties, places, monuments",
    "Ly kings": "G2 kings",
    "Tran kings": "G2 kings",
    "Le so kings": "G2 kings",
    "Ly court, generals, monks": "G3 court, generals, royal family",
    "Tran royal family, generals, officials, scholars": "G3 court, generals, royal family",
    "Ho dynasty and Later Tran": "G3 court, generals, royal family",
    "Lam Son uprising and Le so court": "G3 court, generals, royal family",
    "Foreign rulers and commanders": "G4 foreign rulers, wars, battles",
    "Wars, battles, uprisings, assemblies": "G4 foreign rulers, wars, battles",
    "Texts, laws, institutions": "G5 texts, laws, institutions",
}
LABELS = "dung | sai | bia | trung | dai_tu | mo_ho"


def title_groups(seed_path, articles_path):
    """Map every corpus title to its review group.

    data_vi/articles.txt (final titles after the probe) follows seed_titles.txt line by line,
    so the i-th article takes the section of the i-th non-comment seed line."""
    seed, section = [], None
    for line in open(seed_path, encoding="utf-8"):
        line = line.strip()
        m = re.match(r"#\s*---\s*(.+?)\s*---", line)
        if m:
            section = m.group(1)
        elif line and not line.startswith("#"):
            seed.append(GROUPS[section])
    articles = [l.strip() for l in open(articles_path, encoding="utf-8") if l.strip() and not l.startswith("#")]
    if len(articles) != len(seed):
        raise SystemExit(f"articles.txt has {len(articles)} titles, seed_titles.txt {len(seed)}")
    return dict(zip(articles, seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_group", type=int, default=10)
    ap.add_argument("--n_double", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--openie", default=os.path.join("data_vi", "release", "openie_results_ner.json"))
    ap.add_argument("--out", default=os.path.join(WORK, "openie_review_50.csv"))
    args = ap.parse_args()

    groups = title_groups(os.path.join(WORK, "seed_titles.txt"), os.path.join("data_vi", "articles.txt"))
    pilot = {l.strip() for l in open(os.path.join(WORK, "openie_pilot_ids.txt"), encoding="utf-8") if l.strip()}
    meta = [json.loads(l) for l in open(os.path.join(WORK, "chunks_meta.jsonl"), encoding="utf-8")]
    docs = {d["idx"]: d for d in json.load(open(args.openie, encoding="utf-8"))["docs"]}

    missing = sorted({m["title"] for m in meta if m["title"] not in groups})
    if missing:
        raise SystemExit(f"Titles without a group: {missing}")

    rng = random.Random(args.seed)
    picked = []
    for g in sorted(set(GROUPS.values())):
        pool = sorted((m for m in meta if groups[m["title"]] == g and m["short_id"] not in pilot),
                      key=lambda m: m["short_id"])
        picked += rng.sample(pool, args.n_per_group)
    double = {m["short_id"] for m in rng.sample(picked, args.n_double)}

    rows = []
    for m in sorted(picked, key=lambda m: m["short_id"]):
        d = docs[m["idx"]]
        triples = d["extracted_triples"] or [["", "", ""]]
        for i, t in enumerate(triples):
            t = list(t) + [""] * (3 - len(t))
            rows.append({
                "short_id": m["short_id"], "group": groups[m["title"]],
                "double": "x" if m["short_id"] in double else "",
                "title": m["title"] if i == 0 else "", "section": m["section"] if i == 0 else "",
                "text": d["text"] if i == 0 else "", "triple_no": i + 1 if d["extracted_triples"] else 0,
                "subject": t[0], "relation": t[1], "object": t[2],
                "label": "", "note": "", "missing_facts": "",
            })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(picked)} chunks ({args.n_double} double-labeled), {len(rows)} triples -> {args.out}")
    print(f"labels: {LABELS}")


if __name__ == "__main__":
    main()
