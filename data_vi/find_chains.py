"""Find candidate hop chains for multi-hop questions (M3).

Run from the project root:
    python data_vi/find_chains.py                 # build mention index + candidate chains
    python data_vi/find_chains.py --show c0123    # print one chunk with its outgoing mentions

A chunk "mentions" an article when the article title (without a trailing "(...)") or one of
its redirect aliases appears as a whole phrase. Chains:
    2-hop: chunk A (article X) mentions Y  ->  chunk B of article Y
    3-hop: A mentions Y -> chunk B of Y mentions Z -> chunk C of Z
Very generic articles (dynasties, the state, the capital) are not used as bridges.
"""
import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict

WORK = os.path.join("data_vi", "work")
CORPUS_PATH = os.path.join("reproduce", "dataset", "vnhistory_corpus.json")
GENERIC = {"Nhà Lý", "Nhà Trần", "Nhà Hồ", "Nhà Hậu Trần", "Nhà Lê sơ", "Đại Việt", "Đại Ngu", "Thăng Long",
           "Chăm Pa", "Thời kỳ Bắc thuộc lần thứ tư", "Đại Việt sử ký toàn thư", "Hoàng thành Thăng Long"}


def nfc(text):
    return unicodedata.normalize("NFC", text)


def load():
    corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
    meta = [json.loads(l) for l in open(os.path.join(WORK, "chunks_meta.jsonl"), encoding="utf-8")]
    aliases = json.load(open(os.path.join(WORK, "aliases.json"), encoding="utf-8"))
    for m, c in zip(meta, corpus):
        m["text"] = c["text"]
    return meta, aliases


def name_table(titles, aliases):
    """Surface name -> article title; names shared by two articles are dropped as ambiguous."""
    owners = defaultdict(set)
    for t in titles:
        names = {t, re.sub(r"\s*\(.*\)$", "", t)} | set(aliases.get(t, []))
        for n in names:
            n = nfc(n).strip()
            if len(n.split()) >= 2:
                owners[n].add(t)
    return {n: next(iter(ts)) for n, ts in owners.items() if len(ts) == 1}


def find_mentions(meta, table):
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted(table, key=len, reverse=True)) + r")(?!\w)")
    out = {}
    for m in meta:
        found = Counter()
        for match in pattern.finditer(m["text"]):
            target = table[match.group(1)]
            if target != m["title"]:
                found[target] += 1
        out[m["short_id"]] = found
    return out


def sentences_with(text, names):
    sents = re.split(r"(?<=[.!?…])\s+", text)
    return [s for s in sents if any(re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", s) for n in names)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs="*", help="Short ids of chunks to print")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    meta, aliases = load()
    by_id = {m["short_id"]: m for m in meta}
    titles = sorted({m["title"] for m in meta})
    table = name_table(titles, aliases)
    names_of = defaultdict(list)
    for n, t in table.items():
        names_of[t].append(n)
    mentions = find_mentions(meta, table)

    if args.show:
        for sid in args.show:
            m = by_id[sid]
            print(f"=== {sid} | {m['title']} | {m['section']} | {m['n_tieng']} tiếng\n{m['text']}")
            print("mentions:", dict(mentions[sid]), "\n")
        return

    chunks_of = defaultdict(list)
    for m in meta:
        chunks_of[m["title"]].append(m["short_id"])

    rng = random.Random(args.seed)
    pairs, triples = [], []
    for m in meta:
        for y in mentions[m["short_id"]]:
            if y in GENERIC:
                continue
            for b in chunks_of[y]:
                pairs.append({"a": m["short_id"], "bridge": y, "b": b, "b_is_lead": by_id[b]["section"] == "(lead)"})
    for p in pairs:
        for z in mentions[p["b"]]:
            if z in GENERIC or z in (by_id[p["a"]]["title"], p["bridge"]):
                continue
            for c in chunks_of[z]:
                if by_id[c]["section"] == "(lead)":
                    triples.append({"a": p["a"], "bridge1": p["bridge"], "b": p["b"], "bridge2": z, "c": c})

    with open(os.path.join(WORK, "mentions.json"), "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k, v in mentions.items()}, f, ensure_ascii=False, indent=0)
    for name, rows in (("candidates_2hop.jsonl", pairs), ("candidates_3hop.jsonl", triples)):
        with open(os.path.join(WORK, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    deg = Counter(t for c in mentions.values() for t in c)
    with_out = sum(1 for c in mentions.values() if any(t not in GENERIC for t in c))
    print(f"{len(table)} surface names for {len(titles)} articles; {with_out}/{len(meta)} chunks mention a non-generic article")
    print(f"2-hop candidate pairs: {len(pairs)} ({sum(p['b_is_lead'] for p in pairs)} with a lead chunk as hop 2)")
    print(f"3-hop candidate chains (lead chunk as last hop): {len(triples)}")
    print("Most mentioned articles:", deg.most_common(12))


if __name__ == "__main__":
    main()
