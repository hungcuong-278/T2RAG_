"""Self-check of the section-2 data contract for vnhistory (corpus and OpenIE; questions come later).

Run from the project root:
    python data_vi/validate_dataset.py --corpus
    python data_vi/validate_dataset.py --openie
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter

CORPUS_PATH = os.path.join("reproduce", "dataset", "vnhistory_corpus.json")
OPENIE_PATH = os.path.join("outputs", "vnhistory_univ", "openie_results_ner.json")
DOC_KEYS = {"idx", "title", "text", "passage", "extracted_entities", "extracted_triples"}
HAN = re.compile(r"[\u4e00-\u9fff]")


def passage(item):
    return item["title"] + "\n" + item["text"]


def check_corpus():
    errors = []
    corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        return [f"{CORPUS_PATH} must be a non-empty list"], corpus
    seen = Counter()
    for i, item in enumerate(corpus):
        if set(item) != {"title", "text"}:
            errors.append(f"item {i}: keys must be exactly title, text (got {sorted(item)})")
            continue
        for key in ("title", "text"):
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"item {i}: empty {key}")
            elif not unicodedata.is_normalized("NFC", value):
                errors.append(f"item {i}: {key} is not NFC")
            elif value != value.strip() or "  " in value:
                errors.append(f"item {i}: {key} has extra whitespace")
        if "\n" in item["text"]:
            errors.append(f"item {i}: text contains a newline")
        seen[hashlib.md5(passage(item).encode()).hexdigest()] += 1
    dups = sum(c - 1 for c in seen.values() if c > 1)
    if dups:
        errors.append(f"{dups} duplicate passages")
    lens = [len(it["text"].split()) for it in corpus]
    per_title = Counter(it["title"] for it in corpus)
    print(f"corpus: {len(corpus)} chunks, {len(per_title)} articles, syllables mean {statistics.mean(lens):.0f} "
          f"median {statistics.median(lens):.0f} min {min(lens)} max {max(lens)}, "
          f"chunks/article min {min(per_title.values())} max {max(per_title.values())}")
    return errors, corpus


def check_openie(corpus):
    """OpenIE file must cover the corpus one-to-one, in order, with clean string triples."""
    errors = []
    data = json.load(open(OPENIE_PATH, encoding="utf-8"))
    if set(data) != {"docs", "avg_ent_chars", "avg_ent_words"}:
        errors.append(f"top-level keys must be docs, avg_ent_chars, avg_ent_words (got {sorted(data)})")
    docs = data.get("docs", [])
    if len(docs) != len(corpus):
        errors.append(f"{len(docs)} docs but {len(corpus)} corpus chunks")
    n_tri, n_ent, n_han, empty = 0, 0, 0, []
    for i, (doc, item) in enumerate(zip(docs, corpus)):
        if set(doc) != DOC_KEYS:
            errors.append(f"doc {i}: keys {sorted(doc)}")
            continue
        p = passage(item)
        if doc["passage"] != p or doc["title"] != item["title"] or doc["text"] != item["text"]:
            errors.append(f"doc {i}: title/text/passage differ from the corpus")
        if doc["idx"] != "chunk-" + hashlib.md5(p.encode()).hexdigest():
            errors.append(f"doc {i}: idx is not chunk-md5(passage)")
        for e in doc["extracted_entities"]:
            if not isinstance(e, str) or not e.strip() or not unicodedata.is_normalized("NFC", e):
                errors.append(f"doc {i}: bad entity {e!r}")
        n_ent += len(doc["extracted_entities"])
        for t in doc["extracted_triples"]:
            if not (isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) and x.strip() for x in t)):
                errors.append(f"doc {i}: bad triple {t!r}")
                continue
            if not all(unicodedata.is_normalized("NFC", x) for x in t):
                errors.append(f"doc {i}: triple not NFC {t!r}")
            n_han += any(HAN.search(x) for x in t)
        n_tri += len(doc["extracted_triples"])
        if not doc["extracted_triples"]:
            empty.append(i)
    if empty:
        errors.append(f"{len(empty)} docs without triples (e.g. {empty[:5]})")
    print(f"openie: {len(docs)} docs, {n_ent} entities ({n_ent / max(len(docs), 1):.1f}/doc), "
          f"{n_tri} triples ({n_tri / max(len(docs), 1):.1f}/doc), {n_han} triples with Han characters, "
          f"avg_ent_chars {data.get('avg_ent_chars')}, avg_ent_words {data.get('avg_ent_words')}")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--openie", action="store_true", help="Also check outputs/vnhistory_univ/openie_results_ner.json")
    args = ap.parse_args()
    if not (args.corpus or args.openie):
        ap.error("choose what to check, e.g. --corpus or --openie")
    errors, corpus = check_corpus()
    if args.openie:
        errors += check_openie(corpus)
    for e in errors[:50]:
        print("ERROR", e)
    print("OK" if not errors else f"FAILED: {len(errors)} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
