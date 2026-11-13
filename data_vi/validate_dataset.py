"""Self-check of the section-2 data contract for vnhistory (corpus, OpenIE, questions).

Run from the project root:
    python data_vi/validate_dataset.py --corpus
    python data_vi/validate_dataset.py --openie
    python data_vi/validate_dataset.py --questions                    # reproduce/dataset/vnhistory.json
    python data_vi/validate_dataset.py --questions data_vi/work/vnhistory_preview.json
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
QUESTIONS_PATH = os.path.join("reproduce", "dataset", "vnhistory.json")
DEV_SIZE = 50
DEV_HOPS = {1: 15, 2: 23, 3: 12}
MAX_ANSWER_SYLLABLES = 6
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


def low(text):
    return unicodedata.normalize("NFC", text).lower()


def contains(haystack, needle):
    """Whole-word, case-insensitive match (Vietnamese syllables are separated by spaces)."""
    return re.search(r"(?<!\w)" + re.escape(low(needle)) + r"(?!\w)", low(haystack)) is not None


def is_str(x):
    return isinstance(x, str) and x.strip() != ""


def para_text(p):
    return p["title"] + "\n" + p["paragraph_text"]


def check_one(q, where, passages):
    """Contract 2.3 plus K3, K4, K5, K6, K9 for one question. Returns (errors, hops, final passage)."""
    errors = [f"{where}: missing {k}" for k in
              ("id", "question", "answer", "answer_aliases", "paragraphs", "question_decomposition") if k not in q]
    if errors:
        return errors, None, None
    m = re.fullmatch(r"([123])hop_\d{3}", str(q["id"]))
    hops = int(m.group(1)) if m else None
    if not m:
        errors.append(f"{where}: id must look like 1hop_001 / 2hop_001 / 3hop_001")
    for key in ("question", "answer"):
        if not is_str(q[key]):
            return errors + [f"{where}: {key} must be a non-empty string"], hops, None
        if not unicodedata.is_normalized("NFC", q[key]) or q[key] != q[key].strip():
            errors.append(f"{where}: {key} is not NFC or has outer whitespace")
    if len(q["answer"].split()) > MAX_ANSWER_SYLLABLES:
        errors.append(f"{where}: K4 answer longer than {MAX_ANSWER_SYLLABLES} syllables: {q['answer']!r}")
    aliases = q["answer_aliases"]
    if not isinstance(aliases, list) or not all(is_str(a) for a in aliases):
        errors.append(f"{where}: answer_aliases must be a list of non-empty strings")
        aliases = []
    for a in [q["answer"]] + aliases:
        if contains(q["question"], a):
            errors.append(f"{where}: K9 answer {a!r} appears in the question")

    paras = q["paragraphs"]
    if not isinstance(paras, list) or not paras:
        return errors + [f"{where}: paragraphs must be a non-empty list"], hops, None
    bad_para = False
    for j, p in enumerate(paras):
        if not (isinstance(p, dict) and is_str(p.get("title")) and is_str(p.get("paragraph_text"))
                and isinstance(p.get("is_supporting"), bool)):
            errors.append(f"{where}: paragraph {j} needs title, paragraph_text and a boolean is_supporting")
            bad_para = True
        elif para_text(p) not in passages:
            errors.append(f"{where}: paragraph {j} ({p['title']!r}) is not exactly a corpus passage")
    if bad_para:
        return errors, hops, None
    support = [j for j, p in enumerate(paras) if p["is_supporting"]]
    if len({para_text(paras[j]) for j in support}) != len(support):
        errors.append(f"{where}: K6 two supporting paragraphs are the same passage")

    steps = q["question_decomposition"]
    if not isinstance(steps, list) or not steps:
        return errors + [f"{where}: question_decomposition must be a non-empty list"], hops, None
    if hops is not None and not (hops == len(steps) == len(support)):
        errors.append(f"{where}: K6 id says {hops} hops but {len(steps)} steps "
                      f"and {len(support)} supporting paragraphs")
    used, final = [], None
    for k, st in enumerate(steps, 1):
        if not (isinstance(st, dict) and is_str(st.get("question")) and is_str(st.get("answer"))):
            errors.append(f"{where}: step {k} needs question and answer strings")
            continue
        j = st.get("paragraph_support_idx")
        if j not in support:
            errors.append(f"{where}: step {k} paragraph_support_idx {j!r} is not a supporting paragraph")
            continue
        used.append(j)
        if not contains(para_text(paras[j]), st["answer"]):
            errors.append(f"{where}: K5 step {k} answer {st['answer']!r} not found in its paragraph")
        if k < len(steps) and contains(q["question"], st["answer"]):
            errors.append(f"{where}: K3 bridge answer {st['answer']!r} (step {k}) appears in the question")
        if k == len(steps):
            final = para_text(paras[j])
            if low(st["answer"]) != low(q["answer"]):
                errors.append(f"{where}: last step answer {st['answer']!r} differs from answer {q['answer']!r}")
    if len(set(used)) != len(used):
        errors.append(f"{where}: K6 two steps share one paragraph")
    return errors, hops, final


def check_questions(corpus, path):
    """Contract 2.3 (MuSiQue-style questions) plus K3-K6, K9 and the dev/test split."""
    passages = {passage(it) for it in corpus}
    if not os.path.exists(path):
        return [f"{path} not found (build it with: python data_vi/build_questions.py --build)"]
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, list) or not data:
        return [f"{path} must be a non-empty list"]
    errors, finals, hop_of = [], [], []
    for i, q in enumerate(data):
        if not isinstance(q, dict):
            errors.append(f"q{i}: not an object")
            finals.append(None)
            hop_of.append(None)
            continue
        e, hops, final = check_one(q, f"q{i} {q.get('id')}", passages)
        errors += e
        finals.append(final)
        hop_of.append(hops)
    qs = [q for q in data if isinstance(q, dict)]

    ids = Counter(q.get("id") for q in qs)
    errors += [f"duplicate id {k!r} ({c} times)" for k, c in ids.items() if c > 1]
    texts = Counter(low(q["question"]) for q in qs if is_str(q.get("question")))
    errors += [f"duplicate question {k[:60]!r}" for k, c in texts.items() if c > 1]
    pairs = Counter((low(q["answer"]), f) for q, f in zip(data, finals)
                    if isinstance(q, dict) and is_str(q.get("answer")) and f)
    warnings = [f"answer {a!r} comes from the same final paragraph in {c} questions"
                for (a, _), c in pairs.items() if c > 1]

    # Split: the first DEV_SIZE questions are dev, the rest test; no final paragraph shared across them.
    n_dev = min(DEV_SIZE, len(data))
    dev_hops = Counter(h for h in hop_of[:n_dev] if h)
    if len(data) > DEV_SIZE and dict(dev_hops) != DEV_HOPS:
        errors.append(f"dev hop counts {dict(sorted(dev_hops.items()))}, expected {DEV_HOPS}")
    for i, q in enumerate(data):
        want = "dev" if i < n_dev else "test"
        if isinstance(q, dict) and "split" in q and q["split"] != want:
            errors.append(f"q{i} {q.get('id')}: split is {q['split']!r}, expected {want!r}")
    shared = {f for f in finals[:n_dev] if f} & {f for f in finals[n_dev:] if f}
    if shared:
        errors.append(f"{len(shared)} final-answer paragraphs shared by dev and test "
                      f"(e.g. {next(iter(shared))[:50]!r})")

    for w in warnings[:20]:
        print("WARN", w)

    def fmt(c):
        return "/".join(str(c[h]) for h in (1, 2, 3))
    n_para = sum(len(q.get("paragraphs") or []) for q in qs)
    print(f"questions: {len(data)} in {path} ({fmt(Counter(h for h in hop_of if h))} as 1/2/3-hop), "
          f"dev {fmt(dev_hops)}, test {fmt(Counter(h for h in hop_of[n_dev:] if h))}, "
          f"{n_para / len(data):.1f} paragraphs/question, {len(warnings)} warnings")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--openie", action="store_true", help="Also check outputs/vnhistory_univ/openie_results_ner.json")
    ap.add_argument("--questions", nargs="?", const=QUESTIONS_PATH, default=None, metavar="PATH",
                    help=f"Also check a question file (default {QUESTIONS_PATH})")
    args = ap.parse_args()
    if not (args.corpus or args.openie or args.questions):
        ap.error("choose what to check, e.g. --corpus, --openie or --questions")
    errors, corpus = check_corpus()
    if args.openie:
        errors += check_openie(corpus)
    if args.questions:
        errors += check_questions(corpus, args.questions)
    for e in errors[:50]:
        print("ERROR", e)
    print("OK" if not errors else f"FAILED: {len(errors)} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
