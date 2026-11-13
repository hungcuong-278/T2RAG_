"""Check question drafts and export a review sheet (M3).

Run from the project root:
    python data_vi/build_questions.py --precheck              # automatic checks on all drafts
    python data_vi/build_questions.py --sheet --batch 1       # review sheet (CSV) for one batch
    python data_vi/build_questions.py --build                 # write reproduce/dataset/vnhistory.json
    python data_vi/build_questions.py --build --allow_pending --out data_vi/work/vnhistory_preview.json

Automatic checks (codes follow the K1-K9 criteria of the plan):
  ERR  chunk      a step points to an unknown chunk id
  ERR  hops       hops != number of steps, or two steps share a chunk (K6)
  ERR  K5-step    a step answer does not appear verbatim in that step's passage
  ERR  last       final answer differs from the last step answer
  ERR  K9         the final answer (or an alias) appears in the question
  ERR  K3-bridge  an intermediate answer appears in the question
  ERR  K4         final answer longer than 6 syllables
  ERR  dup        duplicate question text
  WARN shortcut   an entity or year named in a multi-hop question also appears in the final chunk
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_chains  # noqa: E402

WORK = os.path.join("data_vi", "work")
DRAFTS = os.path.join(WORK, "questions_draft.jsonl")
OUT = os.path.join("reproduce", "dataset", "vnhistory.json")
DEV_QUOTA = {1: 15, 2: 23, 3: 12}  # 50 dev questions, the rest is test
SEED = "vnhistory-v1"


def nfc(text):
    return unicodedata.normalize("NFC", text)


def low(text):
    return nfc(text).lower()


def contains(haystack, needle):
    return re.search(r"(?<!\w)" + re.escape(low(needle)) + r"(?!\w)", low(haystack)) is not None


def load_drafts():
    return [json.loads(l) for l in open(DRAFTS, encoding="utf-8") if l.strip()]


def check(drafts):
    meta, aliases = find_chains.load()
    by_id = {m["short_id"]: m for m in meta}
    table = find_chains.name_table(sorted({m["title"] for m in meta}), aliases)
    name_re = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted(table, key=len, reverse=True)) + r")(?!\w)")
    seen_q = Counter(low(d["question"]) for d in drafts)
    results = {}
    for d in drafts:
        errs, warns = [], []
        steps = d["steps"]
        if d["hops"] != len(steps):
            errs.append("hops")
        chunks = [s["chunk"] for s in steps]
        if len(set(chunks)) != len(chunks):
            errs.append("hops(shared chunk)")
        for i, s in enumerate(steps, 1):
            m = by_id.get(s["chunk"])
            if m is None:
                errs.append(f"chunk(step {i})")
                continue
            if not contains(m["title"] + "\n" + m["text"], s["answer"]):
                errs.append(f"K5-step{i}")
        if steps and low(steps[-1]["answer"]) != low(d["answer"]):
            errs.append("last")
        if any(contains(d["question"], a) for a in [d["answer"]] + d["answer_aliases"]):
            errs.append("K9")
        for s in steps[:-1]:
            if contains(d["question"], s["answer"]):
                errs.append("K3-bridge")
        if len(d["answer"].split()) > 6:
            errs.append("K4")
        if seen_q[low(d["question"])] > 1:
            errs.append("dup")
        last = by_id.get(steps[-1]["chunk"]) if steps else None
        if d["hops"] > 1 and last is not None:
            named = {m.group(1) for m in name_re.finditer(d["question"])}
            years = set(re.findall(r"\b1\d{3}\b", d["question"]))
            hits = [n for n in named if contains(last["text"], n)] + [y for y in years if y in last["text"]]
            if hits:
                warns.append("shortcut:" + "|".join(sorted(hits)))
        results[d["draft_id"]] = (errs, warns)
    return results, by_id


# ---------------------------------------------------------------- build (M3.5)

def rank_key(draft_id):
    """Stable pseudo-random rank: adding new drafts does not reshuffle existing ones."""
    return hashlib.md5((SEED + draft_id).encode("utf-8")).hexdigest()


def pick_dev(items):
    """Choose DEV_QUOTA questions per hop count.

    Questions that share a final-step chunk always go to the same split, so dev and
    test never share the chunk that holds the final answer.
    """
    groups = {}
    for d in items:
        groups.setdefault(d["steps"][-1]["chunk"], []).append(d)
    left = dict(DEV_QUOTA)
    dev = set()
    for chunk in sorted(groups, key=lambda c: min(rank_key(d["draft_id"]) for d in groups[c])):
        need = Counter(d["hops"] for d in groups[chunk])
        if all(left.get(h, 0) >= n for h, n in need.items()):
            for h, n in need.items():
                left[h] -= n
            dev.update(d["draft_id"] for d in groups[chunk])
        if not any(left.values()):
            break
    if any(left.values()):
        sys.exit(f"Not enough questions for the dev quota, still missing {left}")
    return dev


def interleave(items):
    """Mix hop counts evenly: each question sits at its relative position within its hop group."""
    by_hop = {}
    for d in sorted(items, key=lambda d: rank_key(d["draft_id"])):
        by_hop.setdefault(d["hops"], []).append(d)
    keyed = [((i + 0.5) / len(g), h, d) for h, g in by_hop.items() for i, d in enumerate(g)]
    return [d for _, _, d in sorted(keyed, key=lambda x: (x[0], x[1]))]


def words(text):
    return set(re.findall(r"\w+", low(text)))


def distractors(d, meta, n):
    """Top-n non-supporting chunks by word overlap with the question (MuSiQue-style negatives)."""
    if n <= 0:
        return []
    used = {s["chunk"] for s in d["steps"]}
    q = words(d["question"])
    scored = [(-len(q & words(m["title"] + " " + m["text"])), m["short_id"]) for m in meta if m["short_id"] not in used]
    return [cid for _, cid in sorted(scored)[:n]]


def build(drafts, results, by_id, meta, out, n_distract, allow_pending):
    pending = [d["draft_id"] for d in drafts if d.get("status") not in ("ok", "loai")]
    if pending and not allow_pending:
        sys.exit(f"{len(pending)} drafts are not reviewed yet ({pending[0]}..{pending[-1]}). "
                 "Fill the review sheet first, or pass --allow_pending for a trial build.")
    items = [d for d in drafts if d.get("status") == "ok"]
    bad = [d["draft_id"] for d in items if results[d["draft_id"]][0]]
    if bad:
        sys.exit(f"Reviewed questions still fail the precheck: {', '.join(bad)}")

    dev = pick_dev(items)
    ordered = interleave([d for d in items if d["draft_id"] in dev]) + \
        interleave([d for d in items if d["draft_id"] not in dev])

    counters = Counter()
    samples = []
    for d in ordered:
        counters[d["hops"]] += 1
        paragraphs = []
        for cid in [s["chunk"] for s in d["steps"]] + distractors(d, meta, n_distract):
            m = by_id[cid]
            paragraphs.append({
                "idx": len(paragraphs),
                "title": m["title"],
                "paragraph_text": m["text"],  # copied verbatim: title + "\n" + text must match the corpus
                "is_supporting": len(paragraphs) < len(d["steps"]),
            })
        aliases = []
        for a in d["answer_aliases"]:
            if a.strip() and low(a) != low(d["answer"]) and a not in aliases:
                aliases.append(a)
        samples.append({
            "id": f"{d['hops']}hop_{counters[d['hops']]:03d}",
            "question": d["question"],
            "answer": d["answer"],
            "answer_aliases": aliases,
            "paragraphs": paragraphs,
            "question_decomposition": [
                {"question": s["question"], "answer": s["answer"], "paragraph_support_idx": i}
                for i, s in enumerate(d["steps"])
            ],
            "split": "dev" if d["draft_id"] in dev else "test",
            "qtype": d["qtype"],
            "draft_id": d["draft_id"],
        })

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    def hops_of(split):
        c = Counter(s["id"][0] for s in samples if s["split"] == split)
        return "/".join(str(c[h]) for h in "123")
    dev_last = {d["steps"][-1]["chunk"] for d in items if d["draft_id"] in dev}
    test_last = {d["steps"][-1]["chunk"] for d in items if d["draft_id"] not in dev}
    print(f"\nWrote {out}: {len(samples)} questions "
          f"(dev {hops_of('dev')}, test {hops_of('test')} as 1/2/3-hop), "
          f"{n_distract} distractors each; skipped {len(drafts) - len(items)} not-ok drafts; "
          f"shared final chunks dev/test: {len(dev_last & test_last)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--sheet", action="store_true", help="Write a CSV review sheet")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--build", action="store_true", help="Write the final question file (only status=ok)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--distractors", type=int, default=0, help="Non-supporting paragraphs per question")
    ap.add_argument("--allow_pending", action="store_true", help="Build even if some drafts are not reviewed")
    args = ap.parse_args()
    drafts = load_drafts()
    if args.batch is not None:
        drafts = [d for d in drafts if d.get("batch") == args.batch]
    results, by_id = check(drafts)

    n_err = sum(1 for e, _ in results.values() if e)
    n_warn = sum(1 for e, w in results.values() if w and not e)
    for d in drafts:
        errs, warns = results[d["draft_id"]]
        if errs or warns:
            print(f"{d['draft_id']} ({d['hops']}-hop) {'ERR ' + ','.join(errs) if errs else ''} "
                  f"{'WARN ' + ','.join(warns) if warns else ''} | {d['question']}")
    print(f"\n{len(drafts)} drafts {dict(Counter(d['hops'] for d in drafts))}: "
          f"{n_err} with errors, {n_warn} with warnings only")

    if args.sheet:
        tag = f"batch{args.batch}" if args.batch is not None else "all"
        path = os.path.join(WORK, f"review_questions_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["draft_id", "hops", "question", "answer", "answer_aliases", "auto_check",
                        "decomposition", "support_1", "support_2", "support_3",
                        "status (ok/sua/loai)", "reason (K1-K9)", "reviewer", "note"])
            for d in drafts:
                errs, warns = results[d["draft_id"]]
                decomp = "\n".join(f"{i}. {s['question']} -> {s['answer']} [{s['chunk']}]" for i, s in enumerate(d["steps"], 1))
                sup = [f"[{s['chunk']}] {by_id[s['chunk']]['title']}\n{by_id[s['chunk']]['text']}" if s["chunk"] in by_id else ""
                       for s in d["steps"]] + ["", "", ""]
                w.writerow([d["draft_id"], d["hops"], d["question"], d["answer"], " | ".join(d["answer_aliases"]),
                            " ".join(["ERR " + ",".join(errs)] if errs else []) + (" WARN " + ",".join(warns) if warns else ""),
                            decomp, sup[0], sup[1], sup[2], d["status"], d["reason"], d["reviewer"], d["note"]])
        print(f"Review sheet: {path}")

    if args.build:
        if args.batch is not None:
            sys.exit("--build uses all batches; drop --batch")
        meta, _ = find_chains.load()
        build(drafts, results, by_id, meta, args.out, args.distractors, args.allow_pending)


if __name__ == "__main__":
    main()
