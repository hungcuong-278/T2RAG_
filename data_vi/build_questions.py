"""Check question drafts and export a review sheet (M3).

Run from the project root:
    python data_vi/build_questions.py --precheck              # automatic checks on all drafts
    python data_vi/build_questions.py --sheet --batch 1       # review sheet (CSV) for one batch

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--sheet", action="store_true", help="Write a CSV review sheet")
    ap.add_argument("--batch", type=int, default=None)
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


if __name__ == "__main__":
    main()
