"""Merge result files of one configuration that was run in batches (--start / --limit) into one file.

    python merge_results.py outputs/results/vnhistory_trag_*_10q_dev50_r1.json -o outputs/results/vnhistory_trag_dev50_r1.json

Records are concatenated (a question that appears twice keeps the later file), sorted by "id",
and the summary is recomputed from the records: EM, F1, EM_lenient, F1_lenient, scores by hop,
recall@5, average LLM calls / tokens, total seconds and errors. Fields that must be identical in
every batch (dataset, method, llm, embedding, answer_format, llm_cache, tag, questions_md5) are
checked; the run stops if they differ, so batches of different configurations are never mixed.
"""
import argparse
import json
import os
import sys

import run_colab as rc

SAME = ("dataset", "method", "llm", "embedding", "answer_format", "llm_cache", "tag", "questions_md5", "fixes")


def merge(datas, run_name=None):
    if not datas:
        raise SystemExit("No result files to merge.")
    first = datas[0]["summary"]
    for d in datas[1:]:
        s = d["summary"]
        diff = [k for k in SAME if s.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"Cannot merge {s.get('run_name')} with {first.get('run_name')}: different {diff}")
    by_qid = {}
    for d in datas:
        for r in d["records"]:
            if d["summary"].get("trace_dir") and not r.get("trace_dir"):
                r["trace_dir"] = d["summary"]["trace_dir"]  # each batch has its own trace folder
            by_qid[str(r.get("qid", r.get("id")))] = r
    records = sorted(by_qid.values(), key=lambda r: (r.get("id") is None, r.get("id")))
    n = len(records)

    def mean(key, digits=4):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        return round(sum(vals) / len(vals), digits) if vals else None

    def weighted(key):
        pairs = [(d["summary"].get(key), d["summary"].get("num_questions") or len(d["records"])) for d in datas]
        pairs = [(v, w) for v, w in pairs if isinstance(v, (int, float))]
        return round(sum(v * w for v, w in pairs) / sum(w for _, w in pairs), 2) if pairs else None

    found5 = sum(r.get("gold_docs_found_first5") or 0 for r in records)
    found = sum(r.get("gold_docs_found") or 0 for r in records)
    total = sum(r.get("gold_docs_total") or 0 for r in records)
    summary = dict(first)
    summary.update({
        "run_name": run_name or first.get("run_name", "") + f"__merged_{n}q",
        "start": min((d["summary"].get("start") or 0) for d in datas),
        "num_questions": n,
        "ExactMatch": mean("em"), "F1": mean("f1"),
        "EM_lenient": mean("em_lenient"), "F1_lenient": mean("f1_lenient"),
        "n_long_answers": sum(bool(r.get("long_answer")) for r in records),
        "n_multi_answers": sum(bool(r.get("multi_answer")) for r in records),
        "by_hops": rc.scores_by_hop(records),
        "recall_first5_docs": round(found5 / total, 4) if total else None,
        "recall_all_retrieved_docs": round(found / total, 4) if total else None,
        "errors": sum(r.get("prediction") == "Error processing query" for r in records),
        "seconds": round(sum(d["summary"].get("seconds") or 0 for d in datas), 1),
        "avg_llm_calls": mean("llm_calls", 2) if any("llm_calls" in r for r in records) else weighted("avg_llm_calls"),
        "avg_cache_hits": mean("cache_hits", 2) if any("cache_hits" in r for r in records) else weighted("avg_cache_hits"),
        "avg_prompt_tokens": mean("prompt_tokens", 1) if any("prompt_tokens" in r for r in records)
        else weighted("avg_prompt_tokens"),
        "avg_completion_tokens": mean("completion_tokens", 1) if any("completion_tokens" in r for r in records)
        else weighted("avg_completion_tokens"),
        "avg_iterations": mean("iterations", 2),
        "max_completion_tokens": max((d["summary"].get("max_completion_tokens") or 0) for d in datas),
        "n_long_calls": sum((d["summary"].get("n_long_calls") or 0) for d in datas),
        "merged_from": [d["summary"].get("run_name") for d in datas],
    })
    return {"summary": summary, "records": records}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args(argv)
    datas = [json.load(open(p, encoding="utf-8")) for p in args.files if os.path.abspath(p) != os.path.abspath(args.out)]
    merged = merge(datas, run_name=os.path.splitext(os.path.basename(args.out))[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    s = merged["summary"]
    print(f"{len(datas)} files, {s['num_questions']} questions: EM {s['ExactMatch']}  F1 {s['F1']}  "
          f"EM_lenient {s['EM_lenient']}  errors {s['errors']} -> {args.out}")
    for h, v in s["by_hops"].items():
        print(f"  {h}-hop ({v['n']} q): EM {v['em']}  F1 {v['f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
