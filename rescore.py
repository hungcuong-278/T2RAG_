"""Score saved predictions again against a new version of the question file, without calling the model.

Use it when only the gold answers / answer_aliases changed (for example vnhistory.json 1.0 -> 1.1):

    python rescore.py outputs/results/vnhistory_trag_*.json --questions reproduce/dataset/vnhistory.json

For every result file (from run_colab.py, or a merged file with the same "summary" / "records" layout)
it writes <name>_rescored.json next to it (or in --out_dir) with:
  * gold_answers, em, f1, em_lenient, f1_lenient of every record recomputed from the new file
    (records are matched by qid; predictions are not touched);
  * ExactMatch, F1, EM_lenient, F1_lenient and by_hops recomputed in the summary;
  * summary.rescored = {questions_md5, previous ExactMatch / F1, number of records whose EM changed}.
The language for normalisation is taken from summary.lang (vi for vnhistory), as in run_colab.py.
"""
import argparse
import json
import os
import sys

import run_colab as rc


def rescore(data, samples_by_qid, questions_md5):
    summary, records = data.get("summary", {}), data.get("records", [])
    lang = summary.get("lang") or ("vi" if str(summary.get("dataset", "")).startswith("vnhistory") else "en")
    os.environ["T2RAG_LANG"] = lang
    missing, changed = [], 0
    for r in records:
        sample = samples_by_qid.get(str(r.get("qid")))
        if sample is None:
            missing.append(r.get("qid"))
            continue
        golds = rc.get_gold_answers([sample])[0]
        em, f1 = rc.em_f1(r.get("prediction", ""), golds)
        em_l, f1_l, multi = rc.lenient_scores(r.get("prediction", ""), golds, lang)
        if float(r.get("em", -1)) != em:
            changed += 1
        r.update({"gold_answers": sorted(golds), "em": em, "f1": round(f1, 4), "em_lenient": em_l,
                  "f1_lenient": round(f1_l, 4), "multi_answer": multi})
        if r.get("hops") is None:
            r["hops"] = rc.hops_of(r.get("qid"))
    if missing:
        raise SystemExit(f"{len(missing)} records have no question with the same qid in the new file "
                         f"(first: {missing[0]!r}); the question file must keep the same ids.")
    n = len(records) or 1
    old = {"ExactMatch": summary.get("ExactMatch"), "F1": summary.get("F1")}
    summary.update({
        "ExactMatch": round(sum(r["em"] for r in records) / n, 4),
        "F1": round(sum(r["f1"] for r in records) / n, 4),
        "EM_lenient": round(sum(r["em_lenient"] for r in records) / n, 4),
        "F1_lenient": round(sum(r["f1_lenient"] for r in records) / n, 4),
        "n_multi_answers": sum(bool(r["multi_answer"]) for r in records),
        "by_hops": rc.scores_by_hop(records),
        "questions_md5": questions_md5,
        "rescored": {"questions_md5": questions_md5, "previous": old, "n_em_changed": changed},
    })
    return data, old, changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="Result JSON files")
    ap.add_argument("--questions", required=True, help="The new question file, e.g. reproduce/dataset/vnhistory.json")
    ap.add_argument("--out_dir", default=None, help="Where to write; default = next to each input file")
    args = ap.parse_args(argv)

    samples = json.load(open(args.questions, encoding="utf-8"))
    by_qid = {str(s.get("id", s.get("_id"))): s for s in samples}
    md5 = rc.file_md5(args.questions)
    print(f"Questions: {args.questions} ({len(samples)} questions, md5 {md5})")
    for path in args.results:
        if path.endswith("_rescored.json"):
            continue
        data = json.load(open(path, encoding="utf-8"))
        data, old, changed = rescore(data, by_qid, md5)
        out_dir = args.out_dir or os.path.dirname(path)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.basename(path)[:-5] + "_rescored.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        s = data["summary"]
        print(f"{os.path.basename(path)}: EM {old['ExactMatch']} -> {s['ExactMatch']}, F1 {old['F1']} -> {s['F1']}, "
              f"EM_lenient {s['EM_lenient']}; {changed} answers changed EM -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
