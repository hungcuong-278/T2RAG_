"""Label the wrong answers of a T2RAG / T2RAG+ run with an error type, using the reasoning traces.

Example (from the repo root):
    python error_analysis.py outputs/results/musique_trag_gemini-3.5-flash_from200_200q.json
    python error_analysis.py <run.json> --compare <other_run.json>      # before / after a fix
    python error_analysis.py <run.json> --sample 30                       # sheet for manual checking

A question counts as correct when EM = 1 or F1 >= --f1_threshold (default 0.5).
Each wrong question gets the FIRST label below that applies:

  sai_ten          (only with --aliases) the prediction is another name of the gold entity
  tra_loi_cuoi_sai the gold answer is already in the final clues / evidence, but the final answer is wrong
  thieu_doan_vang  at least one supporting (gold) passage was never retrieved in any round
  roi_mat_xich     the chain broke: clues were dropped, or unresolved clues remain at the end
  dien_sai_buoc_giua  all gold passages retrieved and the chain is complete, but an intermediate
                   answer (MuSiQue question_decomposition) is missing from the resolved clues
  khac             none of the above

The labels are heuristics: check a sample by hand (--sample) and report the agreement rate.

Outputs (next to the result file unless --out_dir is given):
    errors_<run>.csv          one row per question with label, scores and the final clues
    errors_<run>.md           counts per label, overall and per hop count
    errors_compare_<A>_vs_<B>.md  (with --compare) labels side by side and which errors B fixed
    check_<run>.csv           (with --sample N) random wrong questions with an empty column for a manual label
"""
import argparse
import csv
import json
import os
import random
import re
import string
import unicodedata
from collections import Counter, OrderedDict

LABELS = ["sai_ten", "tra_loi_cuoi_sai", "thieu_doan_vang", "roi_mat_xich", "dien_sai_buoc_giua", "khac"]
LABEL_VI = {
    "dung": "Đúng",
    "sai_ten": "Đúng người, sai tên",
    "tra_loi_cuoi_sai": "Trả lời cuối sai (bộ ba đã có đáp án)",
    "thieu_doan_vang": "Không tìm thấy đoạn vàng",
    "roi_mat_xich": "Rơi mắt xích (bộ ba bị bỏ / chưa giải)",
    "dien_sai_buoc_giua": "Điền sai bước giữa",
    "khac": "Khác",
}


# ---------------------------------------------------------------- text helpers
LANG = {"value": "en"}  # set from the run (summary.lang, or vi for vnhistory*)


def norm(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    if LANG["value"] == "vi":  # Unicode punctuation; keep "an" (a Vietnamese syllable)
        text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
        return " ".join(text.split())
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains(haystack, needle):
    n = norm(needle)
    return bool(n) and f" {n} " in f" {norm(haystack)} "


def clue_text(clues):
    return "\n".join(f"{c.get('subject', '')} {c.get('predicate', '')} {c.get('object', '')}"
                     for c in clues or [])


def has_placeholder(clue):
    return any(re.search(r"(^|\s)\?[A-Za-z0-9_]*($|\s)", str(v)) for v in clue.values())


# ---------------------------------------------------------------- data
def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    out = {}
    for s in samples:
        qid = str(s.get("id", s.get("_id")))
        gold_docs, inter = [], []
        if "paragraphs" in s:
            for p in s["paragraphs"]:
                if p.get("is_supporting", True) is not False:
                    gold_docs.append(p["title"] + "\n" + (p["text"] if "text" in p else p["paragraph_text"]))
        elif "supporting_facts" in s:
            titles = {t for t, _ in s["supporting_facts"]}
            sep = "" if "hotpot" in path else " "
            gold_docs = [t + "\n" + sep.join(sents) for t, sents in s["context"] if t in titles]
        decomp = s.get("question_decomposition") or []
        inter = [d.get("answer", "") for d in decomp[:-1] if d.get("answer")]
        out[qid] = {"gold_docs": list(dict.fromkeys(gold_docs)), "intermediate": inter}
    return out


def load_trace(trace_dir, record):
    name = record.get("trace_file") or f"query_{record['id']}_results.json"
    path = os.path.join(trace_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_aliases(path):
    """{"canonical name": ["other name", ...]} -> list of name groups (normalized)."""
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [{norm(k)} | {norm(v) for v in vs} for k, vs in data.items()]


# ---------------------------------------------------------------- labelling
def analyse(record, trace, gold, alias_groups, f1_threshold):
    info = {"qid": record.get("qid"), "hops": record.get("hops"), "em": record["em"], "f1": record["f1"],
            "question": record.get("question", ""), "prediction": record.get("prediction", ""),
            "gold": " | ".join(record.get("gold_answers", []))}
    correct = record["em"] >= 1.0 or record["f1"] >= f1_threshold
    retrieved, initial, final_clues, evidence, unresolved, dropped = [], [], [], "", 0, None
    if trace:
        for it in trace.get("iterations", []):
            for p in (it.get("step_2_retrieval_and_expansion", {}) or {}).get("retrieved_passages", []) or []:
                if p.get("text") and p["text"] not in retrieved:
                    retrieved.append(p["text"])
        step1 = trace.get("step_1_reasoning") or {}
        initial = step1.get("all_triples") or (step1.get("traceable_clues", []) + step1.get("fuzzy_clues", [])
                                                + step1.get("resolved_clues", []))
        fin = trace.get("final_results") or {}
        final_clues = fin.get("all_final_clues") or []
        evidence = fin.get("final_evidence") or ""
        unresolved = len(fin.get("remaining_fuzzy_clues") or []) + sum(has_placeholder(c) for c in final_clues)
        if isinstance(fin.get("stats"), dict):
            dropped = fin["stats"].get("n_dropped_clues")
    if dropped is None:
        dropped = record.get("n_dropped_clues")
    gold_docs = (gold or {}).get("gold_docs", [])
    found = sum(g in retrieved for g in gold_docs) if trace else record.get("gold_docs_found")
    info.update({"gold_docs_found": found, "gold_docs_total": len(gold_docs) or record.get("gold_docs_total"),
                 "n_initial_clues": len(initial), "n_final_clues": len(final_clues),
                 "n_unresolved": unresolved, "n_dropped": dropped, "has_trace": bool(trace),
                 "final_clues": clue_text(final_clues)[:1500]})
    if correct:
        info["label"] = "dung"
        return info

    golds = record.get("gold_answers", [])
    pred = record.get("prediction", "")
    text = clue_text(final_clues) + "\n" + str(evidence)
    missing_inter = [a for a in (gold or {}).get("intermediate", []) if not contains(text, a)]
    info["missing_intermediate"] = " | ".join(missing_inter)

    label = "khac"
    if alias_groups and any(norm(pred) in g and any(norm(x) in g for x in golds) for g in alias_groups):
        label = "sai_ten"
    elif any(contains(text, g) for g in golds):
        label = "tra_loi_cuoi_sai"
    elif gold_docs and found is not None and found < len(gold_docs):
        label = "thieu_doan_vang"
    elif unresolved > 0 or (dropped or 0) > 0 or (initial and len(final_clues) < len(initial)):
        label = "roi_mat_xich"
    elif missing_inter:
        label = "dien_sai_buoc_giua"
    info["label"] = label
    return info


def label_run(result_path, dataset_path, trace_dir, alias_groups, f1_threshold):
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    s = data["summary"]
    LANG["value"] = s.get("lang") or ("vi" if str(s.get("dataset", "")).startswith("vnhistory") else "en")
    dataset_path = dataset_path or os.path.join("reproduce", "dataset", f"{s['dataset']}.json")
    gold = load_dataset(dataset_path) if os.path.exists(dataset_path) else {}
    trace_dir = trace_dir or s.get("trace_dir") or ""
    rows, n_trace = [], 0
    for r in data["records"]:
        trace = load_trace(trace_dir, r) if trace_dir and os.path.isdir(trace_dir) else None
        n_trace += trace is not None
        rows.append(analyse(r, trace, gold.get(str(r.get("qid"))), alias_groups, f1_threshold))
    return s, rows, n_trace, trace_dir


# ---------------------------------------------------------------- reporting
def count_table(rows, title):
    groups = ["tổng"] + sorted({f"{r['hops']} bước" for r in rows if r.get("hops")})
    order = ["dung"] + LABELS
    lines = [f"### {title}", "", "| Nhãn | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    per = {}
    for g in groups:
        sel = [r for r in rows if g == "tổng" or f"{r.get('hops')} bước" == g]
        wrong = [r for r in sel if r["label"] != "dung"]
        per[g] = (Counter(r["label"] for r in sel), len(sel), len(wrong))
    for lab in order:
        if all(per[g][0][lab] == 0 for g in groups):
            continue
        cells = []
        for g in groups:
            c, n, nw = per[g]
            if lab == "dung":
                cells.append(f"{c[lab]} ({100 * c[lab] / n:.1f}% số câu)" if n else "–")
            else:
                cells.append(f"{c[lab]} ({100 * c[lab] / nw:.1f}% câu sai)" if nw else "–")
        lines.append(f"| {LABEL_VI[lab]} | " + " | ".join(cells) + " |")
    lines.append("| **Số câu / số câu sai** | " + " | ".join(f"{per[g][1]} / {per[g][2]}" for g in groups) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    keys = list(OrderedDict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="Result JSON from run_colab.py (trag or trag_plus)")
    ap.add_argument("--dataset_file", default=None, help="Default: reproduce/dataset/<dataset>.json")
    ap.add_argument("--trace_dir", default=None, help="Default: summary.trace_dir of the result file")
    ap.add_argument("--compare", default=None, help="Second result file (e.g. after a fix) on the same questions")
    ap.add_argument("--compare_trace_dir", default=None)
    ap.add_argument("--aliases", default=None, help='JSON {"name": ["other name", ...]} for the sai_ten label')
    ap.add_argument("--f1_threshold", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=0, help="Write N random wrong questions for manual labelling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)

    with open(args.result, encoding="utf-8") as f:  # language first: aliases are normalised with it
        s0 = json.load(f)["summary"]
    LANG["value"] = s0.get("lang") or ("vi" if str(s0.get("dataset", "")).startswith("vnhistory") else "en")
    alias_groups = load_aliases(args.aliases)
    s, rows, n_trace, trace_dir = label_run(args.result, args.dataset_file, args.trace_dir, alias_groups,
                                            args.f1_threshold)
    out_dir = args.out_dir or os.path.dirname(args.result) or "."
    os.makedirs(out_dir, exist_ok=True)
    name = s.get("run_name") or os.path.splitext(os.path.basename(args.result))[0]
    if n_trace == 0:
        print(f"CẢNH BÁO: không đọc được file vết nào trong '{trace_dir}'. Nhãn chỉ dựa trên file kết quả "
              "(thiếu nhãn rơi mắt xích / trả lời cuối sai). Kiểm tra --trace_dir.")
    elif n_trace < len(rows):
        print(f"Lưu ý: chỉ có vết cho {n_trace}/{len(rows)} câu.")

    write_csv(os.path.join(out_dir, f"errors_{name}.csv"), rows)
    md = f"# Phân tích lỗi: {name}\n\n" + count_table(rows, s.get("method", name))
    md += ("\nNhãn được gán tự động theo thứ tự: sai tên → trả lời cuối sai → không tìm thấy đoạn vàng → "
           "rơi mắt xích → điền sai bước giữa → khác. Cần kiểm tay một mẫu (--sample) để báo độ tin cậy.\n")

    if args.compare:
        s2, rows2, _, _ = label_run(args.compare, args.dataset_file, args.compare_trace_dir, alias_groups,
                                    args.f1_threshold)
        name2 = s2.get("run_name") or os.path.splitext(os.path.basename(args.compare))[0]
        by_q = {r["qid"]: r for r in rows2}
        common = [r for r in rows if r["qid"] in by_q]
        cmp_md = f"# So sánh lỗi: A = {s.get('method')} | B = {s2.get('method')} ({len(common)} câu chung)\n\n"
        cmp_md += count_table(common, f"A = {s.get('method')}") + "\n"
        cmp_md += count_table([by_q[r["qid"]] for r in common], f"B = {s2.get('method')}") + "\n"
        fixed = Counter(r["label"] for r in common if r["label"] != "dung" and by_q[r["qid"]]["label"] == "dung")
        broke = Counter(by_q[r["qid"]]["label"] for r in common
                        if r["label"] == "dung" and by_q[r["qid"]]["label"] != "dung")
        cmp_md += "### B sửa được câu sai nào của A (theo nhãn lỗi ở A)\n\n| Nhãn ở A | Số câu A sai | B sửa đúng |\n|---|---|---|\n"
        wrong_a = Counter(r["label"] for r in common if r["label"] != "dung")
        for lab in LABELS:
            if wrong_a[lab]:
                cmp_md += f"| {LABEL_VI[lab]} | {wrong_a[lab]} | {fixed[lab]} |\n"
        cmp_md += f"\nB làm hỏng {sum(broke.values())} câu A đúng"
        cmp_md += (": " + ", ".join(f"{LABEL_VI[k]} {v}" for k, v in broke.items()) if broke else "") + ".\n"
        path = os.path.join(out_dir, f"errors_compare_{name}_vs_{name2}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cmp_md)
        print(cmp_md)
        print(f"Đã lưu {path}")

    if args.sample:
        wrong = [r for r in rows if r["label"] != "dung"]
        random.Random(args.seed).shuffle(wrong)
        sample = [dict(r, nhan_tay="", ghi_chu="") for r in wrong[:args.sample]]
        if sample:
            write_csv(os.path.join(out_dir, f"check_{name}.csv"), sample)
            md += (f"\nĐã xuất {len(sample)} câu sai ngẫu nhiên ra `check_{name}.csv`: điền cột `nhan_tay` "
                   f"(một trong: {', '.join(LABELS)}), rồi tính tỷ lệ khớp với cột `label`.\n")

    with open(os.path.join(out_dir, f"errors_{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"Đã lưu {os.path.join(out_dir, f'errors_{name}.csv')} và errors_{name}.md")


if __name__ == "__main__":
    main()
