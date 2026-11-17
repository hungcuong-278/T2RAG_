"""Build a Table-1-style results table from run_colab.py result files.

Example (from the repo root):
    python make_table.py --glob "outputs/results/musique_*_from200_200q.json" --baseline trag
    python make_table.py outputs/results/a.json outputs/results/b.json --labels labels.json

Each cell is "EM / F1" (x100); with --metric lenient it is "EM_lenient / F1_lenient" (run_colab.py v6). A dagger (†) means the method beats the baseline on that
metric with p < 0.05 under a paired bootstrap test over the questions both runs answered
(5,000 resamples by default, the same test the T2RAG paper uses). The best value of each
column is in bold.

Outputs (in --out_dir, default "tables"):
    table_main.md   EM / F1 per dataset, split by number of hops, plus the average over datasets
    table_cost.md   LLM calls, tokens, rounds, recall@5 and seconds per question
    table_main.csv  one row per (dataset, method, group) with EM, F1, n, p-values

Repeated runs (--pool_runs): files whose tags differ only by a run suffix (default "_r1", "_r2", ...)
form one row; each question scores the mean over the runs that answered it, and the † test is the
same paired bootstrap on those per-question means. The row label shows the number of runs.
    python make_table.py --glob "results/*test116*_rescored.json" --pool_runs --baseline trag:test116
"""
import argparse
import copy
import csv
import glob
import json
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np


# ---------------------------------------------------------------- loading
VN_QID = re.compile(r"^[123]hop_\d+$")


def load_runs(paths):
    """Return {(dataset, method): {"summary": [...], "records": {qid: record}}}.

    Several files of the same dataset and method (for example two halves of the test set)
    are merged; a question that appears twice keeps the record from the later file."""
    runs = OrderedDict()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        s = data["summary"]
        if not s.get("dataset"):  # v2 merged files have no summary.dataset
            qids = [str(r.get("qid", "")) for r in data["records"]]
            s["dataset"] = "vnhistory" if qids and all(VN_QID.match(q) for q in qids) else \
                (s.get("run_name") or "unknown").split("_")[0]
        s.setdefault("method", "unknown")
        # runs that differ only by --tag (e.g. another S4 threshold) are separate rows
        key = (s["dataset"], s["method"] + (f":{s['tag']}" if s.get("tag") else ""))
        run = runs.setdefault(key, {"summaries": [], "records": OrderedDict(), "files": []})
        run["summaries"].append(s)
        run["files"].append(path)
        for r in data["records"]:
            run["records"][str(r.get("qid", r["id"]))] = r
    return runs


SCORE_KEYS = ("em", "f1", "em_lenient", "f1_lenient")


def pool_runs(runs, suffix=r"_r\d+$"):
    """Join runs whose tags differ only by `suffix`; each question gets the mean score over the runs."""
    pat = re.compile(suffix)
    groups = OrderedDict()
    for (d, m), run in runs.items():
        base_m = pat.sub("", m) if ":" in m else m
        groups.setdefault((d, base_m), []).append(run)
    pooled = OrderedDict()
    for key, members in groups.items():
        if len(members) == 1:
            members[0]["n_runs"] = 1
            pooled[key] = members[0]
            continue
        qids = list(OrderedDict.fromkeys(q for run in members for q in run["records"]))
        records = OrderedDict()
        for q in qids:
            rs = [run["records"][q] for run in members if q in run["records"]]
            rec = copy.deepcopy(rs[0])
            for k in SCORE_KEYS:
                vals = [float(r[k]) for r in rs if r.get(k) is not None]
                rec[k] = float(np.mean(vals)) if vals else None
            rec["n_runs"] = len(rs)
            records[q] = rec
        pooled[key] = {"summaries": [s for run in members for s in run["summaries"]], "records": records,
                       "files": [f for run in members for f in run["files"]], "n_runs": len(members),
                       "member_records": [run["records"] for run in members]}
    return pooled


def group_of(record):
    h = record.get("hops")
    return f"{h} bước" if h else None


# ---------------------------------------------------------------- statistics
def paired_bootstrap_p(a, b, n_boot=5000, seed=0):
    """One-sided p-value for mean(a) > mean(b) on paired per-question scores."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0:
        return None
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.mean(means <= 0))


def mean_or_none(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------- formatting
def fmt(x, decimal=",", nd=1):
    if x is None:
        return "–"
    s = f"{x:.{nd}f}"
    return s.replace(".", decimal) if decimal != "." else s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="Result JSON files from run_colab.py")
    ap.add_argument("--glob", action="append", default=[], help="Glob pattern(s) for result files")
    ap.add_argument("--baseline", default="trag", help="Method name used as the baseline for the † test")
    ap.add_argument("--labels", default=None,
                    help='JSON file {"method name": "row label"}; also fixes the row order')
    ap.add_argument("--out_dir", default="tables")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--decimal", default=",", help="Decimal separator in the tables (default ',')")
    ap.add_argument("--no_groups", action="store_true", help="Only the overall column per dataset")
    ap.add_argument("--metric", default="strict", choices=["strict", "lenient"],
                    help="strict = EM / F1 as in the paper; lenient = EM_lenient / F1_lenient (v6 result files)")
    ap.add_argument("--pool_runs", nargs="?", const=r"_r\d+$", default=None, metavar="REGEX",
                    help=r"Average repeated runs whose tags differ only by this suffix (default _r\d+$)")
    ap.add_argument("--price_in", type=float, default=None, help="USD per million input tokens (adds a cost column)")
    ap.add_argument("--price_out", type=float, default=None, help="USD per million output tokens")
    args = ap.parse_args(argv)
    em_key, f1_key = ("em", "f1") if args.metric == "strict" else ("em_lenient", "f1_lenient")

    def score(r, key):
        v = r.get(key)
        if v is None:
            if args.metric == "lenient":
                raise SystemExit("--metric lenient needs result files from run_colab.py v6 (or rescore.py), "
                                 f"but question {r.get('qid')} has no {key}.")
            return 0.0
        return float(v)

    paths = list(args.files)
    for pattern in args.glob:
        paths += sorted(glob.glob(pattern))
    paths = list(OrderedDict.fromkeys(paths))
    if not paths:
        raise SystemExit("No result files given.")
    runs = load_runs(paths)
    if args.pool_runs:
        runs = pool_runs(runs, args.pool_runs)

    def row_label(m, labels_):
        n_runs = next((run.get("n_runs", 1) for (_, mm), run in runs.items() if mm == m), 1)
        return labels_.get(m, m) + (f" (TB {n_runs} lượt)" if n_runs > 1 else "")

    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            labels = json.load(f, object_pairs_hook=OrderedDict)

    datasets = list(OrderedDict.fromkeys(d for d, _ in runs))
    methods = list(OrderedDict.fromkeys(m for _, m in runs))
    if labels:
        methods = [m for m in labels if m in methods] + [m for m in methods if m not in labels]

    # Columns: per dataset, "Tổng" then each hop group present in that dataset.
    columns = []  # (dataset, group or None)
    for d in datasets:
        columns.append((d, None))
        if not args.no_groups:
            groups = set()
            for (dd, _), run in runs.items():
                if dd == d:
                    groups |= {group_of(r) for r in run["records"].values()} - {None}
            for g in sorted(groups):
                columns.append((d, g))

    # Cell values.
    cells = {}  # (method, dataset, group) -> dict
    rows_csv = []
    for m in methods:
        for d, g in columns:
            run = runs.get((d, m))
            if run is None:
                continue
            recs = {q: r for q, r in run["records"].items() if g is None or group_of(r) == g}
            if not recs:
                continue
            em = 100 * np.mean([score(r, em_key) for r in recs.values()])
            f1 = 100 * np.mean([score(r, f1_key) for r in recs.values()])
            cell = {"em": em, "f1": f1, "n": len(recs), "p_em": None, "p_f1": None}
            base = runs.get((d, args.baseline))
            if base is not None and m != args.baseline:
                common = [q for q in recs if q in base["records"]]
                if common:
                    a_em = [score(recs[q], em_key) for q in common]
                    b_em = [score(base["records"][q], em_key) for q in common]
                    a_f1 = [score(recs[q], f1_key) for q in common]
                    b_f1 = [score(base["records"][q], f1_key) for q in common]
                    cell["p_em"] = paired_bootstrap_p(a_em, b_em, args.n_boot)
                    cell["p_f1"] = paired_bootstrap_p(a_f1, b_f1, args.n_boot)
                    cell["n_common"] = len(common)
            cells[(m, d, g)] = cell
            rows_csv.append({"dataset": d, "method": m, "label": labels.get(m, m), "group": g or "all",
                             "n": len(recs), "EM": round(em, 2), "F1": round(f1, 2),
                             "p_EM_vs_baseline": cell["p_em"], "p_F1_vs_baseline": cell["p_f1"]})

    # Best value per column (bold).
    best = {}
    for d, g in columns:
        vals = [(cells[(m, d, g)]["em"], cells[(m, d, g)]["f1"]) for m in methods if (m, d, g) in cells]
        if vals:
            best[(d, g)] = (max(v[0] for v in vals), max(v[1] for v in vals))

    def cell_text(m, d, g):
        c = cells.get((m, d, g))
        if c is None:
            return "–"
        parts = []
        for key, pkey, bi in (("em", "p_em", 0), ("f1", "p_f1", 1)):
            t = fmt(c[key], args.decimal)
            if c[pkey] is not None and c[pkey] < args.alpha:
                t += "†"
            if abs(c[key] - best[(d, g)][bi]) < 1e-9 and sum((mm, d, g) in cells for mm in methods) > 1:
                t = f"**{t}**"
            parts.append(t)
        return " / ".join(parts)

    # Average over datasets (overall column of each dataset), as in the paper's Table 1.
    avg = {}
    for m in methods:
        ems = [cells[(m, d, None)]["em"] for d in datasets if (m, d, None) in cells]
        f1s = [cells[(m, d, None)]["f1"] for d in datasets if (m, d, None) in cells]
        if len(ems) == len(datasets):
            avg[m] = (float(np.mean(ems)), float(np.mean(f1s)))

    os.makedirs(args.out_dir, exist_ok=True)
    header = ["Phương pháp"] + [f"{d} {'tổng' if g is None else g}" for d, g in columns]
    if len(datasets) > 1:
        header += ["TB EM", "TB F1"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in methods:
        row = [row_label(m, labels)] + [cell_text(m, d, g) for d, g in columns]
        if len(datasets) > 1:
            row += [fmt(avg[m][0], args.decimal), fmt(avg[m][1], args.decimal)] if m in avg else ["–", "–"]
        lines.append("| " + " | ".join(row) + " |")
    n_line = "Số câu: " + "; ".join(
        f"{d} {'tổng' if g is None else g} = {max((cells[(m, d, g)]['n'] for m in methods if (m, d, g) in cells), default=0)}"
        for d, g in columns)
    note = (f"\nÔ = {'EM / F1' if args.metric == 'strict' else 'EM_lenient / F1_lenient'} (×100). **Đậm** = tốt nhất cột. † = hơn `{labels.get(args.baseline, args.baseline)}` "
            f"có ý nghĩa (bootstrap ghép cặp {args.n_boot} lần, p < {args.alpha}). {n_line}.\n")
    main_md = "\n".join(lines) + "\n" + note
    with open(os.path.join(args.out_dir, "table_main.md"), "w", encoding="utf-8") as f:
        f.write(main_md)

    # Cost table.
    priced = args.price_in is not None and args.price_out is not None
    cost_header = ["Phương pháp", "Bộ dữ liệu", "Số câu", "Gọi LLM/câu", "Token vào/câu", "Token ra/câu",
                   "Số vòng TB", "% giải hết", "Recall@5", "Giây/câu", "Câu lỗi"] + (["USD/câu"] if priced else [])
    cost_lines = ["| " + " | ".join(cost_header) + " |", "|" + "---|" * len(cost_header)]
    for m in methods:
        for d in datasets:
            run = runs.get((d, m))
            if run is None:
                continue
            # cost columns read every answered record (all runs when pooled), not the pooled scores
            recs = [r for rr in run.get("member_records", [run["records"]]) for r in rr.values()]
            sums = run["summaries"]
            n = len(recs)
            calls = mean_or_none([r.get("llm_calls") for r in recs])
            if calls is None:  # batch methods (hippo, standard): only the run average exists
                calls = mean_or_none([s.get("avg_llm_calls") for s in sums])
            ptok = mean_or_none([r.get("prompt_tokens") for r in recs]) or mean_or_none(
                [s.get("avg_prompt_tokens") for s in sums])
            ctok = mean_or_none([r.get("completion_tokens") for r in recs]) or mean_or_none(
                [s.get("avg_completion_tokens") for s in sums])
            iters = mean_or_none([r.get("iterations") for r in recs])
            resolved = [r.get("all_resolved") for r in recs if r.get("all_resolved") is not None]
            pct = 100 * float(np.mean(resolved)) if resolved else None
            found5 = sum(r.get("gold_docs_found_first5") or 0 for r in recs)
            total = sum(r.get("gold_docs_total") or 0 for r in recs)
            r5 = 100 * found5 / total if total and any("gold_docs_found_first5" in r for r in recs) else None
            secs = sum(s.get("seconds") or 0 for s in sums) / n if n else None
            errors = sum(r.get("prediction") == "Error processing query" for r in recs)
            usd = ((ptok or 0) * args.price_in + (ctok or 0) * args.price_out) / 1e6 if priced and ptok else None
            cost_lines.append("| " + " | ".join([
                row_label(m, labels), d, str(n), fmt(calls, args.decimal, 2), fmt(ptok, args.decimal, 0),
                fmt(ctok, args.decimal, 0), fmt(iters, args.decimal, 2), fmt(pct, args.decimal),
                fmt(r5, args.decimal), fmt(secs, args.decimal), str(errors)] +
                ([fmt(usd, args.decimal, 4)] if priced else [])) + " |")
    cost_md = "\n".join(cost_lines) + "\n\nRecall@5 = % đoạn vàng nằm trong 5 đoạn đầu được lấy.\n"
    with open(os.path.join(args.out_dir, "table_cost.md"), "w", encoding="utf-8") as f:
        f.write(cost_md)

    with open(os.path.join(args.out_dir, "table_main.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    print(main_md)
    print(cost_md)
    warn = [f"{d}/{m}: {sum(r.get('prediction') == 'Error processing query' for r in run['records'].values())} câu lỗi"
            for (d, m), run in runs.items()
            if any(r.get("prediction") == "Error processing query" for r in run["records"].values())]
    if warn:
        print("CẢNH BÁO, còn câu lỗi (chạy lại lệnh run_colab để bù): " + "; ".join(warn))
    if any(args.baseline == m for _, m in runs) is False:
        print(f"Lưu ý: không có lượt chạy nào tên '{args.baseline}', nên không tính †.")
    print(f"Đã lưu vào {args.out_dir}/table_main.md, table_cost.md, table_main.csv")


if __name__ == "__main__":
    main()
