"""Run T2RAG, T2RAG+ (TRAG_plus), HippoRAG2 or StandardRAG on a slice of a dataset
and save the results to one JSON file.

Examples (from the repo root, after patch_t2rag.py):
    python run_colab.py --dataset musique --method trag --limit 100 --llm_name gemini-3.5-flash
    python run_colab.py --dataset musique --method trag_plus --fixes b,s1,s3 --limit 100 --llm_name gemini-3.5-flash
    python run_colab.py --dataset musique --method trag --start 200 --limit 200 --llm_name gemini-3.5-flash

Differences from the repo's main_gemini.py (the method itself is unchanged):
  * --limit N runs only the first N questions but keeps the full corpus, so the
    OpenIE triplets shipped in outputs/<dataset>_univ are reused (no extra API cost).
  * --max_qa_steps is really passed to TRAG (main_gemini.py always uses 3).
  * --min_interval spaces out LLM calls to stay under free-tier rate limits.
  * --start S skips the first S questions (dev = 0-99, test = 200-399).
  * Counts LLM calls, cache hits and tokens per question (T2RAG / T2RAG+) or in total (others).
  * One LLM cache per dataset is shared by all methods: the cache key is the prompt text,
    so prompts that do not change between configurations are never paid for twice.
  * Prints progress and saves EM, F1, retrieval recall, cost and every prediction
    to outputs/results/<run name>.json; T2RAG traces go to outputs/traces/<run name>/.
"""
import argparse
import json
import logging
import os
import re
import shutil
import string
import threading
import time
import types
from collections import Counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_gold_docs(samples, dataset_name):
    """Same logic as main_gemini.py."""
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_title = set(item[0] for item in sample["supporting_facts"])
            items = [item for item in sample["context"] if item[0] in gold_title]
            sep = "" if dataset_name.startswith("hotpotqa") else " "
            gold_doc = [item[0] + "\n" + sep.join(item[1]) for item in items]
        elif "contexts" in sample:
            gold_doc = [c["title"] + "\n" + c["text"] for c in sample["contexts"] if c["is_supporting"]]
        else:
            paragraphs = [p for p in sample["paragraphs"] if p.get("is_supporting", True) is not False]
            gold_doc = [p["title"] + "\n" + (p["text"] if "text" in p else p["paragraph_text"]) for p in paragraphs]
        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples):
    """Same logic as main_gemini.py."""
    gold_answers = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            gold = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            gold = sample["reference"]
        elif "obj" in sample:
            gold = list({sample["obj"], *[sample["possible_answers"]], sample["o_wiki_title"], *[sample["o_aliases"]]})
        else:
            raise ValueError("Sample has no gold answer")
        gold = {gold} if isinstance(gold, str) else set(gold)
        gold.update(sample.get("answer_aliases", []))
        gold_answers.append(gold)
    return gold_answers


def normalize(text):
    """Same as the repo's normalize_answer; for Vietnamese (T2RAG_LANG=vi): NFC, Unicode
    punctuation removed, and no English article removal ("an" is a Vietnamese syllable)."""
    if os.environ.get("T2RAG_LANG", "en").lower() == "vi":
        import unicodedata
        text = unicodedata.normalize("NFC", str(text)).lower()
        text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
        return " ".join(text.split())
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_f1(prediction, golds):
    """Per-question scores with the same normalization as the repo (for the saved file)."""
    best_em, best_f1 = 0.0, 0.0
    pred_tokens = normalize(prediction).split()
    for gold in golds:
        gold_tokens = normalize(gold).split()
        best_em = max(best_em, float(pred_tokens == gold_tokens))
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same:
            p, r = same / len(pred_tokens), same / len(gold_tokens)
            best_f1 = max(best_f1, 2 * p * r / (p + r))
    return best_em, best_f1


def install_rate_limiter(min_interval):
    """Leave at least `min_interval` seconds between real (uncached) LLM calls."""
    if min_interval <= 0:
        return
    from openai.resources.chat.completions import Completions
    original = Completions.create
    lock, last = threading.Lock(), [0.0]

    def throttled(self, *args, **kwargs):
        with lock:
            wait = last[0] + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()
        return original(self, *args, **kwargs)

    Completions.create = throttled


class LLMCounter:
    """Wraps llm_model.infer to count calls, cache hits and tokens."""

    def __init__(self, llm_model):
        self.reset()
        self.total = {"llm_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        original = llm_model.infer
        counter = self

        def counted_infer(*args, **kwargs):
            result = original(*args, **kwargs)
            meta = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
            hit = bool(result[2]) if isinstance(result, tuple) and len(result) > 2 else False
            add = {"llm_calls": 1, "cache_hits": int(hit),
                   "prompt_tokens": int((meta or {}).get("prompt_tokens", 0) or 0),
                   "completion_tokens": int((meta or {}).get("completion_tokens", 0) or 0)}
            for k, v in add.items():
                counter.current[k] += v
                counter.total[k] += v
            return result

        llm_model.infer = counted_infer

    def reset(self):
        self.current = {"llm_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def snapshot(self):
        return dict(self.current)


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def missing_openie_chunks(openie_path, docs):
    """Passages of the corpus that have no entry in the OpenIE file.

    HippoRAG.index would send these to the LLM with the repo's English OpenIE prompts and then
    rewrite the file, so they are checked before any graph method runs. Keys are computed the
    way HippoRAG does it: chunk-<md5(passage)>, with passage = title + "\n" + text."""
    import hashlib
    with open(openie_path, encoding="utf-8") as f:
        info = json.load(f).get("docs", [])
    have = {"chunk-" + hashlib.md5(d.get("passage", "").encode("utf-8")).hexdigest() for d in info}
    return [d for d in docs if "chunk-" + hashlib.md5(d.encode("utf-8")).hexdigest() not in have]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa",
                    help="popqa, 2wikimultihopqa, musique, hotpotqa, story, medical, sample")
    ap.add_argument("--method", default="trag", choices=["trag", "trag_plus", "standard", "hippo"])
    ap.add_argument("--fixes", default="b,s1,s2,s3,s4",
                    help="trag_plus only: comma list of b,s1,s2,s3,s4,h2a (empty = same as trag)")
    ap.add_argument("--start", type=int, default=0, help="Skip the first N questions")
    ap.add_argument("--limit", type=int, default=50, help="Number of questions (0 = all after --start)")
    ap.add_argument("--llm_name", default="gemini-2.5-flash")
    ap.add_argument("--llm_base_url", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--embedding_name", default="BAAI/bge-m3")
    ap.add_argument("--qa_top_k", type=int, default=5)
    ap.add_argument("--max_qa_steps", type=int, default=3)
    ap.add_argument("--s1_max_steps", type=int, default=5, help="trag_plus + s1: max rounds")
    ap.add_argument("--s2_min_per_clue", type=int, default=2, help="trag_plus + s2: chunks reserved per clue")
    ap.add_argument("--s4_threshold", type=float, default=0.8, help="trag_plus + s4: evidence match threshold")
    ap.add_argument("--tag", default="", help="Extra text added to the run name")
    ap.add_argument("--min_interval", type=float, default=0.0,
                    help="Seconds between LLM calls (free tier: about 60 / requests-per-minute)")
    ap.add_argument("--max_retries", type=int, default=8)
    ap.add_argument("--verbose", action="store_true", help="Show the repo's detailed logs")
    ap.add_argument("--lang", default="auto", choices=["auto", "en", "vi"],
                    help="Answer normalisation and HippoRAG entity keys; auto = vi for vnhistory* datasets")
    ap.add_argument("--allow_openie_extraction", action="store_true",
                    help="hippo only: allow HippoRAG to extract missing triplets with the LLM and rewrite "
                         "the OpenIE file (off by default: the run stops instead)")
    args = ap.parse_args()
    lang = args.lang if args.lang != "auto" else ("vi" if args.dataset.startswith("vnhistory") else "en")
    os.environ["T2RAG_LANG"] = lang  # read by the patched normalize_answer / text_processing

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("openai", "httpx", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise SystemExit("Set GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY first.")

    from src.utils.config_utils import BaseConfig
    from src.evaluation.qa_eval import calculate_overall_qa_metrics
    import numpy as np

    install_rate_limiter(args.min_interval)

    dataset = args.dataset
    corpus = json.load(open(f"reproduce/dataset/{dataset}_corpus.json", encoding="utf-8"))
    samples = json.load(open(f"reproduce/dataset/{dataset}.json", encoding="utf-8"))
    samples = samples[args.start:]
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    if not samples:
        raise SystemExit("No questions selected: check --start and --limit.")
    docs = [f"{d['title']}\n{d['text']}" for d in corpus]
    queries = [s["question"] for s in samples]
    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset)
    except Exception:
        gold_docs = None

    fixes = sorted(f.strip().lower() for f in args.fixes.split(",") if f.strip()) \
        if args.method == "trag_plus" else []
    known = {"b", "s1", "s2", "s3", "s4", "h2a"}
    if set(fixes) - known:
        raise SystemExit(f"Unknown --fixes {sorted(set(fixes) - known)}; use {sorted(known)}")
    method_label = args.method + (("_fx-" + "-".join(fixes)) if args.method == "trag_plus" else "")
    llm_label = args.llm_name.replace("/", "-").replace(":", "-")
    run_name = f"{dataset}_{method_label}_{llm_label}" + (f"_from{args.start}" if args.start else "") \
               + f"_{len(queries)}q" + (f"_{args.tag}" if args.tag else "")

    reusable_dir = f"outputs/{dataset}_univ"
    method_dir = f"outputs/{dataset}_{args.method}"
    has_openie = os.path.exists(os.path.join(reusable_dir, "openie_results_ner.json"))
    if args.method in ("trag", "trag_plus") and not has_openie:
        raise SystemExit(f"No triplets at {reusable_dir}/openie_results_ner.json. "
                         "T2RAG only reads existing OpenIE results; build them first.")

    config = BaseConfig(
        save_dir=method_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=f"{dataset}_univ",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path="src/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        recall_eval_top_k=200,
        linking_top_k=5,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=64,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode="online",
        # HippoRAG keeps at most ~101 neighbours per entity (HippoRAG.py: "num_nns > 100" -> break),
        # so k=200 gives the same synonym edges as the default 2047 with ~10x less RAM
        # (2047 ran out of Colab's 12.7 GB on MuSiQue's ~100k entities).
        synonymy_edge_topk=200,
        max_retry_attempts=args.max_retries,
    )
    config.save_dir = method_dir
    config.exclude_llm_from_paths = True
    config.embedding_base_dir = reusable_dir
    shared_cache = os.path.join("outputs", f"{dataset}_llm_cache")
    os.makedirs(shared_cache, exist_ok=True)

    t0 = time.time()
    print(f"Dataset {dataset}: {len(queries)} questions, {len(docs)} passages in the corpus")
    print(f"Method {method_label} | LLM {args.llm_name} | embedding {args.embedding_name} | lang {lang} | "
          f"questions {args.start}..{args.start + len(queries) - 1}")
    print("Loading models and embeddings (the first run embeds the whole corpus and all triplets; "
          "later runs read them from the cache)...")

    per_question = []
    if args.method in ("trag", "trag_plus"):
        if args.method == "trag":
            from src.TRAG import TRAG
            rag = TRAG(global_config=config, max_qa_steps=args.max_qa_steps)
        else:
            from src.TRAG_plus import TRAGPlus
            rag = TRAGPlus(global_config=config, max_qa_steps=args.max_qa_steps, fixes=fixes,
                           s1_max_steps=args.s1_max_steps, s2_min_per_clue=args.s2_min_per_clue,
                           s4_match_threshold=args.s4_threshold)
        # One trace folder per run, so dev / test runs of a configuration never overwrite each other.
        rag.intermediate_results_dir = os.path.join("outputs", "traces", run_name)
        shutil.rmtree(rag.intermediate_results_dir, ignore_errors=True)
        os.makedirs(rag.intermediate_results_dir, exist_ok=True)
    elif args.method == "hippo":
        # HippoRAG2 (graph + Personalized PageRank) builds its graph from the same OpenIE triplets.
        from src.HippoRAG import HippoRAG
        openie_path = os.path.join(reusable_dir, "openie_results_ner.json")
        openie_md5 = None
        if has_openie:
            missing = missing_openie_chunks(openie_path, docs)
            if missing and not args.allow_openie_extraction:
                raise SystemExit(
                    f"STOP: {len(missing)}/{len(docs)} passages have no triplets in {openie_path}. "
                    "HippoRAG would extract them with the LLM (English prompts) and rewrite the file. "
                    f"First missing passage: {missing[0][:80]!r}. Fix the corpus / OpenIE file, or pass "
                    "--allow_openie_extraction if this is really wanted.")
            openie_md5 = file_md5(openie_path)
            shutil.copy2(openie_path, openie_path + ".bak")
            print(f"OpenIE file OK: every passage has triplets (md5 {openie_md5[:8]}); backup at {openie_path}.bak")
        if not args.allow_openie_extraction:
            config.save_openie = False  # never rewrite the shared OpenIE file
        rag = HippoRAG(global_config=config)
    else:
        from src.StandardRAG import StandardRAG
        rag = StandardRAG(global_config=config)

    # Share one LLM cache per dataset across methods (key = prompt text, so this is safe).
    if hasattr(rag, "llm_model") and hasattr(rag.llm_model, "cache_file_name"):
        rag.llm_model.cache_file_name = os.path.join(
            shared_cache, os.path.basename(rag.llm_model.cache_file_name))
    counter = LLMCounter(rag.llm_model) if hasattr(rag, "llm_model") else None

    if args.method in ("trag", "trag_plus"):
        original_qa = rag.trag_qa  # bound method of the instance's class

        def trag_qa_with_stats(self, query, query_id=None, **kwargs):
            n = len(per_question) + 1
            print(f"[{n}/{len(queries)}] {time.time() - t0:6.0f}s  {query[:80]}", flush=True)
            if counter:
                counter.reset()
            t_q = time.time()
            stats = {"trace_file": f"{query_id}_results.json"}
            try:
                out = original_qa(query, query_id=query_id, **kwargs)
                stats.update({
                    "iterations": out.get("total_iterations"),
                    "all_resolved": str(out.get("completion_status", "")).startswith("all clues"),
                    "n_final_clues": len(out.get("all_final_clues", []) or []),
                    "n_remaining_fuzzy": len(out.get("remaining_fuzzy_clues", []) or []),
                    "n_dropped_clues": out.get("n_dropped_clues"),
                })
                return out
            finally:
                if counter:
                    stats.update(counter.snapshot())
                stats["seconds"] = round(time.time() - t_q, 1)
                per_question.append(stats)

        rag.trag_qa = types.MethodType(trag_qa_with_stats, rag)

    rag.index(docs)
    if args.method == "hippo" and openie_md5:
        now = file_md5(openie_path)
        if now != openie_md5:
            shutil.copy2(openie_path + ".bak", openie_path)
            print(f"WARNING: HippoRAG changed {openie_path} (md5 {openie_md5[:8]} -> {now[:8]}); "
                  "the original file was restored from the backup.")
        else:
            print(f"OpenIE file unchanged after indexing (md5 {now[:8]}).")
    print(f"Index ready after {time.time() - t0:.0f}s. Answering...")
    result = rag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=gold_answers)
    solutions = result[0]

    predictions = [s.answer for s in solutions]
    overall = calculate_overall_qa_metrics(gold_answers=[list(g) for g in gold_answers],
                                           predicted_answers=predictions, aggregation_fn=np.max)
    records, recall_hits, recall5_hits, n_gold, n_docs = [], 0, 0, 0, []
    for i, sol in enumerate(solutions):
        em, f1 = em_f1(sol.answer, gold_answers[i])
        retrieved = list(sol.docs or [])
        n_docs.append(len(retrieved))
        found = found5 = None
        if gold_docs is not None:
            golds = gold_docs[i]
            found = sum(g in retrieved for g in golds)
            found5 = sum(g in retrieved[:5] for g in golds)
            recall_hits += found
            recall5_hits += found5
            n_gold += len(golds)
        qid = samples[i].get("id", samples[i].get("_id", str(i)))
        hops = None
        if isinstance(qid, str) and "hop" in qid.split("_")[0]:
            hops = int(qid.split("_")[0][0])  # MuSiQue ids start with 2hop / 3hop1 / 4hop2 ...
        rec = {"id": args.start + i, "qid": qid, "hops": hops, "question": sol.question,
               "prediction": sol.answer, "gold_answers": sorted(gold_answers[i]), "em": em,
               "f1": round(f1, 4), "num_retrieved_docs": len(retrieved), "gold_docs_found": found,
               "gold_docs_found_first5": found5,
               "gold_docs_total": len(gold_docs[i]) if gold_docs else None}
        if i < len(per_question):
            rec.update(per_question[i])
        records.append(rec)

    errors = sum(p == "Error processing query" for p in predictions)

    def avg(key):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        return round(float(np.mean(vals)), 2) if vals else None

    summary = {
        "run_name": run_name, "dataset": dataset, "method": method_label, "base_method": args.method,
        "fixes": fixes, "tag": args.tag, "lang": lang, "llm": args.llm_name, "start": args.start,
        "embedding": args.embedding_name, "num_questions": len(queries),
        "qa_top_k": args.qa_top_k, "max_qa_steps": args.max_qa_steps,
        "ExactMatch": overall["ExactMatch"], "F1": overall["F1"],
        "recall_first5_docs": round(recall5_hits / n_gold, 4) if n_gold else None,
        "recall_all_retrieved_docs": round(recall_hits / n_gold, 4) if n_gold else None,
        "avg_retrieved_docs": round(float(np.mean(n_docs)), 2) if n_docs else 0,
        "errors": errors, "seconds": round(time.time() - t0, 1),
        "avg_llm_calls": avg("llm_calls"), "avg_cache_hits": avg("cache_hits"),
        "avg_prompt_tokens": avg("prompt_tokens"), "avg_completion_tokens": avg("completion_tokens"),
        "avg_iterations": avg("iterations"),
        "pct_all_resolved": (round(100 * float(np.mean([bool(r.get("all_resolved")) for r in records])), 1)
                             if per_question else None),
        "total_llm": counter.total if counter else None,
        "trace_dir": getattr(rag, "intermediate_results_dir", None),
        "settings": {"s1_max_steps": args.s1_max_steps, "s2_min_per_clue": args.s2_min_per_clue,
                     "s4_threshold": args.s4_threshold} if args.method == "trag_plus" else None,
    }
    if counter and not per_question and len(queries):
        # Methods that answer in batches: only the average per question is known.
        summary["avg_llm_calls"] = round(counter.total["llm_calls"] / len(queries), 2)
        summary["avg_prompt_tokens"] = round(counter.total["prompt_tokens"] / len(queries), 1)
        summary["avg_completion_tokens"] = round(counter.total["completion_tokens"] / len(queries), 1)

    os.makedirs("outputs/results", exist_ok=True)
    out_path = os.path.join("outputs/results", run_name + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n=== Results ===")
    for k, v in summary.items():
        if k not in ("settings", "total_llm"):
            print(f"{k:28s} {v}")
    if errors:
        print(f"\nWARNING: {errors} questions failed (often rate limits). Run the same command again: "
              "answers already received are cached, so only the failed ones call the API.")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
