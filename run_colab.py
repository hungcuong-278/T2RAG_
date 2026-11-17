"""Run T2RAG, T2RAG+ (TRAG_plus), HippoRAG2, StandardRAG or NOR on a slice of a dataset
and save the results to one JSON file.

Examples (from the repo root, after patch_t2rag.py):
    python run_colab.py --dataset musique --method trag --limit 100 --llm_name gemini-3.8-flash
    python run_colab.py --dataset musique --method trag_plus --fixes b,s1,s3 --limit 100 --llm_name gemini-3.8-flash
    python run_colab.py --dataset vnhistory --method trag --start 50 --limit 0 --answer_format short \
        --cache_tag r2 --tag test116_r2 --llm_name gemini-3.8-flash --llm_base_url http://localhost:4000/v1

Differences from the repo's main_gemini.py (the method itself is unchanged):
  * --limit N runs only the first N questions but keeps the full corpus, so the
    OpenIE triplets shipped in outputs/<dataset>_univ are reused (no extra API cost).
  * --max_qa_steps is really passed to TRAG (main_gemini.py always uses 3).
  * --min_interval spaces out LLM calls to stay under free-tier rate limits.
  * --start S skips the first S questions.
  * Counts LLM calls, cache hits and tokens per question (T2RAG / T2RAG+) or in total (others),
    including HippoRAG's fact-reranking calls; every call's token counts are kept for T2RAG.
  * One LLM cache per dataset is shared by all methods: the cache key is the prompt text,
    so prompts that do not change between configurations are never paid for twice.
    --cache_tag NAME uses a separate cache (outputs/<dataset>_llm_cache_NAME), so a repeated
    run really calls the model again and run-to-run variation can be measured.
  * --method nor answers without retrieval (same answer prompt as StandardRAG, no passages).
  * --answer_format short asks every method for one short answer (Vietnamese rules when
    --lang vi); the default "paper" keeps the prompts of the paper. Scored with the same EM/F1,
    plus EM_lenient / F1_lenient (brackets and "năm"/"huyện"-style prefixes ignored, several
    alternatives in one answer counted wrong).
  * Prints progress and saves EM, F1, retrieval recall, cost and every prediction
    to outputs/results/<run name>.json; T2RAG traces go to outputs/traces/<run name>/.
    The summary also records the question / OpenIE file MD5s, the cache used and scores by hop.
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


VI_TONES = "\u0300\u0301\u0303\u0309\u0323"  # huyền, sắc, ngã, hỏi, nặng (combining marks)
VI_VOWELS = "aăâeêioôơuưy"


def vi_tone_canonical(syllable):
    """Put the tone mark of a lower-case Vietnamese syllable on one fixed vowel, so old and new
    placement give the same text: "hoà" / "hòa" -> "hòa", "thuỷ" / "thủy" -> "thủy".
    Rule (old style): a vowel with a hat or horn (ê ô ơ â ă ư; the second one of ươ, uô, iê)
    takes the tone; otherwise the last vowel if a consonant follows, else the first vowel of a
    two-vowel nucleus and the middle one of a three-vowel nucleus. "qu" and "gi" + vowel count
    as consonants. Words that are not Vietnamese syllables come back unchanged except NFC."""
    import unicodedata
    d = unicodedata.normalize("NFD", syllable)
    tones = [ch for ch in d if ch in VI_TONES]
    if len(tones) != 1:
        return unicodedata.normalize("NFC", syllable)
    base = unicodedata.normalize("NFC", "".join(ch for ch in d if ch not in VI_TONES))
    start = 0
    if base.startswith("qu") or (base.startswith("gi") and len(base) > 2 and base[2] in VI_VOWELS):
        start = 2
    idx = [i for i in range(start, len(base)) if base[i] in VI_VOWELS]
    if not idx:
        idx = [i for i in range(len(base)) if base[i] in VI_VOWELS]  # "gì", "quý": the i / y itself
        if not idx:
            return unicodedata.normalize("NFC", syllable)
    # the vowel nucleus is the first run of consecutive vowels
    nucleus = [idx[0]]
    for i in idx[1:]:
        if i == nucleus[-1] + 1:
            nucleus.append(i)
        else:
            break
    hats = [i for i in nucleus if base[i] in "ăâêôơư"]
    if hats:
        pos = hats[-1]
    elif len(nucleus) == 1:
        pos = nucleus[0]
    elif nucleus[-1] + 1 < len(base):  # closed syllable: hoàng, thuyết
        pos = nucleus[-1]
    else:
        pos = nucleus[0] if len(nucleus) == 2 else nucleus[1]
    out = base[:pos + 1] + tones[0] + base[pos + 1:]
    return unicodedata.normalize("NFC", out)


def normalize(text):
    """Same as the repo's normalize_answer; for Vietnamese (T2RAG_LANG=vi): NFC, Unicode
    punctuation removed, no English article removal ("an" is a Vietnamese syllable), and one
    tone placement for every syllable (hoà = hòa, thuỷ = thủy)."""
    if os.environ.get("T2RAG_LANG", "en").lower() == "vi":
        import unicodedata
        text = unicodedata.normalize("NFC", str(text)).lower()
        text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
        return " ".join(vi_tone_canonical(w) for w in text.split())
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


VI_PREFIXES = ("thành phố", "kinh đô", "kinh thành", "năm", "huyện", "tỉnh", "phủ", "châu", "xã", "làng",
               "khoảng")
MULTI_SEPARATORS = re.compile(r"\s(?:hoặc|or)\s|(?<!\d)[,;/]|[,;/](?!\d)", re.IGNORECASE)


def _strip_brackets(text):
    text = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", str(text))
    return " ".join(text.split()).strip(" .")


def _strip_prefix(norm_text):
    for p in VI_PREFIXES:
        if norm_text.startswith(p + " ") and len(norm_text) > len(p) + 1:
            return norm_text[len(p) + 1:]
    return norm_text


def lenient_scores(prediction, golds, lang="en"):
    """EM / F1 after ignoring text in brackets (both sides) and, for Vietnamese, a leading
    "năm" / "huyện" / ... prefix. An answer that lists several alternatives ("A hoặc B", "A, B")
    gets 0 unless a gold answer itself contains that separator.
    Returns (em, f1, multi) where multi tells that the answer listed alternatives."""
    pred = _strip_brackets(prediction)
    golds = [g for g in golds if str(g).strip()]
    multi = bool(MULTI_SEPARATORS.search(pred)) and not any(MULTI_SEPARATORS.search(str(g)) for g in golds)
    if multi:
        return 0.0, 0.0, True
    p = normalize(pred)
    g_norm = [normalize(_strip_brackets(g)) for g in golds]
    if lang == "vi":
        p = _strip_prefix(p)
        g_norm = [_strip_prefix(g) for g in g_norm]
    best_em, best_f1 = 0.0, 0.0
    pt = p.split()
    for g in g_norm:
        gt = g.split()
        best_em = max(best_em, float(pt == gt))
        same = sum((Counter(pt) & Counter(gt)).values())
        if same:
            pr, rc = same / len(pt), same / len(gt)
            best_f1 = max(best_f1, 2 * pr * rc / (pr + rc))
    return best_em, best_f1, False


def hops_of(qid):
    """1hop_001 / 2hop__123 / 3hop1__... -> 1 / 2 / 3; None for other ids."""
    if isinstance(qid, str) and "hop" in qid.split("_")[0] and qid[0].isdigit():
        return int(qid[0])
    return None


def scores_by_hop(records):
    out = {}
    for h in sorted({r["hops"] for r in records if r.get("hops") is not None}):
        rs = [r for r in records if r.get("hops") == h]
        out[str(h)] = {"n": len(rs)}
        for key in ("em", "f1", "em_lenient", "f1_lenient"):
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            out[str(h)][key] = round(sum(vals) / len(vals), 4) if vals else None
    return out


# ---------------------------------------------------------------- answer format (--answer_format short)
PAPER_QA_SYSTEM_START = "As an advanced reading comprehension assistant"  # StandardRAG / HippoRAG / NOR
TRAG_FINAL_START = "Based on the reasoning clues, please answer the following question."  # TRAG.final_qa
TRAG_FORMAT_BLOCK = re.compile(r"Answer format requirements:.*?(?=\n\nAnswer:)", re.DOTALL)
SHORT_RULES = {
    "vi": ("Yêu cầu bắt buộc về đáp án:\n"
           "- Đáp án viết bằng tiếng Việt, chỉ gồm MỘT tên người, địa danh, triều đại, con số hoặc cụm từ ngắn.\n"
           "- Không giải thích, không ghi chú trong ngoặc, không nêu nhiều phương án, không lặp lại câu hỏi.\n"
           "- Hỏi năm thì chỉ ghi số năm (ví dụ: 1288). Hỏi người thì ghi tên như trong tài liệu.\n"
           "- Câu hỏi có/không thì chỉ ghi: có hoặc không.\n"
           "- Nếu tài liệu không đủ thông tin thì ghi đúng hai chữ: Không rõ"),
    "en": ("Answer requirements:\n"
           "- Give ONE entity, date, number or short phrase only.\n"
           "- No explanation, no text in brackets, no list of alternatives, do not repeat the question.\n"
           "- For a year, write only the year (e.g. 1288). For a yes/no question, write only yes or no.\n"
           "- If the information is not available, write exactly: Unknown"),
}


def format_messages(messages, lang):
    """Rewrite the final answer prompts of the three methods; every other prompt is unchanged."""
    rules = SHORT_RULES.get(lang, SHORT_RULES["en"])
    out = [dict(m) for m in messages]
    changed = False
    for m in out:
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if m.get("role") == "system" and content.startswith(PAPER_QA_SYSTEM_START):
            m["content"] = content + "\n\nAfter \"Answer: \" write only the final answer.\n" + rules
            changed = True
        elif m.get("role") == "user" and content.startswith(TRAG_FINAL_START):
            if TRAG_FORMAT_BLOCK.search(content):
                m["content"] = TRAG_FORMAT_BLOCK.sub(lambda _m: rules, content, count=1)
            else:
                m["content"] = content.replace("\nAnswer:", "\n" + rules + "\n\nAnswer:", 1)
            changed = True
    return out if changed else messages


def install_answer_format(llm_model, lang):
    original = llm_model.infer

    def formatted_infer(messages, *args, **kwargs):
        return original(format_messages(messages, lang), *args, **kwargs)

    llm_model.infer = formatted_infer


def make_llm(config):
    from src.llm import _get_llm_class
    return _get_llm_class(config)


def run_nor(config, queries, llm_model):
    """NOR baseline: StandardRAG's answer prompt (same template, same 'Answer:' parsing),
    but no passage is given, so the model answers from what it already knows."""
    from src.prompts.prompt_template_manager import PromptTemplateManager
    ptm = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
    name = f"rag_qa_{config.dataset}"
    if not ptm.is_template_name_valid(name=name):
        name = "rag_qa_musique"
    solutions = []
    for q in queries:
        messages = ptm.render(name=name, prompt_user="Question: " + q + "\nThought: ")
        response = llm_model.infer(messages)
        text = response[0] if isinstance(response, tuple) else response
        answer = str(text).split("Answer:")[-1].strip()
        solutions.append(types.SimpleNamespace(question=q, answer=answer, docs=[]))
    return solutions


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
    """Wraps llm_model.infer to count calls, cache hits and tokens, and keeps every call's
    [prompt_tokens, completion_tokens, cache_hit]. A call whose output is longer than
    `warn_tokens` is reported at once (the model may be stuck in a long hidden reasoning)."""

    def __init__(self, llm_model, warn_tokens=20000):
        self.reset()
        self.total = {"llm_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.max_completion_tokens = 0
        self.long_calls = 0
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
            counter.calls.append([add["prompt_tokens"], add["completion_tokens"], hit])
            counter.max_completion_tokens = max(counter.max_completion_tokens, add["completion_tokens"])
            if warn_tokens and add["completion_tokens"] > warn_tokens:
                counter.long_calls += 1
                print(f"WARNING: one LLM call returned {add['completion_tokens']} output tokens "
                      f"(> {warn_tokens}); check the cost of this run.", flush=True)
            return result

        llm_model.infer = counted_infer

    def reset(self):
        self.current = {"llm_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.calls = []

    def snapshot(self):
        out = dict(self.current)
        out["calls"] = [list(c) for c in self.calls]
        return out


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
    ap.add_argument("--method", default="trag", choices=["trag", "trag_plus", "standard", "hippo", "nor"])
    ap.add_argument("--fixes", default="b,s1,s2,s3,s4",
                    help="trag_plus only: comma list of b,s1,s2,s3,s4,h2a (empty = same as trag)")
    ap.add_argument("--start", type=int, default=0, help="Skip the first N questions")
    ap.add_argument("--limit", type=int, default=50, help="Number of questions (0 = all after --start)")
    ap.add_argument("--qids", default="",
                    help="Comma list of question ids to run instead of --start/--limit (e.g. a pilot "
                         "of 2hop_001,3hop_001); the run name gets _ids<N>")
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
    ap.add_argument("--answer_format", default="paper", choices=["paper", "short"],
                    help="paper = the paper's answer prompts; short = ask for one short answer "
                         "(Vietnamese rules with --lang vi). Use a different --tag for each format.")
    ap.add_argument("--cache_tag", default="",
                    help="Use a separate LLM cache outputs/<dataset>_llm_cache_<tag> (e.g. r2 for a second "
                         "run that must really call the model again); empty = the shared cache")
    ap.add_argument("--warn_tokens", type=int, default=20000,
                    help="Print a warning when one LLM call returns more output tokens than this")
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
        raise SystemExit("Set GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY first "
                         "(with gw_proxy any value works: the proxy adds the real key).")
    if args.cache_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.cache_tag):
        raise SystemExit("--cache_tag may only contain letters, digits, '_', '-' and '.'")

    from src.utils.config_utils import BaseConfig
    from src.evaluation.qa_eval import calculate_overall_qa_metrics
    import numpy as np

    install_rate_limiter(args.min_interval)

    dataset = args.dataset
    corpus = json.load(open(f"reproduce/dataset/{dataset}_corpus.json", encoding="utf-8"))
    questions_path = f"reproduce/dataset/{dataset}.json"
    samples = json.load(open(questions_path, encoding="utf-8"))
    if args.qids:
        wanted = [q.strip() for q in args.qids.split(",") if q.strip()]
        index = {str(s.get("id", s.get("_id"))): i for i, s in enumerate(samples)}
        unknown = [q for q in wanted if q not in index]
        if unknown:
            raise SystemExit(f"Unknown --qids {unknown[:5]}")
        positions = sorted(index[q] for q in wanted)
        samples = [samples[i] for i in positions]
        args.start = 0
    else:
        positions = list(range(args.start, len(samples)))
        samples = samples[args.start:]
        if args.limit and args.limit > 0:
            samples = samples[:args.limit]
        positions = positions[:len(samples)]
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
    known = {"b", "s1", "s2", "s3", "s4", "s5", "h2a"}
    if set(fixes) - known:
        raise SystemExit(f"Unknown --fixes {sorted(set(fixes) - known)}; use {sorted(known)}")
    method_label = args.method + (("_fx-" + "-".join(fixes)) if args.method == "trag_plus" else "")
    llm_label = args.llm_name.replace("/", "-").replace(":", "-")
    run_name = f"{dataset}_{method_label}_{llm_label}" + (f"_from{args.start}" if args.start else "") \
               + (f"_ids{len(queries)}" if args.qids else f"_{len(queries)}q") + (f"_{args.tag}" if args.tag else "")

    reusable_dir = f"outputs/{dataset}_univ"
    method_dir = f"outputs/{dataset}_{args.method}"
    has_openie = os.path.exists(os.path.join(reusable_dir, "openie_results_ner.json"))
    shared_cache = os.path.join("outputs", f"{dataset}_llm_cache" + (f"_{args.cache_tag}" if args.cache_tag else ""))

    # Never overwrite the result of a different configuration that happens to have the same name.
    out_path = os.path.join("outputs/results", run_name + ".json")
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path, encoding="utf-8")).get("summary", {})
        except Exception:
            old = {}
        mine = {"answer_format": args.answer_format, "embedding": args.embedding_name, "llm_cache": shared_cache}
        theirs = {"answer_format": old.get("answer_format", "paper"), "embedding": old.get("embedding"),
                  "llm_cache": old.get("llm_cache", os.path.join("outputs", f"{dataset}_llm_cache"))}
        diff = [k for k in mine if theirs[k] is not None and theirs[k] != mine[k]]
        if diff:
            raise SystemExit(f"STOP: {out_path} already holds a run with a different {', '.join(diff)} "
                             f"({', '.join(f'{k}={theirs[k]}' for k in diff)}). Add a different --tag.")
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
    os.makedirs(shared_cache, exist_ok=True)

    t0 = time.time()
    print(f"Dataset {dataset}: {len(queries)} questions, {len(docs)} passages in the corpus")
    print(f"Method {method_label} | LLM {args.llm_name} | embedding {args.embedding_name} | lang {lang} | "
          f"questions {args.start}..{args.start + len(queries) - 1} | answers {args.answer_format} | "
          f"cache {shared_cache}")
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
    elif args.method == "nor":
        rag = types.SimpleNamespace(llm_model=make_llm(config))
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
    if args.answer_format == "short" and hasattr(rag, "llm_model"):
        install_answer_format(rag.llm_model, lang)
    counter = LLMCounter(rag.llm_model, warn_tokens=args.warn_tokens) if hasattr(rag, "llm_model") else None
    # HippoRAG's fact reranker keeps its own reference to llm_model.infer (src/rerank.py: DSPyFilter
    # stores llm_infer_fn at construction), so it would bypass the counter: point it at the wrapper.
    if counter and getattr(getattr(rag, "rerank_filter", None), "llm_infer_fn", None) is not None:
        rag.rerank_filter.llm_infer_fn = rag.llm_model.infer

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
                if "plus_stats" in out:  # T2RAG+: placeholder bindings and S4 decisions of this question
                    ps = out["plus_stats"]
                    stats.update({"n_bindings": len(out.get("bindings") or {}),
                                  "bindings_applied": ps.get("bindings_applied"),
                                  "s4_rejected": ps.get("s4_rejected"), "s4_accepted": ps.get("s4_accepted"),
                                  "s5_extracted": ps.get("s5_extracted"), "s5_reasked": ps.get("s5_reasked")})
                return out
            finally:
                if counter:
                    stats.update(counter.snapshot())
                stats["seconds"] = round(time.time() - t_q, 1)
                per_question.append(stats)

        rag.trag_qa = types.MethodType(trag_qa_with_stats, rag)

    if args.method != "nor":
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
    if args.method == "nor":
        solutions = run_nor(config, queries, rag.llm_model)
    else:
        result = rag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=gold_answers)
        solutions = result[0]

    predictions = [s.answer for s in solutions]
    overall = calculate_overall_qa_metrics(gold_answers=[list(g) for g in gold_answers],
                                           predicted_answers=predictions, aggregation_fn=np.max)
    records, recall_hits, recall5_hits, n_gold, n_docs = [], 0, 0, 0, []
    for i, sol in enumerate(solutions):
        em, f1 = em_f1(sol.answer, gold_answers[i])
        em_l, f1_l, multi = lenient_scores(sol.answer, gold_answers[i], lang)
        retrieved = list(sol.docs or [])
        n_docs.append(len(retrieved))
        found = found5 = None
        if gold_docs is not None and args.method != "nor":
            golds = gold_docs[i]
            found = sum(g in retrieved for g in golds)
            found5 = sum(g in retrieved[:5] for g in golds)
            recall_hits += found
            recall5_hits += found5
            n_gold += len(golds)
        qid = samples[i].get("id", samples[i].get("_id", str(i)))
        hops = hops_of(qid)  # MuSiQue: 2hop / 3hop1 / 4hop2 ...; vnhistory: 1hop_ / 2hop_ / 3hop_
        rec = {"id": positions[i], "qid": qid, "hops": hops, "question": sol.question,
               "prediction": sol.answer, "gold_answers": sorted(gold_answers[i]), "em": em,
               "f1": round(f1, 4), "em_lenient": em_l, "f1_lenient": round(f1_l, 4),
               "multi_answer": multi, "long_answer": len(str(sol.answer).split()) > 8,
               "num_retrieved_docs": len(retrieved), "gold_docs_found": found,
               "gold_docs_found_first5": found5,
               "gold_docs_total": len(gold_docs[i]) if gold_docs and args.method != "nor" else None}
        if i < len(per_question):
            rec.update(per_question[i])
        records.append(rec)

    errors = sum(p == "Error processing query" for p in predictions)

    def avg(key, digits=None):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        if not vals:
            return None
        return round(float(np.mean(vals)), digits if digits is not None else (4 if key.endswith("lenient") else 2))

    summary = {
        "run_name": run_name, "dataset": dataset, "method": method_label, "base_method": args.method,
        "fixes": fixes, "tag": args.tag, "lang": lang, "llm": args.llm_name, "start": args.start,
        "embedding": args.embedding_name, "num_questions": len(queries), "qids": args.qids or None,
        "qa_top_k": args.qa_top_k, "max_qa_steps": args.max_qa_steps,
        "answer_format": args.answer_format, "llm_cache": shared_cache,
        "questions_md5": file_md5(questions_path),
        "openie_md5": file_md5(os.path.join(reusable_dir, "openie_results_ner.json")) if has_openie else None,
        # EM / F1 = mean of the per-question scores below (run_colab.normalize); the repo's own
        # evaluation is kept for reference (it differs only for Vietnamese tone placement)
        "ExactMatch": avg("em", 4), "F1": avg("f1", 4),
        "ExactMatch_repo": overall["ExactMatch"], "F1_repo": overall["F1"],
        "EM_lenient": avg("em_lenient"), "F1_lenient": avg("f1_lenient"),
        "n_long_answers": sum(bool(r["long_answer"]) for r in records),
        "n_multi_answers": sum(bool(r["multi_answer"]) for r in records),
        "by_hops": scores_by_hop(records),
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
        "plus_totals": ({k: sum(r.get(k) or 0 for r in records)
                         for k in ("n_bindings", "bindings_applied", "s4_rejected", "s4_accepted", "n_dropped_clues",
                                   "s5_extracted", "s5_reasked")}
                        if args.method == "trag_plus" else None),
        "max_completion_tokens": counter.max_completion_tokens if counter else None,
        "n_long_calls": counter.long_calls if counter else None,
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n=== Results ===")
    for k, v in summary.items():
        if k not in ("settings", "total_llm", "by_hops"):
            print(f"{k:28s} {v}")
    if errors:
        print(f"\nWARNING: {errors} questions failed (often rate limits). Run the same command again: "
              "answers already received are cached, so only the failed ones call the API.")
    for h, v in summary["by_hops"].items():
        print(f"  {h}-hop ({v['n']:3d} q)   EM {v['em']}   F1 {v['f1']}   EM_lenient {v['em_lenient']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
