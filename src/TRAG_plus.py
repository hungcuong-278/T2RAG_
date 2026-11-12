"""T2RAG+ : the official T2RAG with switchable fixes. Put this file at src/TRAG_plus.py.

The original src/TRAG.py is not modified. Every change lives in the subclass TRAGPlus and is
switched on by a flag in `fixes`; with no flags the class behaves like TRAG (same prompts,
same retrieval, same loop), so ablations are easy and the baseline stays intact.

Flags
  b   bug fixes found in the official code:
        L2  clues with 2+ placeholders are no longer dropped by the "ENFORCE" step (TRAG.py 1236-1245)
        L4  placeholders are counted as tokens (?, ?A, ?director), not as '?' characters (TRAG.py 1064)
        L5  a proposition found in several passages keeps every passage (TRAG.py 206 overwrote them)
        L6  question words are removed as whole words, not substrings ("whose" -> "se") (TRAG.py 481)
  s1  named placeholders (?A, ?B) + a binding table that substitutes values into every clue;
      unresolved clues are kept for the next round instead of being dropped (TRAG.py 739-740,
      1234, 1247); up to s1_max_steps rounds, stopping early when a round makes no progress
  s2  hybrid retrieval: BM25 on passages + vector search on propositions, fused with RRF;
      each searchable clue is guaranteed s2_min_per_clue passages of its own
  s3  the final answer step also reads up to 3 evidence passages (TRAG.py final_qa reads clues only)
  s4  each resolved value must come with a quoted evidence sentence that is checked against the
      passages; unsupported values are not locked and are searched again (best with s1)
  h2a reserved for the graph fallback (Huong 2), not implemented yet
"""
import contextlib
import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .TRAG import TRAG

logger = logging.getLogger(__name__)

FIELDS = ("subject", "predicate", "object")
# A placeholder is "?" or "?Name" standing as its own token (not the "?" ending a real title).
PLACEHOLDER_TOKEN = re.compile(r"(?<![\w?])\?[A-Za-z0-9_]*(?![\w?])")
QUESTION_WORDS = re.compile(r"\b(what|who|whom|whose|where|when|why|how|which)\b", re.IGNORECASE)
WORD = re.compile(r"\w+", re.UNICODE)


# --------------------------------------------------------------------------- small helpers
def placeholders_in(text: str) -> List[str]:
    return PLACEHOLDER_TOKEN.findall(text or "")


def clue_placeholders(clue: Dict[str, str]) -> List[str]:
    out = []
    for f in FIELDS:
        out.extend(placeholders_in(clue.get(f, "")))
    return out


def count_placeholders(clue: Dict[str, str]) -> int:
    """L4: count placeholder tokens, not '?' characters."""
    return len(clue_placeholders(clue))


def count_question_marks(clue: Dict[str, str]) -> int:
    """The original rule (TRAG.py): every '?' character counts."""
    return sum(str(v).count("?") for v in clue.values())


def strip_placeholders(text: str) -> str:
    return " ".join(PLACEHOLDER_TOKEN.sub(" ", text or "").split())


def clue_key(clue: Dict[str, str]) -> Tuple[str, str, str]:
    return tuple(clue.get(f, "").strip().lower() for f in FIELDS)


def clean(clue: Dict[str, str]) -> Dict[str, str]:
    """Keep only subject / predicate / object (other code sums over clue.values())."""
    return {f: str(clue.get(f, "")).strip() for f in FIELDS}


def tokens(text: str) -> List[str]:
    return WORD.findall((text or "").lower())


def norm_text(text: str) -> str:
    return " ".join(tokens(text))


# --------------------------------------------------------------------------- BM25 (no extra package)
class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = np.zeros(len(docs), dtype=np.float32)
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            tf = Counter(tokens(d))
            self.doc_len[i] = sum(tf.values())
            for t, c in tf.items():
                self.postings[t].append((i, c))
        self.n = len(docs)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for t in set(tokens(query)):
            if t not in self.postings:
                continue
            idf = self.idf[t]
            for i, c in self.postings[t]:
                denom = c + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                s[i] += idf * c * (self.k1 + 1) / denom
        return s


def rrf(rankings: List[List[Any]], k: int = 60) -> Dict[Any, float]:
    fused: Dict[Any, float] = defaultdict(float)
    for ranking in rankings:
        for r, item in enumerate(ranking):
            fused[item] += 1.0 / (k + r + 1)
    return fused


# --------------------------------------------------------------------------- prompt texts
S1_DECOMPOSE_REPLACEMENTS = [
    ('3. Use "?" as placeholder for unknown entities',
     '3. Use NAMED placeholders for unknown entities: ?A, ?B, ?C ... Each unknown entity gets its own name, '
     'and the SAME unknown entity must use the SAME name in every triple'),
    ('Triple: France | has capital | ?', 'Triple: France | has capital | ?A'),
    ('Triples: ? | won Best Picture | 2020\n           ? | is directed by | ?',
     'Triples: ?A | won Best Picture | 2020\n           ?A | is directed by | ?B'),
    ('?directorA | was born in | ?\n           ?directorB | was born in | ?',
     '?directorA | was born in | ?birthA\n           ?directorB | was born in | ?birthB'),
    ('[List each triple on a new line in format: subject | predicate | object]',
     '[List each triple on a new line in format: subject | predicate | object, using named placeholders]'),
]

S1_RESOLVE_NOTE = """
        Placeholder rules:
        - Keep the placeholder names (?A, ?B, ...). Never rename them.
        - After each Fully Resolved Clue and each Newly Traceable Clue, add one line listing the placeholders you filled, e.g.
          Binding: ?A = Ermengarde of Tours
"""

S4_RESOLVE_NOTE = """
        Evidence rules:
        - After each Fully Resolved Clue, add one line that copies, word for word, the sentence from the Context Passages that supports it, e.g.
          Evidence: "Ermengarde of Tours died on 20 March 851."
        - If no passage supports the value, write Evidence: NOT FOUND and do not guess the entity.
"""


class TRAGPlus(TRAG):
    def __init__(self, global_config=None, max_qa_steps: int = 3, fixes=(), s1_max_steps: int = 5,
                 s2_min_per_clue: int = 2, s2_max_k: int = 10, s4_match_threshold: float = 0.8, **kwargs):
        self.fixes = frozenset(f.strip().lower() for f in fixes if f and f.strip())
        unknown = self.fixes - {"b", "s1", "s2", "s3", "s4", "h2a"}
        if unknown:
            raise ValueError(f"Unknown fixes: {sorted(unknown)}")
        if "h2a" in self.fixes:
            raise NotImplementedError("h2a (graph fallback) is not implemented yet")
        self.s1_max_steps = s1_max_steps
        self.s2_min_per_clue = s2_min_per_clue
        self.s2_max_k = s2_max_k
        self.s4_match_threshold = s4_match_threshold
        self._bm25: Optional[BM25] = None
        self.prop_passage_ids: Optional[List[str]] = None
        self.prop_passage_texts: Optional[List[str]] = None
        self._reset_query_state()
        super().__init__(global_config=global_config, max_qa_steps=max_qa_steps, **kwargs)
        if self.fixes & {"b", "s2"}:
            self._build_proposition_sources()

    # ------------------------------------------------------------------ helpers
    def on(self, flag: str) -> bool:
        return flag in self.fixes

    def _count(self, clue: Dict[str, str]) -> int:
        return count_placeholders(clue) if (self.on("b") or self.on("s1")) else count_question_marks(clue)

    def _reset_query_state(self):
        self._bindings: Dict[str, str] = {}
        self._evidence: Dict[Tuple[str, str, str], str] = {}
        self._stats = {"n_dropped_clues": 0, "s4_rejected": 0, "s4_accepted": 0, "bindings_applied": 0}
        self._last_prompt_passages: List[str] = []

    @contextlib.contextmanager
    def _rewrite_prompt(self, rewrite):
        """Temporarily rewrite the prompt the parent class sends (keeps any wrapper on infer)."""
        original = self.llm_model.infer

        def patched(messages, *args, **kwargs):
            messages = [dict(m) for m in messages]
            for m in messages:
                if m.get("role") == "user":
                    m["content"] = rewrite(m["content"])
            return original(messages, *args, **kwargs)

        self.llm_model.infer = patched
        try:
            yield
        finally:
            self.llm_model.infer = original

    # ------------------------------------------------------------------ L5: proposition -> every passage
    def _build_proposition_sources(self):
        """One source passage per stored proposition occurrence (same order as the embedding matrix)."""
        docs = (self.openie_results or {}).get("docs", [])
        ids, texts = [], []
        for doc in docs:
            doc_idx, passage = doc.get("idx", ""), doc.get("passage", "")
            if not doc_idx:
                continue
            for triple in doc.get("extracted_triples", []):
                if len(triple) == 3:
                    ids.append(str(doc_idx))
                    texts.append(passage)
        if len(ids) == len(self.all_propositions):
            self.prop_passage_ids, self.prop_passage_texts = ids, texts
        else:
            logger.warning(f"Proposition sources not aligned ({len(ids)} vs {len(self.all_propositions)}); "
                           "falling back to the original mapping")
            self.prop_passage_ids = [str(self.proposition_to_passage.get(p, {}).get("passage_id", "unknown"))
                                     for p in self.all_propositions]
            self.prop_passage_texts = [self.proposition_to_passage.get(p, {}).get("passage", "")
                                       for p in self.all_propositions]
        self.passage_text_by_id: Dict[str, str] = {}
        for pid, text in zip(self.prop_passage_ids, self.prop_passage_texts):
            self.passage_text_by_id.setdefault(pid, text)
        for doc in docs:  # passages without triples can still be found by BM25
            if doc.get("idx"):
                self.passage_text_by_id.setdefault(str(doc["idx"]), doc.get("passage", ""))

    def _source(self, idx: int) -> Tuple[str, str]:
        if self.on("b") and self.prop_passage_ids is not None:
            return self.prop_passage_ids[idx], self.prop_passage_texts[idx]
        info = self.proposition_to_passage.get(self.all_propositions[idx], {})
        return str(info.get("passage_id", "unknown")), info.get("passage", "")

    # ------------------------------------------------------------------ step 1 (S1, L4)
    def _categorize(self, triples: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        out = {"fuzzy_clues": [], "traceable_clues": [], "resolved_clues": []}
        for t in triples:
            n = self._count(t)
            key = "fuzzy_clues" if n >= 2 else "traceable_clues" if n == 1 else "resolved_clues"
            out[key].append(t)
        return out

    def _name_anonymous(self, triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Give every anonymous '?' its own name so the binding table can track it."""
        used = {p for t in triples for p in clue_placeholders(t) if p != "?"}
        counter = [0]

        def fresh(_m):
            while True:
                counter[0] += 1
                name = f"?V{counter[0]}"
                if name not in used:
                    used.add(name)
                    return name

        named = []
        for t in triples:
            named.append({f: re.sub(r"(?<![\w?])\?(?![\w?])", fresh, t.get(f, "")) for f in FIELDS})
        return named

    def reason_and_form_triples(self, query: str) -> Dict[str, Any]:
        if not self.on("s1"):
            result = super().reason_and_form_triples(query)
            if self.on("b"):
                result.update(self._categorize(result["all_triples"]))
            return result

        def rewrite(text):
            for old, new in S1_DECOMPOSE_REPLACEMENTS:
                if old in text:
                    text = text.replace(old, new)
            return text

        with self._rewrite_prompt(rewrite):
            result = super().reason_and_form_triples(query)
        triples = self._name_anonymous([clean(t) for t in result["all_triples"]])
        result["all_triples"] = triples
        result.update(self._categorize(triples))
        return result

    # ------------------------------------------------------------------ step 2.1 retrieval (L6, L5, S2)
    def _query_texts(self, clues: List[Dict[str, str]], query: str) -> Tuple[List[str], str]:
        clue_queries = []
        for clue in clues:
            parts = [strip_placeholders(clue.get(f, "")) for f in FIELDS]
            text = " ".join(p for p in parts if p)
            if text:
                clue_queries.append(text)
        if self.on("b"):
            q = QUESTION_WORDS.sub(" ", query).replace("?", " ")
            q = " ".join(q.split())
        else:  # the original behaviour (TRAG.py 478-484)
            q = query
            for w in ["what", "who", "where", "when", "why", "how", "which", "whose"]:
                q = q.lower().replace(w, "").strip()
            q = q.replace("?", "").strip()
        return clue_queries, q

    def retrieve_and_double_check(self, traceable_resolved_clues: List[Dict[str, str]], query: str) -> Dict[str, Any]:
        if not (self.on("b") or self.on("s2")):
            return super().retrieve_and_double_check(traceable_resolved_clues, query)
        empty = {"retrieved_passages": [], "retrieved_propositions": [], "passage_count": 0, "proposition_count": 0}
        if self.proposition_embeddings is None or len(self.all_propositions) == 0:
            return empty
        clue_queries, q = self._query_texts(traceable_resolved_clues, query)
        queries = clue_queries + ([q] if q else [])
        if not queries:
            return empty
        from .prompts.linking import get_query_instruction
        q_emb = self.embedding_model.batch_encode(queries, instruction=get_query_instruction("query_to_passage"),
                                                  norm=True)
        sims = np.dot(self.proposition_embeddings, np.asarray(q_emb).T)  # (n_props, n_queries)
        k = getattr(self.global_config, "qa_top_k", 5)
        if self.on("s2"):
            return self._hybrid_select(sims, queries, n_clue_queries=len(clue_queries), k=k)
        return self._original_select(sims, k)

    def _prop_entry(self, idx: int, score: float) -> Dict[str, Any]:
        pid, text = self._source(idx)
        return {"proposition": self.all_propositions[idx], "similarity": float(score),
                "passage_id": pid, "passage_text": text}

    def _original_select(self, sims: np.ndarray, k: int) -> Dict[str, Any]:
        """Same selection as TRAG.py 512-560, but each proposition keeps its own passage (L5)."""
        best = sims.max(axis=1)
        passages, props, seen = [], [], set()
        for idx in np.argsort(best)[::-1]:
            if len(passages) >= k:
                break
            if best[idx] <= 0:
                continue
            entry = self._prop_entry(int(idx), best[idx])
            props.append(entry)
            if entry["passage_text"] and entry["passage_id"] not in seen and entry["passage_id"] != "unknown":
                passages.append({"text": entry["passage_text"], "passage_id": entry["passage_id"],
                                 "score": float(best[idx])})
                seen.add(entry["passage_id"])
        return {"retrieved_passages": passages, "retrieved_propositions": props,
                "passage_count": len(passages), "proposition_count": len(props)}

    def _passage_ranking_from_props(self, col: np.ndarray, depth: int = 50) -> List[str]:
        ranking, seen = [], set()
        for idx in np.argsort(col)[::-1]:
            if col[idx] <= 0 or len(ranking) >= depth:
                break
            pid, _ = self._source(int(idx))
            if pid != "unknown" and pid not in seen:
                ranking.append(pid)
                seen.add(pid)
        return ranking

    def _bm25_ranking(self, text: str, depth: int = 50) -> List[str]:
        if self._bm25 is None:
            if self.prop_passage_ids is None:
                self._build_proposition_sources()
            self._bm25_ids = list(self.passage_text_by_id.keys())
            self._bm25 = BM25([self.passage_text_by_id[i] for i in self._bm25_ids])
        scores = self._bm25.scores(text)
        top = np.argsort(scores)[::-1][:depth]
        return [self._bm25_ids[i] for i in top if scores[i] > 0]

    def _hybrid_select(self, sims: np.ndarray, queries: List[str], n_clue_queries: int, k: int) -> Dict[str, Any]:
        per_query = []
        for j, text in enumerate(queries):
            vec = self._passage_ranking_from_props(sims[:, j])
            lex = self._bm25_ranking(text)
            fused = rrf([vec, lex])
            per_query.append(sorted(fused, key=fused.get, reverse=True))
        k_eff = k
        if n_clue_queries * self.s2_min_per_clue > k:
            k_eff = min(n_clue_queries * self.s2_min_per_clue, max(self.s2_max_k, k))
        chosen: List[str] = []
        # reserved share: each searchable clue first gets its own best passages
        for j in range(n_clue_queries):
            taken = 0
            for pid in per_query[j]:
                if taken >= self.s2_min_per_clue or len(chosen) >= k_eff:
                    break
                if pid not in chosen:
                    chosen.append(pid)
                taken += 1
        # remaining slots: one global fused ranking over every query (clues + original question)
        global_scores = rrf(per_query)
        for pid in sorted(global_scores, key=global_scores.get, reverse=True):
            if len(chosen) >= k_eff:
                break
            if pid not in chosen:
                chosen.append(pid)
        chosen_set = set(chosen)
        passages = [{"text": self.passage_text_by_id.get(pid, ""), "passage_id": pid,
                     "score": float(global_scores.get(pid, 0.0))} for pid in chosen]
        # propositions shown to the LLM: the 5 best per query, plus any from a chosen passage in the top 30
        props, seen_p = [], set()
        for j in range(sims.shape[1]):
            col = sims[:, j]
            order = np.argsort(col)[::-1][:30]
            for rank, idx in enumerate(order):
                idx = int(idx)
                if col[idx] <= 0 or idx in seen_p:
                    continue
                pid, _ = self._source(idx)
                if rank < 5 or pid in chosen_set:
                    props.append(self._prop_entry(idx, col[idx]))
                    seen_p.add(idx)
        return {"retrieved_passages": passages, "retrieved_propositions": props,
                "passage_count": len(passages), "proposition_count": len(props), "k_effective": k_eff}

    # ------------------------------------------------------------------ step 2.2 resolution (S1, S4)
    def _parse_plus(self, response: str):
        """Like TRAG._parse_combined_clue_response, but also reads Binding: and Evidence: lines."""
        resolved, traceable = [], []
        section, clue, meta = None, {}, {}

        def flush():
            if section and len(clue) == 3:
                (resolved if section == "resolved" else traceable).append((dict(clue), dict(meta)))

        for raw in (response or "").splitlines():
            line = raw.strip()
            if line.startswith("Fully Resolved Clue") or line.startswith("Newly Traceable Clue"):
                flush()
                section = "resolved" if line.startswith("Fully") else "traceable"
                clue, meta = {}, {}
            elif line.startswith("Subject:"):
                clue["subject"] = line.split("Subject:", 1)[1].strip()
            elif line.startswith("Predicate:"):
                clue["predicate"] = line.split("Predicate:", 1)[1].strip()
            elif line.startswith("Object:"):
                clue["object"] = line.split("Object:", 1)[1].strip()
            elif line.startswith("Binding:"):
                for part in re.split(r"[;,]\s*(?=\?)", line.split("Binding:", 1)[1]):
                    if "=" in part:
                        var, val = part.split("=", 1)
                        var, val = var.strip(), val.strip().strip('"').strip()
                        if PLACEHOLDER_TOKEN.fullmatch(var) and val and not placeholders_in(val):
                            meta.setdefault("bindings", {})[var] = val
            elif line.startswith("Evidence:"):
                meta["evidence"] = line.split("Evidence:", 1)[1].strip().strip('"').strip()
        flush()
        return resolved, traceable

    @staticmethod
    def _field_match(inp: str, out: str) -> float:
        a, b = inp.strip().lower(), out.strip().lower()
        if placeholders_in(inp) and strip_placeholders(inp) == "":
            return 0.5  # a pure placeholder can become anything
        if a == b:
            return 1.0
        sa = strip_placeholders(a)
        if sa and (sa in b or b in sa):
            return 0.75
        return 0.0

    def _match_input(self, out: Dict[str, str], inputs: List[Dict[str, str]],
                     bindings: Optional[Dict[str, str]] = None) -> Optional[int]:
        """Index of the input clue this LLM output most likely rewrites (None if no good match)."""
        best, best_i = 0.0, None
        for i, inp in enumerate(inputs):
            score = sum(self._field_match(inp.get(f, ""), out.get(f, "")) for f in FIELDS)
            if bindings and set(bindings) & set(clue_placeholders(inp)):
                score += 1.0
            if score > best:
                best, best_i = score, i
        return best_i if best >= 1.5 else None

    @staticmethod
    def _derive_bindings(inp: Dict[str, str], out: Dict[str, str]) -> Dict[str, str]:
        b = {}
        for f in FIELDS:
            ph = placeholders_in(inp.get(f, ""))
            val = out.get(f, "").strip()
            if len(ph) == 1 and strip_placeholders(inp.get(f, "")) == "" and val and not placeholders_in(val):
                b[ph[0]] = val
        return b

    def _substitute(self, clue: Dict[str, str]) -> Dict[str, str]:
        if not self._bindings:
            return clue
        new = {}
        for f in FIELDS:
            text = clue.get(f, "")
            new[f] = PLACEHOLDER_TOKEN.sub(lambda m: self._bindings.get(m.group(0), m.group(0)), text)
        if new != clue:
            self._stats["bindings_applied"] += 1
        return new

    def _evidence_ok(self, quote: str, passages: List[str], value: str) -> bool:
        q = tokens(quote)
        if not q or norm_text(quote) in ("not found", "none"):
            return False
        supported = False
        for p in passages:
            pt = tokens(p)
            if norm_text(quote) in " ".join(pt):
                supported = True
                break
            ps = set(pt)
            if sum(t in ps for t in q) / len(q) >= self.s4_match_threshold:
                supported = True
                break
        if not supported:
            return False
        vt = tokens(value)
        if not vt:
            return True
        qs = set(q)
        return sum(t in qs for t in vt) / len(vt) >= self.s4_match_threshold

    def _resolution_prompt(self, traceable_clues, fuzzy_clues, retrieval_results, current_resolved_clues, query):
        """The original prompt (TRAG.py 631-725) plus the S1 / S4 notes."""
        context_propositions = "\n".join(f"- {p.get('proposition', '')}"
                                         for p in retrieval_results["retrieved_propositions"])
        top_k = len(retrieval_results.get("retrieved_passages", [])) if self.on("s2") else \
            getattr(self.global_config, "qa_top_k", 5)
        top_passages = retrieval_results.get("retrieved_passages", [])[:top_k]
        self._last_prompt_passages = [p.get("text", "") for p in top_passages]
        context_passages = "\n".join(f"- {p.get('text', '')}" for p in top_passages)
        fuzzy_clues_text = ""
        for i, c in enumerate(fuzzy_clues):
            fuzzy_clues_text += f"""
            Fuzzy Clue {i+1}:
            Subject: {c['subject']}
            Predicate: {c['predicate']}
            Object: {c['object']}
            """
        resolved_clues_context = ""
        if current_resolved_clues:
            resolved_clues_context = "\n\nPreviously Resolved Clues (use these to replace ? placeholders):\n" + \
                "\n".join(f"- {c.get('subject', '')} | {c.get('predicate', '')} | {c.get('object', '')}"
                          for c in current_resolved_clues)
        if self.on("s1") and self._bindings:
            resolved_clues_context += "\n\nKnown placeholder values:\n" + \
                "\n".join(f"- {k} = {v}" for k, v in self._bindings.items())
        traceable_clues_text = ""
        for i, c in enumerate(traceable_clues):
            traceable_clues_text += f"""
Traceable Clue {i+1}:
Subject: {c['subject']}
Predicate: {c['predicate']}
Object: {c['object']}
"""
        example_extra = ""
        if self.on("s1"):
            example_extra += "\n        Binding: ?A = Ermengarde of Tours"
        if self.on("s4"):
            example_extra += '\n        Evidence: "Lothair II was the son of Emperor Lothair I and Ermengarde of Tours."'
        notes = (S1_RESOLVE_NOTE if self.on("s1") else "") + (S4_RESOLVE_NOTE if self.on("s4") else "")
        return f"""Example:
        Context Propositions:
        {context_propositions[:1000]}

        Fully Resolved Clue 1:
        Subject: Lothair II
        Predicate: has mother
        Object: Ermengarde of Tours{example_extra}

        Newly Traceable Clue 1:
        Subject: Ermengarde of Tours
        Predicate: died on
        Object: {"?B" if self.on("s1") else "20 March 851"}

        ---

        Now apply the same process to the following clues:
        Use the context passages and propositions to resolve any '?' placeholders with as much detail as possible, grounding your answers in the passage content.
        Instructions:
        1. For traceable clues (one '?'), replace '?' with the correct entity to fully resolve it, including any relevant attributes.
        2. For fuzzy clues (multiple '?'), generate a Newly Traceable Clue by replacing one of the placeholders with the correct entity, including any relevant context.
{notes}
        Original Query: {query}

        Traceable Clues:
        {traceable_clues_text}

        Fuzzy Clues:
        {fuzzy_clues_text}

        Context Passages:
        {context_passages}

        Context Propositions:
        {context_propositions}

        {resolved_clues_context}

        Return two lists in this format:
        Fully Resolved Clue 1:
        Subject: ...
        Predicate: ...
        Object: ...

        Newly Traceable Clue 1:
        Subject: ...
        Predicate: ...
        Object: ...

        (Continue numbering accordingly)"""

    def resolve_clues(self, traceable_clues, fuzzy_clues, retrieval_results, current_resolved_clues, query):
        if not (self.on("s1") or self.on("s4")):
            return super().resolve_clues(traceable_clues, fuzzy_clues, retrieval_results, current_resolved_clues, query)
        if not retrieval_results["retrieved_passages"] and not retrieval_results["retrieved_propositions"]:
            return {"resolved_clues": [], "newly_traceable_clues": [],
                    "remaining_traceable_clues": list(traceable_clues), "remaining_fuzzy_clues": list(fuzzy_clues)}
        prompt = self._resolution_prompt(traceable_clues, fuzzy_clues, retrieval_results, current_resolved_clues, query)
        response = self.llm_model.infer([{"role": "user", "content": prompt}])
        text = response[0] if isinstance(response, tuple) else response
        resolved_out, traceable_out = self._parse_plus(text)

        inputs = list(traceable_clues) + list(fuzzy_clues)
        consumed = set()
        rejected = set()
        resolved, newly = [], []
        new_bindings: Dict[str, str] = {}
        for out, meta in resolved_out:
            i = self._match_input(out, inputs, meta.get("bindings"))
            b = dict(meta.get("bindings", {}))
            if i is not None:
                for var, val in self._derive_bindings(inputs[i], out).items():
                    b.setdefault(var, val)
            if self.on("s4"):
                value = " ".join(b.values()) if b else (out.get("object", "") if i is None else
                                                        " ".join(self._derive_bindings(inputs[i], out).values()))
                if not self._evidence_ok(meta.get("evidence", ""), self._last_prompt_passages, value):
                    self._stats["s4_rejected"] += 1
                    if i is not None:
                        rejected.add(i)
                    continue
                self._stats["s4_accepted"] += 1
                self._evidence[clue_key(out)] = meta.get("evidence", "")
            resolved.append(clean(out))
            if i is not None:
                consumed.add(i)
            new_bindings.update(b)
        for out, meta in traceable_out:
            i = self._match_input(out, inputs, meta.get("bindings"))
            b = dict(meta.get("bindings", {}))
            if i is not None:
                for var, val in self._derive_bindings(inputs[i], out).items():
                    b.setdefault(var, val)
                consumed.add(i)
            newly.append(clean(out))
            new_bindings.update(b)

        if self.on("s1"):
            for var, val in new_bindings.items():
                self._bindings.setdefault(var, val)
        n_t = len(traceable_clues)
        unmatched = [i for i in range(len(inputs)) if i not in consumed]
        if self.on("s1"):
            remaining_traceable = [inputs[i] for i in unmatched if i < n_t]
            remaining_fuzzy = [inputs[i] for i in unmatched if i >= n_t]
        else:  # s4 alone: only the rejected clues are searched again (original keeps nothing)
            remaining_traceable = [inputs[i] for i in unmatched if i < n_t and i in rejected]
            remaining_fuzzy = []
        if self.on("s1"):
            resolved = [self._substitute(c) for c in resolved]
            newly = [self._substitute(c) for c in newly]
            remaining_traceable = [self._substitute(c) for c in remaining_traceable]
            remaining_fuzzy = [self._substitute(c) for c in remaining_fuzzy]
        return {"resolved_clues": self._deduplicate_clues(resolved),
                "newly_traceable_clues": self._deduplicate_clues(newly),
                "remaining_traceable_clues": self._deduplicate_clues(remaining_traceable),
                "remaining_fuzzy_clues": self._deduplicate_clues(remaining_fuzzy)}

    # ------------------------------------------------------------------ loop condition (L4)
    def has_unresolved_clues(self, fuzzy_clues, traceable_clues) -> bool:
        if not (self.on("b") or self.on("s1")):
            return super().has_unresolved_clues(fuzzy_clues, traceable_clues)
        return any(count_placeholders(c) >= 2 for c in fuzzy_clues) or \
            any(count_placeholders(c) == 1 for c in traceable_clues)

    # ------------------------------------------------------------------ step 3 (S3)
    def _evidence_passages(self, clues, all_retrieval_results, limit: int = 3, max_chars: int = 600) -> List[str]:
        values = set()
        for c in clues:
            for f in ("subject", "object"):
                v = strip_placeholders(c.get(f, ""))
                if v:
                    values.add(v.lower())
        scored, seen = [], set()
        order = 0
        for rr in reversed(all_retrieval_results or []):  # latest round first
            for p in rr.get("retrieved_passages", []):
                text = p.get("text", "")
                if not text or text in seen:
                    continue
                seen.add(text)
                low = text.lower()
                hits = sum(1 for v in values if v in low)
                scored.append((-hits, order, text))
                order += 1
        scored.sort()
        return [t[:max_chars] for _, _, t in scored[:limit]]

    def final_qa(self, query, all_traceable_resolved_clues, all_retrieval_results) -> str:
        if not self.on("s3"):
            return super().final_qa(query, all_traceable_resolved_clues, all_retrieval_results)
        quotes = [self._evidence[clue_key(c)] for c in all_traceable_resolved_clues if clue_key(c) in self._evidence]
        passages = self._evidence_passages(all_traceable_resolved_clues, all_retrieval_results)
        evidence = "\n".join([f'- "{q}"' for q in quotes] + [f"- {p}" for p in passages])
        self._last_final_evidence = evidence

        def rewrite(text):
            if not evidence:
                return text
            return text.replace("\n\nInstructions:\n1. Analyze the question step by step",
                                "\n\nSupporting Evidence (passages the clues came from):\n" + evidence +
                                "\n\nInstructions:\n1. Analyze the question step by step, checking the clues "
                                "against the supporting evidence", 1)

        with self._rewrite_prompt(rewrite):
            return super().final_qa(query, all_traceable_resolved_clues, all_retrieval_results)

    # ------------------------------------------------------------------ main loop (L2, S1)
    def _count_dropped(self, inputs, clues_result) -> int:
        """How many input clues disappear this round: neither rewritten by the LLM nor kept
        (kept clues may have had placeholders substituted, so they are matched, not compared)."""
        outputs = (clues_result["resolved_clues"] + clues_result["newly_traceable_clues"] +
                   clues_result["remaining_traceable_clues"] + clues_result["remaining_fuzzy_clues"])
        free = list(range(len(inputs)))
        for out in outputs:
            i = self._match_input(out, [inputs[j] for j in free])
            if i is not None:
                free.pop(i)
        return len(free)

    def trag_qa(self, query: str, query_id: str = None, **kwargs) -> Dict[str, Any]:
        if not self.fixes:
            return super().trag_qa(query, query_id=query_id, **kwargs)
        if query_id is None:
            query_id = f"query_{hash(query) % 10000}"
        self._reset_query_state()

        r = self.reason_and_form_triples(query)
        self.save_iteration_results(query_id, query, {
            "step": 1, "step_name": "reasoning_and_triple_formation", "input_query": query,
            "reasoning": r["reasoning"], "all_triples": r["all_triples"], "fuzzy_clues": r["fuzzy_clues"],
            "traceable_clues": r["traceable_clues"], "resolved_clues": r["resolved_clues"],
            "propositions": r["propositions"], "fixes": sorted(self.fixes)})

        traceable = list(r["traceable_clues"])
        resolved = list(r["resolved_clues"])
        fuzzy = list(r["fuzzy_clues"])
        all_results = []
        iteration = 0
        max_steps = self.s1_max_steps if self.on("s1") else self.max_qa_steps
        termination = "all_resolved"
        while iteration < max_steps and self.has_unresolved_clues(fuzzy, traceable):
            iteration += 1
            in_t, in_f, in_r = list(traceable), list(fuzzy), list(resolved)
            retrieval = self.retrieve_and_double_check(in_t, query)
            all_results.append(retrieval)
            res = self.resolve_clues(traceable, fuzzy, retrieval, resolved, query)
            dropped = self._count_dropped(in_t + in_f, res)
            self._stats["n_dropped_clues"] += dropped

            resolved.extend(res["resolved_clues"])
            candidates = res["newly_traceable_clues"] + res["remaining_traceable_clues"]
            if self.on("b") or self.on("s1"):
                candidates += res["remaining_fuzzy_clues"]
                if self.on("s1"):
                    candidates = [self._substitute(c) for c in candidates]
                    resolved = [self._substitute(c) for c in resolved]
                cat = self._categorize(self._deduplicate_clues(candidates))
                traceable, fuzzy = cat["traceable_clues"], cat["fuzzy_clues"]
                resolved.extend(cat["resolved_clues"])  # L2: 2+ placeholders go back to fuzzy
            else:  # original ENFORCE step (TRAG.py 1236-1247)
                traceable = [c for c in candidates if count_question_marks(c) == 1]
                resolved.extend(c for c in candidates if count_question_marks(c) == 0)
                fuzzy = res["remaining_fuzzy_clues"]
            progress = len(res["resolved_clues"]) + len(res["newly_traceable_clues"])

            self.save_iteration_results(query_id, query, {
                "iteration": iteration,
                "step_2_retrieval_and_expansion": {
                    "input_traceable_clues": in_t, "input_resolved_clues": in_r,
                    "retrieved_passages": retrieval["retrieved_passages"],
                    "retrieved_propositions": retrieval["retrieved_propositions"][:50],
                    "retrieved_count": len(retrieval["retrieved_passages"]),
                    "k_effective": retrieval.get("k_effective")},
                "step_3_clue_resolution": {
                    "input_traceable_clues": in_t, "input_fuzzy_clues": in_f,
                    "resolved_clues": res["resolved_clues"], "newly_traceable_clues": res["newly_traceable_clues"],
                    "remaining_traceable_clues": res["remaining_traceable_clues"],
                    "remaining_fuzzy_clues": res["remaining_fuzzy_clues"],
                    "dropped_this_round": dropped, "bindings": dict(self._bindings)},
                "iteration_summary": {"traceable_left": len(traceable), "fuzzy_left": len(fuzzy),
                                      "progress": progress}})
            if not self.has_unresolved_clues(fuzzy, traceable):
                termination = "all_resolved"
                break
            if self.on("s1") and progress == 0:
                termination = "no_progress"
                break
        else:
            if self.has_unresolved_clues(fuzzy, traceable):
                termination = "max_qa_steps_reached"

        if self.on("s1"):
            resolved = [self._substitute(c) for c in resolved]
            traceable = [self._substitute(c) for c in traceable]
        resolved = self._deduplicate_clues(resolved)
        all_final = resolved + traceable
        answer = self.final_qa(query, all_final, all_results)
        unresolved = self.has_unresolved_clues(fuzzy, traceable)
        status = "all clues fully resolved" if not unresolved else \
            f"{termination} with unresolved clues remaining (fuzzy: {len(fuzzy)}, traceable: {len(traceable)})"
        self.save_iteration_results(query_id, query, {
            "step": "final", "step_name": "final_qa", "total_iterations": iteration,
            "all_resolved_clues": resolved, "all_traceable_clues": traceable, "all_final_clues": all_final,
            "remaining_fuzzy_clues": fuzzy, "final_answer": answer, "termination": termination,
            "completion_status": "all_resolved" if not unresolved else termination,
            "bindings": dict(self._bindings), "final_evidence": getattr(self, "_last_final_evidence", None),
            "stats": dict(self._stats)})
        return {"answer": answer, "all_resolved_clues": resolved, "all_traceable_clues": traceable,
                "all_final_clues": all_final, "all_retrieval_results": all_results, "total_iterations": iteration,
                "remaining_fuzzy_clues": fuzzy, "completion_status": status,
                "n_dropped_clues": self._stats["n_dropped_clues"], "s4_rejected": self._stats["s4_rejected"],
                "bindings": dict(self._bindings)}
