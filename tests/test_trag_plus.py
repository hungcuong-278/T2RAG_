"""Unit tests for src/TRAG_plus.py with a fake LLM and fake embeddings (no network needed).

Run from the repo root:  python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import fake_env as fe  # noqa: E402


def build(cls, rules, fixes=None, qa_top_k=5, **kwargs):
    llm = fe.FakeLLM(rules)
    fe.install_fakes(llm)
    config = fe.make_config(fe.tmpdir(), qa_top_k=qa_top_k)
    if fixes is None:
        rag = cls(global_config=config, max_qa_steps=3)
    else:
        rag = cls(global_config=config, max_qa_steps=3, fixes=fixes, **kwargs)
    return rag, llm


class TestHelpers(unittest.TestCase):
    def test_l4_count_placeholders(self):
        from src.TRAG_plus import count_placeholders, count_question_marks
        clue = {"subject": "Who's Afraid of Virginia Woolf?", "predicate": "is directed by", "object": "?A"}
        self.assertEqual(count_question_marks(clue), 2)   # original rule counts the title's '?'
        self.assertEqual(count_placeholders(clue), 1)     # fixed rule counts only the placeholder
        self.assertEqual(count_placeholders({"subject": "?", "predicate": "born in", "object": "?year"}), 2)

    def test_bm25_vietnamese(self):
        from src.TRAG_plus import BM25
        docs = ["Trần Hưng Đạo chỉ huy trận Bạch Đằng năm 1288.", "Nhà Lý trị vì Đại Việt.",
                "Trần Liễu sinh năm 1211."]
        scores = BM25(docs).scores("Trần Liễu sinh năm nào")
        self.assertEqual(int(scores.argmax()), 2)

    def test_rrf(self):
        from src.TRAG_plus import rrf
        fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
        self.assertAlmostEqual(fused["a"], fused["b"])
        self.assertGreater(fused["a"], fused["c"])


class TestEquivalence(unittest.TestCase):
    def test_no_fixes_is_the_original(self):
        """With no flags, TRAGPlus must send exactly the same prompts and give the same answer."""
        from src.TRAG import TRAG
        from src.TRAG_plus import TRAGPlus
        rag_a, llm_a = build(TRAG, fe.chain_rules(named=False))
        out_a = rag_a.trag_qa(fe.QUESTION, query_id="q0")
        rag_b, llm_b = build(TRAGPlus, fe.chain_rules(named=False), fixes=[])
        out_b = rag_b.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out_a["answer"], out_b["answer"])
        self.assertEqual(llm_a.prompts, llm_b.prompts)


class TestBugFixes(unittest.TestCase):
    def test_original_loses_the_chain(self):
        """Reproduces bug L1: after round 1 the fuzzy clues are dropped, so the answer is lost."""
        from src.TRAG import TRAG
        rag, llm = build(TRAG, fe.chain_rules(named=False))
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out["answer"], "unknown")
        self.assertEqual(out["total_iterations"], 1)

    def test_b_alone_counts_the_dropped_clues(self):
        from src.TRAG_plus import TRAGPlus
        rag, _ = build(TRAGPlus, fe.chain_rules(named=False, with_evidence=False), fixes=["b"])
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out["answer"], "unknown")   # b alone does not fix L1 ...
        self.assertEqual(out["n_dropped_clues"], 2)   # ... but it measures it

    def test_l6_question_words(self):
        from src.TRAG_plus import TRAGPlus
        rag, _ = build(TRAGPlus, fe.chain_rules(), fixes=["b"])
        _, q = rag._query_texts([], "Whose son showed the way to Somewhere?")
        self.assertIn("showed", q)
        self.assertIn("Somewhere", q)
        rag.fixes = frozenset()
        _, q_orig = rag._query_texts([], "Whose son showed the way to Somewhere?")
        self.assertNotIn("showed", q_orig)  # the original turns it into "sed"

    def test_l5_duplicate_propositions_keep_every_passage(self):
        from src.TRAG_plus import TRAGPlus
        rag, _ = build(TRAGPlus, fe.chain_rules(), fixes=["b"])
        prop = "Tran Hung Dao is prince of Tran dynasty"
        sources = {rag.prop_passage_ids[i] for i, p in enumerate(rag.all_propositions) if p == prop}
        self.assertEqual(len(sources), 2)
        res = rag.retrieve_and_double_check(
            [{"subject": "Tran Hung Dao", "predicate": "is prince of", "object": "?"}], "prince of Tran dynasty")
        texts = [p["text"] for p in res["retrieved_passages"]]
        self.assertIn(fe.passage(1), texts)
        self.assertIn(fe.passage(5), texts)


class TestS1(unittest.TestCase):
    def test_s1_follows_the_whole_chain(self):
        from src.TRAG_plus import TRAGPlus
        rag, llm = build(TRAGPlus, fe.chain_rules(named=True, with_evidence=False), fixes=["b", "s1"])
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out["answer"], fe.GOLD)
        self.assertEqual(out["bindings"].get("?A"), "Tran Hung Dao")
        self.assertEqual(out["bindings"].get("?B"), "Tran Lieu")
        self.assertEqual(out["total_iterations"], 3)
        self.assertEqual(out["n_dropped_clues"], 0)

    def test_s1_rewrites_the_decomposition_prompt(self):
        from src.TRAG_plus import TRAGPlus
        rag, llm = build(TRAGPlus, fe.chain_rules(named=True, with_evidence=False), fixes=["s1"])
        rag.trag_qa(fe.QUESTION, query_id="q0")
        first = llm.prompts[0]
        self.assertIn("NAMED placeholders", first)
        self.assertIn("?A | won Best Picture | 2020", first)
        self.assertIn("?directorA | was born in | ?birthA", first)

    def test_s1_names_anonymous_placeholders(self):
        from src.TRAG_plus import TRAGPlus
        rag, _ = build(TRAGPlus, fe.chain_rules(named=False, with_evidence=False), fixes=["s1"])
        named = rag._name_anonymous([{"subject": "?", "predicate": "has father", "object": "?"}])
        self.assertEqual(named[0]["subject"], "?V1")
        self.assertEqual(named[0]["object"], "?V2")

    def test_s1_stops_when_no_progress(self):
        from src.TRAG_plus import TRAGPlus
        rules = fe.chain_rules(named=True)[:1] + [(["Based on the reasoning clues"], "unknown")]
        rag, llm = build(TRAGPlus, rules, fixes=["b", "s1"])
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out["total_iterations"], 1)  # nothing resolved -> stop instead of 5 rounds


class TestS2(unittest.TestCase):
    def test_each_clue_gets_its_own_passages(self):
        from src.TRAG_plus import TRAGPlus
        rag, _ = build(TRAGPlus, fe.chain_rules(), fixes=["b", "s2"], qa_top_k=2, s2_min_per_clue=1)
        clues = [{"subject": "Mongol fleet", "predicate": "led by", "object": "?"},
                 {"subject": "Tran Lieu", "predicate": "was born in", "object": "?"}]
        res = rag.retrieve_and_double_check(clues, "Who led the Mongol fleet and when was Tran Lieu born?")
        texts = [p["text"] for p in res["retrieved_passages"]]
        self.assertIn(fe.passage(3), texts)
        self.assertIn(fe.passage(2), texts)

    def test_k_grows_when_many_clues(self):
        from src.TRAG_plus import TRAGPlus
        rag, _ = build(TRAGPlus, fe.chain_rules(), fixes=["s2"], qa_top_k=2, s2_min_per_clue=2)
        clues = [{"subject": "Mongol fleet", "predicate": "led by", "object": "?"},
                 {"subject": "Tran Lieu", "predicate": "was born in", "object": "?"}]
        res = rag.retrieve_and_double_check(clues, "question")
        self.assertEqual(res["k_effective"], 4)


class TestS3(unittest.TestCase):
    def test_final_prompt_contains_evidence(self):
        from src.TRAG_plus import TRAGPlus
        rag, llm = build(TRAGPlus, fe.chain_rules(named=True, with_evidence=False), fixes=["b", "s1", "s3"])
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        final_prompt = [p for p in llm.prompts if "Based on the reasoning clues" in p][-1]
        self.assertIn("Supporting Evidence", final_prompt)
        self.assertIn("born in 1211", final_prompt)
        self.assertEqual(out["answer"], fe.GOLD)


class TestS4(unittest.TestCase):
    def test_supported_values_are_accepted(self):
        from src.TRAG_plus import TRAGPlus
        rag, llm = build(TRAGPlus, fe.chain_rules(named=True), fixes=["b", "s1", "s4"])
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out["answer"], fe.GOLD)
        self.assertEqual(out["s4_rejected"], 0)
        self.assertIn("Evidence rules", [p for p in llm.prompts if "Traceable Clue 1" in p][0])

    def test_unsupported_value_is_not_locked(self):
        from src.TRAG_plus import TRAGPlus
        rag, llm = build(TRAGPlus, fe.chain_rules(named=True, bad_evidence=True), fixes=["b", "s1", "s4"])
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        self.assertNotIn("1300", out["answer"])
        self.assertGreaterEqual(out["s4_rejected"], 1)
        self.assertNotIn("?C", out["bindings"])
        # the clue stays open (still has its placeholder) instead of being locked to a wrong value
        self.assertTrue(any("?C" in c["object"] for c in out["all_traceable_clues"]))


class TestAllFixes(unittest.TestCase):
    def test_full_configuration(self):
        from src.TRAG_plus import TRAGPlus
        rag, llm = build(TRAGPlus, fe.chain_rules(named=True), fixes=["b", "s1", "s2", "s3", "s4"])
        out = rag.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out["answer"], fe.GOLD)
        self.assertEqual(out["n_dropped_clues"], 0)

    def test_unknown_flag_is_rejected(self):
        from src.TRAG_plus import TRAGPlus
        with self.assertRaises(ValueError):
            build(TRAGPlus, fe.chain_rules(), fixes=["s9"])


if __name__ == "__main__":
    unittest.main()
