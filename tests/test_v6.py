"""v6: lenient scoring, short-answer prompts, NOR, per-run cache, Vietnamese in T2RAG+, rescore."""
import json
import os
import shutil
import sys
import types
import unicodedata
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.getcwd()
sys.path.insert(0, REPO)

import fake_env as fe  # noqa: E402
from test_run_colab import make_dataset  # noqa: E402


class LangEnv:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self.old = os.environ.get("T2RAG_LANG")
        os.environ["T2RAG_LANG"] = self.lang

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("T2RAG_LANG", None)
        else:
            os.environ["T2RAG_LANG"] = self.old


class TestLenientScores(unittest.TestCase):
    def score(self, pred, golds, lang="vi"):
        import run_colab as rc
        with LangEnv(lang):
            return rc.lenient_scores(pred, golds, lang), rc.em_f1(pred, golds)

    def test_prefix_and_brackets(self):
        (em_l, _, multi), (em, _) = self.score("Năm 1383", ["1383"])
        self.assertEqual((em, em_l, multi), (0.0, 1.0, False))
        (em_l, _, _), (em, _) = self.score("Trần Khâm (Trần Nhân Tông)", ["Trần Khâm"])
        self.assertEqual((em, em_l), (0.0, 1.0))
        (em_l, _, _), _ = self.score("huyện Đông Triều", ["Đông Triều"])
        self.assertEqual(em_l, 1.0)
        (em_l, _, _), _ = self.score("huyện Đông Triều", ["Đông Triều"], lang="en")
        self.assertEqual(em_l, 0.0)  # prefixes are Vietnamese only

    def test_several_alternatives_are_wrong(self):
        for pred in ("An Sinh vương hoặc Khâm Minh Đại vương", "R'cam Mal, Jaya Simhavarman III",
                     "Trần Nhật Phụ; Trần Đoan Ngọ"):
            (em_l, f1_l, multi), _ = self.score(pred, ["An Sinh vương", "Jaya Simhavarman III", "Trần Nhật Phụ"])
            self.assertEqual((em_l, f1_l, multi), (0.0, 0.0, True), pred)

    def test_commas_that_are_not_alternatives(self):
        (em_l, _, multi), _ = self.score("1,000", ["1000"], lang="en")
        self.assertFalse(multi)
        (em_l, _, multi), _ = self.score("Thăng Long, Hà Nội", ["Thăng Long, Hà Nội"])
        self.assertEqual((em_l, multi), (1.0, False))

    def test_by_hop(self):
        import run_colab as rc
        recs = [{"hops": 1, "em": 1.0, "f1": 1.0, "em_lenient": 1.0, "f1_lenient": 1.0},
                {"hops": 2, "em": 0.0, "f1": 0.5, "em_lenient": 1.0, "f1_lenient": 1.0},
                {"hops": 2, "em": 1.0, "f1": 1.0, "em_lenient": 1.0, "f1_lenient": 1.0}]
        by = rc.scores_by_hop(recs)
        self.assertEqual(by["2"], {"n": 2, "em": 0.5, "f1": 0.75, "em_lenient": 1.0, "f1_lenient": 1.0})
        self.assertEqual(rc.hops_of("3hop_007"), 3)
        self.assertEqual(rc.hops_of("4hop2__123"), 4)
        self.assertIsNone(rc.hops_of("5a8b57f25542995d1e6f1371"))


TRAG_FINAL = """Based on the reasoning clues, please answer the following question.

Question: Q?

Key Reasoning Clues:
Clue 1: A | b | c

Instructions:
1. Analyze the question step by step
2. Use the reasoning clues to understand what information is needed
3. Provide ONLY a concise answer

Answer format requirements:
- For WH questions (who/what/where/when): Provide the exact entity, date, full name, or full place name only
- For yes/no questions: Answer only "yes" or "no"
- No explanations, reasoning, or additional text
- One entity or fact only

Answer:"""


class TestAnswerFormat(unittest.TestCase):
    def test_trag_final_prompt(self):
        import run_colab as rc
        out = rc.format_messages([{"role": "user", "content": TRAG_FINAL}], "vi")[0]["content"]
        self.assertIn("Không rõ", out)
        self.assertNotIn("One entity or fact only", out)
        self.assertTrue(out.endswith("\n\nAnswer:"))
        self.assertIn("Key Reasoning Clues:\nClue 1: A | b | c", out)

    def test_paper_qa_system_prompt(self):
        import run_colab as rc
        msgs = [{"role": "system", "content": rc.PAPER_QA_SYSTEM_START + " ... Conclude with \"Answer: \"."},
                {"role": "user", "content": "Wikipedia Title: X\n\nQuestion: Q\nThought: "}]
        out = rc.format_messages(msgs, "en")
        self.assertIn("Answer requirements", out[0]["content"])
        self.assertEqual(out[1], msgs[1])
        self.assertNotIn("Answer requirements", msgs[0]["content"])  # input not modified

    def test_other_prompts_unchanged(self):
        import run_colab as rc
        msgs = [{"role": "user", "content": "You are tasked with reasoning about a question ..."}]
        self.assertIs(rc.format_messages(msgs, "vi"), msgs)


class TestCounter(unittest.TestCase):
    def test_per_call_tokens_and_warning(self):
        import run_colab as rc
        llm = types.SimpleNamespace(infer=lambda m: ("word " * 30, {"prompt_tokens": 7, "completion_tokens": 30}, False))
        c = rc.LLMCounter(llm, warn_tokens=20)
        llm.infer([{"role": "user", "content": "x"}])
        llm.infer([{"role": "user", "content": "y"}])
        snap = c.snapshot()
        self.assertEqual(snap["calls"], [[7, 30, False], [7, 30, False]])
        self.assertEqual((c.long_calls, c.max_completion_tokens), (2, 30))
        c.reset()
        self.assertEqual(c.snapshot()["calls"], [])


class TestVietnameseTragPlus(unittest.TestCase):
    def query_text(self, q):
        from src.TRAG_plus import TRAGPlus
        dummy = types.SimpleNamespace(on=lambda f: f == "b")
        with LangEnv("vi"):
            return TRAGPlus._query_texts(dummy, [], q)[1]

    def test_question_words(self):
        self.assertEqual(self.query_text("Ai là người chỉ huy trận Bạch Đằng năm 1288?"),
                         "là người chỉ huy trận Bạch Đằng năm 1288")
        self.assertEqual(self.query_text("Hồ Quý Ly mất ở đâu?"), "Hồ Quý Ly mất ở")
        self.assertEqual(self.query_text("Trần Thái Tông lên ngôi khi nào?"), "Trần Thái Tông lên ngôi")
        self.assertEqual(self.query_text("Nhà Lý dời đô vào năm nào?"), "Nhà Lý dời đô vào năm")

    def test_bm25_syllable_pairs_and_nfc(self):
        from src.TRAG_plus import BM25, tokens
        docs = ["Trần Hưng Đạo đánh quân Nguyên", "Hưng Yên là một tỉnh, có đạo Phật", "Đạo giáo ở Hưng Yên"]
        pairs, plain = BM25(docs, syllable_pairs=True), BM25(docs)
        sp, sn = pairs.scores("Hưng Đạo"), plain.scores("Hưng Đạo")
        self.assertEqual(int(sp.argmax()), 0)
        self.assertGreater(sp[0] - max(sp[1], sp[2]), sn[0] - max(sn[1], sn[2]))
        decomposed = unicodedata.normalize("NFD", "Trần Hưng Đạo")
        self.assertEqual(tokens(decomposed), ["trần", "hưng", "đạo"])
        self.assertGreater(float(pairs.scores(decomposed)[0]), 0)


class TestVietnamesePlaceholdersAndNFC(unittest.TestCase):
    """v6.1: Vietnamese placeholder names (?Người, ?NămA) and NFC in every text comparison."""

    def test_unicode_placeholder_names(self):
        from src.TRAG_plus import placeholders_in, count_placeholders, strip_placeholders
        clue = {"subject": "?Người", "predicate": "lên ngôi năm", "object": "?NămA"}
        with LangEnv("vi"):
            self.assertEqual(placeholders_in("?Người | lên ngôi năm | ?NămA"), ["?Người", "?NămA"])
            self.assertEqual(count_placeholders(clue), 2)
            self.assertEqual(strip_placeholders("con của ?Người"), "con của")
            self.assertEqual(placeholders_in("Ai là vua? ?A"), ["?A"])  # the "?" ending a title is not one
        with LangEnv("en"):
            self.assertEqual(placeholders_in("?Người | lên ngôi năm | ?NămA"), [])  # English: unchanged rule
            self.assertEqual(placeholders_in("?A | born in | ?year1"), ["?A", "?year1"])

    def test_binding_and_substitution_with_vietnamese_names(self):
        from src.TRAG_plus import TRAGPlus
        response = ("Fully Resolved Clue 1:\nSubject: Trần Thừa\nPredicate: là cha của\nObject: Trần Thái Tông\n"
                    "Binding: ?Người = Trần Thái Tông\n")
        with LangEnv("vi"):
            resolved, _ = TRAGPlus._parse_plus(types.SimpleNamespace(), response)
            self.assertEqual(resolved[0][1]["bindings"], {"?Người": "Trần Thái Tông"})
            dummy = types.SimpleNamespace(_bindings={"?Người": "Trần Thái Tông"}, _stats={"bindings_applied": 0})
            out = TRAGPlus._substitute(dummy, {"subject": "?Người", "predicate": "có con là", "object": "?B"})
            self.assertEqual(out["subject"], "Trần Thái Tông")
            self.assertEqual(out["object"], "?B")
            self.assertEqual(dummy._stats["bindings_applied"], 1)

    def test_nfc_in_matching(self):
        from src.TRAG_plus import TRAGPlus, clue_key
        nfd = unicodedata.normalize("NFD", "Trần Thái Tông")
        with LangEnv("vi"):
            self.assertEqual(TRAGPlus._field_match(nfd, "Trần Thái Tông"), 1.0)
            self.assertEqual(clue_key({"subject": nfd, "predicate": "là", "object": "vua"}),
                             clue_key({"subject": "Trần Thái Tông", "predicate": "là", "object": "vua"}))
            results = [{"retrieved_passages": [{"text": "Nhà Lý dời đô ra Thăng Long."},
                                               {"text": "Trần Thái Tông là vua đầu tiên của nhà Trần."}]}]
            clues = [{"subject": nfd, "predicate": "là", "object": "?A"}]
            out = TRAGPlus._evidence_passages(types.SimpleNamespace(), clues, results, limit=1)
            self.assertEqual(out, ["Trần Thái Tông là vua đầu tiên của nhà Trần."])

    def test_no_flags_same_prompts_in_vietnamese(self):
        from src.TRAG import TRAG
        from src.TRAG_plus import TRAGPlus
        from test_trag_plus import build
        with LangEnv("vi"):
            rag_a, llm_a = build(TRAG, fe.chain_rules(named=False))
            out_a = rag_a.trag_qa(fe.QUESTION, query_id="q0")
            rag_b, llm_b = build(TRAGPlus, fe.chain_rules(named=False), fixes=[])
            out_b = rag_b.trag_qa(fe.QUESTION, query_id="q0")
        self.assertEqual(out_a["answer"], out_b["answer"])
        self.assertEqual(llm_a.prompts, llm_b.prompts)

    def test_vietnamese_note_asks_for_latin_placeholder_names(self):
        from src.TRAG_plus import VI_NAMES_NOTE
        self.assertIn("?A, ?B, ?C", VI_NAMES_NOTE)


class TestRunColabV6(unittest.TestCase):
    def setUp(self):
        self.root = fe.tmpdir()
        make_dataset(self.root)
        self.llm = fe.FakeLLM(fe.chain_rules(named=True) + [
            (["Question: ", "Thought: "], "Thought: I remember it.\nAnswer: 1211")])
        fe.install_fakes(self.llm)
        os.environ.setdefault("GOOGLE_API_KEY", "fake")
        self.old_cwd, self.old_argv = os.getcwd(), sys.argv
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.old_cwd)
        sys.argv = self.old_argv
        shutil.rmtree(self.root, ignore_errors=True)

    def run_main(self, argv):
        import run_colab
        sys.argv = ["run_colab.py", "--dataset", "fake", "--llm_name", "fake-llm",
                    "--embedding_name", "fake-embedding"] + argv
        run_colab.main()
        files = sorted(os.listdir("outputs/results"), key=lambda f: os.path.getmtime(os.path.join("outputs/results", f)))
        with open(os.path.join("outputs/results", files[-1])) as f:
            return files[-1], json.load(f)

    def test_nor(self):
        import run_colab
        old = run_colab.make_llm
        run_colab.make_llm = lambda config: self.llm
        try:
            name, data = self.run_main(["--method", "nor", "--limit", "2"])
        finally:
            run_colab.make_llm = old
        s, recs = data["summary"], data["records"]
        self.assertIn("_nor_", name)
        self.assertEqual(recs[0]["prediction"], "1211")
        self.assertEqual(recs[0]["em"], 1.0)
        self.assertIsNone(s["recall_first5_docs"])
        self.assertEqual(s["avg_llm_calls"], 1.0)
        self.assertTrue(all("Wikipedia Title: Battle" not in p for p in self.llm.prompts))

    def test_short_answers_cache_tag_and_summary(self):
        name, data = self.run_main(["--method", "trag_plus", "--fixes", "b,s1,s3", "--limit", "2",
                                    "--answer_format", "short", "--cache_tag", "r2"])
        s, recs = data["summary"], data["records"]
        self.assertEqual(s["answer_format"], "short")
        self.assertTrue(s["llm_cache"].endswith("fake_llm_cache_r2"))
        self.assertTrue(os.path.isdir(s["llm_cache"]))
        self.assertEqual(len(s["questions_md5"]), 32)
        self.assertEqual(len(s["openie_md5"]), 32)
        self.assertEqual(set(s["by_hops"]), {"2", "3"})
        self.assertIn("EM_lenient", s)
        final_prompts = [p for p in self.llm.prompts if p.startswith("Based on the reasoning clues")]
        self.assertTrue(final_prompts and all("Answer requirements" in p for p in final_prompts))
        self.assertEqual(recs[0]["prediction"], fe.GOLD)
        self.assertEqual(len(recs[0]["calls"]), recs[0]["llm_calls"])
        self.assertGreater(recs[0]["n_bindings"], 0)          # s1 named the placeholders and bound them
        self.assertIn("s4_rejected", recs[0])
        self.assertEqual(s["plus_totals"]["n_bindings"], sum(r["n_bindings"] for r in recs))

    def test_batches_merge_to_the_full_run(self):
        import merge_results
        _, full = self.run_main(["--method", "trag_plus", "--fixes", "b,s1", "--limit", "2", "--tag", "full"])
        _, a = self.run_main(["--method", "trag_plus", "--fixes", "b,s1", "--start", "0", "--limit", "1", "--tag", "m"])
        _, b = self.run_main(["--method", "trag_plus", "--fixes", "b,s1", "--start", "1", "--limit", "1", "--tag", "m"])
        m = merge_results.merge([b, a])
        s = m["summary"]
        self.assertEqual([r["qid"] for r in m["records"]], [r["qid"] for r in full["records"]])
        for key in ("ExactMatch", "F1", "EM_lenient", "by_hops", "recall_first5_docs", "avg_llm_calls"):
            self.assertEqual(s[key], full["summary"][key], key)
        self.assertEqual(s["num_questions"], 2)
        # counters are summed over the batches (v6.1 copied them from the first batch)
        self.assertEqual(s["total_llm"]["llm_calls"],
                         a["summary"]["total_llm"]["llm_calls"] + b["summary"]["total_llm"]["llm_calls"])
        self.assertEqual(s["plus_totals"], full["summary"]["plus_totals"])
        self.assertEqual(s["pct_all_resolved"], full["summary"]["pct_all_resolved"])
        self.assertEqual(s["avg_retrieved_docs"], full["summary"]["avg_retrieved_docs"])
        with self.assertRaises(SystemExit):
            merge_results.merge([a, full])  # different --tag
        # the merged file keeps one trace folder per batch, so error_analysis finds every trace
        import error_analysis
        files = sorted(f for f in os.listdir("outputs/results") if f.endswith("_m.json"))
        merge_results.main([os.path.join("outputs/results", f) for f in files] + ["-o", "outputs/results/merged_m.json"])
        merged = json.load(open("outputs/results/merged_m.json"))
        self.assertEqual(merged["summary"]["run_name"], "merged_m")
        _, rows, n_trace, _ = error_analysis.label_run("outputs/results/merged_m.json", "reproduce/dataset/fake.json",
                                                        None, {}, 0.8)
        self.assertEqual((len(rows), n_trace), (2, 2))

    def test_qids_selects_questions(self):
        all_ids = [s["id"] for s in json.load(open("reproduce/dataset/fake.json"))]
        name, data = self.run_main(["--method", "trag", "--qids", all_ids[1], "--tag", "pilot"])
        self.assertIn("_ids1_pilot", name)
        self.assertEqual([r["qid"] for r in data["records"]], [all_ids[1]])
        self.assertEqual(data["records"][0]["id"], 1)
        with self.assertRaises(SystemExit):
            self.run_main(["--method", "trag", "--qids", "nope"])

    def test_refuses_to_overwrite_other_configuration(self):
        self.run_main(["--method", "trag", "--limit", "1"])
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--method", "trag", "--limit", "1", "--answer_format", "short"])
        self.assertIn("different answer_format", str(ctx.exception))
        self.run_main(["--method", "trag", "--limit", "1"])  # same configuration: rerun is fine
        self.run_main(["--method", "trag", "--limit", "1", "--answer_format", "short", "--tag", "short"])


class TestMakeTableLenient(unittest.TestCase):
    def test_lenient_columns(self):
        import make_table
        from test_reporting import write_result, rec, read
        root = fe.tmpdir()
        try:
            p = lambda n: os.path.join(root, n)  # noqa: E731
            write_result(p("a.json"), "vnhistory", "trag",
                         [rec(f"2hop_{i}", 0.0, 0.5, 2, em_lenient=1.0, f1_lenient=1.0) for i in range(4)])
            make_table.main([p("a.json"), "--out_dir", p("t1"), "--no_groups"])
            self.assertIn("0,0 / 50,0", read(p("t1/table_main.md")))
            make_table.main([p("a.json"), "--out_dir", p("t2"), "--no_groups", "--metric", "lenient"])
            md = read(p("t2/table_main.md"))
            self.assertIn("100,0 / 100,0", md)
            self.assertIn("EM_lenient / F1_lenient", md)
            write_result(p("old.json"), "vnhistory", "trag", [rec("2hop_9", 1.0, 1.0)])
            with self.assertRaises(SystemExit):
                make_table.main([p("old.json"), "--out_dir", p("t3"), "--metric", "lenient"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestRescore(unittest.TestCase):
    def test_new_alias_changes_em(self):
        import rescore
        data = {"summary": {"dataset": "vnhistory", "lang": "vi", "ExactMatch": 0.0, "F1": 0.5},
                "records": [{"qid": "2hop_003", "prediction": "Lý Càn Đức", "em": 0.0, "f1": 0.5,
                             "gold_answers": ["Lý Nhân Tông"]}]}
        samples = {"2hop_003": {"id": "2hop_003", "answer": "Lý Nhân Tông", "answer_aliases": ["Lý Càn Đức"]}}
        out, old, changed = rescore.rescore(data, samples, "x" * 32)
        self.assertEqual(changed, 1)
        self.assertEqual(out["records"][0]["em"], 1.0)
        self.assertEqual(out["records"][0]["hops"], 2)
        self.assertEqual(out["summary"]["ExactMatch"], 1.0)
        self.assertEqual(out["summary"]["rescored"]["previous"], {"ExactMatch": 0.0, "F1": 0.5})

    def test_unknown_qid_stops(self):
        import rescore
        data = {"summary": {"lang": "vi"}, "records": [{"qid": "9hop_999", "prediction": "x"}]}
        with self.assertRaises(SystemExit):
            rescore.rescore(data, {}, "x" * 32)


if __name__ == "__main__":
    unittest.main()
