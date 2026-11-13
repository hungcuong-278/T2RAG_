"""Tests for make_table.py and error_analysis.py.

1. Unit tests on hand-made result files (exact numbers, bootstrap, merging).
2. End to end: run_colab.py runs T2RAG and T2RAG+ on the fake 3-hop corpus, then both tools
   read the real result files and reasoning traces.
"""
import csv
import json
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import fake_env as fe  # noqa: E402


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def write_result(path, dataset, method, records, **summary):
    s = {"dataset": dataset, "method": method, "run_name": os.path.splitext(os.path.basename(path))[0],
         "seconds": 10.0}
    s.update(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": s, "records": records}, f)


def rec(qid, em, f1, hops=2, **kw):
    r = {"id": int(qid.split("_")[-1]), "qid": qid, "hops": hops, "em": em, "f1": f1, "question": qid,
         "prediction": "x", "gold_answers": ["y"]}
    r.update(kw)
    return r


class TestMakeTable(unittest.TestCase):
    def setUp(self):
        self.root = fe.tmpdir()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_values_dagger_bold_and_merge(self):
        import make_table
        base = [rec(f"2hop_{i}", 0.0, 0.2, 2) for i in range(30)] + [rec(f"3hop_{i}", 1.0, 1.0, 3) for i in range(10)]
        better = [rec(f"2hop_{i}", 1.0, 1.0, 2) for i in range(30)] + [rec(f"3hop_{i}", 1.0, 1.0, 3) for i in range(10)]
        p = lambda n: os.path.join(self.root, n)  # noqa: E731
        write_result(p("base.json"), "musique", "trag", base)
        # the improved method is split over two files (e.g. two halves of the test set)
        write_result(p("b1.json"), "musique", "trag_plus_fx-b-s1", better[:20], avg_llm_calls=5.0)
        write_result(p("b2.json"), "musique", "trag_plus_fx-b-s1", better[20:], avg_llm_calls=5.0)
        out = p("tables")
        make_table.main([p("base.json"), p("b1.json"), p("b2.json"), "--out_dir", out, "--n_boot", "2000"])
        md = read(os.path.join(out, "table_main.md"))
        lines = [line for line in md.splitlines() if line.startswith("| trag")]
        base_line = [line for line in lines if line.startswith("| trag |")][0]
        plus_line = [line for line in lines if line.startswith("| trag_plus")][0]
        # overall: base EM 25.0, F1 (30*0.2+10)/40 = 40.0 ; plus 100/100
        self.assertIn("25,0 / 40,0", base_line)
        self.assertIn("**100,0†** / **100,0†**", plus_line)
        # 3-hop column: equal scores -> no dagger
        self.assertIn("**100,0** / **100,0**", plus_line.split("|")[-2])
        rows = list(csv.DictReader(read(os.path.join(out, "table_main.csv")).splitlines()))
        all_plus = [r for r in rows if r["method"] == "trag_plus_fx-b-s1" and r["group"] == "all"][0]
        self.assertEqual(all_plus["n"], "40")
        self.assertLess(float(all_plus["p_EM_vs_baseline"]), 0.05)
        cost = read(os.path.join(out, "table_cost.md"))
        self.assertIn("5,00", cost)

    def test_bootstrap(self):
        from make_table import paired_bootstrap_p
        self.assertEqual(paired_bootstrap_p([1, 1, 1], [1, 1, 1]), 1.0)
        self.assertLess(paired_bootstrap_p([1] * 50, [0] * 50), 0.01)
        self.assertGreater(paired_bootstrap_p([0] * 50, [1] * 50), 0.99)
        noisy = paired_bootstrap_p([1, 0] * 10, [0, 1] * 10)
        self.assertGreater(noisy, 0.3)

    def test_average_over_datasets_and_labels(self):
        import make_table
        p = lambda n: os.path.join(self.root, n)  # noqa: E731
        write_result(p("a.json"), "musique", "trag", [rec("2hop_1", 1.0, 1.0)])
        write_result(p("b.json"), "vnhistory", "trag", [rec("2hop_1", 0.0, 0.5)])
        with open(p("labels.json"), "w") as f:
            json.dump({"trag": "T²RAG gốc"}, f)
        out = p("t")
        make_table.main([p("a.json"), p("b.json"), "--labels", p("labels.json"), "--out_dir", out, "--no_groups"])
        md = read(os.path.join(out, "table_main.md"))
        self.assertIn("| T²RAG gốc | 100,0 / 100,0 | 0,0 / 50,0 | 50,0 | 75,0 |", md)


def make_dataset(root):
    os.makedirs(os.path.join(root, "reproduce", "dataset"), exist_ok=True)
    corpus = [{"title": t, "text": x} for t, x in fe.DOCS]
    para = lambda i: {"title": fe.DOCS[i][0], "paragraph_text": fe.DOCS[i][1], "is_supporting": True}  # noqa: E731
    samples = [{"id": "3hop1__bachdang", "question": fe.QUESTION, "answer": fe.GOLD, "answer_aliases": [],
                "paragraphs": [para(0), para(1), para(2)],
                "question_decomposition": [
                    {"question": "Who commanded the Battle of Bach Dang in 1288?", "answer": "Tran Hung Dao"},
                    {"question": "Who is the father of #1?", "answer": "Tran Lieu"},
                    {"question": "When was #2 born?", "answer": fe.GOLD}]},
               {"id": "2hop__fleet", "question": "Who led the fleet the Battle of Bach Dang in 1288 fought against?",
                "answer": "Omar Khan", "answer_aliases": [], "paragraphs": [para(0), para(3)]}]
    json.dump(corpus, open(os.path.join(root, "reproduce", "dataset", "fake_corpus.json"), "w"))
    json.dump(samples, open(os.path.join(root, "reproduce", "dataset", "fake.json"), "w"))
    os.makedirs(os.path.join(root, "outputs", "fake_univ"), exist_ok=True)
    json.dump(fe.openie_json(), open(os.path.join(root, "outputs", "fake_univ", "openie_results_ner.json"), "w"))


class TestEndToEnd(unittest.TestCase):
    """run_colab (trag, trag_plus) -> error_analysis -> make_table, on the fake corpus."""

    @classmethod
    def setUpClass(cls):
        import run_colab
        cls.root = fe.tmpdir()
        make_dataset(cls.root)
        old_cwd, old_argv = os.getcwd(), sys.argv
        os.environ.setdefault("GOOGLE_API_KEY", "fake")
        cls.results = {}
        try:
            os.chdir(cls.root)
            for method, extra in (("trag", []), ("trag_plus", ["--fixes", "b,s1,s3"])):
                fe.install_fakes(fe.FakeLLM(fe.chain_rules(named=True)))
                sys.argv = ["run_colab.py", "--dataset", "fake", "--llm_name", "fake-llm",
                            "--embedding_name", "fake-embedding", "--method", method, "--limit", "2"] + extra
                run_colab.main()
            for name in os.listdir("outputs/results"):
                key = "plus" if "trag_plus" in name else "trag"
                cls.results[key] = os.path.join("outputs", "results", name)
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def in_root(self, fn, *a):
        old = os.getcwd()
        os.chdir(self.root)
        try:
            return fn(*a)
        finally:
            os.chdir(old)

    def labels(self, key):
        path = os.path.join(self.root, "outputs", "results")
        name = os.path.splitext(os.path.basename(self.results[key]))[0]
        rows = list(csv.DictReader(read(os.path.join(path, f"errors_{name}.csv")).splitlines()))
        return {r["qid"]: r for r in rows}

    def test_error_labels_from_traces(self):
        import error_analysis
        self.in_root(error_analysis.main, [self.results["trag"], "--compare", self.results["plus"], "--sample", "5"])
        self.in_root(error_analysis.main, [self.results["plus"]])
        orig, plus = self.labels("trag"), self.labels("plus")
        # Original T2RAG loses the 3-hop chain after round 1 (bug L1) -> broken chain.
        self.assertEqual(orig["3hop1__bachdang"]["label"], "roi_mat_xich")
        self.assertEqual(orig["3hop1__bachdang"]["has_trace"], "True")
        self.assertEqual(plus["3hop1__bachdang"]["label"], "dung")
        cmp_files = [f for f in os.listdir(os.path.join(self.root, "outputs", "results")) if f.startswith("errors_compare")]
        self.assertEqual(len(cmp_files), 1)
        text = read(os.path.join(self.root, "outputs", "results", cmp_files[0]))
        # The fake LLM has no script for the 2-hop question, so both runs miss it; B fixes the 3-hop one.
        self.assertIn("| Rơi mắt xích (bộ ba bị bỏ / chưa giải) | 2 | 1 |", text)

    def test_make_table_on_real_runs(self):
        import make_table
        out = os.path.join(self.root, "tables")
        self.in_root(make_table.main, [self.results["trag"], self.results["plus"], "--out_dir", out, "--n_boot", "500"])
        md = read(os.path.join(out, "table_main.md"))
        self.assertIn("fake 3 bước", md)
        cost = read(os.path.join(out, "table_cost.md"))
        plus_row = [line for line in cost.splitlines() if line.startswith("| trag_plus")][0]
        self.assertIn("| fake | 2 |", plus_row)


class TestLabelRules(unittest.TestCase):
    def test_rule_order(self):
        from error_analysis import analyse
        gold = {"gold_docs": ["T\nA", "U\nB"], "intermediate": ["Mid"]}
        trace = lambda passages, clues, fuzzy=(): {  # noqa: E731
            "iterations": [{"step_2_retrieval_and_expansion": {"retrieved_passages": [{"text": t} for t in passages]}}],
            "step_1_reasoning": {"all_triples": [{}, {}]},
            "final_results": {"all_final_clues": clues, "remaining_fuzzy_clues": list(fuzzy)}}
        r = {"qid": "q", "em": 0.0, "f1": 0.0, "prediction": "wrong", "gold_answers": ["Gold"]}
        c = lambda s, o: {"subject": s, "predicate": "p", "object": o}  # noqa: E731
        # answer already in the clues -> final answer error, even if a gold passage is missing
        self.assertEqual(analyse(r, trace(["T\nA"], [c("X", "Gold"), c("Mid", "z")]), gold, [], 0.5)["label"],
                         "tra_loi_cuoi_sai")
        self.assertEqual(analyse(r, trace(["T\nA"], [c("X", "y"), c("Mid", "z")]), gold, [], 0.5)["label"],
                         "thieu_doan_vang")
        self.assertEqual(analyse(r, trace(["T\nA", "U\nB"], [c("X", "y")]), gold, [], 0.5)["label"], "roi_mat_xich")
        self.assertEqual(analyse(r, trace(["T\nA", "U\nB"], [c("X", "y"), c("W", "z")]), gold, [], 0.5)["label"],
                         "dien_sai_buoc_giua")
        self.assertEqual(analyse(r, trace(["T\nA", "U\nB"], [c("X", "y"), c("Mid", "z")]), gold, [], 0.5)["label"],
                         "khac")
        groups = [{"tran quoc tuan", "tran hung dao"}]
        r2 = dict(r, prediction="Trần Quốc Tuấn".replace("ầ", "a").replace("ố", "o").replace("ấ", "a"),
                  gold_answers=["Tran Hung Dao"])
        self.assertEqual(analyse(r2, trace([], []), gold, groups, 0.5)["label"], "sai_ten")
        self.assertEqual(analyse(dict(r, f1=0.6), None, gold, [], 0.5)["label"], "dung")


class TestCheckDataset(unittest.TestCase):
    def setUp(self):
        self.root = fe.tmpdir()
        make_dataset(self.root)
        # contract ids: 3hop_ / 2hop_ prefixes
        path = os.path.join(self.root, "reproduce", "dataset", "fake.json")
        data = json.load(open(path))
        data[0]["id"], data[1]["id"] = "3hop_001", "2hop_002"
        json.dump(data, open(path, "w"))
        self.qpath = path

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_good_dataset_passes(self):
        import check_vnhistory
        self.assertEqual(check_vnhistory.main(["--dataset", "fake", "--root", self.root, "--dev_size", "1"]), 0)

    def test_openie_md5(self):
        import check_vnhistory
        import hashlib
        oie = os.path.join(self.root, "outputs", "fake_univ", "openie_results_ner.json")
        md5 = hashlib.md5(open(oie, "rb").read()).hexdigest()
        base = ["--dataset", "fake", "--root", self.root, "--dev_size", "1"]
        self.assertEqual(check_vnhistory.main(base + ["--openie_md5", md5[:8] + "…"]), 0)
        self.assertEqual(check_vnhistory.main(base + ["--openie_md5", "00000000"]), 1)

    def test_broken_dataset_fails(self):
        import check_vnhistory
        data = json.load(open(self.qpath))
        data[0]["id"] = "q001"                                   # no hop prefix
        data[1]["paragraphs"][0]["paragraph_text"] += " extra"   # not a corpus passage
        json.dump(data, open(self.qpath, "w"))
        oie_path = os.path.join(self.root, "outputs", "fake_univ", "openie_results_ner.json")
        oie = json.load(open(oie_path))
        oie["docs"][0]["extracted_triples"].append(["only", "two"])
        json.dump(oie, open(oie_path, "w"))
        self.assertEqual(check_vnhistory.main(["--dataset", "fake", "--root", self.root]), 1)


if __name__ == "__main__":
    unittest.main()
