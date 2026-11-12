"""End-to-end test of run_colab.py (trag and trag_plus) on the tiny fake dataset."""
import json
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.getcwd()
sys.path.insert(0, REPO)

import fake_env as fe  # noqa: E402


def make_dataset(root):
    os.makedirs(os.path.join(root, "reproduce", "dataset"), exist_ok=True)
    corpus = [{"title": t, "text": x} for t, x in fe.DOCS]
    samples = [{"id": "3hop1__bachdang", "question": fe.QUESTION, "answer": fe.GOLD, "answer_aliases": [],
                "paragraphs": [{"title": fe.DOCS[i][0], "paragraph_text": fe.DOCS[i][1], "is_supporting": True}
                               for i in (0, 1, 2)]},
               {"id": "2hop__fleet", "question": "Who led the fleet the Battle of Bach Dang in 1288 fought against?",
                "answer": "Omar Khan", "answer_aliases": [],
                "paragraphs": [{"title": fe.DOCS[i][0], "paragraph_text": fe.DOCS[i][1], "is_supporting": True}
                               for i in (0, 3)]}]
    json.dump(corpus, open(os.path.join(root, "reproduce", "dataset", "fake_corpus.json"), "w"))
    json.dump(samples, open(os.path.join(root, "reproduce", "dataset", "fake.json"), "w"))
    os.makedirs(os.path.join(root, "outputs", "fake_univ"), exist_ok=True)
    json.dump(fe.openie_json(), open(os.path.join(root, "outputs", "fake_univ", "openie_results_ner.json"), "w"))


class TestRunColab(unittest.TestCase):
    def run_main(self, argv):
        import run_colab
        root = fe.tmpdir()
        make_dataset(root)
        llm = fe.FakeLLM(fe.chain_rules(named=True))
        fe.install_fakes(llm)
        old_cwd, old_argv = os.getcwd(), sys.argv
        os.environ.setdefault("GOOGLE_API_KEY", "fake")
        try:
            os.chdir(root)
            sys.argv = ["run_colab.py", "--dataset", "fake", "--llm_name", "fake-llm",
                        "--embedding_name", "fake-embedding"] + argv
            run_colab.main()
            results = os.listdir(os.path.join(root, "outputs", "results"))
            with open(os.path.join(root, "outputs", "results", results[0])) as f:
                data = json.load(f)
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv
            shutil.rmtree(root, ignore_errors=True)
        return results[0], data

    def test_trag_plus_run(self):
        name, data = self.run_main(["--method", "trag_plus", "--fixes", "b,s1,s3", "--limit", "2"])
        self.assertIn("trag_plus_fx-b-s1-s3", name)
        s, recs = data["summary"], data["records"]
        self.assertEqual(s["fixes"], ["b", "s1", "s3"])
        self.assertEqual(recs[0]["hops"], 3)
        self.assertEqual(recs[1]["hops"], 2)
        self.assertEqual(recs[0]["prediction"], fe.GOLD)
        for key in ("llm_calls", "prompt_tokens", "iterations", "n_dropped_clues", "trace_file", "gold_docs_found_first5"):
            self.assertIn(key, recs[0])
        self.assertEqual(recs[0]["llm_calls"], 5)  # decompose + 3 rounds + final answer
        self.assertIsNotNone(s["avg_llm_calls"])

    def test_start_offset(self):
        name, data = self.run_main(["--method", "trag", "--start", "1", "--limit", "1"])
        self.assertIn("_from1_1q", name)
        self.assertEqual(data["records"][0]["qid"], "2hop__fleet")
        self.assertEqual(data["records"][0]["id"], 1)


if __name__ == "__main__":
    unittest.main()
