"""Vietnamese handling (T2RAG_LANG=vi) in the patched repo code and in our own scripts.
English behaviour must stay exactly as in the paper's code."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())


class LangEnv:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self.old = os.environ.get("T2RAG_LANG")
        os.environ["T2RAG_LANG"] = self.lang

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("T2RAG_LANG", None)
        else:
            os.environ["T2RAG_LANG"] = self.old


class TestRepoPatches(unittest.TestCase):
    def test_normalize_answer(self):
        from src.utils.eval_utils import normalize_answer
        with LangEnv("en"):
            self.assertEqual(normalize_answer("The Tràng An!"), "tràng")  # original English rule
        with LangEnv("vi"):
            self.assertEqual(normalize_answer("Tràng An"), "tràng an")
            self.assertEqual(normalize_answer("“Lê Lợi”."), "lê lợi")
            decomposed = "Tràng An"  # NFD form
            self.assertEqual(normalize_answer(decomposed), "tràng an")

    def test_text_processing(self):
        from src.utils.misc_utils import text_processing
        with LangEnv("en"):
            self.assertEqual(text_processing("Trần Hưng Đạo"), "tr n h ng   o")  # the original bug
        with LangEnv("vi"):
            self.assertEqual(text_processing("Trần Hưng Đạo"), "trần hưng đạo")
            self.assertEqual(text_processing(["Lý Thái Tổ (1009)", "Thăng-Long"]),
                             ["lý thái tổ 1009", "thăng long"])


class TestOurScripts(unittest.TestCase):
    def test_run_colab_scores(self):
        import run_colab
        with LangEnv("vi"):
            self.assertEqual(run_colab.em_f1("Tràng An", ["tràng an"])[0], 1.0)
            self.assertEqual(run_colab.em_f1("An Nam", ["Tràng An"])[0], 0.0)
        with LangEnv("en"):
            self.assertEqual(run_colab.em_f1("the Beatles", ["Beatles"])[0], 1.0)

    def test_error_analysis_norm(self):
        import error_analysis
        old = error_analysis.LANG["value"]
        try:
            error_analysis.LANG["value"] = "vi"
            self.assertTrue(error_analysis.contains("Vua đóng đô ở Tràng An.", "Tràng An"))
            self.assertFalse(error_analysis.contains("Vua đóng đô ở Tràng.", "Tràng An"))
        finally:
            error_analysis.LANG["value"] = old


if __name__ == "__main__":
    unittest.main()
