"""Fake LLM, fake embedding and a tiny 3-hop corpus, so T2RAG / T2RAG+ can be tested
without network, GPU or API key.  Used by tests/test_trag_plus.py and tests/test_run_colab.py.

Run from the repo root (the folder that contains src/):
    python -m unittest discover tests -v
"""
import hashlib
import json
import os
import re
import tempfile

import numpy as np

# ---------------------------------------------------------------- tiny corpus (chain of 3 hops)
DOCS = [
    ("Battle of Bach Dang (1288)",
     "The Battle of Bach Dang in 1288 was commanded by Tran Hung Dao against the Mongol fleet."),
    ("Tran Hung Dao", "Tran Hung Dao was a prince of the Tran dynasty. His father was Tran Lieu."),
    ("Tran Lieu", "Tran Lieu was born in 1211 and died in 1251."),
    ("Mongol fleet", "The Mongol fleet was led by Omar Khan."),
    ("Ly dynasty", "The Ly dynasty ruled Dai Viet before the Tran dynasty."),
    ("Tran dynasty", "Tran Hung Dao is prince of Tran dynasty, the family that followed the Ly dynasty."),
]
TRIPLES = [
    [["Battle of Bach Dang 1288", "commanded by", "Tran Hung Dao"],
     ["Battle of Bach Dang 1288", "fought against", "Mongol fleet"]],
    [["Tran Hung Dao", "is prince of", "Tran dynasty"], ["Tran Hung Dao", "has father", "Tran Lieu"]],
    [["Tran Lieu", "was born in", "1211"], ["Tran Lieu", "died in", "1251"]],
    [["Mongol fleet", "led by", "Omar Khan"]],
    [["Ly dynasty", "ruled before", "Tran dynasty"]],
    [["Tran Hung Dao", "is prince of", "Tran dynasty"]],  # same proposition as doc 2 (bug L5)
]
QUESTION = "When was the father of the commander of the Battle of Bach Dang in 1288 born?"
GOLD = "1211"


def passage(i):
    return f"{DOCS[i][0]}\n{DOCS[i][1]}"


def openie_json():
    docs = []
    for i in range(len(DOCS)):
        p = passage(i)
        docs.append({"idx": "chunk-" + hashlib.md5(p.encode()).hexdigest(), "passage": p,
                     "extracted_entities": [], "extracted_triples": TRIPLES[i]})
    return {"docs": docs, "avg_ent_chars": 0, "avg_ent_words": 0}


# ---------------------------------------------------------------- fakes
class FakeLLM:
    """rules: list of (keys, response). The first rule whose keys all appear in the prompt wins.
    response may be a string or a function(prompt) -> string. Every prompt is recorded."""

    def __init__(self, rules, default="unknown"):
        self.rules, self.default, self.prompts = rules, default, []

    def infer(self, messages, *args, **kwargs):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        out = self.default
        for keys, response in self.rules:
            if all(k in prompt for k in keys):
                out = response(prompt) if callable(response) else response
                break
        meta = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(out.split()), "finish_reason": "stop"}
        return out, meta, False


class FakeEmbedding:
    """Bag-of-words hashing vectors: texts sharing words get a high cosine."""
    dim = 512

    def __init__(self, *args, **kwargs):
        pass

    def batch_encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class FakeStore:
    def __init__(self, *args, **kwargs):
        self.items = []

    def insert_strings(self, texts):
        self.items.extend(texts)

    def get_all_ids(self):
        return list(range(len(self.items)))

    def get_embeddings(self, ids):
        return FakeEmbedding().batch_encode([self.items[i] for i in ids])


def install_fakes(llm):
    """Make TRAG build the fakes instead of real models (patches names inside src.TRAG)."""
    import importlib
    # importlib, not "import src.TRAG as m": src/__init__.py exports a class named TRAG.
    trag_module = importlib.import_module("src.TRAG")
    trag_module._get_llm_class = lambda config: llm
    trag_module._get_embedding_model_class = lambda embedding_model_name=None: FakeEmbedding
    trag_module.EmbeddingStore = FakeStore


def make_config(root, qa_top_k=5):
    """A BaseConfig whose OpenIE file is the tiny corpus. Returns (config, root)."""
    from src.utils.config_utils import BaseConfig
    univ = os.path.join(root, "fake_univ")
    os.makedirs(univ, exist_ok=True)
    with open(os.path.join(univ, "openie_results_ner.json"), "w") as f:
        json.dump(openie_json(), f)
    config = BaseConfig(save_dir=os.path.join(root, "run"), llm_name="fake-llm", dataset="fake_univ",
                        embedding_model_name="fake-embedding", qa_top_k=qa_top_k, openie_mode="online",
                        max_qa_steps=3)
    config.exclude_llm_from_paths = True
    config.embedding_base_dir = univ
    return config


# ---------------------------------------------------------------- LLM scripts
def decompose_response(named):
    if named:
        return ("Reasoning: find the commander, then his father, then the birth year.\nTriples:\n"
                "Battle of Bach Dang 1288 | commanded by | ?A\n?A | has father | ?B\n?B | was born in | ?C\n")
    return ("Reasoning: find the commander, then his father, then the birth year.\nTriples:\n"
            "Battle of Bach Dang 1288 | commanded by | ?\n? | has father | ?\n? | was born in | ?\n")


def chain_rules(named=True, with_evidence=True, bad_evidence=False):
    """An LLM that answers one hop per round and never rewrites the fuzzy clues itself
    (the situation where the original code loses the chain, bug L1)."""
    ev1 = 'Evidence: "The Battle of Bach Dang in 1288 was commanded by Tran Hung Dao against the Mongol fleet."'
    ev2 = 'Evidence: "His father was Tran Lieu."'
    ev3 = 'Evidence: "Tran Lieu was born in 1211 and died in 1251."'
    if bad_evidence:
        ev3 = 'Evidence: "Tran Lieu was born in the year 1300 in Thang Long."'
    if not with_evidence:
        ev1 = ev2 = ev3 = ""
    hop1 = f"""Fully Resolved Clue 1:
Subject: Battle of Bach Dang 1288
Predicate: commanded by
Object: Tran Hung Dao
{"Binding: ?A = Tran Hung Dao" if named else ""}
{ev1}
"""
    hop2 = f"""Fully Resolved Clue 1:
Subject: Tran Hung Dao
Predicate: has father
Object: Tran Lieu
{"Binding: ?B = Tran Lieu" if named else ""}
{ev2}
"""
    hop3 = f"""Fully Resolved Clue 1:
Subject: Tran Lieu
Predicate: was born in
Object: {"1300" if bad_evidence else "1211"}
{"Binding: ?C = " + ("1300" if bad_evidence else "1211") if named else ""}
{ev3}
"""

    def final(prompt):
        clues = prompt.split("Key Reasoning Clues:")[1].split("Instructions:")[0]
        return "1211" if "1211" in clues else "unknown"

    return [
        (["extracting the necessary knowledge triples"], decompose_response(named)),
        (["Traceable Clue 1:\nSubject: Tran Lieu\nPredicate: was born in"], hop3),
        (["Traceable Clue 1:\nSubject: Tran Hung Dao\nPredicate: has father"], hop2),
        (["Traceable Clue 1:\nSubject: Battle of Bach Dang 1288\nPredicate: commanded by"], hop1),
        (["Based on the reasoning clues"], final),
    ]


def tmpdir():
    return tempfile.mkdtemp(prefix="t2rag_test_")
