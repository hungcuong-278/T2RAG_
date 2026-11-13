"""Patch the official T2RAG repo so it runs on Google Colab with only an API key.

Run once from the repo root:  python patch_t2rag.py
Safe to run again: each patch is skipped if it is already applied.

What it changes (the T2RAG method itself is untouched):
  1. Optional heavy dependencies (vllm, gritlm, boto3, litellm) are imported lazily,
     so `pip install -r requirements.txt` (which would replace Colab's torch) is not needed.
  2. Adds a SentenceTransformer embedding class, so small embedding models
     (BAAI/bge-m3, intfloat/multilingual-e5-*) run on Colab's free T4 GPU.
     The paper's NV-Embed-v2 (about 7B parameters) does not fit on a T4.
  3. LLM calls retry with exponential backoff, so rate limits (HTTP 429) on
     free API tiers slow the run down instead of turning answers into errors.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(ROOT, "src", "TRAG.py")):
    ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "src", "TRAG.py")):
    sys.exit("Run this script from the T2RAG repo root (the folder that contains src/TRAG.py).")


def patch(rel_path, old, new, marker=None):
    path = os.path.join(ROOT, rel_path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    marker = marker or new
    if marker in text:
        print(f"  already patched: {rel_path}")
        return
    if old not in text:
        sys.exit(f"Cannot patch {rel_path}: expected text not found. The repo may have changed.")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new, 1))
    print(f"  patched: {rel_path}")


def lazy(line, names):
    stub = "; ".join(f"{n} = None" for n in names)
    return f"try:\n    {line}\nexcept ImportError:  # optional dependency (colab patch)\n    {stub}"


print("Making optional dependencies lazy...")
patch("src/llm/vllm_offline.py", "from vllm import SamplingParams, LLM",
      lazy("from vllm import SamplingParams, LLM", ["SamplingParams", "LLM"]))
patch("src/embedding_model/GritLM.py", "from gritlm import GritLM",
      lazy("from gritlm import GritLM", ["GritLM"]))
patch("src/embedding_model/Cohere.py", "import boto3\nfrom botocore.exceptions import ClientError",
      "try:\n    import boto3\n    from botocore.exceptions import ClientError\n"
      "except ImportError:  # optional dependency (colab patch)\n    boto3 = None; ClientError = Exception")
patch("src/llm/bedrock_llm.py", "import litellm\n",
      lazy("import litellm", ["litellm"]) + "\n")

print("Registering the SentenceTransformer embedding class...")
patch("src/embedding_model/__init__.py",
      "from .Cohere import CohereEmbeddingModel\n",
      "from .Cohere import CohereEmbeddingModel\n"
      "from .SentenceTransformerModel import SentenceTransformerEmbeddingModel\n")
patch("src/embedding_model/__init__.py",
      '    assert False, f"Unknown embedding model name: {embedding_model_name}"',
      "    elif any(k in embedding_model_name.lower() for k in "
      "(\"bge\", \"e5\", \"minilm\", \"mpnet\", \"sentence-transformers\", \"vietnamese\")):\n"
      "        return SentenceTransformerEmbeddingModel\n"
      '    assert False, f"Unknown embedding model name: {embedding_model_name}"',
      marker="return SentenceTransformerEmbeddingModel")

st_path = os.path.join(ROOT, "src", "embedding_model", "SentenceTransformerModel.py")
with open(st_path, "w", encoding="utf-8") as f:
    f.write('''"""SentenceTransformer embedding backend (added by patch_t2rag.py for Colab)."""
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.embedding_model_name, device=device, trust_remote_code=True)
        if device == "cuda":
            self.model.half()
        # Short triplets and passages: 512 tokens is plenty and keeps a T4 fast.
        self.model.max_seq_length = min(self.model.max_seq_length or 512, 512)
        # e5 models expect "query: " / "passage: " prefixes.
        self.is_e5 = "e5" in self.embedding_model_name.lower()
        self.batch_size = 128 if device == "cuda" else 32
        self.embedding_config = EmbeddingConfig.from_dict({
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "encode_params": {"batch_size": self.batch_size},
        })
        logger.info(f"Loaded {self.embedding_model_name} on {device}")

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if t and t.strip() else " " for t in texts]
        if self.is_e5:
            prefix = "query: " if kwargs.get("instruction") else "passage: "
            texts = [prefix + t for t in texts]
        norm = kwargs.get("norm", self.embedding_config.norm)
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=bool(norm),
                                convert_to_numpy=True, show_progress_bar=len(texts) > 2000)
        return np.asarray(emb, dtype=np.float32)
''')
print("  wrote: src/embedding_model/SentenceTransformerModel.py")

print("Adding exponential backoff for rate limits...")
patch("src/llm/openai_gpt.py",
      "from tenacity import retry, stop_after_attempt, wait_fixed",
      "from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential")
patch("src/llm/openai_gpt.py",
      "dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))",
      "dynamic_retry = retry(stop=stop_after_attempt(max_retries), "
      "wait=wait_exponential(multiplier=2, min=5, max=120))")
patch("src/llm/openai_gpt.py",
      'self.max_retries = kwargs.get("max_retries", 2)',
      'self.max_retries = kwargs.get("max_retries", getattr(global_config, "max_retry_attempts", 2))')
patch("src/llm/openai_gpt.py",
      "http_client=client, max_retries=self.max_retries)",
      "http_client=client, max_retries=2)")

print("Vietnamese support (only active when T2RAG_LANG=vi; English runs are unchanged)...")
# EM/F1: the repo drops "a/an/the" (but "an" is a Vietnamese syllable: Tràng An, Lê Văn An),
# removes only ASCII punctuation and does not NFC-normalise.
patch("src/utils/eval_utils.py",
      "import re\nimport string\n",
      "import os\nimport re\nimport string\nimport unicodedata\n",
      marker="import unicodedata\n")
patch("src/utils/eval_utils.py",
      "    return white_space_fix(remove_articles(remove_punc(lower(answer))))",
      "    if os.environ.get(\"T2RAG_LANG\", \"en\").lower() == \"vi\":  # colab patch: Vietnamese answers\n"
      "        text = unicodedata.normalize(\"NFC\", answer).lower()\n"
      "        text = \"\".join(\" \" if unicodedata.category(ch).startswith(\"P\") else ch for ch in text)\n"
      "        return white_space_fix(text)\n"
      "    return white_space_fix(remove_articles(remove_punc(lower(answer))))",
      marker="# colab patch: Vietnamese answers")
# HippoRAG graph keys: text_processing keeps only [A-Za-z0-9], so "Trần Hưng Đạo" became "tr n h ng o".
patch("src/utils/misc_utils.py",
      "import numpy as np\nimport re\n",
      "import numpy as np\nimport os\nimport re\nimport unicodedata\n",
      marker="import unicodedata\n")
patch("src/utils/misc_utils.py",
      "    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()",
      "    if os.environ.get(\"T2RAG_LANG\", \"en\").lower() == \"vi\":  # colab patch: keep Vietnamese letters\n"
      "        text = unicodedata.normalize(\"NFC\", text).lower()\n"
      "        return \" \".join(re.sub(r'[^\\w ]', ' ', text).replace('_', ' ').split())\n"
      "    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()",
      marker="# colab patch: keep Vietnamese letters")

# HippoRAG skips synonym search for entities with <= 2 ASCII letters ("Hồ", "Lý" have only one).
patch("src/HippoRAG.py",
      "            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:",
      "            if len(re.sub(r'[\\W_]' if os.environ.get('T2RAG_LANG', 'en').lower() == 'vi' "
      "else '[^A-Za-z0-9]', '', entity)) > 2:  # colab patch: Vietnamese letters count",
      marker="# colab patch: Vietnamese letters count")

print("Done.")
