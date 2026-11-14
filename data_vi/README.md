# vnhistory dataset (data session)

Vietnamese history corpus (Lý – Trần – Hồ – Lê sơ, 1009–1527) and its OpenIE triples, built to the file contract in `KE_HOACH_CHIA_VIEC_2_PHIEN.md`, section 2.

## Files

| Path | Content |
|---|---|
| `reproduce/dataset/vnhistory_corpus.json` | 896 passages from 158 Vietnamese Wikipedia articles, `[{"title", "text"}]`, NFC, mean 136 syllables |
| `data_vi/release/openie_results_ner.json` | OpenIE for all 896 passages: 12,015 entities, 10,815 triples (12.1 per passage). MD5 `f6ff5da78046518fc3458c495fd2895a` |
| `data_vi/work/chunks_meta.jsonl` | Per-passage metadata: short id (`c0000`–`c0895`), section, source page and revision |
| `data_vi/articles.txt` | The 158 article titles |
| `reproduce/dataset/vnhistory.json` | 166 reviewed questions (58 one-hop, 73 two-hop, 35 three-hop), MuSiQue layout; dev = first 50 (15/23/12), test = 116 (43/50/23). MD5 `b07cfa13dc9abc164da85f5b201471f5` |
| `data_vi/*.py` | Pipeline scripts (below) |

Questions: 178 drafts were written in three batches and reviewed by hand against criteria K1–K9; 166 were kept and 12 were rejected (ambiguous answer, a step that one passage alone answers, or a bridge name missing from the next passage). Five pairs of questions reach the same answer in the same final passage through different reasoning chains (d013/d152, d042/d166, d056/d145, d059/d167, d121/d164); both questions of each pair are kept.

## Use with the T²RAG code

```bash
mkdir -p outputs/vnhistory_univ
cp data_vi/release/openie_results_ner.json outputs/vnhistory_univ/
python check_vnhistory.py          # corpus, OpenIE and questions: all pass
```

`HippoRAG.index` re-extracts (with English prompts) any passage missing from this file and overwrites it. Check the MD5 after a run; copy the release file back if it changed.

## OpenIE details

- Model `gemini-3.8-flash` through a reseller gateway (`https://1gw.gwai.cloud`, Anthropic Messages format), default thinking, temperature 0 when accepted.
- Two calls per passage as in the paper (NER, then triples), with Vietnamese prompts `prompts_vi.py` version `v2` (one-shot example; rules: no Han characters, no generic titles as entities, keep ages and durations).
- Each doc has `idx`, `title`, `text`, `passage`, `extracted_entities`, `extracted_triples`; `idx = "chunk-" + md5(title + "\n" + text)`.
- Quality on the full run: 0 parse failures, 0 triples with Han characters, 2 triples whose subject and object are both absent from the passage.
- Cost: about 27 USD by the script estimate (1,817 calls, 33 transient errors retried), 162 minutes with 3 workers.

## Rebuild

Run from the repo root. API keys go in a `.env` file (`GWAI_API_KEY=...`, `GWAI_API_KEY_2=...`); never commit it.

```bash
python data_vi/fetch_articles.py probe       # resolve titles, aliases and links (MediaWiki API)
python data_vi/fetch_articles.py download    # raw extracts to data_vi/work/raw/
python data_vi/build_corpus.py --max_per_article 6
python data_vi/validate_dataset.py --corpus
python data_vi/openie_vi.py run --all        # cached and resumable; rotates keys by budget
python data_vi/openie_vi.py build            # writes outputs/vnhistory_univ/openie_results_ner.json
python data_vi/validate_dataset.py --openie
```

`find_chains.py` and `build_questions.py` support writing multi-hop questions (bridge candidates, pre-checks, review sheets). Drafts live in `data_vi/work/questions_draft.jsonl` (not released).

```bash
python data_vi/build_questions.py --precheck        # automatic checks on all drafts
python data_vi/build_questions.py --build           # writes reproduce/dataset/vnhistory.json (reviewed drafts only)
python data_vi/validate_dataset.py --questions      # contract 2.3, criteria K3-K6, K9, dev/test split
```

Question file layout:

- Only drafts with `status=ok` are exported; `--build` refuses to run while some drafts are unreviewed (`--allow_pending` for a trial build).
- The first 50 questions are dev (15 one-hop, 23 two-hop, 12 three-hop, interleaved), the rest is test. Dev and test never share the passage that holds the final answer.
- Supporting paragraphs come first, in step order, copied verbatim from the corpus. `--distractors N` adds N non-supporting paragraphs (most word overlap with the question); default 0.
- Extra fields beyond the contract: `split`, `qtype` (`single` / `chain`), `draft_id`, `idx` in each paragraph, `paragraph_support_idx` in each decomposition step.

## Notes for the code session

- `normalize_answer` drops "an" as an English article; it is a common Vietnamese syllable.
- No NFC normalization before comparing answers.
- HippoRAG `text_processing` keeps only ASCII characters, which breaks Vietnamese entity matching.
- `run_colab.py` needs an Anthropic-format client to use the same gateway (it rejects `/v1/chat/completions`).
