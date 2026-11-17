# T²RAG cho câu hỏi lịch sử Việt Nam (NCKH)

Đề tài: **Ứng dụng và cải thiện T²RAG cho hỗ trợ tìm kiếm, trả lời thông tin lịch sử**
(phạm vi dữ liệu: Lý – Trần – Hồ – Lê sơ, 1009–1527).

Repo này chứa **code nhóm viết thêm** trên nền phương pháp T²RAG:

- Bài báo: Gong et al., *Beyond Chunks and Graphs: Retrieval-Augmented Generation through Triplet-Driven Thinking*,
  Findings of ACL 2026 (arXiv: https://arxiv.org/abs/2508.02435).
- Code gốc của tác giả (giấy phép MIT): https://github.com/Emory-Melody/T2RAG, phiên bản dùng trong phân tích: commit `72c43bd`.
  Repo này **không chép lại** code gốc; khi chạy, code nhóm được chép đè vào bản code gốc.

## Nội dung

| File / thư mục | Nội dung |
|---|---|
| `src/TRAG_plus.py` | T²RAG cải tiến (lớp con của `TRAG`, không sửa code gốc). Bật từng phần bằng `--fixes`: `b` (sửa lỗi code gốc L2, L4, L5, L6), `s1` (giữ bộ ba chưa giải + bảng biến), `s2` (tìm kiếm lai BM25 + vector), `s3` (trả lời có bằng chứng), `s4` (kiểm chứng giá trị), `s5` (bảo vệ câu trả lời: lấy đáp án ngắn sau "Answer:", hỏi lại một lần khi model trả về đoạn suy luận hoặc "NOT FOUND"; mặc định tắt) |
| `run_colab.py` | Chạy một phương pháp (`trag`, `trag_plus`, `hippo`, `standard`, `nor` = không truy hồi) trên một phần bộ dữ liệu; lưu EM/F1, EM/F1 nới (`EM_lenient`), Recall@5, điểm theo số bước, số lần gọi LLM và token từng lượt gọi, MD5 dữ liệu, vết suy luận. `--answer_format short`: yêu cầu đáp án ngắn (tiếng Việt khi `--lang vi`); `--cache_tag`: cache LLM riêng cho từng lượt chạy |
| `merge_results.py` | Gộp các lô của cùng một cấu hình (`--start` / `--limit`) thành một file kết quả |
| `rescore.py` | Chấm lại kết quả đã có theo file câu hỏi mới (ví dụ `vnhistory.json` bản 1.1), không gọi model |
| `compare_methods.py` | So sánh hai lượt chạy theo từng câu và theo số bước |
| `make_table.py` | Bảng kết quả kiểu Table 1 của bài báo (EM / F1 theo số bước, kiểm định bootstrap) và bảng chi phí; `--metric lenient` cho EM/F1 nới |
| `error_analysis.py` | Gán nhãn lỗi tự động cho câu sai từ vết suy luận; so sánh trước / sau |
| `check_vnhistory.py` | Kiểm tra định dạng bộ dữ liệu lịch sử trước khi chạy |
| `patch_t2rag.py` | Vá code gốc để chạy trên Google Colab (không đổi phương pháp) |
| `tests/` | 53 bài test dùng LLM giả (không tốn API) |
| `notebooks/T2RAG_Colab_v6.ipynb` | Notebook Colab chạy toàn bộ thí nghiệm (sinh bởi `build_notebook_v6.py`): lấy code bằng `git clone`, gọi LLM qua `data_vi/gw_proxy.py` hoặc thẳng Google, chạy theo lô. Bản cũ: `T2RAG_Colab_v2.ipynb` |
| `docs/` | Hướng dẫn chạy và kế hoạch thí nghiệm |
| `data_vi/`, `reproduce/dataset/vnhistory_corpus.json` | Bộ dữ liệu lịch sử Việt Nam: kho 896 đoạn, bộ ba OpenIE, script dựng dữ liệu (xem `data_vi/README.md`) |

## Chạy trên Google Colab

1. Mở `notebooks/T2RAG_Colab_v6.ipynb` trên Colab, chọn GPU T4. Không cần tải zip lên Drive: ô cài đặt tự `git clone` repo này.
2. Thêm secret `GWAI_API_KEY` (gateway; có thể thêm `GWAI_API_KEY_2`…) hoặc `GOOGLE_API_KEY`, bật *Notebook access*. **Không ghi API key vào code.**
3. Chạy Phần 0 (cài đặt; ô kiểm tra phải in `OK`; bật gateway; chọn model `gemini-3.8-flash`), rồi các phần sau theo kế hoạch.

Kiểm tra code trên máy (cần bản code gốc đã vá và các thư viện của nó):

```bash
git clone https://github.com/Emory-Melody/T2RAG.git && cd T2RAG
cp -r <repo này>/* .
python patch_t2rag.py
python -m unittest discover tests
```

## Trạng thái

- Code v6 qua 53/53 test bằng LLM giả.
- Đã có: Standard RAG, HippoRAG2, T²RAG gốc trên `vnhistory` (dev 50 và test 116 câu, 2 lượt, `gemini-3.8-flash`).
- Đang làm: GĐ1 (T²RAG so với HippoRAG2 trên MuSiQue), bộ tiếng Anh, ablation T²RAG+ trên `vnhistory`.
