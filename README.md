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
| `src/TRAG_plus.py` | T²RAG cải tiến (lớp con của `TRAG`, không sửa code gốc). Bật từng phần bằng `--fixes`: `b` (sửa lỗi code gốc L2, L4, L5, L6), `s1` (giữ bộ ba chưa giải + bảng biến), `s2` (tìm kiếm lai BM25 + vector), `s3` (trả lời có bằng chứng), `s4` (kiểm chứng giá trị) |
| `run_colab.py` | Chạy một phương pháp (`trag`, `trag_plus`, `hippo`, `standard`) trên một phần bộ dữ liệu; lưu EM/F1/Recall, số lần gọi LLM, vết suy luận |
| `compare_methods.py` | So sánh hai lượt chạy theo từng câu và theo số bước |
| `make_table.py` | Bảng kết quả kiểu Table 1 của bài báo (EM / F1 theo số bước, kiểm định bootstrap) và bảng chi phí |
| `error_analysis.py` | Gán nhãn lỗi tự động cho câu sai từ vết suy luận; so sánh trước / sau |
| `check_vnhistory.py` | Kiểm tra định dạng bộ dữ liệu lịch sử trước khi chạy |
| `patch_t2rag.py` | Vá code gốc để chạy trên Google Colab (không đổi phương pháp) |
| `tests/` | 29 bài test dùng LLM giả (không tốn API) |
| `notebooks/T2RAG_Colab_v2.ipynb` | Notebook Colab chạy toàn bộ thí nghiệm (sinh bởi `build_notebook_v2.py`) |
| `docs/` | Hướng dẫn chạy và kế hoạch thí nghiệm |

## Chạy trên Google Colab

Xem chi tiết trong `docs/HUONG_DAN_CHAY_COLAB_v4.md`. Tóm tắt:

1. Nén các file code của repo này thành `t2rag_code_v4.zip` (hoặc lấy bản zip của nhóm) và tải lên Google Drive vào `MyDrive/T2RAG_code/`.
2. Mở `notebooks/T2RAG_Colab_v2.ipynb` trên Colab, chọn GPU T4.
3. Thêm secret `GOOGLE_API_KEY` (key Gemini của bạn) và bật *Notebook access*. **Không ghi API key vào code.**
4. Chạy Phần 0 (cài đặt; ô kiểm tra phải in `OK`), rồi các phần sau theo kế hoạch.

Kiểm tra code trên máy (cần bản code gốc đã vá và các thư viện của nó):

```bash
git clone https://github.com/Emory-Melody/T2RAG.git && cd T2RAG
cp -r <repo này>/* .
python patch_t2rag.py
python -m unittest discover tests
```

## Trạng thái

- Code đã qua 29/29 test bằng LLM giả; chưa có kết quả chạy với Gemini thật.
- Đang làm: GĐ1 (so sánh T²RAG với HippoRAG2 trên MuSiQue) và ablation S1–S4.
