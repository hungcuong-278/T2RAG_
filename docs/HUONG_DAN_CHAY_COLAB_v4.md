# HƯỚNG DẪN CHẠY TRÊN COLAB: code v4 (GĐ0, GĐ2, GĐ3, GĐ4 đã xong)

## 0. Có gì mới so với v3

| File | Việc | Trạng thái |
|---|---|---|
| `make_table.py` (**mới**) | Gom các file kết quả thành bảng kiểu Table 1 của bài báo: ô "EM / F1", tách theo số bước, **in đậm** số tốt nhất, **†** khi hơn T²RAG gốc có ý nghĩa (bootstrap ghép cặp 5.000 lần, p < 0,05). Có thêm bảng chi phí: số lần gọi LLM/câu, token, số vòng, Recall@5, giây/câu. | Test đạt |
| `error_analysis.py` (**mới**) | Đọc vết suy luận và tự gán nhãn lỗi cho từng câu sai. Có so sánh trước/sau (`--compare`) và xuất mẫu để kiểm tay (`--sample 30`). | Test đạt |
| `check_vnhistory.py` (**mới**) | Kiểm tra 3 file dữ liệu lịch sử của phiên kia có đúng hợp đồng không, **trước khi tốn API**. | Test đạt |
| `build_notebook_v2.py` → `T2RAG_Colab_v2.ipynb` (**mới**) | Một notebook cho mọi giai đoạn: cài đặt → GĐ1 → ablation → tập báo cáo → bảng → `vnhistory`. | Đã sinh |
| `run_colab.py` | Thêm trường `tag` vào `summary`, để các lượt chỉ khác `--tag` (ví dụ ngưỡng S4) thành các hàng riêng trong bảng | Test đạt |
| `src/TRAG_plus.py`, `compare_methods.py`, `patch_t2rag.py` | Không đổi | |

**Kiểm tra:** giải nén zip vào một bản repo gốc sạch, chạy `patch_t2rag.py`, rồi chạy test: **29/29 test đạt** (dùng LLM giả, không tốn API).

**Chưa chạy với Gemini thật.** Nhánh `hippo` vẫn chưa chạy lần nào.

## 1. Cách dùng (khuyến nghị: notebook)

1. Tải **`t2rag_code_v4.zip`** lên Drive, vào thư mục `MyDrive/T2RAG_code/`.
2. Mở **`T2RAG_Colab_v2.ipynb`** trên Colab (*File → Upload notebook*), chọn GPU T4, thêm secret `GOOGLE_API_KEY`.
3. Chạy lần lượt từng phần. Notebook đã ghi sẵn khi nào cần dán kết quả cho mình.

Nếu bạn vẫn dùng notebook cũ: thay zip v3 bằng v4 ở lệnh `unzip`, rồi chạy thêm các lệnh ở mục 2.

## 2. Các lệnh mới

```bash
# Bảng kết quả tập chỉnh (mọi lượt 100 câu MuSiQue)
python make_table.py --glob "outputs/results/musique_*_100q*.json" --baseline trag --out_dir tables/dev

# Bảng cuối trên tập báo cáo, đổi tên hàng bằng labels.json
python make_table.py --glob "outputs/results/musique_*_from200_200q.json" --baseline trag --labels labels.json --out_dir tables/test

# Phân tích lỗi một lượt chạy, so sánh trước/sau, xuất 30 câu để kiểm tay
python error_analysis.py outputs/results/musique_trag_gemini-3.5-flash_100q.json \
       --compare outputs/results/musique_trag_plus_fx-b-s1-s3_gemini-3.5-flash_100q.json --sample 30

# Kiểm tra dữ liệu lịch sử (sau khi chép 3 file của phiên dữ liệu vào repo)
python check_vnhistory.py
```

`labels.json` ví dụ:
```json
{"standard": "Standard RAG", "hippo": "HippoRAG2", "trag": "T²RAG (gốc)",
 "trag_plus_fx-b-s1": "T²RAG + sửa lỗi + S1", "trag_plus_fx-b-s1-s2-s3": "T²RAG+"}
```

## 3. Nhãn lỗi của `error_analysis.py`

Một câu tính là **đúng** khi EM = 1 hoặc F1 ≥ 0,5. Câu sai nhận **nhãn đầu tiên khớp** theo thứ tự:

| Thứ tự | Nhãn | Nghĩa |
|---|---|---|
| 1 | `sai_ten` | Trả lời đúng người nhưng khác tên. Chỉ có khi truyền file `--aliases` (cần cho dữ liệu Lý – Trần: Trần Quốc Tuấn = Trần Hưng Đạo). |
| 2 | `tra_loi_cuoi_sai` | Đáp án đúng **đã nằm trong bộ ba hoặc bằng chứng** nhưng câu trả lời cuối vẫn sai |
| 3 | `thieu_doan_vang` | Có ít nhất một đoạn vàng **không được lấy** ở vòng nào |
| 4 | `roi_mat_xich` | Bộ ba bị bỏ, hoặc còn ô trống chưa giải lúc kết thúc (lỗi L1 của bản gốc) |
| 5 | `dien_sai_buoc_giua` | Đã đủ đoạn vàng, chuỗi đầy đủ, nhưng thiếu đáp án trung gian (MuSiQue có sẵn `question_decomposition`) |
| 6 | `khac` | Các trường hợp còn lại |

"Trả lời cuối sai" được xét **trước** "thiếu đoạn vàng". Lý do: nếu bộ ba đã chứa đáp án thì lỗi nằm ở bước trả lời, không phải ở bước tìm kiếm. Đây là thay đổi so với thứ tự trong kế hoạch ban đầu.

Đây là nhãn tự động, nên phải **kiểm tay 30 câu** (`--sample 30`, điền cột `nhan_tay`) và ghi tỷ lệ khớp vào báo cáo.

## 4. Thứ tự việc bạn cần chạy (không đổi)

1. **GĐ1:** notebook phần 1. Chạy thử hippo 20 câu trước; nếu lỗi thì dán log.
2. **GĐ2:** notebook phần 2, rồi dán bảng `tables/dev/table_main.md` và `table_cost.md`.
3. Mình chọn cấu hình tốt nhất cùng bạn, sau đó mới chạy phần 3 (tập báo cáo) **một lần**.
