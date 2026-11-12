# KẾ HOẠCH CHI TIẾT: PHIÊN "CODE VÀ THÍ NGHIỆM" (NCKH T²RAG lịch sử Việt Nam)

> Tài liệu làm việc của phiên này. Mỗi việc ghi rõ: **mục tiêu, file, thiết kế đến từng hàm, cách kiểm tra, việc bạn cần chạy trên Colab, tiêu chí xong**.
> Đọc kèm: `BAN_GIAO_T2RAG.md` (bản đồ code, lỗi L1–L8) và `KE_HOACH_CHIA_VIEC_2_PHIEN.md` (hợp đồng dữ liệu với phiên kia).
> Lập ngày 12/11/2026. Số dòng code theo repo gốc, commit `72c43bd`.

---

## 0. Tóm tắt một trang

**Mục tiêu của phiên:** có một bộ code chạy được trên Colab, gồm:
1. **T²RAG cải tiến** (`TRAG_plus.py`): bật hoặc tắt từng sửa đổi S1–S4 và sửa lỗi, không đụng vào code gốc.
2. **Đo lường đầy đủ:** EM/F1/Recall theo số bước, số lần gọi LLM và token mỗi câu, p-value bootstrap.
3. **Bảng kết quả kiểu Table 1** và **bảng phân loại lỗi**, sinh tự động từ file kết quả.
4. **Số liệu để quyết định có làm Hướng 2 (đồ thị) hay không.**

**Thứ tự làm:**

| Giai đoạn | Việc | Ai làm | Ra gì |
|---|---|---|---|
| **GĐ0** Hạ tầng | 0.1 thêm `--start`, đếm LLM, ghi `trace`; 0.2 bộ test giả | Phiên này | `run_colab.py` v3, `tests/` |
| **GĐ1** Kiểm chứng Hướng 2 | Smoke test hippo → 200 câu MuSiQue trag và hippo → so sánh | **Bạn chạy Colab**, phiên này sửa lỗi | Quyết định H2: làm / không làm |
| **GĐ2** T²RAG cải tiến | `TRAG_plus.py`: B (lỗi) → S1 → S3 → S2 → S4 | Phiên này viết, bạn chạy tập chỉnh | Cấu hình tốt nhất |
| **GĐ3** Công cụ báo cáo | `make_table.py`, `error_analysis.py` | Phiên này | Bảng A, bảng lỗi |
| **GĐ4** Notebook | Cập nhật `build_notebook.py` | Phiên này | `T2RAG_Colab.ipynb` v2 |
| **GĐ5** (tùy chọn) H2a | Đồ thị khi bế tắc | Phiên này, nếu GĐ1 ủng hộ | `--fixes h2a` |
| **GĐ6** Gộp | `check_vnhistory.py`, chạy trên `vnhistory`, bàn giao v2 | Cả hai | Bảng B, `BAN_GIAO_T2RAG_v2.md` |

GĐ1 (bạn chạy) và GĐ2 (mình viết code) **làm song song được**.

---

## 1. Nguyên tắc làm việc

1. **Không sửa `src/TRAG.py`.** Mọi thay đổi nằm trong lớp con `TRAGPlus(TRAG)` ở `src/TRAG_plus.py`. Nhờ vậy:
   - T²RAG gốc luôn chạy được để làm baseline;
   - mỗi sửa đổi bật/tắt bằng cờ `--fixes`, nên làm ablation dễ.
2. **Không sửa dữ liệu** (`reproduce/dataset/vnhistory*`, `data_vi/`). Đó là việc của phiên kia.
3. **Code kiểm tra trước trong sandbox** bằng LLM giả và embedding giả, vì sandbox không có mạng và không cài được gói. Chạy thật luôn làm trên **Colab của bạn**.
4. **Mỗi sửa đổi chỉ giữ lại khi tăng điểm trên tập chỉnh.** Số cuối cùng chỉ lấy trên tập báo cáo, chạy **một lần**.
5. Mỗi bước xong thì cập nhật mục 9 (nhật ký) của file này.

---

## 2. Dữ liệu và ngân sách chạy

### 2.1 Chia tập MuSiQue (tiếng Anh, có nhãn 2/3/4 bước)

| Tập | Câu | Dùng cho | Số câu theo bước |
|---|---|---|---|
| **GĐ1 so sánh** | 0–199 | trag và hippo (kiểm chứng H2) | 103 / 58 / 39 |
| **Tập chỉnh (dev)** | 0–99 | Chọn S nào giữ, chỉnh ngưỡng S4 | đếm sau |
| **Tập báo cáo (test)** | 200–399 | Bảng A cuối cùng, chạy một lần | đếm sau |

Cần thêm tham số `--start` (việc 0.1) để chạy được đoạn 200–399.

Với `vnhistory` (sau khi gộp): theo hợp đồng, 50 câu đầu là dev, phần còn lại là test.

### 2.2 Ước lượng số lần gọi LLM (để tính quota Gemini)

| Phương pháp | Gọi LLM mỗi câu | 200 câu |
|---|---|---|
| T²RAG gốc | 1 phân rã + ≤3 điền + 1 trả lời ≈ **3–5** | ~800 |
| HippoRAG2 | 1 lọc sự thật + 1 trả lời ≈ **2** | ~400 |
| T²RAG + S1 (tối đa 5 vòng) | ≈ 4–7 | ~1.100 |
| S2, S3, S4 | **không thêm lần gọi** (chỉ thêm token) | như trên |

- Cache của repo giúp **chạy lại không tốn thêm**: câu nào đã có đáp án trong cache thì không gọi lại.
- Tính cả ablation, tổng cộng khoảng **8.000–10.000 lần gọi**. **Bạn cần kiểm tra quota/ngày của gemini-3.5-flash** trong Google AI Studio. Nếu thiếu, giảm tập báo cáo xuống 150 câu, hoặc chạy rải ra nhiều ngày (dùng `--min_interval`).

---

## 3. GĐ0: Hạ tầng

### Việc 0.1: `run_colab.py` v3
| Thay đổi | Chi tiết |
|---|---|
| `--start S` | Lấy `samples[S : S+limit]`. Tên file kết quả thêm `_from{S}`. `qid` vẫn lấy từ id gốc, nên so sánh giữa các lần chạy vẫn khớp. |
| `--method trag_plus --fixes b,s1,s2,s3,s4` | Tạo `TRAGPlus(global_config, max_qa_steps, fixes=set(...))`. Tên file thêm `_fx-b-s1...`; `summary` thêm trường `fixes`. |
| **Đếm LLM** (N4) | Bọc `rag.llm_model.infer`: mỗi lần gọi ghi `calls += 1`, `cache_hits += (kết quả[2] is True)`, cộng `prompt_tokens` và `completion_tokens` từ `metadata`. Với trag/trag_plus, bộ đếm reset đầu mỗi `trag_qa`, nên có số **theo từng câu**. Với hippo, `rag_qa` chạy theo lô, nên chỉ có **tổng và trung bình**. |
| Trường mới trong `records` | `llm_calls`, `prompt_tokens`, `completion_tokens`, `cache_hits`, `iterations`, `completion_status`, `n_dropped_clues` (lấy từ kết quả `trag_qa`). |
| Trường mới trong `summary` | `avg_llm_calls`, `avg_prompt_tokens`, `avg_completion_tokens`, `avg_iterations`, `pct_all_resolved`, `start`, `fixes`. |
| Thư mục vết | Mỗi cấu hình ghi `intermediate_results` vào thư mục riêng (`outputs/<ds>_<method>_<fixes>/...`) để `error_analysis.py` đọc được. |
| Map `query_i` → `qid` | `rag_qa` đặt tên file vết theo thứ tự trong lượt chạy (`query_0`…). Ghi thêm `trace_file` vào mỗi record để tra ngược. |

**Kiểm tra:** chạy giả bằng `tests/fake_env.py` trên 3 câu; file JSON phải có đủ trường mới.
**Xong khi:** chạy giả qua được cả ba nhánh `trag`, `trag_plus` và `standard`.

### Việc 0.2: bộ test giả (`tests/`)
- **`tests/fake_env.py`:**
  - `FakeLLM`: trả lời theo kịch bản dạng dict "chuỗi khóa trong prompt → đáp án", có trả `metadata` token giả.
  - `FakeEmbedding`: vector băm từ token (hashing trick, 256 chiều, chuẩn hóa), không cần tải model.
  - Kho văn bản 6 đoạn và OpenIE giả cho chuỗi 3 bước: "Trận Bạch Đằng 1288 → Trần Hưng Đạo → cha là Trần Liễu".
- **`tests/test_trag_plus.py`**, mỗi sửa đổi có ít nhất một test:
  - bộ ba mờ **không bị xóa** sau vòng 1 (S1);
  - biến `?A` được thay vào bộ ba sau (S1);
  - mỗi bộ ba tìm được có ≥2 chunk riêng (S2);
  - `final_qa` nhận được câu bằng chứng (S3);
  - giá trị không có bằng chứng thì **không bị khóa** (S4);
  - các lỗi L4/L5/L6 (nhóm B).
- Chạy bằng `python -m unittest discover tests -v`. Không cần mạng.

---

## 4. GĐ1: Kiểm chứng Hướng 2 (T²RAG so với HippoRAG2)

| Bước | Bạn chạy trên Colab | Phiên này làm |
|---|---|---|
| 1.1 | `python run_colab.py --dataset musique --method hippo --limit 20 --llm_name gemini-3.5-flash` | Đọc lỗi bạn dán và sửa `run_colab.py`/config. Hay lỗi ở: `python-igraph`, trường config HippoRAG cần mà chưa có, đường dẫn `embedding_base_dir`. |
| 1.2 | `... --method trag --limit 200` và `... --method hippo --limit 200` | |
| 1.3 | `python compare_methods.py <trag.json> <hippo.json>` | Đọc file `only_hippo_correct_vs_trag.csv` để xem vì sao đồ thị đúng: tìm thấy chunk mà T²RAG không thấy, hay T²RAG bị mất mắt xích. |

**Quy tắc quyết định:**
- **Làm H2a** nếu **EM trần − EM T²RAG ≥ 3 điểm** ở nhóm 3–4 bước **và** trong các câu "chỉ hippo đúng" có ≥ 1/3 câu T²RAG **không tìm thấy chunk vàng**. Khi đó đồ thị thực sự bù được phần tìm kiếm.
- Nếu không đạt thì **bỏ H2**. Trong báo cáo ghi: "đã kiểm chứng, đồ thị không bù được điểm yếu của T²RAG trên dữ liệu này", kèm bảng. Đây vẫn là một kết quả có giá trị.

**Xong khi:** có bảng `compare_methods` (tổng và theo 2/3/4 bước) và một đoạn kết luận ghi vào mục 9.

---

## 5. GĐ2: `src/TRAG_plus.py`, T²RAG cải tiến

### 5.1 Khung lớp
```python
class TRAGPlus(TRAG):
    def __init__(self, global_config, max_qa_steps=3, fixes=frozenset(), s1_max_steps=5,
                 s2_bm25_weight=1.0, s2_min_per_clue=2, s4_match_threshold=0.8): ...
    # ghi đè:
    _prepare_proposition_embeddings   # L5 (b)
    reason_and_form_triples           # S1: biến có tên
    retrieve_and_double_check         # L6 (b), S2
    resolve_clues                     # S1 (bảng biến, giữ phần chưa giải), S4 (bằng chứng)
    has_unresolved_clues              # L4 (b)
    final_qa                          # S3
    trag_qa                           # vòng lặp mới: L2 (b), S1, ghi thêm vết
```
Cờ `fixes` gồm: `b` (sửa lỗi), `s1`, `s2`, `s3`, `s4`, `h2a`. Cờ nào tắt thì gọi lại hàm gốc bằng `super()`, để kết quả **trùng hệ thống gốc**. Có test kiểm điều này.

### 5.2 Nhóm B: sửa lỗi code (làm đầu tiên vì rẻ và không đổi prompt)
| Lỗi | Sửa thế nào |
|---|---|
| **L5**: mệnh đề trùng bị ghi đè (dòng 206) | Thêm mảng song song `self.prop_sources[i]` = passage của **lần xuất hiện thứ i** (vì `all_propositions` đã giữ mọi lần xuất hiện, thẳng hàng với ma trận embedding). Khi tìm, lấy passage theo chỉ số `idx` thay vì tra dict theo chuỗi. |
| **L6**: xóa từ để hỏi theo chuỗi con (dòng 481–484) | `re.sub(r"\b(what\|who\|whom\|whose\|where\|when\|why\|how\|which)\b", " ", q, flags=re.I)`. Giữ nguyên chữ hoa (bản gốc hạ hết chữ thường). |
| **L4**: đếm `?` theo ký tự (dòng 1064, ENFORCE) | Hàm `count_placeholders(clue)`: đếm các **trường** có giá trị khớp `^\?[A-Za-z0-9_]*$`, hoặc chứa token `?X` đứng riêng. Dấu `?` trong tên thật (ví dụ tên phim) không bị đếm. |
| **L2**: ENFORCE bỏ bộ ba ≥2 `?` (dòng 1236–1245) | Bộ ba ≥2 ô trống chuyển về danh sách **mờ** thay vì bỏ. |

**Kỳ vọng:** thay đổi nhỏ, chủ yếu là sửa đúng. Vẫn phải chạy dev để chắc không làm giảm điểm.

### 5.3 S1: bảng biến và giữ bộ ba chưa giải (sửa L1, đích chính cho câu 3–4 bước)
1. **Prompt phân rã** (ghi đè `reason_and_form_triples`): bắt LLM đặt tên cho mọi ô trống (`?A`, `?B`, …). Ví dụ trong prompt chuyển sang dạng có tên, vẫn giữ định dạng `subject | predicate | object` để dùng lại parser gốc. Nếu LLM vẫn trả `?` vô danh, code tự đặt tên theo thứ tự.
2. **Bảng biến** `self._bindings = {"?A": "Trần Hưng Đạo"}` cho mỗi câu hỏi.
3. **Prompt điền** (ghi đè `resolve_clues`): giữ nguyên các khối `Fully Resolved Clue` / `Newly Traceable Clue`, thêm **một dòng mới** sau mỗi khối: `Binding: ?A = <giá trị>`. Parser gốc bỏ qua dòng lạ, nên vẫn an toàn; parser mới đọc thêm dòng `Binding:`.
4. **Code tự thay biến:** khi `?A` có giá trị, thay vào **mọi** bộ ba còn chứa `?A`, rồi phân loại lại bằng `count_placeholders`. Như vậy bộ ba mờ tự thành bộ ba tìm được, không phụ thuộc LLM có viết lại hay không.
5. **Giữ phần chưa giải:** `remaining = cũ − đã giải − đã nâng cấp`. Đối chiếu qua tên biến, nếu không có tên biến thì dùng `_clues_similar`. Làm đúng như Algorithm 1 dòng 13.
6. **Số vòng:** dừng khi hết ô trống, hoặc khi đến `s1_max_steps` (mặc định 5), hoặc khi **một vòng không giải thêm được gì** (tránh lặp vô ích, đỡ tốn LLM).
7. **Vết mới:** `bindings`, `dropped_clues` (bản gốc tính số bị bỏ để so sánh), `iterations`.

**Test:** chuỗi 3 bước giả. Bản gốc mất bộ ba thứ 3 sau vòng 1; bản S1 giữ được và giải được ở vòng 2–3.
**Kỳ vọng:** EM nhóm 3–4 bước tăng; số lần gọi LLM tăng nhẹ.

### 5.4 S3: trả lời có bằng chứng (rẻ, làm thứ hai)
- Mỗi bộ ba đã giải lưu thêm `evidence`. Nguồn: câu bằng chứng do LLM trích (nếu có S4); nếu không, lấy mệnh đề và passage có điểm cao nhất trong lần tìm đã dùng để giải nó.
- `final_qa` mới đọc: (a) danh sách bộ ba như cũ; (b) tối đa **3 passage bằng chứng** khác nhau, cắt mỗi passage ≤ 600 ký tự.
- Giữ nguyên yêu cầu định dạng đáp án ("One entity or fact only"), để EM vẫn so được với bản gốc.

**Test:** prompt `final_qa` chứa đoạn bằng chứng; khi tắt `s3` thì prompt trùng bản gốc.

### 5.5 S2: tìm kiếm lai và chia suất
1. **BM25 trên chunk,** tự viết bằng numpy (sandbox và Colab không cần thêm gói). Tách từ đơn giản: chữ thường, bỏ dấu câu. Tiếng Việt giữ nguyên dấu, tách theo khoảng trắng, chưa tách từ ghép.
2. **Mỗi bộ ba tìm được** tạo 2 danh sách: vector trên mệnh đề (như cũ, quy về chunk) và BM25 trên chunk (truy vấn = các phần đã biết của bộ ba). Gộp bằng **RRF**: `score = Σ 1/(60 + hạng)`.
3. **Chia suất:** mỗi bộ ba lấy trước `s2_min_per_clue` (mặc định 2) chunk tốt nhất của nó. Các suất còn lại xét chung (gồm cả truy vấn từ câu hỏi gốc) cho đến khi đủ `qa_top_k`. Nếu số bộ ba × 2 > k thì tăng k lên `min(2 × số bộ ba, 10)`. Số chunk thực tế ghi vào vết.
4. Định dạng trả về giữ như hàm gốc (`retrieved_passages`, `retrieved_propositions`), nên các bước sau không phải đổi.

**Test:** 2 bộ ba, một bộ ba "mạnh" chiếm hết top-5 ở bản gốc; ở bản S2, bộ ba còn lại có ≥2 chunk.
**Kỳ vọng:** `recall_first5_docs` tăng.

### 5.6 S4: kiểm chứng nhẹ trước khi khóa giá trị
1. Prompt điền thêm yêu cầu: sau mỗi `Fully Resolved Clue` có dòng `Evidence: "<câu trích nguyên văn từ Context Passages>"`, hoặc `Evidence: NOT FOUND`.
2. Code so khớp câu trích với các passage đã đưa vào prompt: chuẩn hóa (chữ thường, bỏ khoảng trắng thừa), rồi tính `difflib.SequenceMatcher` trên cửa sổ trượt **hoặc** tỷ lệ token chung ≥ `s4_match_threshold` (mặc định 0,8). Thêm điều kiện: giá trị điền vào phải xuất hiện trong câu trích (so mờ).
3. Không khớp hoặc `NOT FOUND`: **không khóa**, bộ ba giữ ô trống và được tìm lại ở vòng sau (cần S1). Ghi vào vết `rejected_by_s4`.
4. **Chỉnh ngưỡng trên dev:** thử 0,7 / 0,8 / 0,9. Đo tỷ lệ bị từ chối, và trên một mẫu nhỏ kiểm tay xem bao nhiêu câu bị từ chối oan.

**Rủi ro:** kiểm quá chặt làm mất giá trị đúng. Vì vậy S4 phải đứng sau S1 (có vòng sau để tìm lại).
**Không tốn thêm lần gọi LLM**, chỉ thêm token đầu ra.

### 5.7 Thứ tự chạy ablation trên tập dev (MuSiQue 0–99)
| Lượt | `--fixes` | Giữ lại nếu |
|---|---|---|
| A0 | (trag gốc) | baseline |
| A1 | `b` | EM không giảm quá 1 điểm |
| A2 | `b,s1` | EM 3–4 bước tăng |
| A3 | `b,s1,s3` | EM tăng hoặc không đổi |
| A4 | `b,s1,s3,s2` | Recall@5 tăng và EM không giảm |
| A5 | `b,s1,s3,s2,s4` (thử 3 ngưỡng) | EM tăng hoặc lỗi bịa giảm |

Cấu hình tốt nhất (gọi là **T²RAG+**) và các cấu hình trung gian được chạy **một lần** trên test (200–399) để làm Bảng A.

---

## 6. GĐ3: Công cụ báo cáo

### Việc 6.1: `make_table.py` (N5)
- **Đầu vào:** danh sách file kết quả, hoặc `--glob 'outputs/results/musique_*_from200.json'`; `--baseline trag` (phương pháp làm mốc so sánh); `--labels labels.json` (tùy chọn, đổi tên hàng).
- **Xử lý:**
  - nhóm theo (dataset, phương pháp + fixes);
  - tính EM/F1 tổng và theo bước (lấy từ trường `hops`), Recall@5, số lần gọi LLM trung bình, token trung bình, thời gian mỗi câu;
  - **bootstrap ghép cặp 5.000 lần** trên các câu chung với baseline: p = tỷ lệ mẫu có Δ ≤ 0. Đánh dấu † nếu p < 0,05, như bài báo.
- **Đầu ra:**
  - `tables/table_A.md` (EM / F1 dạng `56,6 / 62,4`, in đậm số tốt nhất mỗi cột);
  - `tables/table_A.csv`;
  - `tables/table_cost.md` (số lần gọi và token).
- **Kiểm tra:** chạy trên file kết quả giả; kiểm tay một ô.

### Việc 6.2: `error_analysis.py` (N6)
- **Đầu vào:** file kết quả, thư mục `intermediate_results` tương ứng, file dataset (để lấy đoạn vàng và `question_decomposition`).
- **Nhãn lỗi tự động** cho mỗi câu sai (đúng = EM 1 hoặc F1 ≥ 0,5), xét **theo thứ tự**, gán nhãn đầu tiên khớp:
  1. **Không tìm thấy:** không có đoạn vàng nào trong mọi passage đã lấy qua các vòng.
  2. **Rơi mắt xích:** vết cuối còn bộ ba chưa giải, hoặc `dropped_clues > 0`.
  3. **Điền sai bước giữa:** có đáp án trung gian của `question_decomposition` (MuSiQue có sẵn) không xuất hiện trong bộ ba đã giải, dù đoạn vàng đã được lấy.
  4. **Trả lời cuối sai:** đáp án vàng có trong bộ ba đã giải hoặc bằng chứng, nhưng dự đoán sai.
  5. **Khác.**
- **Đầu ra:**
  - `errors_<run>.csv` (mỗi câu một dòng: nhãn, câu hỏi, dự đoán, đáp án, bộ ba cuối);
  - bảng tỷ lệ lỗi theo nhãn và theo số bước;
  - bảng **trước và sau** khi so hai lượt chạy (ví dụ A0 với T²RAG+).
- Kèm hướng dẫn **kiểm tay 30 câu** để đo độ tin của nhãn tự động; ghi tỷ lệ khớp vào báo cáo.

---

## 7. GĐ4–GĐ6

### GĐ4: notebook v2 (N7)
Sửa `build_notebook.py` để notebook có:
- cell ghi tất cả file: `patch_t2rag.py`, `run_colab.py`, `src/TRAG_plus.py`, `compare_methods.py`, `make_table.py`, `error_analysis.py`;
- cell kiểm tra quota và tự chọn model;
- cell theo từng giai đoạn: GĐ1, ablation dev, chạy test, làm bảng, phân tích lỗi;
- cell `--dataset vnhistory`;
- cell sao lưu lên Drive sau mỗi lần chạy.

### GĐ5: H2a, chỉ làm nếu GĐ1 đạt quy tắc ở mục 4
- Cờ `h2a` trong `TRAGPlus`. Khi vòng lặp kết thúc mà **còn bộ ba chưa giải**, hoặc **một vòng không tìm thấy gì**, thì:
  - lấy thực thể trong bộ ba đã giải và trong câu hỏi làm điểm xuất phát;
  - gọi `graph_search_with_fact_entities` hoặc `run_ppr` của HippoRAG2 (dùng chung đồ thị đã dựng ở GĐ1) để lấy top-5 chunk;
  - đưa các chunk này vào `final_qa` (cùng chỗ với bằng chứng của S3).
- Bỏ bước `rerank_facts` để không tốn thêm lần gọi LLM.
- Ghi vào vết: số câu kích hoạt H2a, và EM của riêng các câu đó.

### GĐ6: gộp với phiên dữ liệu
1. `check_vnhistory.py`: kiểm tra hợp đồng mục 2 của file chia việc:
   - id có tiền tố số bước;
   - đoạn hỗ trợ nằm trong corpus;
   - mỗi passage có `idx` trong file OpenIE;
   - mỗi bộ ba có 3 phần tử;
   - in thống kê.
2. Chạy trên `vnhistory`: Standard, HippoRAG2, T²RAG gốc, T²RAG+ (và các bản trung gian), có thể thêm H2a. Dev dùng để xác nhận lại ngưỡng S4; test dùng để lấy Bảng B.
3. Có thể cần **prompt tiếng Việt** cho Bước 1, `resolve_clues` và `final_qa`. Làm thành cờ `--prompt_lang vi` trong `TRAGPlus`, **giữ nguyên các từ khóa tiếng Anh mà parser cần**.
4. Xuất `BAN_GIAO_T2RAG_v2.md`.

---

## 8. Mốc hoàn thành và việc bạn cần làm

| Mốc | Phiên này giao | Bạn cần làm |
|---|---|---|
| **M-A** | `run_colab.py` v3 + `tests/` + hướng dẫn GĐ1 | Chạy smoke test hippo 20 câu, dán log |
| **M-B** | Sửa lỗi hippo (nếu có) | Chạy 200 câu trag và hippo, rồi `compare_methods`, dán bảng |
| **M-C** | `TRAG_plus.py` (B + S1 + S3) có test | Chạy A0–A3 trên dev (100 câu), dán bảng |
| **M-D** | S2 + S4 | Chạy A4–A5 trên dev |
| **M-E** | `make_table.py`, `error_analysis.py`, notebook v2 | Chạy test 200–399 cho các cấu hình đã chọn |
| **M-F** | Bảng A, bảng lỗi, (H2a) | Gửi báo cáo của phiên dữ liệu khi xong |
| **M-G** | Gộp, Bảng B, `BAN_GIAO_T2RAG_v2.md` | |

---

## 9. Nhật ký và quyết định (cập nhật dần)

| Ngày | Việc | Kết quả / quyết định |
|---|---|---|
| 12/11/2026 | Lập kế hoạch | Thống nhất GĐ0–GĐ6. Bắt đầu GĐ0. |
| 12/11/2026 | **GĐ0 xong**: `run_colab.py` v3 (`--start`, `trag_plus --fixes`, đếm LLM/token, cache LLM dùng chung theo bộ dữ liệu, vết ở `outputs/traces/<run>`); `tests/fake_env.py` | Test end-to-end với LLM giả đạt. Câu 3 bước tốn 5 lần gọi LLM với `b,s1,s3`. |
| 12/11/2026 | **GĐ2 viết xong** `src/TRAG_plus.py`: B (L2, L4, L5, L6), S1, S2, S3, S4 | **21/21 test đạt.** Không bật cờ thì prompt trùng bản gốc. Bản gốc mất chuỗi 3 bước; `b,s1` giải đúng. Chưa chạy Gemini thật. Bàn giao `t2rag_code_v3.zip` + `HUONG_DAN_CHAY_COLAB_v3.md`. |
| 12/11/2026 | Nhóm chốt phạm vi dữ liệu | **Lý – Trần – Hồ – Lê sơ (1009–1527), khoảng 120 bài Wikipedia tiếng Việt.** Giai đoạn liền mạch, các bài liên kết dày, dễ tạo câu 2–3 bước. Ảnh hưởng tới code: mục 11. |
| 12/11/2026 | **GĐ3 xong**: `make_table.py` (bảng EM/F1 theo bước, †, in đậm, bảng chi phí) và `error_analysis.py` (6 nhãn lỗi, so sánh trước/sau, mẫu kiểm tay) | Thứ tự nhãn đổi so với mục 6.2: **"trả lời cuối sai" xét trước "không tìm thấy"** (đáp án đã nằm trong bộ ba thì lỗi ở bước trả lời). Thêm nhãn `sai_ten` (dùng `--aliases`) cho dữ liệu Lý – Trần. |
| 12/11/2026 | **GĐ4 xong**: `build_notebook_v2.py` → `T2RAG_Colab_v2.ipynb`. **Làm sớm một phần GĐ6**: `check_vnhistory.py` | `run_colab.py` lưu thêm `tag`. Zip `t2rag_code_v4.zip` thử trên repo gốc sạch + patch: **29/29 test đạt**. Bàn giao kèm `HUONG_DAN_CHAY_COLAB_v4.md`. |
| | Chờ bạn | GĐ1 (hippo 20 câu, rồi 200 câu) và ablation A0–A5 trên dev 100 câu. Phiên này chỉ còn GĐ5 (tùy kết quả GĐ1) và GĐ6 (khi có dữ liệu). |

## 10. Rủi ro và cách xử lý
| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Hết quota Gemini | Lỗi 429 kéo dài, nhiều câu "Error processing query" | `--min_interval`, chạy lại (có cache), giảm test xuống 150 câu, chạy rải nhiều ngày |
| Nhánh hippo lỗi cài đặt | `ImportError` igraph, thiếu config | Sửa theo log ở GĐ1. Nếu quá khó thì báo cáo HippoRAG2 bằng số của bài báo kèm ghi chú. |
| S1 làm LLM đặt tên biến lung tung | Nhiều bộ ba không có tên biến | Code tự đặt tên, và có thể quay về `_clues_similar` |
| S4 từ chối oan | Tỷ lệ từ chối > 30%, EM giảm | Hạ ngưỡng, chỉ bật S4 cho bộ ba cuối chuỗi |
| Kết quả dao động do LLM | Chạy lại cho số khác | `temperature=0` (kiểm config), dùng cache, báo bootstrap CI |
| Lệch định dạng dữ liệu với phiên kia | `check_vnhistory.py` báo lỗi | Sửa bằng script chuyển đổi, không sửa tay |

## 11. Ảnh hưởng của phạm vi Lý – Trần – Hồ – Lê sơ (1009–1527) tới phần code

| Đặc điểm dữ liệu | Ảnh hưởng | Xử lý (làm ở GĐ6, sau khi gộp) |
|---|---|---|
| Một người nhiều tên: tên húy, miếu hiệu, tước hiệu (Trần Quốc Tuấn = Hưng Đạo Vương = Trần Hưng Đạo; Lê Lợi = Lê Thái Tổ; Hồ Quý Ly = Lê Quý Ly) | EM chấm sai khi trả lời đúng người nhưng khác tên; S4 so khớp giá trị dễ trượt | Phiên dữ liệu điền đủ `answer_aliases`. S4 so khớp theo danh sách bí danh nếu có file `aliases_vi.json`. `error_analysis.py` thêm nhãn "đúng người, sai tên". |
| Trùng miếu hiệu giữa các triều (Lý / Trần / Lê **Thái Tổ**; nhiều **Thái Tông**, **Nhân Tông**, **Thánh Tông**) | Truy vấn mơ hồ, tìm nhầm triều | Prompt phân rã yêu cầu giữ tên triều ("Trần Nhân Tông", không viết "Nhân Tông"). S2 (BM25) giúp khớp đúng cả cụm. |
| Năm, niên hiệu, can chi ("1288", "năm 1288", "Mậu Tý") | Cùng một đáp án nhưng nhiều cách viết | Đáp án vàng ghi dạng số năm, cách viết khác đưa vào `answer_aliases`. Báo cáo thêm F1 bên cạnh EM. |
| Tiếng Việt có dấu, BM25 tách theo âm tiết | "Trần", "Hưng", "Đạo" thành ba từ riêng | Chuẩn hóa Unicode **NFC** trước khi tìm và so khớp. Thử thêm cặp 2 âm tiết (bigram) cho BM25 trên tập chỉnh. |
| Kho nhỏ (~120 bài) | Tìm vét cạn đủ nhanh, tốn ít quota | Không cần FAISS. |
| Chỉ có câu 1–3 bước (không có 4 bước) | Mức tăng của S1 có thể nhỏ hơn trên MuSiQue | Bảng B tách cột 1/2/3 bước. Kết luận về S1 dựa trên cả MuSiQue và `vnhistory`. |
| Prompt gốc toàn tiếng Anh | LLM có thể dịch tên riêng hoặc trả lời bằng tiếng Anh | Cờ `--prompt_lang vi` cho Bước 1, `resolve_clues`, `final_qa`. **Giữ nguyên** các từ khóa parser cần (`Fully Resolved Clue`, `Newly Traceable Clue`, `Subject:`, `Predicate:`, `Object:`, `Binding:`, `Evidence:`). |
