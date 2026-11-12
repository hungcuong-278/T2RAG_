"""Generate T2RAG_Colab_v2.ipynb: one notebook for every stage of the plan (GĐ1-GĐ3, then vnhistory).

    python build_notebook_v2.py [output path]

The notebook does not embed the code: it unzips t2rag_code_v4.zip from Google Drive
(MyDrive/T2RAG_code/), so updating the code only means replacing the zip.
"""
import json
import sys

CODE_ZIP = "t2rag_code_v4.zip"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(f"""
# T²RAG cho câu hỏi lịch sử: notebook thí nghiệm (v2)

Notebook này chạy mọi giai đoạn trong kế hoạch "Code và thí nghiệm":

| Phần | Nội dung | Tốn API? |
|---|---|---|
| 0 | Cài đặt, kiểm tra code bằng test giả, chọn model | Không (trừ 1 lần gọi thử) |
| 1 | **GĐ1**: so sánh T²RAG với HippoRAG2 trên MuSiQue (quyết định Hướng 2) | Có |
| 2 | **GĐ2**: ablation S1–S4 trên tập chỉnh (MuSiQue câu 0–99) | Có |
| 3 | **Tập báo cáo** (MuSiQue câu 200–399) và **bảng kết quả + phân tích lỗi** | Có / Không |
| 4 | Dữ liệu lịch sử `vnhistory` (sau khi gộp với phiên dữ liệu) | Có |

**Chuẩn bị:**
1. Tải `{CODE_ZIP}` lên Google Drive, vào thư mục `MyDrive/T2RAG_code/`.
2. Thêm secret `GOOGLE_API_KEY` (biểu tượng chìa khóa bên trái, bật *Notebook access*).
3. Chọn GPU: *Runtime → Change runtime type → T4 GPU*.

Mọi kết quả và cache LLM được sao lưu vào `MyDrive/T2RAG_outputs/`. Câu nào đã có đáp án trong cache thì chạy lại **không tốn API**.
""")

md("## 0. Cài đặt")
code("""
import os
from google.colab import userdata, drive
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
drive.mount('/content/drive')
""")
code(f"""
%%bash
set -e
cd /content
[ -d T2RAG ] || git clone -q https://github.com/Emory-Melody/T2RAG.git
pip install -q openai tenacity filelock python-igraph sentence-transformers
rm -rf t2rag_code && mkdir t2rag_code && cd t2rag_code
unzip -q -o /content/drive/MyDrive/T2RAG_code/{CODE_ZIP}
cp -r */* /content/T2RAG/ 2>/dev/null || cp -r * /content/T2RAG/
cd /content/T2RAG && python patch_t2rag.py
mkdir -p /content/drive/MyDrive/T2RAG_outputs && rsync -a /content/drive/MyDrive/T2RAG_outputs/ outputs/
echo "Xong. Code trong /content/T2RAG"
""")
md("Kiểm tra code bằng LLM giả (khoảng 10 giây, **không tốn API**). Phải thấy `OK` ở dòng cuối.")
code("""
!cd /content/T2RAG && python -m unittest discover tests 2>&1 | tail -3
""")
md("Chọn model và gọi thử một lần. Mọi phương pháp phải dùng **cùng một model** thì mới so sánh được.")
code("""
from openai import OpenAI
client = OpenAI(api_key=os.environ['GOOGLE_API_KEY'],
                base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
try:
    print([m.id.replace('models/', '') for m in client.models.list() if 'flash' in m.id])
except Exception as e:
    print('Không lấy được danh sách model:', e)

LLM_NAME = 'gemini-3.5-flash'  #@param {type:"string"}
MIN_INTERVAL = 0  #@param {type:"number"}
reply = client.chat.completions.create(model=LLM_NAME, temperature=0,
        messages=[{'role': 'user', 'content': 'Ai lãnh đạo trận Bạch Đằng năm 1288? Trả lời một tên.'}])
print(LLM_NAME, '→', reply.choices[0].message.content)
COMMON = f'--llm_name {LLM_NAME} --min_interval {MIN_INTERVAL}'
""")
md("""
Hàm sao lưu: **chạy sau mỗi lệnh dài** (Colab có thể ngắt phiên bất cứ lúc nào).

Nếu gặp lỗi 429 (vượt quota), đặt `MIN_INTERVAL = 6` ở ô trên rồi chạy lại đúng lệnh vừa lỗi: chỉ các câu lỗi mới gọi API.
""")
code("""
def backup():
    os.system('rsync -a /content/T2RAG/outputs/ /content/drive/MyDrive/T2RAG_outputs/')
    print('Đã sao lưu lên Drive.')
%cd /content/T2RAG
""")

md("""
## 1. GĐ1: T²RAG so với HippoRAG2 (có nên làm Hướng 2?)

**1.1 Chạy thử HippoRAG2 với 20 câu.** Nhánh này chưa từng chạy thật. Nếu lỗi, copy toàn bộ log gửi lại.
""")
code("""
!python run_colab.py --dataset musique --method hippo --limit 20 {COMMON}
backup()
""")
md("**1.2 Chạy 200 câu cho cả hai, rồi so sánh.** 200 câu đầu gồm 103 câu 2 bước, 58 câu 3 bước, 39 câu 4 bước.")
code("""
!python run_colab.py --dataset musique --method trag  --limit 200 {COMMON}
backup()
!python run_colab.py --dataset musique --method hippo --limit 200 {COMMON}
backup()
""")
code("""
!python compare_methods.py outputs/results/musique_trag_{LLM_NAME}_200q.json outputs/results/musique_hippo_{LLM_NAME}_200q.json
""")
md("""
**Quy tắc quyết định (kế hoạch mục 4):** làm Hướng 2 (H2a) nếu "EM trần − EM T²RAG" ≥ 3 điểm ở nhóm 3–4 bước, **và** ít nhất 1/3 số câu "chỉ hippo đúng" là câu T²RAG không tìm thấy đoạn vàng. Ô dưới kiểm tra điều kiện thứ hai.
""")
code("""
!python error_analysis.py outputs/results/musique_trag_{LLM_NAME}_200q.json
import pandas as pd
only_b = pd.read_csv('outputs/results/only_hippo_correct_vs_trag.csv')
lab = pd.read_csv(f'outputs/results/errors_musique_trag_{LLM_NAME}_200q.csv')
lab['qid'] = lab['qid'].astype(str); only_b['qid'] = only_b['qid'].astype(str)
print(only_b.merge(lab[['qid', 'label']], on='qid')['label'].value_counts())
""")

md("""
## 2. GĐ2: ablation trên tập chỉnh (MuSiQue câu 0–99)

Lượt A0 là T²RAG gốc. Mỗi lượt sau thêm một sửa đổi:
- `b`: sửa lỗi code;
- `s1`: giữ bộ ba chưa giải và dùng bảng biến;
- `s3`: trả lời có bằng chứng;
- `s2`: tìm kiếm lai;
- `s4`: kiểm chứng giá trị.

Sửa đổi nào làm điểm giảm thì bỏ.
""")
code("""
DEV = '--dataset musique --limit 100'
!python run_colab.py {DEV} --method trag {COMMON}
!python run_colab.py {DEV} --method trag_plus --fixes b {COMMON}
!python run_colab.py {DEV} --method trag_plus --fixes b,s1 {COMMON}
!python run_colab.py {DEV} --method trag_plus --fixes b,s1,s3 {COMMON}
backup()
""")
code("""
!python run_colab.py {DEV} --method trag_plus --fixes b,s1,s2,s3 {COMMON}
!python run_colab.py {DEV} --method trag_plus --fixes b,s1,s2,s3,s4 {COMMON}
!python run_colab.py {DEV} --method trag_plus --fixes b,s1,s2,s3,s4 --s4_threshold 0.7 --tag th07 {COMMON}
!python run_colab.py {DEV} --method trag_plus --fixes b,s1,s2,s3,s4 --s4_threshold 0.9 --tag th09 {COMMON}
backup()
""")
md("Bảng tập chỉnh (có tách theo số bước). Dán bảng này cho phiên Code.")
code("""
!python make_table.py --glob "outputs/results/musique_*_100q*.json" --baseline trag --out_dir tables/dev --n_boot 2000
""")
md("Phân tích lỗi trước và sau, so T²RAG gốc với một cấu hình (sửa tên file nếu cần):")
code("""
!python error_analysis.py outputs/results/musique_trag_{LLM_NAME}_100q.json --compare outputs/results/musique_trag_plus_fx-b-s1-s3_{LLM_NAME}_100q.json
""")

md("""
## 3. Tập báo cáo (MuSiQue câu 200–399) và bảng cuối

Chỉ chạy **sau khi đã chọn cấu hình ở phần 2**, và chỉ chạy **một lần**. Điền vào `BEST` cấu hình tốt nhất. Có thể thêm các cấu hình trung gian để có bảng ablation.
""")
code("""
BEST = 'b,s1,s2,s3'  #@param {type:"string"}
TEST = '--dataset musique --start 200 --limit 200'
!python run_colab.py {TEST} --method standard {COMMON}
!python run_colab.py {TEST} --method hippo {COMMON}
!python run_colab.py {TEST} --method trag {COMMON}
!python run_colab.py {TEST} --method trag_plus --fixes b,s1 {COMMON}
!python run_colab.py {TEST} --method trag_plus --fixes {BEST} {COMMON}
backup()
""")
code("""
import json
labels = {'standard': 'Standard RAG', 'hippo': 'HippoRAG2', 'trag': 'T²RAG (gốc)',
          'trag_plus_fx-b-s1': 'T²RAG + sửa lỗi + S1',
          'trag_plus_fx-' + '-'.join(sorted(BEST.split(','))): 'T²RAG+ (' + BEST + ')'}
json.dump(labels, open('labels.json', 'w'), ensure_ascii=False)
!python make_table.py --glob "outputs/results/musique_*_from200_200q.json" --baseline trag --labels labels.json --out_dir tables/test
!python error_analysis.py outputs/results/musique_trag_{LLM_NAME}_from200_200q.json --sample 30
backup()
""")
md("Bảng nằm ở `tables/test/table_main.md` và `table_cost.md`. File `check_*.csv` là 30 câu để kiểm tay nhãn lỗi.")

md("""
## 4. Dữ liệu lịch sử `vnhistory` (sau khi gộp)

Cần ba file của phiên dữ liệu:
- `reproduce/dataset/vnhistory.json`
- `reproduce/dataset/vnhistory_corpus.json`
- `outputs/vnhistory_univ/openie_results_ner.json`

Chép ba file vào đúng chỗ (hoặc lên Drive rồi rsync), kiểm tra định dạng, rồi chạy như phần 2–3. 50 câu đầu là tập chỉnh.
""")
code("""
!python check_vnhistory.py 2>/dev/null || echo "check_vnhistory.py sẽ có ở bản code sau khi gộp"
""")
code("""
VN_DEV  = '--dataset vnhistory --limit 50'
VN_TEST = '--dataset vnhistory --start 50 --limit 0'
!python run_colab.py {VN_DEV} --method trag {COMMON}
!python run_colab.py {VN_DEV} --method trag_plus --fixes {BEST} {COMMON}
backup()
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = sys.argv[1] if len(sys.argv) > 1 else "T2RAG_Colab_v2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Wrote", out)
