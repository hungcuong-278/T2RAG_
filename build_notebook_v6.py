"""Generate notebooks/T2RAG_Colab_v6.ipynb.

    python build_notebook_v6.py [output path]

Changes from v2: the code comes from the team repository with git clone (no zip on Drive),
the LLM is called through data_vi/gw_proxy.py (gateway, Colab Secrets GWAI_API_KEY...) or
directly through Google, long runs go in resumable batches that are merged with
merge_results.py, and the vnhistory part runs each T2RAG+ configuration twice with
separate caches (--cache_tag) and short answers (--answer_format short).
"""
import json
import sys

TEAM_REPO = "https://github.com/hungcuong-278/T2RAG_.git"
PAPER_REPO = "https://github.com/Emory-Melody/T2RAG.git"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# T²RAG cho câu hỏi lịch sử: notebook thí nghiệm (v6)

| Phần | Nội dung | Tốn API? |
|---|---|---|
| 0 | Cài đặt (lấy code từ GitHub nhóm), test giả, bật gateway, chọn model | Không (trừ vài lần gọi thử) |
| 1 | **GĐ1**: T²RAG so với HippoRAG2 trên MuSiQue 200 câu (quyết định Hướng 2) | Có |
| 2 | **Bộ tiếng Anh, tập test** (câu 200–399): Standard, HippoRAG2, T²RAG | Có |
| 3 | **vnhistory**: NOR, ablation T²RAG+ trên dev (mỗi cấu hình 2 lượt), rồi test | Có |
| 4 | Bảng kết quả và phân tích lỗi | Không |

**Chuẩn bị (một lần):**
1. Colab Secrets (biểu tượng chìa khóa bên trái, bật *Notebook access* cho từng secret):
   - chạy qua gateway: `GWAI_API_KEY`, có thể thêm `GWAI_API_KEY_2`, `GWAI_API_KEY_3`, `GWAI_API_KEY_4` (hết tiền key này thì tự sang key sau);
   - hoặc gọi thẳng Google: `GOOGLE_API_KEY`.
2. *Runtime → Change runtime type → T4 GPU*.

**Không cần tải zip lên Drive nữa:** ô cài đặt lấy code mới nhất từ GitHub của nhóm. Code trên GitHub đổi thì chỉ cần chạy lại phần 0.

Kết quả và cache LLM được sao lưu vào `MyDrive/T2RAG_outputs/`. Câu đã có trong cache thì chạy lại **không tốn tiền**.
""")

md("## 0. Cài đặt")
code("""
import os, glob, json, subprocess, time
from google.colab import drive
drive.mount('/content/drive')
TEAM_BRANCH = 'main'  #@param {type:"string"}
os.environ['TEAM_BRANCH'] = TEAM_BRANCH  # branch of the team repo to clone (e.g. a branch under review)
""")
code(f"""
%%bash
set -e
cd /content
[ -d T2RAG ] || git clone -q {PAPER_REPO}
rm -rf team && git clone -q --depth 1 -b "${{TEAM_BRANCH:-main}}" {TEAM_REPO} team
echo "Code nhóm: nhánh ${{TEAM_BRANCH:-main}}, commit $(git -C team rev-parse --short HEAD) - $(git -C team log -1 --format=%s)"
pip install -q openai tenacity filelock python-igraph sentence-transformers anthropic
rsync -a --exclude .git --exclude README.md --exclude notebooks /content/team/ /content/T2RAG/
cd /content/T2RAG && python patch_t2rag.py
mkdir -p /content/drive/MyDrive/T2RAG_outputs && rsync -a /content/drive/MyDrive/T2RAG_outputs/ outputs/
mkdir -p outputs/vnhistory_univ && cp data_vi/release/openie_results_ner.json outputs/vnhistory_univ/
echo "Xong. Code trong /content/T2RAG"
""")
md("Kiểm tra code bằng LLM giả (**không tốn API**). Dòng cuối phải là `OK`; nếu là `FAILED` thì dừng lại và gửi log.")
code("""
!cd /content/T2RAG && python -m unittest discover tests 2>&1 | tail -3
""")
md("""
**Bật gateway** (bỏ qua ô này nếu gọi thẳng Google). Bộ chuyển `gw_proxy.py` chạy nền ở `localhost:4000`.
Nó được tách khỏi ô đang chạy (`start_new_session=True`), nên bấm dừng một ô khác không làm nó chết.
""")
code("""
from google.colab import userdata
env = dict(os.environ)
for name in ['GWAI_API_KEY', 'GWAI_API_KEY_2', 'GWAI_API_KEY_3', 'GWAI_API_KEY_4']:
    try:
        env[name] = userdata.get(name)
    except Exception:
        pass
os.environ['OPENAI_API_KEY'] = 'local-proxy'   # run_colab.py chỉ cần một giá trị bất kỳ; proxy gắn key thật
proxy = subprocess.Popen(['python', 'data_vi/gw_proxy.py', '--port', '4000', '--timeout', '180'],
                         cwd='/content/T2RAG', env=env, start_new_session=True,
                         stdout=open('/content/gw_proxy.log', 'a'), stderr=subprocess.STDOUT)
for _ in range(30):
    time.sleep(1)
    if os.system('curl -s localhost:4000/v1/models > /dev/null') == 0:
        break
print(open('/content/gw_proxy.log').read()[-400:])
""")
md("""
**Chọn model và gọi thử.** Nhóm chốt `gemini-3.8-flash`. Mọi phương pháp, mọi người phải dùng **cùng một model**.
`LLM_MODE = 'gateway'` (qua proxy) hoặc `'google'` (key Google).
""")
code("""
from openai import OpenAI
from google.colab import userdata
LLM_MODE = 'gateway'  #@param ['gateway', 'google']
LLM_NAME = 'gemini-3.8-flash'  #@param {type:"string"}
MIN_INTERVAL = 0  #@param {type:"number"}
if LLM_MODE == 'gateway':
    BASE, KEY = 'http://localhost:4000/v1', 'local-proxy'
else:
    os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
    BASE, KEY = 'https://generativelanguage.googleapis.com/v1beta/openai/', os.environ['GOOGLE_API_KEY']
client = OpenAI(api_key=KEY, base_url=BASE)
reply = client.chat.completions.create(model=LLM_NAME, temperature=0,
        messages=[{'role': 'user', 'content': 'Ai lãnh đạo trận Bạch Đằng năm 1288? Trả lời một tên.'}])
print(LLM_NAME, '→', reply.choices[0].message.content)
COMMON = f'--llm_name {LLM_NAME} --llm_base_url {BASE} --min_interval {MIN_INTERVAL}'
""")
md("""
Hàm dùng chung:
- `backup()`: sao lưu lên Drive (chạy sau mỗi lệnh dài);
- `usage()`: số lượt gọi, token và lỗi của gateway từ lúc bật proxy (để tính tiền);
- `run_batches(...)`: chạy theo lô (mặc định 10 câu; MuSiQue và các bộ tiếng Anh dùng lô 50 vì mỗi lô phải nạp lại mô hình khoảng 2 phút), mỗi lô xong là sao lưu; bị ngắt thì **chạy lại đúng ô đó**, lô đã xong được bỏ qua; đủ lô thì gộp thành một file `<dataset>_<method>_<model>_<tag>.json`.

Giá `gemini-3.8-flash`: 0,75 USD / 1 triệu token vào, 3,75 USD / 1 triệu token ra.
""")
code("""
%cd /content/T2RAG

def backup():
    os.system('rsync -a /content/T2RAG/outputs/ /content/drive/MyDrive/T2RAG_outputs/')
    print('Đã sao lưu lên Drive.')

def usage(price_in=0.75, price_out=3.75):
    try:
        u = json.loads(subprocess.check_output(['curl', '-s', 'localhost:4000/v1/usage']))
        cost = u.get('in', 0) / 1e6 * price_in + u.get('out', 0) / 1e6 * price_out
        print(u, f'≈ {cost:.3f} USD từ lúc bật proxy')
    except Exception as e:
        print('Không đọc được /v1/usage (proxy chưa bật?):', e)

LLM_LABEL = LLM_NAME.replace('/', '-').replace(':', '-')

def run_batches(dataset, method, start, total, tag, extra='', size=10):
    fixes = ''
    if method == 'trag_plus':
        parts = extra.split('--fixes')
        fixes = parts[1].split()[0] if len(parts) > 1 else 'b,s1,s2,s3,s4'
    label = method + ('_fx-' + '-'.join(sorted(f for f in fixes.split(',') if f)) if method == 'trag_plus' else '')
    files = []
    for s in range(start, start + total, size):
        n = min(size, start + total - s)
        name = f'{dataset}_{label}_{LLM_LABEL}' + (f'_from{s}' if s else '') + f'_{n}q_{tag}.json'
        path = f'outputs/results/{name}'
        if not os.path.exists(path):
            print(f'--- lô {s}..{s + n - 1} ---')
            os.system(f'python run_colab.py --dataset {dataset} --method {method} --start {s} --limit {n} '
                      f'--tag {tag} {extra} {COMMON}')
            backup()
        if not os.path.exists(path):
            raise SystemExit(f'Lô {s}..{s + n - 1} chưa xong ({path}). Chạy lại ô này.')
        files.append(path)
    out = f'outputs/results/{dataset}_{label}_{LLM_LABEL}_{tag}.json'
    os.system('python merge_results.py ' + ' '.join(files) + f' -o {out}')
    backup()
    return out
""")

md("""
## 1. GĐ1: T²RAG so với HippoRAG2 trên MuSiQue (có nên làm Hướng 2?)

**1.1 Chạy thử 5 câu để đo tiền** (T²RAG gọi LLM nhiều lần mỗi câu). Nhân kết quả `usage()` lên 40 là ước tính cho 200 câu.
""")
code("""
usage()
!python run_colab.py --dataset musique --method trag --limit 5 --tag pilot {COMMON}
usage()
""")
md("**1.2 Chạy 200 câu đầu cho cả hai** (103 câu 2 bước, 58 câu 3 bước, 39 câu 4 bước), theo lô.")
code("""
GD1_TRAG = run_batches('musique', 'trag', 0, 200, 'gd1', size=50)
GD1_HIPPO = run_batches('musique', 'hippo', 0, 200, 'gd1', size=50)
usage()
""")
code("""
!python compare_methods.py {GD1_TRAG} {GD1_HIPPO}
!python make_table.py {GD1_TRAG} {GD1_HIPPO} --baseline hippo:gd1 --out_dir tables/gd1
""")
md("""
**Quy tắc quyết định:** làm Hướng 2 (H2a) nếu "EM trần − EM T²RAG" ≥ 3 điểm ở nhóm 3–4 bước, **và** ít nhất 1/3 số câu "chỉ HippoRAG đúng" là câu T²RAG không tìm thấy đoạn vàng (`thieu_doan_vang`). Ô dưới kiểm tra điều kiện thứ hai.
""")
code("""
!python error_analysis.py {GD1_TRAG}
import pandas as pd
only_b = pd.read_csv('outputs/results/only_hippo_correct_vs_trag.csv')
lab = pd.read_csv('outputs/results/errors_' + os.path.basename(GD1_TRAG)[:-5] + '.csv')  # merged file name = run_name
lab['qid'] = lab['qid'].astype(str); only_b['qid'] = only_b['qid'].astype(str)
print(only_b.merge(lab[['qid', 'label']], on='qid')['label'].value_counts())
""")

md("""
## 2. Bộ tiếng Anh, tập test (câu 200–399)

Chia việc: điền vào `DATASETS` những bộ **bạn** chạy (ví dụ An: `musique`; Hưng: `2wikimultihopqa,hotpotqa`).
Prompt giữ đúng như bài báo (không dùng `--answer_format short` cho tiếng Anh).
""")
code("""
DATASETS = 'musique'  #@param {type:"string"}
EN_TEST = {}
for ds in [d.strip() for d in DATASETS.split(',') if d.strip()]:
    for m in ['standard', 'hippo', 'trag']:
        EN_TEST[(ds, m)] = run_batches(ds, m, 200, 200, 'test', size=50)
    usage()
""")

md("""
## 3. vnhistory: T²RAG+ (dữ liệu lịch sử)

Cùng điều kiện với các lượt đã có (Standard RAG, HippoRAG2, T²RAG gốc): `gemini-3.8-flash`, `BAAI/bge-m3`,
`--answer_format paper` (mặc định), `vnhistory.json` v1.1 MD5 `1c088232…` (v1.0 `b07cfa13…` chỉ khác đáp án thay thế:
kết quả cũ chấm lại bằng `rescore.py`, không cần chạy lại), OpenIE MD5 `f6ff5da7…`.
Mỗi cấu hình một tag riêng, mỗi lượt một cache riêng (`--cache_tag r1`, `r2`).
Key gateway còn ít: **chạy thử 5 câu, gửi số liệu cho nhóm duyệt chi phí rồi mới chạy dev.**
""")
code("""
!python check_vnhistory.py --openie_md5 f6ff5da7
!md5sum reproduce/dataset/vnhistory.json outputs/vnhistory_univ/openie_results_ner.json
""")
md("**3.1 Chạy thử 5 câu dev** (3 câu 2 bước, 2 câu 3 bước), đủ cả 5 cải tiến.")
code("""
PILOT_IDS = '2hop_001,2hop_002,2hop_003,3hop_001,3hop_002'  #@param {type:"string"}
usage()
!python run_colab.py --dataset vnhistory --method trag_plus --fixes b,s1,s2,s3,s4 --qids {PILOT_IDS} --tag pilot5_plus_full --cache_tag pilot {COMMON}
usage()
backup()
r = json.load(open(sorted(glob.glob('outputs/results/vnhistory_trag_plus_fx-b-s1-s2-s3-s4_*_ids5_pilot5_plus_full.json'))[-1]))
s = r['summary']
print({k: s[k] for k in ['ExactMatch', 'F1', 'avg_llm_calls', 'avg_prompt_tokens', 'avg_completion_tokens',
                         'max_completion_tokens', 'plus_totals', 'errors']})
for x in r['records']:
    print(x['qid'], '| đáp:', x['prediction'], '| đúng:', x['gold_answers'][:3], '| gọi LLM:', x['llm_calls'],
          '| bindings:', x.get('n_bindings'), '/ applied', x.get('bindings_applied'), '| s4 từ chối:', x.get('s4_rejected'))
""")
md("""
Vết chi tiết (giá trị gán cho `?A`, `?B`… và bằng chứng S4) của câu đầu: `bindings` cho biết placeholder nào đã được gán,
`stats.s4_rejected` là số lần S4 từ chối. Kiểm tra tên tiếng Việt có giữ nguyên dấu không.
""")
code("""
tdir = s['trace_dir']
for f in sorted(os.listdir(tdir))[:2]:
    t = json.load(open(os.path.join(tdir, f)))
    fin = t.get('final_results', t)
    print(f, '\\n  bindings:', fin.get('bindings'), '\\n  stats:', fin.get('stats'))
""")
md("""
**3.2 Ablation trên dev 50 câu** (sau khi nhóm duyệt chi phí). Mặc định 3 cấu hình; chạy lượt `r1` trước,
lượt `r2` sau nếu còn tiền (điền `RUNS = 'r1,r2'`).
""")
code("""
CONFIGS = 'b,s1 | b,s1,s3 | b,s1,s2,s3,s4'  #@param {type:"string"}
RUNS = 'r1'  #@param {type:"string"}
DEV_RUNS = {}
for cfg in [c.strip() for c in CONFIGS.split('|') if c.strip()]:
    for r in [x.strip() for x in RUNS.split(',') if x.strip()]:
        tag = f"dev50_plus_{cfg.replace(',', '_')}_{r}"
        DEV_RUNS[(cfg, r)] = run_batches('vnhistory', 'trag_plus', 0, 50, tag, extra=f'--cache_tag {r} --fixes {cfg}')
        usage()
""")
md("""
So với T²RAG gốc: chép file kết quả T²RAG gốc dev (định dạng `paper`) vào `outputs/results/` rồi điền tên tag của nó vào
`BASE_TAG`. Chọn cấu hình theo **trung bình 2 lượt và so từng câu** (kiểm định dấu), không theo một con số đơn lẻ.
""")
code("""
BASE_TAG = 'dev50'  #@param {type:"string"}
!python make_table.py --glob "outputs/results/vnhistory_*_{LLM_LABEL}_dev50_*.json" --glob "outputs/results/vnhistory_trag_*{BASE_TAG}*.json" --baseline trag:{BASE_TAG} --out_dir tables/vn_dev --n_boot 2000
""")
md("**3.3 Tập test 116 câu, 2 lượt** cho cấu hình tốt nhất. Điền `BEST`.")
code("""
BEST = 'b,s1,s3'  #@param {type:"string"}
TEST_RUNS = {}
for r in ['r1', 'r2']:
    TEST_RUNS[r] = run_batches('vnhistory', 'trag_plus', 50, 116, f"test116_plus_{BEST.replace(',', '_')}_{r}",
                               extra=f'--cache_tag {r} --fixes {BEST}')
    usage()
""")

md("""
**3.4 S5 (bảo vệ câu trả lời) trên dev.** S5 chỉ thêm một lượt gọi khi câu trả lời cuối là một đoạn suy luận dài
hoặc "NOT FOUND"/"không rõ"; mọi lời gọi khác trùng prompt với lượt full nên lấy từ cache `r1`/`r2` (gần như không tốn tiền).
Chỉ chạy test với S5 **sau khi** dev cho thấy S5 có ích (so từng câu với lượt full cùng cache).
""")
code("""
S5_CFG = 'b,s1,s2,s3,s4,s5'  #@param {type:"string"}
S5_RUNS = 'r1'  #@param {type:"string"}
DEV_RUNS = globals().get('DEV_RUNS', {})
for r in [x.strip() for x in S5_RUNS.split(',') if x.strip()]:
    DEV_RUNS[(S5_CFG, r)] = run_batches('vnhistory', 'trag_plus', 0, 50, f"dev50_plus_{S5_CFG.replace(',', '_')}_{r}",
                                        extra=f'--cache_tag {r} --fixes {S5_CFG}')
    usage()
    # same run without s5 (same cache): which questions changed
    full = glob.glob(f"outputs/results/vnhistory_trag_plus_fx-b-s1-s2-s3-s4_{LLM_LABEL}_dev50_plus_b_s1_s2_s3_s4_{r}.json")
    if full:
        subprocess.run(['python', 'compare_methods.py', DEV_RUNS[(S5_CFG, r)], full[0]])
    else:
        print('Chưa có lượt full dev', r, 'để so sánh')
""")

md("""
## 4. Bảng và phân tích lỗi

Khi `vnhistory.json` lên bản mới (chỉ đổi đáp án thay thế), chấm lại mà **không gọi model**:
`python rescore.py <các file kết quả> --questions reproduce/dataset/vnhistory.json`.
""")
code("""
TEST_BASE_TAG = 'test116'  #@param {type:"string"}
# merged files only (chunk files are named ..._from50_10q_test116...), rescored on the current vnhistory.json;
# --pool_runs averages r1 / r2 per question and tests on the per-question means
!python rescore.py outputs/results/vnhistory_*_{LLM_LABEL}_test116_*.json outputs/results/vnhistory_trag_{LLM_LABEL}_*{TEST_BASE_TAG}*.json --questions reproduce/dataset/vnhistory.json --out_dir outputs/results/rescored
!python make_table.py --glob "outputs/results/rescored/*.json" --pool_runs --baseline trag:{TEST_BASE_TAG} --price_in 0.75 --price_out 3.75 --out_dir tables/vn_test
TEST_RUNS = globals().get('TEST_RUNS', {})
T_BEST = TEST_RUNS.get('r1') or (sorted(glob.glob(f"outputs/results/vnhistory_trag_plus_*_{LLM_LABEL}_test116_plus_*_r1.json")) or [''])[-1]
if T_BEST:
    subprocess.run(['python', 'error_analysis.py', T_BEST])
backup()
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = sys.argv[1] if len(sys.argv) > 1 else "notebooks/T2RAG_Colab_v6.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Wrote", out)
