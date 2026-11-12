"""Check that a new dataset follows the file contract agreed with the data session
(KE_HOACH_CHIA_VIEC_2_PHIEN.md, section 2), before spending any API call on it.

    python check_vnhistory.py                 # dataset "vnhistory"
    python check_vnhistory.py --dataset other

Checks:
  corpus     reproduce/dataset/<ds>_corpus.json: list of {"title", "text"}, no empty or duplicate passages
  questions  reproduce/dataset/<ds>.json: id starts with 1hop_/2hop_/3hop_..., question and short answer present,
             answer_aliases is a list, every supporting paragraph is exactly a corpus passage
  openie     outputs/<ds>_univ/openie_results_ner.json: every corpus passage has an entry whose idx is
             chunk-<md5(passage)>, and every triple has exactly 3 non-empty parts
Text is compared after Unicode NFC normalisation; a mismatch that disappears after NFC is reported
as a warning because the code compares raw strings.
Exit code 0 = no errors (warnings allowed), 1 = errors.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter


def nfc(s):
    return unicodedata.normalize("NFC", s)


def md5_idx(passage):
    return "chunk-" + hashlib.md5(passage.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vnhistory")
    ap.add_argument("--root", default=".", help="Repo root")
    ap.add_argument("--max_answer_words", type=int, default=8)
    ap.add_argument("--dev_size", type=int, default=50)
    args = ap.parse_args(argv)

    errors, warnings = [], []
    err = errors.append
    warn = warnings.append
    ds = args.dataset
    corpus_path = os.path.join(args.root, "reproduce", "dataset", f"{ds}_corpus.json")
    q_path = os.path.join(args.root, "reproduce", "dataset", f"{ds}.json")
    oie_path = os.path.join(args.root, "outputs", f"{ds}_univ", "openie_results_ner.json")

    # ---------------- corpus
    passages = []
    if not os.path.exists(corpus_path):
        err(f"Thiếu file kho văn bản {corpus_path}")
    else:
        corpus = json.load(open(corpus_path, encoding="utf-8"))
        if not isinstance(corpus, list):
            err("Kho văn bản phải là một danh sách [{title, text}, ...]")
            corpus = []
        for i, d in enumerate(corpus):
            if not isinstance(d, dict) or not d.get("title") or not d.get("text"):
                err(f"Đoạn {i} thiếu title hoặc text")
                continue
            passages.append(f"{d['title']}\n{d['text']}")
        dup = [p for p, c in Counter(passages).items() if c > 1]
        if dup:
            warn(f"{len(dup)} đoạn bị trùng hoàn toàn (ví dụ: {dup[0][:60]!r})")
        lens = [len(p.split()) for p in passages]
        if lens:
            short = sum(n < 20 for n in lens)
            long_ = sum(n > 400 for n in lens)
            if short:
                warn(f"{short} đoạn ngắn hơn 20 từ")
            if long_:
                warn(f"{long_} đoạn dài hơn 400 từ (hợp đồng: khoảng 100–250 từ)")
    passage_set = set(passages)
    passage_nfc = {nfc(p) for p in passages}

    # ---------------- questions
    hop_counter = Counter()
    if not os.path.exists(q_path):
        err(f"Thiếu file câu hỏi {q_path}")
        samples = []
    else:
        samples = json.load(open(q_path, encoding="utf-8"))
    ids = Counter()
    for i, s in enumerate(samples):
        sid = str(s.get("id", ""))
        ids[sid] += 1
        where = f"Câu {i} (id={sid!r})"
        m = re.match(r"^(\d)hop_", sid)
        if not m:
            err(f"{where}: id phải bắt đầu bằng 1hop_ / 2hop_ / 3hop_")
        else:
            hop_counter[int(m.group(1))] += 1
        if not s.get("question"):
            err(f"{where}: thiếu question")
        ans = s.get("answer")
        if not ans or not isinstance(ans, str):
            err(f"{where}: thiếu answer (chuỗi)")
        elif len(ans.split()) > args.max_answer_words:
            warn(f"{where}: đáp án dài {len(ans.split())} từ, EM sẽ khó đúng: {ans[:60]!r}")
        if not isinstance(s.get("answer_aliases", []), list):
            err(f"{where}: answer_aliases phải là danh sách")
        paras = s.get("paragraphs")
        if not paras:
            err(f"{where}: thiếu paragraphs")
            continue
        sup = [p for p in paras if p.get("is_supporting", True) is not False]
        if not sup:
            err(f"{where}: không có đoạn nào is_supporting = true")
        if m and len(sup) < int(m.group(1)):
            warn(f"{where}: {m.group(1)} bước nhưng chỉ có {len(sup)} đoạn hỗ trợ")
        for p in sup:
            text = p.get("paragraph_text", p.get("text", ""))
            full = f"{p.get('title', '')}\n{text}"
            if full in passage_set:
                continue
            if nfc(full) in passage_nfc:
                warn(f"{where}: đoạn hỗ trợ {p.get('title')!r} chỉ khớp sau chuẩn hóa NFC; hãy chuẩn hóa cả hai file")
            else:
                err(f"{where}: đoạn hỗ trợ {p.get('title')!r} không trùng từng ký tự với đoạn nào trong kho "
                    "(recall sẽ bằng 0)")
    for sid, c in ids.items():
        if c > 1:
            err(f"id {sid!r} bị lặp {c} lần")

    # ---------------- OpenIE
    n_triples = n_bad = 0
    if not os.path.exists(oie_path):
        err(f"Thiếu file bộ ba {oie_path}")
    else:
        oie = json.load(open(oie_path, encoding="utf-8"))
        docs = oie.get("docs") if isinstance(oie, dict) else None
        if docs is None:
            err('File bộ ba phải có dạng {"docs": [...]}')
            docs = []
        by_passage = {}
        for j, d in enumerate(docs):
            p = d.get("passage", "")
            by_passage[p] = d
            if d.get("idx") != md5_idx(p):
                err(f"Bộ ba, mục {j}: idx {d.get('idx')!r} khác chunk-<md5(passage)>")
            for t in d.get("extracted_triples", []):
                n_triples += 1
                if not (isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) and x.strip() for x in t)):
                    n_bad += 1
        missing = [p for p in passages if p not in by_passage]
        if missing:
            err(f"{len(missing)} đoạn trong kho không có trong file bộ ba (ví dụ: {missing[0][:60]!r})")
        extra = [p for p in by_passage if p not in passage_set]
        if extra:
            warn(f"{len(extra)} mục trong file bộ ba không thuộc kho văn bản")
        if n_bad:
            err(f"{n_bad}/{n_triples} bộ ba không có đúng 3 phần khác rỗng (TRAG sẽ bỏ qua)")
        no_triples = sum(1 for d in docs if not d.get("extracted_triples"))
        if no_triples:
            warn(f"{no_triples} đoạn không có bộ ba nào")

    # ---------------- report
    print(f"Bộ dữ liệu: {ds}")
    print(f"  Đoạn văn: {len(passages)}")
    print(f"  Câu hỏi: {len(samples)}  theo bước: " + ", ".join(f"{h} bước = {c}" for h, c in sorted(hop_counter.items())))
    if samples:
        dev = Counter(int(m.group(1)) for s in samples[:args.dev_size]
                      if (m := re.match(r"^(\d)hop_", str(s.get("id", "")))))
        test = Counter(int(m.group(1)) for s in samples[args.dev_size:]
                       if (m := re.match(r"^(\d)hop_", str(s.get("id", "")))))
        print(f"  Tập chỉnh (câu 0–{args.dev_size - 1}): " + ", ".join(f"{h} bước = {c}" for h, c in sorted(dev.items())))
        print(f"  Tập báo cáo (câu {args.dev_size} trở đi): " + ", ".join(f"{h} bước = {c}" for h, c in sorted(test.items())))
    print(f"  Bộ ba: {n_triples}" + (f" (TB {n_triples / len(passages):.1f}/đoạn)" if passages else ""))
    for w in warnings[:30]:
        print("  CẢNH BÁO:", w)
    if len(warnings) > 30:
        print(f"  ... và {len(warnings) - 30} cảnh báo khác")
    for e in errors[:30]:
        print("  LỖI:", e)
    if len(errors) > 30:
        print(f"  ... và {len(errors) - 30} lỗi khác")
    print("KẾT QUẢ:", "ĐẠT" if not errors else f"{len(errors)} LỖI, cần sửa trước khi chạy")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
