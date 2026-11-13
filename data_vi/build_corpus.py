"""Build reproduce/dataset/vnhistory_corpus.json from the raw Wikipedia extracts (M2).

Run from the project root (after fetch_articles.py download):
    python data_vi/build_corpus.py --stats              # dry run: size per cap, nothing written
    python data_vi/build_corpus.py --max_per_article 7  # write the corpus

Steps: drop reference / opinion / pop-culture sections, clean the text, cut each leaf
section into chunks of about 120-220 syllables (never across sections or mid-sentence),
then keep at most N chunks per article, picked round-robin over sections so long
articles keep their breadth (e.g. a king's laws and wars, not only his childhood).
Each corpus item is {"title": article title, "text": chunk}; passage = title + "\\n" + text.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import unicodedata
from collections import Counter, OrderedDict

RAW_DIR = os.path.join("data_vi", "work", "raw")
CORPUS_PATH = os.path.join("reproduce", "dataset", "vnhistory_corpus.json")
META_PATH = os.path.join("data_vi", "work", "chunks_meta.jsonl")

# Sections dropped with all their subsections (compared after lower-casing and NFC).
DROP_SECTIONS = {
    # references and navigation
    "xem thêm", "chú thích", "tham khảo", "liên kết ngoài", "ghi chú", "đọc thêm", "thư mục",
    "tài liệu tham khảo", "chú giải", "sách tham khảo", "sách", "tài liệu", "nguồn", "nguồn tham khảo",
    "hình ảnh", "thư viện ảnh", "liên kết", "ghi chú và tham khảo", "chú thích và tham khảo",
    # opinions, legends, modern commemoration, pop culture
    "nhận định", "đánh giá", "bình luận", "nhận xét", "giai thoại", "truyền thuyết",
    "trong văn hóa đại chúng", "trong văn hoá đại chúng", "văn hóa đại chúng", "trong nghệ thuật",
    "tưởng niệm", "tôn vinh", "hậu thế ghi công", "ghi công", "đền thờ", "di tích", "hình ảnh công cộng",
    "kịch nói", "phim ảnh", "điện ảnh", "truyền hình", "tiểu thuyết",
}

# Heading patterns dropped as well: footnotes, opinions, worship, modern naming and monuments.
DROP_PATTERNS = re.compile(
    r"cước chú|nhận định|đánh giá|nhìn nhận|đón nhận|văn hóa đại chúng|văn hoá đại chúng|hình ảnh trong văn hóa"
    r"|^tưởng nhớ|^đền thờ|^đền |thờ tự|thờ phụng|được thờ|được tôn thờ|^lễ hội|^tên đường|^tên trường"
    r"|^các công trình gắn|^di sản ngày nay|^thời hiện đại|công nhận và bảo tồn|^ghi danh|sách dùng để viết bài"
    r"|vinh danh")

HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"“(]?[A-ZÀ-ỸĐ0-9])")
SENT_END = re.compile(r"[.!?…:;]$")
CLOSERS = "\"”’'»)]"


def nfc(text):
    return unicodedata.normalize("NFC", text)


def clean_line(line):
    line = nfc(line).replace("\t", " ")
    line = re.sub(r"\[(\d+|[a-z]|cần dẫn nguồn|ghi chú \d+)\]", "", line)
    line = re.sub(r"\s+([.,;:!?])", r"\1", line)
    line = re.sub(r"\(\s*\)", "", line)
    return re.sub(r"\s+", " ", line).strip()


def close_line(line):
    """List items and quotes often end a line without a full stop; add one so lines do not run together."""
    core = line.rstrip(CLOSERS).rstrip()
    return line if SENT_END.search(core) else line + "."


def is_name_list(text, min_items=20, max_median=3):
    """True for chunks that are only long comma-separated name lists (e.g. ward names)."""
    items = [x.strip() for x in re.split(r"[,;]", text) if x.strip()]
    return len(items) >= min_items and statistics.median(len(x.split()) for x in items) <= max_median


def n_syll(text):
    return len(text.split())


def parse_sections(extract):
    """Split an extract into leaf sections: [(heading path, [lines])]; lead section has path ()."""
    sections, path, lines = [], (), []
    for raw in extract.split("\n"):
        m = HEADING.match(raw.strip())
        if m:
            if lines:
                sections.append((path, lines))
            level, name = len(m.group(1)), clean_line(m.group(2))
            path = path[:max(0, level - 2)] + (name,)
            lines = []
            continue
        line = clean_line(raw)
        if line:
            lines.append(close_line(line))
    if lines:
        sections.append((path, lines))
    return sections


def drop_reason(path):
    """Return the heading that causes this section to be dropped, or None."""
    for name in path:
        low = name.lower()
        if low in DROP_SECTIONS or DROP_PATTERNS.search(low):
            return low
    return None


def chunk_lines(lines, target, hard_max, min_len):
    """Greedy packing of lines (split into sentences when too long) into chunks."""
    # A short line ending with ':' introduces what follows (e.g. "Hậu phi:"); keep it with the next line.
    merged, header = [], ""
    for line in lines:
        if line.endswith(":") and n_syll(line) <= 15:
            header = (header + " " + line).strip()
            continue
        merged.append((header + " " + line).strip() if header else line)
        header = ""
    if header:
        merged.append(header)
    units = []
    for line in merged:
        if n_syll(line) <= hard_max:
            units.append(line)
            continue
        sent_buf = ""
        for sent in SENT_SPLIT.split(line):
            cand = (sent_buf + " " + sent).strip()
            if sent_buf and n_syll(cand) > target:
                units.append(sent_buf)
                sent_buf = sent
            else:
                sent_buf = cand
        if sent_buf:
            units.append(sent_buf)
    chunks, cur = [], ""
    for unit in units:
        cand = (cur + " " + unit).strip()
        if cur and n_syll(cand) > target:
            # Do not leave an introducing sentence ("Sử chép:") at the end of a chunk; move it forward.
            carry = ""
            if cur.endswith(":"):
                sents = SENT_SPLIT.split(cur)
                if len(sents) > 1 and n_syll(sents[-1] + " " + unit) <= hard_max:
                    carry, cur = sents[-1], " ".join(sents[:-1])
            chunks.append(cur)
            cur = (carry + " " + unit).strip()
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    # Merge a short tail into the previous chunk when it still fits under the hard cap.
    if len(chunks) >= 2 and n_syll(chunks[-1]) < min_len and n_syll(chunks[-2] + " " + chunks[-1]) <= hard_max:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail
    return chunks


def round_robin(section_chunks, cap, lead_first):
    """Pick up to `cap` chunks: lead chunks first, then one chunk per section per round, spread evenly."""
    picked = []
    queues = [list(enumerate(chs)) for _, chs in section_chunks]
    if section_chunks and section_chunks[0][0] == ():
        for item in queues[0][:lead_first]:
            picked.append((0, item[0]))
        queues[0] = queues[0][lead_first:]
    while len(picked) < cap and any(queues):
        live = [s for s, q in enumerate(queues) if q]
        room = cap - len(picked)
        if len(live) > room:
            # More sections than free slots: take evenly spaced sections across the article.
            live = [live[round(i * (len(live) - 1) / max(1, room - 1))] for i in range(room)] if room > 1 else live[:1]
        for s in dict.fromkeys(live):
            picked.append((s, queues[s].pop(0)[0]))
    return sorted(picked)  # keep document order


def build(args):
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No raw files in {RAW_DIR}; run fetch_articles.py download first.")
    articles, drop_stats = [], Counter()
    for path in files:
        raw = json.load(open(path, encoding="utf-8"))
        kept = []
        for sec_path, lines in parse_sections(raw["extract"]):
            reason = drop_reason(sec_path)
            if reason:
                drop_stats[reason] += sum(map(n_syll, lines))
                continue
            chs = []
            for c in chunk_lines(lines, args.target, args.hard_max, args.min_len):
                if n_syll(c) < args.drop_below:
                    drop_stats["<short chunk>"] += n_syll(c)
                elif is_name_list(c):
                    drop_stats["<name list>"] += n_syll(c)
                else:
                    chs.append(c)
            if chs:
                kept.append((sec_path, chs))
        articles.append((raw, kept))
    return articles, drop_stats


def select(articles, cap, lead_first):
    items, seen = [], set()
    for raw, kept in articles:
        title = nfc(raw["title"])
        for order, (s, c) in enumerate(round_robin(kept, cap, lead_first)):
            text = kept[s][1][c]
            passage = title + "\n" + text
            idx = "chunk-" + hashlib.md5(passage.encode()).hexdigest()
            if idx in seen:
                continue
            seen.add(idx)
            items.append({"title": title, "text": text, "idx": idx, "section": " > ".join(kept[s][0]) or "(lead)",
                          "order": order, "n_tieng": n_syll(text), "pageid": raw["pageid"], "revid": raw["revid"]})
    return items


def describe(items, label):
    lens = [it["n_tieng"] for it in items]
    per_art = Counter(it["title"] for it in items)
    print(f"{label}: {len(items)} chunks from {len(per_art)} articles | syllables mean {statistics.mean(lens):.0f}, "
          f"median {statistics.median(lens):.0f}, min {min(lens)}, max {max(lens)} | "
          f"chunks/article median {statistics.median(per_art.values()):.0f}, max {max(per_art.values())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_article", type=int, default=7)
    ap.add_argument("--lead_first", type=int, default=2, help="Lead-section chunks taken before round-robin")
    ap.add_argument("--target", type=int, default=200, help="Soft maximum syllables per chunk")
    ap.add_argument("--hard_max", type=int, default=250, help="Hard maximum syllables per chunk")
    ap.add_argument("--min_len", type=int, default=60, help="Merge a section's last chunk if shorter")
    ap.add_argument("--drop_below", type=int, default=20, help="Drop chunks shorter than this")
    ap.add_argument("--stats", action="store_true", help="Only print sizes for several caps")
    args = ap.parse_args()

    articles, drop_stats = build(args)
    all_items = select(articles, 10 ** 6, args.lead_first)
    print(f"Dropped syllables by section: {dict(drop_stats.most_common(12))}")
    describe(all_items, "no cap")
    if args.stats:
        for cap in (4, 5, 6, 7, 8, 10, 12):
            describe(select(articles, cap, args.lead_first), f"cap {cap:2d}")
        return

    items = select(articles, args.max_per_article, args.lead_first)
    describe(items, f"cap {args.max_per_article}")
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump([{"title": it["title"], "text": it["text"]} for it in items], f, ensure_ascii=False, indent=1)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            meta = {"short_id": f"c{i:04d}", **{k: v for k, v in it.items() if k != "text"}}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"Wrote {CORPUS_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()
