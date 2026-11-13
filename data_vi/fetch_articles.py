"""Probe and download Vietnamese Wikipedia articles for the vnhistory corpus (M1/M2).

Run from the repo root:
    python data_vi/fetch_articles.py probe    --seeds data_vi/work/seed_titles.txt
    python data_vi/fetch_articles.py download --articles data_vi/articles.txt

probe:    resolves every seed title (redirects, missing pages, disambiguation pages),
          collects in-set links and redirect aliases, and suggests an article list.
download: saves the plain-text extract of each article to data_vi/work/raw/ (resumable).
Only the standard MediaWiki API is used; no LLM calls.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from urllib.parse import quote

import requests

API = "https://vi.wikipedia.org/w/api.php"
WORK = os.path.join("data_vi", "work")
USER_AGENT = "T2RAG-vnhistory/0.1 (student research project on Vietnamese history QA; python-requests)"


def nfc(text):
    return unicodedata.normalize("NFC", text).strip()


def read_titles(path):
    """One title per line; '#' starts a comment; duplicates are dropped, order is kept."""
    titles, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            title = nfc(line.split("#", 1)[0])
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
    return titles


def slug(title):
    """File-system safe name that keeps Vietnamese letters."""
    return re.sub(r"[^\w]+", "_", title).strip("_")


class Api:
    """Small MediaWiki client with polite pacing and retries."""

    def __init__(self, sleep=0.2, max_tries=6):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.sleep = sleep
        self.max_tries = max_tries
        self.calls = 0

    def get(self, params):
        params = {**params, "action": "query", "format": "json", "formatversion": "2", "maxlag": "5"}
        for attempt in range(self.max_tries):
            try:
                resp = self.session.get(API, params=params, timeout=30)
                self.calls += 1
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                data = resp.json()
                if "error" in data:
                    if data["error"].get("code") == "maxlag":
                        raise RuntimeError("server lag")
                    sys.exit(f"Wikipedia API error: {data['error']}")
                time.sleep(self.sleep)
                return data
            except (requests.RequestException, RuntimeError, ValueError) as err:
                wait = min(60, 2 ** (attempt + 1))
                print(f"  retry {attempt + 1}/{self.max_tries} in {wait}s ({err})", flush=True)
                time.sleep(wait)
        sys.exit("Wikipedia API keeps failing; try again later.")

    def get_all(self, params):
        """Follow API continuation and yield every response."""
        cont = {}
        while True:
            data = self.get({**params, **cont})
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]


def resolve(api, titles):
    """Map each seed title to its page: ok / redirect / missing / disambig."""
    results = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = api.get({"titles": "|".join(batch), "redirects": "1",
                        "prop": "info|pageprops", "ppprop": "disambiguation"})
        query = data.get("query", {})
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        pages = {p["title"]: p for p in query.get("pages", [])}
        for seed in batch:
            name = normalized.get(seed, seed)
            target = redirects.get(name, name)
            page = pages.get(target)
            row = {"seed": seed, "title": target, "pageid": None, "bytes": 0, "status": "missing"}
            if page and not page.get("missing") and not page.get("invalid"):
                row.update(pageid=page["pageid"], bytes=page.get("length", 0))
                if "disambiguation" in page.get("pageprops", {}):
                    row["status"] = "disambig"
                elif target != seed:
                    row["status"] = "redirect"
                else:
                    row["status"] = "ok"
            results[seed] = row
    return results


def links_and_aliases(api, pageid):
    """Article-namespace links of a page and the redirect titles that point to it."""
    links, aliases = set(), set()
    for data in api.get_all({"pageids": str(pageid), "prop": "links|redirects", "pllimit": "max",
                             "plnamespace": "0", "rdlimit": "max", "rdnamespace": "0"}):
        for page in data.get("query", {}).get("pages", []):
            links.update(nfc(l["title"]) for l in page.get("links", []))
            aliases.update(nfc(r["title"]) for r in page.get("redirects", []))
    return links, aliases


def search(api, text, limit=3):
    data = api.get({"list": "search", "srsearch": text, "srlimit": str(limit), "srnamespace": "0"})
    return [nfc(s["title"]) for s in data.get("query", {}).get("search", [])]


def cmd_probe(args):
    api = Api(sleep=args.sleep)
    seeds = read_titles(args.seeds)
    print(f"{len(seeds)} seed titles. Resolving...", flush=True)
    rows = resolve(api, seeds)

    # Unique pages (two seeds may point to the same article, e.g. a personal name and a temple name).
    pages = {}
    for r in rows.values():
        if r["status"] in ("ok", "redirect", "disambig") and r["title"] not in pages:
            pages[r["title"]] = r
    print(f"{len(pages)} distinct pages. Collecting links and aliases...", flush=True)
    links, aliases = {}, {}
    for i, (title, r) in enumerate(pages.items(), 1):
        links[title], aliases[title] = links_and_aliases(api, r["pageid"])
        if i % 20 == 0 or i == len(pages):
            print(f"  {i}/{len(pages)} pages, {api.calls} API calls", flush=True)

    # Name -> canonical title, so that links through redirects still count.
    articles = [t for t, r in pages.items() if r["status"] != "disambig"]
    canon = {}
    for t in articles:
        canon[t] = t
        for a in aliases[t]:
            canon.setdefault(a, t)
    out_links = {t: sorted({canon[l] for l in links[t] if l in canon and canon[l] != t}) for t in articles}
    in_links = {t: set() for t in articles}
    for t, outs in out_links.items():
        for o in outs:
            in_links[o].add(t)

    report, suggested = [], []
    for seed in seeds:
        r = rows[seed]
        t = r["title"]
        degree = len(set(out_links.get(t, [])) | in_links.get(t, set()))
        note = ""
        if r["status"] == "missing":
            note = "search: " + " | ".join(search(api, seed))
        elif r["status"] == "disambig":
            options = [l for l in sorted(links[t]) if l.startswith(seed.split(" (")[0])][:8]
            note = "options: " + " | ".join(options or sorted(links[t])[:8])
        else:
            if r["bytes"] < args.min_bytes:
                note = f"short (<{args.min_bytes} bytes)"
            if degree < args.min_links:
                note = (note + "; " if note else "") + f"few links in set ({degree})"
            if not note and t not in suggested:
                suggested.append(t)
        report.append({"seed": seed, "status": r["status"], "title": t, "pageid": r["pageid"],
                       "bytes": r["bytes"], "out_in_set": len(out_links.get(t, [])),
                       "in_in_set": len(in_links.get(t, set())), "degree": degree,
                       "aliases": len(aliases.get(t, [])), "note": note})

    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "probe_report.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    with open(os.path.join(WORK, "aliases.json"), "w", encoding="utf-8") as f:
        json.dump({t: sorted(aliases[t]) for t in articles}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(WORK, "link_graph.json"), "w", encoding="utf-8") as f:
        json.dump(out_links, f, ensure_ascii=False, indent=1)
    with open(os.path.join(WORK, "articles_suggested.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(suggested) + "\n")

    def show(label, items):
        print(f"\n{label}: {len(items)}")
        for it in items:
            print("  - " + it)

    show("MISSING (fix the title)", [f"{r['seed']}  ->  {r['note']}" for r in report if r["status"] == "missing"])
    show("DISAMBIGUATION (pick one)", [f"{r['seed']}  ->  {r['note']}" for r in report if r["status"] == "disambig"])
    show("REDIRECTED (ok, real title differs)", [f"{r['seed']}  ->  {r['title']}" for r in report if r["status"] == "redirect"])
    show("FLAGGED (short or weakly linked)", [f"{r['title']}  ({r['bytes']} bytes, degree {r['degree']}): {r['note']}"
                                             for r in report if r["status"] in ("ok", "redirect") and r["note"]])
    sizes = sorted(pages[t]["bytes"] for t in suggested)
    print(f"\nSuggested articles: {len(suggested)}  (total {sum(sizes) / 1e6:.1f} MB of wikitext, "
          f"median {sizes[len(sizes) // 2] if sizes else 0} bytes)")
    print(f"API calls: {api.calls}. Files written to {WORK}/: probe_report.csv, aliases.json, "
          f"link_graph.json, articles_suggested.txt")


def cmd_download(args):
    api = Api(sleep=args.sleep)
    titles = read_titles(args.articles)
    raw_dir = os.path.join(WORK, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    t0, done, skipped, empty = time.time(), 0, 0, []
    for i, title in enumerate(titles, 1):
        path = os.path.join(raw_dir, slug(title) + ".json")
        if os.path.exists(path) and not args.force:
            skipped += 1
            continue
        data = api.get({"titles": title, "redirects": "1", "prop": "extracts|info|revisions",
                        "explaintext": "1", "exsectionformat": "wiki", "exlimit": "1",
                        "rvprop": "ids|timestamp"})
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            print(f"[{i}/{len(titles)}] MISSING: {title}", flush=True)
            empty.append(title)
            continue
        page = pages[0]
        rev = (page.get("revisions") or [{}])[0]
        record = {"title": nfc(page["title"]), "requested_title": title, "pageid": page["pageid"],
                  "revid": rev.get("revid"), "timestamp": rev.get("timestamp"),
                  "url": "https://vi.wikipedia.org/wiki/" + quote(page["title"].replace(" ", "_")),
                  "length_bytes": page.get("length"), "extract": nfc(page.get("extract", ""))}
        if len(record["extract"]) < 500:
            empty.append(title)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)
        done += 1
        print(f"[{i}/{len(titles)}] {time.time() - t0:5.0f}s  {len(record['extract']):7d} chars  {record['title']}",
              flush=True)
    print(f"\nDownloaded {done}, already present {skipped}, problems {len(empty)}.")
    for t in empty:
        print(f"  check: {t} (missing or extract shorter than 500 chars)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--seeds", default=os.path.join(WORK, "seed_titles.txt"))
    p.add_argument("--min_bytes", type=int, default=8000, help="Flag articles shorter than this (wikitext bytes)")
    p.add_argument("--min_links", type=int, default=3, help="Flag articles linked to fewer in-set articles")
    p.add_argument("--sleep", type=float, default=0.2)
    d = sub.add_parser("download")
    d.add_argument("--articles", default=os.path.join("data_vi", "articles.txt"))
    d.add_argument("--force", action="store_true", help="Download again even if the file exists")
    d.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    cmd_probe(args) if args.cmd == "probe" else cmd_download(args)


if __name__ == "__main__":
    main()
