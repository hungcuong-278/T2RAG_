"""Remove the newest rows of an LLM cache (SQLite, table `cache`) so those prompts are asked again.

Used once on 2026-11-18: the gateway answered with hidden "thinking" from that evening, and those
answers went into caches that also hold the earlier, non-thinking answers. Rows are removed by
insertion order (SQLite rowid), so pass exactly the number of calls made after the change.

    python purge_cache_tail.py outputs/vnhistory_llm_cache_r1 --last 3            # dry run: show rows
    python purge_cache_tail.py outputs/vnhistory_llm_cache_r1 --last 3 --apply    # back up, then delete

A dry run prints the rows that would go and the newest row that stays; --apply first copies the
SQLite file to <file>.bak_<time>.
"""
import argparse
import glob
import json
import os
import shutil
import sqlite3
import time


def describe(row):
    rowid, message, metadata = row
    try:
        meta = json.loads(metadata or "{}")
    except ValueError:
        meta = {}
    out = meta.get("completion_tokens", meta.get("usage", {}).get("completion_tokens") if isinstance(meta.get("usage"), dict) else None)
    text = " ".join(str(message or "").split())
    return f"rowid {rowid:>6}  out_tokens {str(out):>6}  len {len(str(message or '')):>5}  {text[:60]!r}"


def purge(db, last, apply):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    rows = c.execute("SELECT rowid, message, metadata FROM cache ORDER BY rowid DESC LIMIT ?", (last + 1,)).fetchall()
    drop, keep = rows[:last], rows[last:]
    print(f"{db}: {total} rows; {len(drop)} newest would be removed")
    for r in reversed(drop):
        print("  - " + describe(r))
    if keep:
        print("  newest row kept: " + describe(keep[0]))
    if apply and drop:
        conn.close()
        backup = f"{db}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(db, backup)
        conn = sqlite3.connect(db)
        conn.execute(f"DELETE FROM cache WHERE rowid IN ({','.join('?' * len(drop))})", [r[0] for r in drop])
        conn.commit()
        left = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        print(f"  removed {total - left} rows; backup {backup}")
    conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dirs", nargs="+", help="Cache folders (each holds one *_cache.sqlite)")
    ap.add_argument("--last", type=int, required=True, help="Number of newest rows to remove")
    ap.add_argument("--apply", action="store_true", help="Really delete (default: dry run)")
    args = ap.parse_args(argv)
    for d in args.cache_dirs:
        dbs = sorted(glob.glob(os.path.join(d, "*_cache.sqlite")))
        if len(dbs) != 1:
            raise SystemExit(f"{d}: expected one *_cache.sqlite, found {dbs}")
        purge(dbs[0], args.last, args.apply)


if __name__ == "__main__":
    main()
