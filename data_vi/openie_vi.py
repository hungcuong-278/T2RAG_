"""Vietnamese OpenIE for the vnhistory corpus (M4): 2 LLM calls per chunk (NER, then triples).

Run from the project root (Windows PowerShell or any shell):
    pip install openai
    python data_vi/openie_vi.py check                          # which models answer with this key (1 call each)
    python data_vi/openie_vi.py run --pilot 20                 # pilot on 20 sampled chunks
    python data_vi/openie_vi.py review                         # pilot results -> Markdown for reading
    python data_vi/openie_vi.py run --all --min_interval 6     # whole corpus (resumable)
    python data_vi/openie_vi.py build                          # write outputs/vnhistory_univ/openie_results_ner.json

The API key is read from the variable named by --api_key_env (default GOOGLE_API_KEY); if it is
not set, the script asks for it (input hidden). --base_url selects the OpenAI-compatible endpoint:
Google's own (default) or a gateway that resells Gemini.
Every chunk result is cached under data_vi/cache/openie/<PROMPT_VERSION>/<md5>.json, so an
interrupted run continues where it stopped and a finished chunk is never paid for twice.
"""
import argparse
import getpass
import hashlib
import json
import os
import random
import re
import sys
import threading
import traceback
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts_vi  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese output in a Windows console
except Exception:
    pass

LOG_DIR = os.path.join("data_vi", "work", "logs")


class Tee:
    """Mirror everything printed to the console into a log file (so runs can be reviewed later)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.console = sys.stdout
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.file.write(text)
            self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

CORPUS_PATH = os.path.join("reproduce", "dataset", "vnhistory_corpus.json")
OUT_PATH = os.path.join("outputs", "vnhistory_univ", "openie_results_ner.json")
CACHE_ROOT = os.path.join("data_vi", "cache", "openie")
ENV_FILE = ".env"   # project-root secrets file (never commit it)
PILOT_IDS = os.path.join("data_vi", "work", "openie_pilot_ids.txt")
META_PATH = os.path.join("data_vi", "work", "chunks_meta.jsonl")
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GATEWAY_BASE_URL = "https://1gw.gwai.cloud/v1"   # reseller; the group's current key only serves Claude models there
BASE_URL = GOOGLE_BASE_URL
DEFAULT_KEY_ENV = "GOOGLE_API_KEY"


def nfc(text):
    return unicodedata.normalize("NFC", text)


def load_corpus():
    corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
    meta = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]
    chunks = []
    for c, m in zip(corpus, meta):
        passage = c["title"] + "\n" + c["text"]
        chunks.append({"short_id": m["short_id"], "title": c["title"], "text": c["text"], "passage": passage,
                       "idx": "chunk-" + hashlib.md5(passage.encode()).hexdigest(), "section": m["section"]})
    return chunks


THINKING_BODY = {"default": {}, "off": {"thinking": {"type": "disabled"}}, "low": {"output_config": {"effort": "low"}}}
RUN = {"model": None, "thinking": "default"}   # set from the command line; selects the cache folder


def cache_dir():
    tag = RUN["model"] + ("" if RUN["thinking"] == "default" else f"__thinking-{RUN['thinking']}")
    return os.path.join(CACHE_ROOT, prompts_vi.PROMPT_VERSION, tag)


def cache_path(chunk):
    return os.path.join(cache_dir(), chunk["idx"][len("chunk-"):] + ".json")


# ---------------------------------------------------------------- parsing

def parse_json_obj(text, key):
    """Return the list under `key` from a model reply; tolerate code fences and extra text."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    candidates = [t] + re.findall(r"\{.*\}", t, flags=re.S)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get(key), list):
            return obj[key]
    return None


def clean_entities(raw):
    seen, out = set(), []
    for e in raw or []:
        if isinstance(e, str):
            e = nfc(e).strip()
            if e and e not in seen:
                seen.add(e)
                out.append(e)
    return out


def clean_triples(raw):
    seen, out = set(), []
    for t in raw or []:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        t = [nfc(str(x)).strip() for x in t]
        if not all(t) or tuple(t) in seen:
            continue
        seen.add(tuple(t))
        out.append(t)
    return out


def ungrounded(triples, passage):
    """Triples whose subject and object both do not appear in the passage (possible hallucination)."""
    low = passage.lower()
    return [t for t in triples if t[0].lower() not in low and t[2].lower() not in low]


# ---------------------------------------------------------------- LLM calls

SPEND_PATH = os.path.join(CACHE_ROOT, "key_spend.json")
EXHAUSTED_HINTS = ("insufficient", "quota", "credit", "balance", "limit exceeded", "cost limit", "exceeded",
                   "payment", "expired", "billing")


def read_keys(env_name):
    """All keys whose variable name starts with env_name (GWAI_API_KEY, GWAI_API_KEY_2, ...), env first, then .env."""
    found = {}
    if os.path.exists(ENV_FILE):
        for line in open(ENV_FILE, encoding="utf-8-sig"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, value = line.partition("=")
            if name.strip().startswith(env_name) and value.strip():
                found[name.strip()] = value.strip().strip('"').strip("'")
    for name, value in os.environ.items():
        if name.startswith(env_name) and value.strip():
            found[name] = value.strip()
    return [found[n] for n in sorted(found, key=lambda n: (len(n), n))]


def fingerprint(key):
    return f"...{key[-6:]}" if len(key) > 6 else "(short)"


class KeyRing:
    """Rotates through API keys; each key is retired at `budget` USD of estimated spend or on a quota error."""

    def __init__(self, args):
        keys = read_keys(args.api_key_env)
        if not keys:
            keys = [get_key(args.api_key_env)]
        self.keys, self.args, self.i = keys, args, 0
        self.lock = threading.Lock()
        self.spend = json.load(open(SPEND_PATH, encoding="utf-8")) if os.path.exists(SPEND_PATH) else {}
        self.dead = set()
        print(f"{len(keys)} key(s) from {args.api_key_env}*: " + ", ".join(
            f"{fingerprint(k)} spent ${self.spend.get(fingerprint(k), 0):.2f}" for k in keys), flush=True)
        self._select(start=0)

    def _usable(self, k):
        return fingerprint(k) not in self.dead and self.spend.get(fingerprint(k), 0) < self.args.key_budget

    def _select(self, start):
        for j in list(range(start, len(self.keys))) + list(range(0, start)):
            if self._usable(self.keys[j]):
                self.i = j
                self.client = make_client(self.args.api_format, self.args.base_url, self.keys[j])
                print(f"  using key {fingerprint(self.keys[j])} "
                      f"(spent ${self.spend.get(fingerprint(self.keys[j]), 0):.2f} of ${self.args.key_budget:.2f})", flush=True)
                return
        raise StopIteration("all keys used up: add a new key to .env and run the same command again")

    @property
    def current(self):
        return fingerprint(self.keys[self.i])

    def snapshot(self):
        """(client, fingerprint) of the key a call will use; the call is billed to that key even if
        another thread switches keys while it is running."""
        with self.lock:
            if not self._usable(self.keys[self.i]):
                self._select(start=self.i + 1)
            return self.client, self.current

    def charge(self, fp, t_in, t_out):
        cost = t_in / 1e6 * self.args.price_in + t_out / 1e6 * self.args.price_out
        with self.lock:
            self.spend[fp] = round(self.spend.get(fp, 0) + cost, 6)
            os.makedirs(CACHE_ROOT, exist_ok=True)
            with open(SPEND_PATH, "w", encoding="utf-8") as f:
                json.dump(self.spend, f, indent=1)
            if fp == self.current and self.spend[fp] >= self.args.key_budget:
                print(f"  key {fp} reached ${self.spend[fp]:.2f}; switching key", flush=True)
                self._select(start=self.i + 1)
        return cost

    def retire(self, fp, reason):
        with self.lock:
            if fp in self.dead:
                return
            self.dead.add(fp)
            print(f"  key {fp} retired ({reason[:120]})", flush=True)
            if fp == self.current:
                self._select(start=self.i + 1)


def error_detail(err, limit=400):
    """Status code plus the body the server sent back (never contains the key)."""
    body = getattr(err, "body", None)
    if not body:
        resp = getattr(err, "response", None)
        body = getattr(resp, "text", "") if resp is not None else ""
    text = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body or "")
    text = re.sub(r"\s+", " ", text)
    return f"{getattr(err, 'status_code', '')} {type(err).__name__}: {text[:limit] or str(err)[:limit]}"


def get_key(env_name):
    """API key from the given environment variable, or typed in with hidden input.

    Prints only the length and the shape of the key (never the key itself), so a bad paste
    (empty, wrong key, stray quotes or control characters) is visible at once.
    """
    key, source = os.getenv(env_name), "environment variable"
    if not key and os.path.exists(ENV_FILE):
        # Optional local secrets file: lines like GWAI_API_KEY=sk-...; listed in .gitignore.
        for line in open(ENV_FILE, encoding="utf-8-sig"):
            name, _, value = line.strip().partition("=")
            if name.strip() == env_name and value.strip():
                key, source = value.strip(), ENV_FILE
    if not key:
        key, source = getpass.getpass(f"{env_name} (input hidden): "), "hidden input"
    key = "".join(ch for ch in key if ch.isprintable()).strip().strip('"').strip("'")
    warn = " (Anthropic-format key: use api_format 'anthropic')" if key.startswith("sk-ant-") else ""
    print(f"Key from {source}: {len(key)} characters{warn}")
    return key


def make_client(api_format, base_url, key, timeout=300):
    """OpenAI-compatible client (Google endpoint) or Anthropic Messages client (reseller gateway)."""
    if api_format == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=key, base_url=base_url, max_retries=0, timeout=timeout)
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=timeout)


_SEND_TEMPERATURE = {"anthropic": True}


def complete(client, api_format, model, messages, json_mode=False, reasoning_effort=None, max_tokens=32000,
             extra=None):
    """One chat call in either format; returns (text, input_tokens, output_tokens, stop_reason).

    Thinking models spend part of max_tokens on hidden reasoning, so the budget is generous."""
    if api_format == "anthropic":
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": turns}
        if system:
            kwargs["system"] = system
        # Recent anthropic SDKs dropped the `temperature` argument; send it as a raw body field
        # instead, and stop sending it if the gateway rejects it.
        body = dict(extra or {})
        if _SEND_TEMPERATURE["anthropic"]:
            body["temperature"] = 0
        if body:
            kwargs["extra_body"] = body
        try:
            resp = client.messages.create(**kwargs)
        except Exception as err:
            if "temperature" not in str(err) or "temperature" not in kwargs.get("extra_body", {}):
                raise
            _SEND_TEMPERATURE["anthropic"] = False
            print("  gateway rejected temperature=0; continuing with its default sampling", flush=True)
            kwargs["extra_body"].pop("temperature")
            resp = client.messages.create(**kwargs)
        if not getattr(resp, "content", None):
            raise RuntimeError("empty response from the gateway")
        text = "".join(getattr(b, "text", "") or "" for b in resp.content)
        usage = getattr(resp, "usage", None)
        return (text, getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0,
                getattr(resp, "stop_reason", None))
    kwargs = {"model": model, "messages": messages, "temperature": 0}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if extra:
        kwargs["extra_body"] = dict(extra)
    resp = client.chat.completions.create(**kwargs)
    usage = getattr(resp, "usage", None)
    return (resp.choices[0].message.content or "", getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0, getattr(resp.choices[0], "finish_reason", None))


class Caller:
    def __init__(self, args):
        self.ring = KeyRing(args)
        self.args = args
        self.cost = 0.0
        self.calls = self.errors = self.tokens_in = self.tokens_out = 0
        self.last = 0.0
        self.lock = threading.Lock()
        self.json_mode = args.api_format != "anthropic"

    def chat(self, messages, label=""):
        a = self.args
        one_line = a.workers > 1
        for attempt in range(a.max_tries):
            with self.lock:
                wait = self.last + a.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
                if a.max_calls and self.calls >= a.max_calls:
                    raise StopIteration("max_calls reached")
                self.calls += 1
                n = self.calls
            client, fp = self.ring.snapshot()
            start = time.time()
            if not one_line:
                print(f"    {label} call {n} started {time.strftime('%H:%M:%S')} ...", end="", flush=True)
            try:
                text, t_in, t_out, stop = complete(client, a.api_format, a.model, messages, json_mode=self.json_mode,
                                                   reasoning_effort=a.reasoning_effort, extra=a.extra_body)
                self.last_stop = stop
                done = f"done in {time.time() - start:4.1f}s (tokens in {t_in}, out {t_out}, stop {stop})"
                print(f"    {label} call {n} {done}" if one_line else f" {done}", flush=True)
                cost = self.ring.charge(fp, t_in, t_out)
                with self.lock:
                    self.tokens_in += t_in
                    self.tokens_out += t_out
                    self.cost += cost
                return text
            except StopIteration:
                raise
            except Exception as err:  # rate limits, server errors, unsupported parameters
                with self.lock:
                    self.errors += 1
                print(f"    {label} call {n} failed after {time.time() - start:4.1f}s" if one_line
                      else f" failed after {time.time() - start:4.1f}s", flush=True)
                msg = error_detail(err)
                status = getattr(err, "status_code", None)
                if status in (401, 402, 403, 429) and any(h in msg.lower() for h in EXHAUSTED_HINTS):
                    self.ring.retire(fp, msg)
                    continue
                if status == 400 and self.json_mode and "response_format" in msg:
                    print("  JSON mode not accepted by the endpoint; continuing without it", flush=True)
                    self.json_mode = False
                    continue
                if status in (400, 401, 403, 404):
                    raise SystemExit(f"API error: {msg}")
                backoff = min(120, 5 * 2 ** attempt)
                print(f"  {msg[:160]}: retry {attempt + 1}/{a.max_tries} in {backoff}s", flush=True)
                time.sleep(backoff)
        raise RuntimeError("LLM call failed after retries")


def ask_json(caller, messages, key, label, tries=3):
    """Call until the reply parses as JSON with `key` (a reply cut off mid-way is asked again)."""
    for k in range(tries):
        raw = caller.chat(messages, label=label if k == 0 else f"{label} (re-ask {k})")
        parsed = parse_json_obj(raw, key)
        if parsed is not None:
            return raw, parsed, getattr(caller, "last_stop", None), k
        print(f"    {label}: reply not valid JSON (stop {getattr(caller, 'last_stop', None)}, "
              f"{len(raw)} chars); asking again", flush=True)
    return raw, None, getattr(caller, "last_stop", None), tries


def process(chunk, caller):
    passage = chunk["passage"]
    sid = chunk["short_id"]
    ner_raw, ner, ner_stop, ner_re = ask_json(caller, prompts_vi.ner_messages(passage), "named_entities", f"{sid} NER    ")
    entities = clean_entities(ner)
    tri_raw, tri, tri_stop, tri_re = ask_json(caller, prompts_vi.triple_messages(passage, entities), "triples",
                                              f"{sid} triples")
    triples = clean_triples(tri)
    return {"idx": chunk["idx"], "short_id": sid, "passage": passage,
            "prompt_version": prompts_vi.PROMPT_VERSION, "model": caller.args.model,
            "ner_raw": ner_raw, "triples_raw": tri_raw, "entities": entities, "triples": triples,
            "parse_ok": ner is not None and tri is not None, "stop": [ner_stop, tri_stop], "re_asks": ner_re + tri_re,
            "ungrounded": ungrounded(triples, passage), "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def is_bad(path):
    """A cached result whose NER or triple reply could not be parsed (e.g. cut off)."""
    rec = json.load(open(path, encoding="utf-8"))
    return parse_json_obj(rec.get("ner_raw"), "named_entities") is None or \
        parse_json_obj(rec.get("triples_raw"), "triples") is None


# ---------------------------------------------------------------- commands

def pick_pilot(chunks, n, seed):
    """n chunks from n different articles, half lead sections and half body sections."""
    rng = random.Random(seed)
    leads = [c for c in chunks if c["section"] == "(lead)"]
    bodies = [c for c in chunks if c["section"] != "(lead)"]
    rng.shuffle(leads)
    rng.shuffle(bodies)
    picked, titles = [], set()
    for pool, k in ((leads, n // 2), (bodies, n - n // 2)):
        for c in pool:
            if len([p for p in picked if (p["section"] == "(lead)") == (pool is leads)]) >= k:
                break
            if c["title"] not in titles:
                titles.add(c["title"])
                picked.append(c)
    return picked


def cmd_run(args):
    chunks = load_corpus()
    if args.pilot:
        todo = pick_pilot(chunks, args.pilot, args.seed)
        os.makedirs(os.path.dirname(PILOT_IDS), exist_ok=True)
        with open(PILOT_IDS, "w", encoding="utf-8") as f:
            f.write("\n".join(c["short_id"] for c in todo) + "\n")
    elif args.ids:
        wanted = set(args.ids)
        todo = [c for c in chunks if c["short_id"] in wanted]
    else:
        todo = chunks
    pending = [c for c in todo if not os.path.exists(cache_path(c))
               or (args.redo_bad and is_bad(cache_path(c)))]
    print(f"Prompt {prompts_vi.PROMPT_VERSION} | model {args.model} | thinking {args.thinking} | "
          f"workers {args.workers} | {len(todo)} chunks selected, "
          f"{len(todo) - len(pending)} already cached, {len(pending)} to do ({2 * len(pending)} LLM calls)", flush=True)
    if not pending:
        return
    caller = Caller(args)
    os.makedirs(os.path.dirname(cache_path(pending[0])), exist_ok=True)
    t0, done, failed, finished = time.time(), 0, [], 0

    def report(chunk, rec):
        nonlocal done, finished
        finished += 1
        if rec is None:
            failed.append(chunk["short_id"])
            return
        done += 1
        elapsed = time.time() - t0
        eta = elapsed / finished * (len(pending) - finished)
        flag = "" if rec["parse_ok"] else "  PARSE-PROBLEM"
        print(f"[{finished}/{len(pending)}] {chunk['short_id']} {len(rec['entities']):2d} ent {len(rec['triples']):2d} tri"
              f"{'  ' + str(len(rec['ungrounded'])) + ' ungrounded' if rec['ungrounded'] else ''}{flag} | "
              f"calls {caller.calls} err {caller.errors} | ${caller.cost:.2f} on key {caller.ring.current} | "
              f"{elapsed / 60:5.1f} min, ETA {eta / 60:5.1f} min", flush=True)

    stop = threading.Event()

    def safe_process(chunk):
        if stop.is_set():
            return None
        try:
            rec = process(chunk, caller)
            # Save right here in the worker so a finished chunk is never lost
            with open(cache_path(chunk), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            return rec
        except StopIteration as err:
            stop.set()
            print(f"    stopping: {err}", flush=True)
            return None
        except RuntimeError as err:
            print(f"    {chunk['short_id']} FAILED: {err}", flush=True)
            return None
        except Exception:
            print(f"    {chunk['short_id']} CRASHED:\n{traceback.format_exc()}", flush=True)
            return None

    try:
        if args.workers <= 1:
            for chunk in pending:
                if stop.is_set():
                    break
                report(chunk, safe_process(chunk))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(safe_process, c): c for c in pending}
                for fut in as_completed(futures):
                    try:
                        rec = fut.result()
                        if rec is not None or not stop.is_set():
                            report(futures[fut], rec)
                    except Exception:
                        print(f"    report error for {futures[fut]['short_id']}:\n{traceback.format_exc()}", flush=True)
                    if stop.is_set():
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
    except StopIteration:
        print(f"Stopped: --max_calls {args.max_calls} reached. Run the same command again later to continue.")
    except KeyboardInterrupt:
        print("Interrupted. Finished chunks are cached; run the same command again to continue.")
    print(f"\nDone {done}, failed {len(failed)} {failed if failed else ''} | LLM calls {caller.calls}, "
          f"errors {caller.errors}, tokens in {caller.tokens_in}, out {caller.tokens_out}, "
          f"estimated cost ${caller.cost:.3f}, {(time.time() - t0) / 60:.1f} min")
    print("Spend per key: " + ", ".join(f"{fp} ${v:.2f}" for fp, v in caller.ring.spend.items()))


def load_cached(chunks):
    out = {}
    for c in chunks:
        p = cache_path(c)
        if os.path.exists(p):
            out[c["short_id"]] = json.load(open(p, encoding="utf-8"))
    return out


def cmd_review(args):
    chunks = {c["short_id"]: c for c in load_corpus()}
    ids = [l.strip() for l in open(PILOT_IDS, encoding="utf-8") if l.strip()] if not args.ids else args.ids
    cached = load_cached([chunks[i] for i in ids])
    path = os.path.join("data_vi", "work", f"openie_pilot_{os.path.relpath(cache_dir(), CACHE_ROOT).replace(os.sep, '_')}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Pilot OpenIE {os.path.relpath(cache_dir(), CACHE_ROOT)}\n\n")
        for sid in ids:
            rec = cached.get(sid)
            c = chunks[sid]
            f.write(f"## {sid} | {c['title']} | {c['section']}\n\n{c['text']}\n\n")
            if rec is None:
                f.write("(not processed yet)\n\n")
                continue
            f.write("**Thực thể:** " + "; ".join(rec["entities"]) + "\n\n**Bộ ba:**\n\n")
            for t in rec["triples"]:
                mark = "  (?)" if t in rec["ungrounded"] else ""
                f.write(f"- {t[0]} | {t[1]} | {t[2]}{mark}\n")
            f.write("\nNhận xét:\n\n")
    tri = [len(r["triples"]) for r in cached.values()]
    ent = [len(r["entities"]) for r in cached.values()]
    if tri:
        print(f"{len(cached)}/{len(ids)} processed | triples/chunk mean {sum(tri) / len(tri):.1f} "
              f"(min {min(tri)}, max {max(tri)}) | entities/chunk mean {sum(ent) / len(ent):.1f} | "
              f"parse problems {sum(parse_json_obj(r['ner_raw'], 'named_entities') is None or parse_json_obj(r['triples_raw'], 'triples') is None for r in cached.values())} | "
              f"ungrounded triples {sum(len(r['ungrounded']) for r in cached.values())}")
    print(f"Review file: {path}")


def cmd_check(args):
    """List Flash models visible to the key and send one tiny request to each candidate."""
    key = get_key(args.api_key_env)
    base = args.base_url.rstrip("/")
    if args.api_format == "anthropic" or "googleapis.com" in base:
        candidates = [base]
    else:
        candidates = [base, base[:-3]] if base.endswith("/v1") else [base, base + "/v1"]
    for url in candidates:
        print(f"Base URL {url} | format {args.api_format}")
        client = make_client(args.api_format, url, key, timeout=60)
        try:
            names = sorted(m.id for m in client.models.list())
            flash = [n for n in names if "flash" in n.lower()]
            print(f"  {len(names)} models listed; Flash: {', '.join(flash) or '(none)'}")
            if names and not flash:
                print("  first models:", ", ".join(names[:40]))
        except Exception as err:
            print(f"  model listing failed: {error_detail(err, 200)}")
        ok = False
        for model in args.models:
            try:
                text, t_in, t_out, _ = complete(client, args.api_format, model, [
                    {"role": "user", "content": 'Trả lời đúng một từ JSON: {"ok": true}'}], max_tokens=1024,
                    extra=args.extra_body)
                ok = True
                print(f"  OK   {model:24s} reply={text.strip()[:40]!r} tokens in/out={t_in}/{t_out}")
            except Exception as err:
                print(f"  FAIL {model:24s} {error_detail(err)}")
            time.sleep(args.pause)
        if ok:
            print(f"\nWorking base URL: {url}")
            return
    print("\nNo base URL worked; check the key and the gateway documentation.")


def cmd_build(args):
    chunks = load_corpus()
    cached = load_cached(chunks)
    missing = [c["short_id"] for c in chunks if c["short_id"] not in cached]
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} chunks have no OpenIE result yet (e.g. {missing[:5]}). "
                         "Finish `run --all` first, or pass --allow_missing for a test file.")
    docs = []
    for c in chunks:
        rec = cached.get(c["short_id"], {"entities": [], "triples": []})
        docs.append({"idx": c["idx"], "title": c["title"], "text": c["text"], "passage": c["passage"],
                     "extracted_entities": rec["entities"], "extracted_triples": rec["triples"]})
    ents = [e for d in docs for e in d["extracted_entities"]]
    out = {"docs": docs,
           "avg_ent_chars": round(sum(len(e) for e in ents) / len(ents), 4) if ents else 0,
           "avg_ent_words": round(sum(len(e.split()) for e in ents) / len(ents), 4) if ents else 0}
    path = args.out or OUT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    n_tri = sum(len(d["extracted_triples"]) for d in docs)
    print(f"Wrote {path}: {len(docs)} docs, {n_tri} triples ({n_tri / len(docs):.1f}/chunk), "
          f"{len(missing)} chunks without results")


CONFIG_PATH = os.path.join("data_vi", "llm_config.json")
DEFAULT_CONFIG = {"model": "gemini-3.8-flash", "api_format": "anthropic", "base_url": "https://1gw.gwai.cloud",
                  "api_key_env": "GWAI_API_KEY", "min_interval": 2.0, "max_calls": 0, "reasoning_effort": None,
                  "workers": 4, "thinking": "default", "extra_body": {},
                  "price_in": 0.75, "price_out": 3.75, "key_budget": 9.5}
# Google AI Studio instead: {"api_format": "openai", "base_url": GOOGLE_BASE_URL, "api_key_env": "GOOGLE_API_KEY",
#                            "min_interval": 13.0}  (free tier: 5 requests/minute, 20/day for Flash)


def load_config():
    """Settings shared by all commands; the file holds no secret (only the NAME of the key variable)."""
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        print(f"Created {CONFIG_PATH} with default settings")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(json.load(open(CONFIG_PATH, encoding="utf-8")))
    return cfg


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("run", "check", "build"):
        log_path = os.path.join(LOG_DIR, f"openie_{sys.argv[1]}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        sys.stdout = Tee(log_path)
        print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} | python {' '.join(sys.argv)} | log: {log_path}")
    cfg = load_config()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", type=int, help="Sample this many chunks (saved to openie_pilot_ids.txt)")
    g.add_argument("--ids", nargs="+", help="Process these short ids (e.g. c0001 c0002)")
    g.add_argument("--all", action="store_true", help="Process the whole corpus")
    r.add_argument("--model", default=cfg["model"])
    r.add_argument("--base_url", default=cfg["base_url"], help="Endpoint (Google or the reseller gateway)")
    r.add_argument("--api_format", default=cfg["api_format"], choices=["openai", "anthropic"])
    r.add_argument("--api_key_env", default=cfg["api_key_env"], help="Environment variable holding the key")
    r.add_argument("--min_interval", type=float, default=cfg["min_interval"], help="Seconds between calls (60 / RPM)")
    r.add_argument("--max_calls", type=int, default=cfg["max_calls"], help="Stop after this many calls (daily quota guard)")
    r.add_argument("--max_tries", type=int, default=6)
    r.add_argument("--reasoning_effort", default=cfg["reasoning_effort"], help="Optional: low / medium / high (if supported)")
    r.add_argument("--seed", type=int, default=2026)
    r.add_argument("--redo_bad", action="store_true", help="Also redo cached chunks whose replies were cut off")
    r.add_argument("--workers", type=int, default=cfg["workers"], help="Chunks processed in parallel")
    r.add_argument("--price_in", type=float, default=cfg["price_in"], help="USD per 1M input tokens")
    r.add_argument("--price_out", type=float, default=cfg["price_out"], help="USD per 1M output tokens")
    r.add_argument("--key_budget", type=float, default=cfg["key_budget"], help="Retire a key at this estimated spend")
    r.add_argument("--thinking", default=cfg["thinking"], choices=list(THINKING_BODY),
                   help="Model thinking: default, off, or low (if the endpoint supports it)")
    r.add_argument("--extra_body", type=json.loads, default=cfg["extra_body"],
                   help='Extra request fields as JSON, e.g. \'{"thinking": {"type": "disabled"}}\'')
    c = sub.add_parser("check")
    c.add_argument("--models", nargs="+", default=[cfg["model"]])
    c.add_argument("--base_url", default=cfg["base_url"])
    c.add_argument("--api_format", default=cfg["api_format"], choices=["openai", "anthropic"])
    c.add_argument("--api_key_env", default=cfg["api_key_env"])
    c.add_argument("--pause", type=float, default=2.0, help="Seconds between test calls")
    c.add_argument("--extra_body", type=json.loads, default=cfg["extra_body"])
    c.add_argument("--thinking", default=cfg["thinking"], choices=list(THINKING_BODY))
    v = sub.add_parser("review")
    v.add_argument("--ids", nargs="+")
    v.add_argument("--model", default=cfg["model"])
    v.add_argument("--thinking", default=cfg["thinking"], choices=list(THINKING_BODY))
    b = sub.add_parser("build")
    b.add_argument("--model", default=cfg["model"])
    b.add_argument("--thinking", default=cfg["thinking"], choices=list(THINKING_BODY))
    b.add_argument("--allow_missing", action="store_true")
    b.add_argument("--out", default=None)
    args = ap.parse_args()
    if hasattr(args, "thinking"):
        RUN.update(model=getattr(args, "model", cfg["model"]) if args.cmd != "check" else cfg["model"],
                   thinking=args.thinking)
        args.extra_body = {**THINKING_BODY[args.thinking], **(getattr(args, "extra_body", None) or {})}
    {"run": cmd_run, "check": cmd_check, "review": cmd_review, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
