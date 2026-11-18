"""Local OpenAI-compatible endpoint in front of the reseller gateway (Anthropic Messages format).

run_colab.py / HippoRAG talk to an OpenAI chat-completions endpoint; the gateway only accepts
/v1/messages. This shim translates one into the other without touching the experiment code.

    GWAI_API_KEY=... python gw_proxy.py --port 4000
    python run_colab.py ... --llm_base_url http://localhost:4000/v1 --llm_name gemini-3.8-flash

Keys come from GWAI_API_KEY, GWAI_API_KEY_2, ... (environment only; never printed).
A key that hits the gateway cost limit (429 cost_limit_exceeded) is dropped and the call is
retried with the next key. max_tokens is always raised to --max_tokens because the model
spends most of its output budget on hidden thinking.

Failed attempts are retried a few times with short pauses (--retries), then reported to the
client as an error so its own retry logic takes over; each failure is logged with its status and
message, and /v1/usage counts failures by kind. Client errors (400/401/403/404) are not retried,
except a bare 400 "bad request", which the gateway returns at random and which gets one retry.

--timeout is a read timeout of the HTTP client, so a reply that trickles in never hits it (calls of
290-990 s were seen). --deadline caps the whole attempt: past it the proxy stops waiting, logs a
"deadline" error and retries (the abandoned call may still finish and be billed by the gateway).

--thinking off (default) sends thinking={"type": "disabled"}. From 2026-11-18 evening the gateway's
gemini-3.8-flash thinks by default (hidden reasoning tokens, billed as output); all results before that
were produced without thinking, so it is switched off to keep runs comparable.
"""
import argparse
import collections
import concurrent.futures
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from anthropic import Anthropic

GATEWAY = "https://1gw.gwai.cloud"


def load_keys():
    names = sorted(n for n in os.environ if n.startswith("GWAI_API_KEY") and os.environ[n].strip())
    return [os.environ[n].strip() for n in names]


class Upstream:
    def __init__(self, keys, max_tokens, timeout, retries, deadline=None, thinking="off"):
        if not keys:
            sys.exit("No GWAI_API_KEY* variable set")
        self.keys, self.i, self.lock = keys, 0, threading.Lock()
        self.max_tokens, self.timeout, self.retries = max_tokens, timeout, retries
        self.deadline = deadline or timeout
        self.thinking = thinking
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        self.usage = {"calls": 0, "errors": 0, "in": 0, "out": 0}
        self.error_kinds = collections.Counter()

    def client(self):
        with self.lock:
            if self.i >= len(self.keys):
                raise RuntimeError("all gateway keys are used up")
            return self.i, Anthropic(api_key=self.keys[self.i], base_url=GATEWAY, max_retries=0, timeout=self.timeout)

    def drop(self, idx):
        with self.lock:
            if self.i == idx:
                self.i += 1
                print(f"[gw_proxy] key #{idx + 1} reached its cost limit; switching to key #{self.i + 1}", flush=True)

    def chat(self, body):
        msgs = body.get("messages", [])
        system = "\n\n".join(m["content"] if isinstance(m["content"], str) else
                             "".join(p.get("text", "") for p in m["content"]) for m in msgs if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] in ("user", "assistant")]
        kwargs = {"model": body["model"], "max_tokens": self.max_tokens, "messages": turns,
                  "extra_body": {"temperature": body.get("temperature", 0) or 0}}
        if self.thinking == "off":
            kwargs["extra_body"]["thinking"] = {"type": "disabled"}
        if system:
            kwargs["system"] = system
        bad_request_retried = False
        for attempt in range(self.retries + 1):
            idx, cl = self.client()
            start = time.time()
            try:
                future = self.pool.submit(cl.messages.create, **kwargs)
                try:
                    resp = future.result(timeout=self.deadline)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"no complete reply within the {self.deadline:.0f}s deadline") from None
                text = "".join(getattr(b, "text", "") or "" for b in resp.content or [])
                if not text:
                    raise RuntimeError("empty response from the gateway")
                u = resp.usage
                with self.lock:
                    self.usage["calls"] += 1
                    self.usage["in"] += u.input_tokens or 0
                    self.usage["out"] += u.output_tokens or 0
                return text, u.input_tokens or 0, u.output_tokens or 0, resp.stop_reason
            except Exception as err:  # gateway errors: cost limit, overload, empty reply, timeout
                status = getattr(err, "status_code", None)
                msg = " ".join(str(getattr(err, "body", "") or err).split())
                kind = f"{status or type(err).__name__}"
                with self.lock:
                    self.usage["errors"] += 1
                    self.error_kinds[kind] += 1
                print(f"[gw_proxy] {time.strftime('%H:%M:%S')} attempt {attempt + 1} failed after "
                      f"{time.time() - start:.1f}s: {kind} {msg[:200]}", flush=True)
                if status == 429 and "cost_limit" in msg:
                    self.drop(idx)
                    continue
                # The gateway sometimes answers a valid request with a bare 400 "bad request";
                # the same request succeeds when resent, so allow one retry for that case only.
                transient_400 = status == 400 and "bad request" in msg.lower() and not bad_request_retried
                if transient_400:
                    bad_request_retried = True
                elif status in (400, 401, 403, 404):
                    raise
                if attempt < self.retries:
                    time.sleep(2 * 2 ** attempt)
        raise RuntimeError("gateway call failed after retries")


def make_handler(up, model_name):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                # The client gave up (its own timeout) before the gateway answered
                print(f"[gw_proxy] {time.strftime('%H:%M:%S')} client closed the connection before the reply",
                      flush=True)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self.send_json(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            elif self.path.rstrip("/").endswith("/usage"):
                self.send_json(200, {**up.usage, "error_kinds": dict(up.error_kinds)})
            else:
                self.send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self.send_json(404, {"error": {"message": "not found"}})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            n = max(1, int(body.get("n", 1) or 1))
            try:
                outs = [up.chat(body) for _ in range(n)]
            except Exception as err:
                code = getattr(err, "status_code", None) or 502
                return self.send_json(code, {"error": {"message": str(err)[:500], "type": "gateway_error"}})
            choices = [{"index": k, "message": {"role": "assistant", "content": t},
                        "finish_reason": "length" if stop == "max_tokens" else "stop"}
                       for k, (t, _, _, stop) in enumerate(outs)]
            t_in, t_out = sum(o[1] for o in outs), sum(o[2] for o in outs)
            self.send_json(200, {"id": f"chatcmpl-{int(time.time() * 1000)}", "object": "chat.completion",
                                 "created": int(time.time()), "model": body.get("model", model_name),
                                 "choices": choices,
                                 "usage": {"prompt_tokens": t_in, "completion_tokens": t_out,
                                           "total_tokens": t_in + t_out}})
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4000)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--timeout", type=float, default=300, help="Seconds to wait for one gateway reply")
    ap.add_argument("--retries", type=int, default=3, help="Quick retries inside the proxy before reporting an error")
    ap.add_argument("--deadline", type=float, default=None,
                    help="Total seconds for one attempt, however slowly the reply arrives (default: --timeout)")
    ap.add_argument("--thinking", choices=["off", "default"], default="off",
                    help="off: ask the gateway for no hidden reasoning (as in all runs before 2026-11-18 evening); "
                         "default: send nothing and let the gateway decide")
    args = ap.parse_args()
    keys = load_keys()
    up = Upstream(keys, args.max_tokens, args.timeout, args.retries, args.deadline, args.thinking)
    print(f"[gw_proxy] {len(keys)} key(s); thinking={args.thinking}; listening on http://localhost:{args.port}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(up, args.model)).serve_forever()


if __name__ == "__main__":
    main()
