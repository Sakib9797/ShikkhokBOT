# scripts/llm_client.py
"""Thin client for any OpenAI-compatible chat completions endpoint.

Two supported providers, selected with `--provider` or `LLM_PROVIDER`:

  * **local** (default) — vLLM, Ollama, LM Studio, llama.cpp, TGI on your own
    GPU box. Free, private, no rate limits, but you supply the hardware.
  * **groq** — Groq's hosted OpenAI-compatible API. Very fast, no GPU needed,
    but it is a metered service with rate limits and it sees your data.

Both run the same prompts and the same validators, so a corpus generated on one
is directly comparable to one generated on the other. `_cot_model` records
which model produced each chain either way.

    python scripts/llm_client.py --check                    # local
    python scripts/llm_client.py --check --provider groq    # Groq

Groq needs a key. Put it in `.env` as `GROQ_API_KEY=gsk_...` — that file is
git-ignored, and the key must never be committed or pasted into a script.

Four things a hosted or local endpoint may do differently, all handled here:

  * **Structured output support varies.** vLLM takes `guided_json`; OpenAI and
    Groq's newer models take `response_format: json_schema`; most others manage
    `json_object`; some manage neither. `negotiate()` tries the modes this
    provider plausibly supports, once, and remembers which worked.

  * **Reasoning models emit their scratchpad inline.** Qwen3 and friends wrap it
    in `<think>...</think>` before the real answer. That is stripped before JSON
    parsing, along with markdown fences. On Groq, `reasoning_format="hidden"`
    suppresses it server-side instead, which is cheaper.

  * **Hosted APIs rate-limit.** Groq's free tier will 429 under the worker
    counts a local server happily absorbs. `--rpm` paces requests client-side,
    and the SDK retries 429s with backoff on top of that.

  * **`json_object` mode requires the word "JSON" in the prompt.** The Bengali
    system prompt in `cot_core.py` includes it deliberately; do not remove it.
"""
import json
import os
import pathlib
import re
import sys
import threading
import time

# provider presets: base_url, the env var holding its key, whether a key is
# required, the json modes worth trying, and a sensible default model
PROVIDERS = {
    "local": {
        "base_url": "http://localhost:8000/v1",
        "key_env": "LLM_API_KEY",
        "key_required": False,
        "modes": ("schema", "guided", "object", "none"),
        "model": "Qwen/Qwen3-32B-AWQ",
        "rpm": 0,                     # no client-side pacing
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "key_required": True,
        # Groq rejects vLLM's guided_json; json_schema works on its newer
        # models, json_object on most of the rest
        "modes": ("schema", "object", "none"),
        "model": "qwen/qwen3-32b",
        "rpm": 55,                    # free tier is commonly 60 rpm; leave headroom
    },
}
DEFAULT_PROVIDER = "local"

# Kept for backwards compatibility with anything importing these directly.
DEFAULT_BASE_URL = PROVIDERS["local"]["base_url"]
DEFAULT_MODEL = PROVIDERS["local"]["model"]

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class LLMError(Exception):
    """Server said no, or said something we cannot parse."""


class RateLimitedError(LLMError):
    """Provider refused for quota reasons — worth surfacing, not retrying blindly."""


def strip_reasoning(text: str) -> str:
    """Drop a reasoning model's <think> scratchpad and any markdown fence."""
    text = _THINK.sub("", text)
    # an unclosed <think> means the model never emitted the answer
    if "<think>" in text and "</think>" not in text:
        raise LLMError("truncated inside <think> — raise max_tokens")
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    return text.strip()


def parse_json(text: str):
    """Parse the model's JSON, tolerating prose wrapped around it."""
    cleaned = strip_reasoning(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # last resort: the outermost {...} in the response
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"unparseable json: {exc}") from exc
    raise LLMError("no json object in response")


class RateLimiter:
    """Spaces requests to at most `rpm` per minute across all worker threads."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            due = max(now, self._next)
            self._next = due + self.interval
        delay = due - now
        if delay > 0:
            time.sleep(delay)


class LLMClient:
    """One model on one provider, one negotiated json mode, rate-limited."""

    def __init__(self, base_url=None, model=None, api_key=None, timeout=600.0,
                 max_retries=None, temperature=0.3, max_tokens=2048,
                 json_mode=None, provider=None, rpm=None, reasoning_format=None):
        from openai import OpenAI

        self.provider = (provider or os.environ.get("LLM_PROVIDER")
                         or DEFAULT_PROVIDER).lower()
        if self.provider not in PROVIDERS:
            raise LLMError(f"unknown provider {self.provider!r}; "
                           f"choose from {sorted(PROVIDERS)}")
        spec = PROVIDERS[self.provider]

        # LLM_BASE_URL / LLM_MODEL describe the setup declared by LLM_PROVIDER.
        # Switching provider on the command line must not drag them along: a
        # localhost URL or a vLLM model name is meaningless on Groq.
        env_provider = (os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
        env_applies = self.provider == env_provider
        self.base_url = (base_url
                         or (os.environ.get("LLM_BASE_URL") if env_applies else None)
                         or spec["base_url"])
        self.model = (model
                      or os.environ.get(f"{self.provider.upper()}_MODEL")
                      or (os.environ.get("LLM_MODEL") if env_applies else None)
                      or spec["model"])
        self.modes = spec["modes"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = json_mode          # None = negotiate on first use
        self.reasoning_format = reasoning_format

        # A provider that needs a real key must not be satisfied by the generic
        # placeholder local servers use; that would surface as a baffling 401.
        key = api_key or os.environ.get(spec["key_env"])
        if not spec["key_required"]:
            key = key or os.environ.get("LLM_API_KEY")
        if key and key.strip().upper() in ("EMPTY", "NONE", "CHANGEME", "YOUR_KEY"):
            key = None
        if spec["key_required"] and not key:
            raise LLMError(
                f"{self.provider} needs an API key. Put {spec['key_env']}=... in .env "
                f"(git-ignored). Never hard-code it in a script or commit it.")

        # hosted providers rate-limit, so retry harder there than on localhost
        if max_retries is None:
            max_retries = 6 if spec["key_required"] else 3
        self.limiter = RateLimiter(spec["rpm"] if rpm is None else rpm)

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=key or "EMPTY",    # local servers ignore it, the SDK insists
            timeout=timeout,
            max_retries=max_retries,
        )

    # --- plumbing ----------------------------------------------------------

    def _kwargs(self, mode, schema):
        """Per-mode request extras for the same logical 'return this JSON'."""
        extra_body = {}
        # Groq suppresses <think> server-side, which saves output tokens
        if self.provider == "groq" and self.reasoning_format:
            extra_body["reasoning_format"] = self.reasoning_format

        if mode == "schema":
            out = {"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "cot", "schema": schema, "strict": True},
            }}
        elif mode == "guided":                     # vLLM's own flag
            extra_body["guided_json"] = schema
            out = {}
        elif mode == "object":
            out = {"response_format": {"type": "json_object"}}
        else:
            out = {}                               # prompt-only; parser copes

        if extra_body:
            out["extra_body"] = extra_body
        return out

    def _call(self, messages, schema, mode, max_tokens=None):
        self.limiter.wait()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **self._kwargs(mode, schema),
            )
        except Exception as exc:
            # surface quota exhaustion distinctly: retrying harder will not help
            if type(exc).__name__ == "RateLimitError":
                raise RateLimitedError(
                    f"rate/quota limit on {self.provider}: {str(exc)[:200]}") from exc
            raise
        choice = resp.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            raise LLMError(f"empty content, finish_reason={choice.finish_reason}")
        if choice.finish_reason == "length":
            raise LLMError("truncated — raise max_tokens")
        return parse_json(text), resp.usage

    # --- public ------------------------------------------------------------

    def negotiate(self, messages, schema):
        """Find the strongest json mode this provider accepts. Runs once."""
        errors = {}
        for mode in self.modes:
            try:
                out = self._call(messages, schema, mode)
                self.mode = mode
                return out
            except LLMError:
                raise                              # endpoint worked, model didn't
            except Exception as exc:               # unsupported param -> next mode
                errors[mode] = f"{type(exc).__name__}: {str(exc)[:120]}"
        raise LLMError(f"no usable json mode on {self.base_url}: {errors}")

    def complete(self, system, user, schema, max_tokens=None):
        """Return (parsed_json, usage). Negotiates the json mode if needed."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        if self.mode is None:
            return self.negotiate(messages, schema)
        return self._call(messages, schema, self.mode, max_tokens)


def load_env(path=".env"):
    """Minimal .env reader — avoids a python-dotenv dependency."""
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v and not os.environ.get(k.strip()):
            os.environ[k.strip()] = v


def add_provider_args(ap):
    """Shared CLI surface, so every script selects a backend the same way."""
    ap.add_argument("--provider", default=None, choices=sorted(PROVIDERS),
                    help=f"default: $LLM_PROVIDER or {DEFAULT_PROVIDER}")
    ap.add_argument("--model", default=None, help="default: $LLM_MODEL or the provider's")
    ap.add_argument("--base-url", default=None, help="override the provider's endpoint")
    ap.add_argument("--rpm", type=int, default=None,
                    help="client-side requests/minute cap (0 = unlimited)")
    ap.add_argument("--reasoning-format", default=None,
                    choices=["parsed", "raw", "hidden"],
                    help="Groq only: 'hidden' suppresses <think> and saves tokens")
    return ap


def client_from_args(args, **overrides):
    """Build an LLMClient from the shared args above."""
    return LLMClient(
        base_url=args.base_url, model=args.model, provider=args.provider,
        rpm=args.rpm, reasoning_format=getattr(args, "reasoning_format", None),
        **overrides)


def check(base_url=None, model=None, provider=None, rpm=None, reasoning_format=None):
    """Connectivity probe: list models, then round-trip a tiny schema."""
    load_env()
    try:
        c = LLMClient(base_url, model, provider=provider, rpm=rpm,
                      reasoning_format=reasoning_format)
    except LLMError as exc:
        sys.exit(str(exc))
    print(f"provider : {c.provider}")
    print(f"base_url : {c.base_url}")
    print(f"model    : {c.model}")
    try:
        served = sorted(m.id for m in c._client.models.list().data)
        shown = served if len(served) <= 12 else served[:12] + [f"... (+{len(served)-12})"]
        print(f"served   : {shown}")
        if c.model not in served and served:
            print(f"  ! '{c.model}' is not in the served list — "
                  f"pass --model with one of the above")
    except Exception as exc:
        hint = ("is the server running, and is the port open to this machine?"
                if c.provider == "local" else
                "is GROQ_API_KEY correct and still valid?")
        sys.exit(f"cannot reach {c.base_url}: {type(exc).__name__}: {exc}\n{hint}")

    schema = {"type": "object",
              "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    try:
        out, usage = c.complete("Reply with JSON only.", 'Return {"ok": true}', schema)
        print(f"json mode: {c.mode}")
        print(f"round-trip OK: {out}")
        if usage:
            print(f"usage    : in={usage.prompt_tokens} out={usage.completion_tokens}")
    except Exception as exc:
        sys.exit(f"round-trip failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    add_provider_args(ap)
    a = ap.parse_args()
    check(a.base_url, a.model, a.provider, a.rpm, a.reasoning_format)
