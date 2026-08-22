# scripts/llm_client.py
"""Thin client for a local OpenAI-compatible LLM server.

Talks to whatever is serving the 5090 box — vLLM, Ollama, LM Studio,
llama.cpp's server, TGI — since they all expose `/v1/chat/completions`.
Nothing here is Anthropic-specific and no paid API is involved.

    # on the 5090 machine
    vllm serve Qwen/Qwen3-32B-AWQ --host 0.0.0.0 --port 8000 \
        --max-model-len 8192 --gpu-memory-utilization 0.92

    # from this machine
    python scripts/llm_client.py --check

Point `LLM_BASE_URL` at that box in `.env` (e.g. `http://192.168.1.50:8000/v1`),
or run the pipeline directly on the 5090 with the default localhost.

Two things local servers do that a hosted API does not, both handled here:

  * **Structured output support varies.** vLLM takes `guided_json`, newer
    builds take OpenAI-style `response_format: json_schema`, some only manage
    `json_object`, and llama.cpp may manage neither. `negotiate()` tries them
    in order once and remembers which worked, so a run pays the discovery cost
    a single time and every later request uses the mode the server accepts.

  * **Reasoning models emit their scratchpad inline.** Qwen3 and friends wrap
    it in `<think>...</think>` ahead of the real answer. That is stripped
    before JSON parsing, along with markdown fences.
"""
import json
import os
import pathlib
import re
import sys

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B-AWQ"

# json-mode strategies, most to least capable
MODES = ("schema", "guided", "object", "none")

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class LLMError(Exception):
    """Server said no, or said something we cannot parse."""


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


class LLMClient:
    """One local model, one negotiated json mode, retried on transport errors."""

    def __init__(self, base_url=None, model=None, api_key=None, timeout=600.0,
                 max_retries=3, temperature=0.3, max_tokens=2048, json_mode=None):
        from openai import OpenAI
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = json_mode          # None = negotiate on first use
        self._client = OpenAI(
            base_url=self.base_url,
            # local servers ignore the key but the SDK insists on one
            api_key=api_key or os.environ.get("LLM_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )

    # --- plumbing ----------------------------------------------------------

    def _kwargs(self, mode, schema):
        """Per-mode request extras for the same logical 'return this JSON'."""
        if mode == "schema":
            return {"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "cot", "schema": schema, "strict": True},
            }}
        if mode == "guided":                       # vLLM's own flag
            return {"extra_body": {"guided_json": schema}}
        if mode == "object":
            return {"response_format": {"type": "json_object"}}
        return {}                                  # prompt-only; parser copes

    def _call(self, messages, schema, mode, max_tokens=None):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **self._kwargs(mode, schema),
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            raise LLMError(f"empty content, finish_reason={choice.finish_reason}")
        if choice.finish_reason == "length":
            raise LLMError("truncated — raise max_tokens")
        return parse_json(text), resp.usage

    # --- public ------------------------------------------------------------

    def negotiate(self, messages, schema):
        """Find the strongest json mode this server accepts. Runs once."""
        errors = {}
        for mode in MODES:
            try:
                out = self._call(messages, schema, mode)
                self.mode = mode
                return out
            except LLMError:
                raise                              # server worked, model didn't
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


def check(base_url=None, model=None):
    """Connectivity probe: list models, then round-trip a tiny schema."""
    load_env()
    c = LLMClient(base_url, model)
    print(f"base_url : {c.base_url}")
    print(f"model    : {c.model}")
    try:
        served = [m.id for m in c._client.models.list().data]
        print(f"served   : {served}")
        if c.model not in served and served:
            print(f"  ! '{c.model}' is not in the served list — "
                  f"pass --model with one of the above")
    except Exception as exc:
        sys.exit(f"cannot reach {c.base_url}: {type(exc).__name__}: {exc}\n"
                 f"is the server running, and is the port open to this machine?")

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
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    a = ap.parse_args()
    check(a.base_url, a.model)
