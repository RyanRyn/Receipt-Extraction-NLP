"""Local LLM layer: a small open-weight instruct model doing receipt NER.

Design notes
------------
* **Fully local and free.** Weights are pulled once from HuggingFace and run on
  the machine - no API key, no token, no subscription, and no data leaves the
  computer (which matters for a DMS holding financial documents).

* **Prompt tokens are cheap, generated tokens are not.** On CPU, decoding runs
  at a few tokens/second while prefill is near-instant, so we spend freely on
  few-shot demonstrations and ask for *compact* single-line JSON back. The
  few-shot pair deliberately covers one clean Bahasa Melayu receipt and one
  OCR-corrupted English one, which is what teaches the model both the Malay
  field vocabulary and the digit-repair behaviour.

* **Greedy decoding.** Extraction must be reproducible for a report, so
  sampling is off.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time

import hashlib
from pathlib import Path

from dms.config import (
    CACHE_DIR,
    LLM_MAX_NEW_TOKENS_FULL,
    LLM_MAX_NEW_TOKENS_HEADER,
    detect_device,
    resolve_model,
)

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an information extraction engine for Malaysian receipts (resit). "
    "The text is produced by OCR, so it contains errors, and it may be written "
    "in Bahasa Melayu, in English, or in a mixture of both. "
    "You reply with ONE compact JSON object on a single line and nothing else - "
    "no markdown, no code fence, no explanation."
)

HEADER_FIELDS = (
    '{"merchant":str|null,"address":str|null,"phone":str|null,"tax_id":str|null,'
    '"invoice_no":str|null,"date":"YYYY-MM-DD"|null,"time":"HH:MM"|null,'
    '"currency":str|null,"subtotal":num|null,"tax":num|null,"total":num|null,'
    '"paid":num|null,"change":num|null,"payment_method":str|null}'
)

FULL_FIELDS = (
    '{"merchant":str|null,"address":str|null,"phone":str|null,"tax_id":str|null,'
    '"invoice_no":str|null,"date":"YYYY-MM-DD"|null,"time":"HH:MM"|null,'
    '"currency":str|null,"subtotal":num|null,"tax":num|null,"total":num|null,'
    '"paid":num|null,"change":num|null,"payment_method":str|null,'
    '"items":[{"name":str,"qty":num|null,"unit_price":num|null,"amount":num|null}]}'
)

RULES = """Rules:
- Every value must be a literal: a plain number, a quoted string, or null.
  NEVER write a calculation. Write "unit_price": 18.13 or null, never
  "unit_price": 54.40 / 3 - that is not valid JSON.
- Copy values from the receipt. Never invent a value that is not there; use null.
- Malay labels: JUMLAH BESAR/JUMLAH = total, JUMLAH KECIL = subtotal, CUKAI = tax,
  TUNAI = cash paid, BAKI = change, TARIKH = date, MASA = time, NO. RESIT = invoice_no.
- Malay months: Mac=March, Mei=May, Ogos=August, Oktober=October, Disember=December.
- Repair obvious OCR damage in numbers: "22-60"->22.60, "Z1.32"->21.32, "3O.OO"->30.00.
- Amounts are plain numbers without the RM prefix."""

_FEWSHOT_MS_IN = """KEDAI SERBANEKA HARAPAN
No 8, Jalan Melur 2, 81100 Johor Bahru, Johor
Tel: 07-3345 678
RESIT JUALAN   No. Resit: A-0912
Tarikh: 05/02/2023   Masa: 09:15 pagi
Roti Gardenia 1 x 3.50 3.50
Susu Dutch Lady 2 x 6.20 12.40
Jumlah Kecil 15.90
Cukai SST 6% 0.95
JUMLAH BESAR 16.85
Tunai 20.00
Baki 3.15"""

_FEWSHOT_MS_OUT = (
    '{"merchant":"KEDAI SERBANEKA HARAPAN","address":"No 8, Jalan Melur 2, 81100 '
    'Johor Bahru, Johor","phone":"07-3345 678","tax_id":null,"invoice_no":"A-0912",'
    '"date":"2023-02-05","time":"09:15","currency":"MYR","subtotal":15.90,"tax":0.95,'
    '"total":16.85,"paid":20.00,"change":3.15,"payment_method":"CASH",'
    '"items":[{"name":"Roti Gardenia","qty":1,"unit_price":3.50,"amount":3.50},'
    '{"name":"Susu Dutch Lady","qty":2,"unit_price":6.20,"amount":12.40}]}'
)

_FEWSHOT_EN_IN = """99 SPEED MART SDN BHD
LOT 2, JLN SS2/24, 47300 PETALING JAYA, SELANGOR
GST ID : 000123456789
TAX INVOICE   Inv: R0O0012345
Mineral Water 6OOml 1.OO X 2.5O 2.50
TOTAL 2-50
CASH 5.00
CHANGE 2-5O
Rabu, 12-07-2017 Tiae 18;40"""

_FEWSHOT_EN_OUT = (
    '{"merchant":"99 SPEED MART SDN BHD","address":"LOT 2, JLN SS2/24, 47300 '
    'PETALING JAYA, SELANGOR","phone":null,"tax_id":"000123456789",'
    '"invoice_no":"R000012345","date":"2017-07-12","time":"18:40","currency":"MYR",'
    '"subtotal":null,"tax":null,"total":2.50,"paid":5.00,"change":2.50,'
    '"payment_method":"CASH","items":[{"name":"Mineral Water 600ml","qty":1,'
    '"unit_price":2.50,"amount":2.50}]}'
)


def _strip_items(compact_json: str) -> str:
    """Reuse the few-shot answers in header-only mode by dropping ``items``."""
    return re.sub(r',"items":\[.*\]\}$', "}", compact_json)


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------

# A small model sometimes answers with a *calculation* rather than a value,
# e.g. `"unit_price": 54.40 / 3`. That is not JSON, and one such field used to
# discard the whole extraction - merchant, date, total and all.
_ARITHMETIC = re.compile(
    r":\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*(?=[,}\]])"
)

# Top-level scalar fields, used to salvage something from a reply that will not
# parse as a whole.
_SCALAR_FIELD = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*'
    r'("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false|null)'
)


def _evaluate_arithmetic(blob: str) -> str:
    """Replace ``<num> <op> <num>`` value expressions with their result."""
    def repl(match: re.Match) -> str:
        a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
        try:
            value = {"+": lambda: a + b, "-": lambda: a - b,
                     "*": lambda: a * b, "/": lambda: a / b}[op]()
        except ZeroDivisionError:
            return ": null"
        return f": {round(value, 2)}"
    return _ARITHMETIC.sub(repl, blob)


def salvage_fields(text: str) -> dict:
    """Recover whatever top-level scalars we can from an unparseable reply.

    A partial record - merchant, date and total without the line items - is far
    more useful than discarding the reply entirely because one nested field was
    malformed. The first occurrence of a key wins, and since the model emits the
    header fields before the ``items`` array, that is the top-level one.
    """
    out: dict = {}
    for match in _SCALAR_FIELD.finditer(text):
        key, raw = match.group(1), match.group(2)
        if key in out:
            continue
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            continue
    return out


def parse_json_object(raw: str) -> dict | None:
    """Pull the first JSON object out of a model reply, repairing common slips."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        blob = text[start:end]
    except ValueError:
        blob = text                       # truncated: no closing brace at all

    attempts = [
        blob,
        _evaluate_arithmetic(blob),                          # 54.40 / 3 -> 18.13
        re.sub(r",\s*([}\]])", r"\1", blob),                 # trailing commas
        re.sub(r"\bNone\b", "null", blob),
        _evaluate_arithmetic(re.sub(r",\s*([}\]])", r"\1", blob)),
    ]
    for attempt in attempts:
        try:
            value = json.loads(attempt)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    try:                                                     # python-dict style
        value = ast.literal_eval(blob)
        if isinstance(value, dict):
            return value
    except (ValueError, SyntaxError):
        pass

    # Last resort: keep the fields we can still read.
    rescued = salvage_fields(_evaluate_arithmetic(text))
    if rescued:
        rescued["_partial"] = True
        return rescued
    return None


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------

class LocalLLMExtractor:
    """Wraps a small instruct model as a receipt information extractor."""

    _loaded: dict[str, tuple] = {}     # repo id -> (tokenizer, model)

    def __init__(self, model: str | None = None, use_fewshot: bool = True,
                 verbose: bool = True, cache: bool = False):
        self.repo = resolve_model(model)
        self.use_fewshot = use_fewshot
        self.verbose = verbose
        self.device, self.dtype = detect_device()
        self._load_error: str | None = None
        # Decoding is deterministic, so identical input always yields identical
        # output. Caching it makes an ablation over merge policies instant
        # instead of re-running inference for every configuration.
        self.cache_enabled = cache
        self._cache: dict[str, dict] | None = None
        safe = self.repo.replace("/", "_")
        self._cache_path = Path(CACHE_DIR) / f"llm_{safe}.json"

    # -- response cache -------------------------------------------------
    def _cache_key(self, text: str, mode: str) -> str:
        # The prompt is part of the key: editing a prompt changes the output,
        # and a key that ignored it would silently serve stale extractions.
        prompt_fingerprint = hashlib.sha256(
            "".join((SYSTEM_PROMPT, RULES, HEADER_FIELDS, FULL_FIELDS,
                     _FEWSHOT_MS_OUT, _FEWSHOT_EN_OUT)).encode()
        ).hexdigest()[:8]
        digest = hashlib.sha256(
            f"{self.repo}\x00{mode}\x00{self.use_fewshot}\x00"
            f"{prompt_fingerprint}\x00{text}".encode()
        ).hexdigest()
        return digest[:32]

    def _load_cache(self) -> dict:
        if self._cache is None:
            if self._cache_path.exists():
                try:
                    self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    self._cache = {}
            else:
                self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is not None:
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    # -- model handling -------------------------------------------------
    def _ensure_loaded(self):
        if self.repo in LocalLLMExtractor._loaded:
            return LocalLLMExtractor._loaded[self.repo]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device == "cpu":
            # Physical cores; hyper-threads hurt more than help for GEMM here.
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

        if self.verbose:
            print(f"  [llm] loading {self.repo} on {self.device} ({self.dtype}) ...")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(self.repo)
        model = AutoModelForCausalLM.from_pretrained(self.repo, dtype=self.dtype)
        model.to(self.device)
        model.eval()
        if self.verbose:
            print(f"  [llm] ready in {time.time() - t0:.1f}s")

        LocalLLMExtractor._loaded[self.repo] = (tok, model)
        return tok, model

    @classmethod
    def unload(cls, repo: str | None = None) -> None:
        """Free a loaded model's VRAM.

        Needed when comparing several models in one process: an 8 GB card holds
        one 3B model in float16 with little to spare, so the previous one must
        be released before the next is loaded.
        """
        import gc

        import torch

        targets = [repo] if repo else list(cls._loaded)
        for key in targets:
            if key in cls._loaded:
                del cls._loaded[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def available(self) -> bool:
        """True when the model can be loaded (weights cached or downloadable)."""
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:
            self._load_error = str(exc)
            if self.verbose:
                print(f"  [llm] unavailable: {exc}")
            return False

    # -- prompting ------------------------------------------------------
    def _build_messages(self, ocr_text: str, mode: str) -> list[dict]:
        fields = FULL_FIELDS if mode == "full" else HEADER_FIELDS
        instruction = f"Extract this JSON schema:\n{fields}\n{RULES}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.use_fewshot:
            for src, ans in ((_FEWSHOT_MS_IN, _FEWSHOT_MS_OUT),
                             (_FEWSHOT_EN_IN, _FEWSHOT_EN_OUT)):
                out = ans if mode == "full" else _strip_items(ans)
                messages.append({"role": "user",
                                 "content": f"{instruction}\n\nRECEIPT:\n{src}"})
                messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user",
                         "content": f"{instruction}\n\nRECEIPT:\n{ocr_text}"})
        return messages

    def _apply_template(self, tok, messages) -> str:
        try:                       # Qwen3 et al. default to a <think> preamble
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # -- public API -----------------------------------------------------
    def extract(self, ocr_text: str, mode: str = "full",
                max_new_tokens: int | None = None) -> dict:
        """Extract receipt fields from OCR text.

        ``mode="header"`` skips line items, which roughly halves generation
        time - use it for batch runs and evaluation.
        Returns ``{}`` when the model is unavailable or emits unparseable JSON.
        """
        import torch

        if not ocr_text or not ocr_text.strip():
            return {}

        key = self._cache_key(ocr_text, mode) if self.cache_enabled else None
        if key is not None:
            hit = self._load_cache().get(key)
            if hit is not None:
                if self.verbose:
                    print("  [llm] cache hit")
                return dict(hit)

        try:
            tok, model = self._ensure_loaded()
        except Exception as exc:
            self._load_error = str(exc)
            if self.verbose:
                print(f"  [llm] load failed, skipping LLM layer: {exc}")
            return {}

        budget = max_new_tokens or (
            LLM_MAX_NEW_TOKENS_FULL if mode == "full" else LLM_MAX_NEW_TOKENS_HEADER
        )
        prompt = self._apply_template(tok, self._build_messages(ocr_text, mode))
        enc = tok(prompt, return_tensors="pt").to(self.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        generated = out[0][enc["input_ids"].shape[1]:]
        elapsed = time.time() - t0
        reply = tok.decode(generated, skip_special_tokens=True)

        if self.verbose:
            speed = len(generated) / elapsed if elapsed else 0.0
            print(f"  [llm] {len(generated)} tokens in {elapsed:.1f}s "
                  f"({speed:.1f} tok/s)")

        data = parse_json_object(reply)
        if data is None:
            if self.verbose:
                truncated = len(generated) >= budget
                reason = ("hit the token limit, so the reply is incomplete"
                          if truncated else "the reply is complete but malformed")
                print(f"  [llm] could not parse JSON - {reason}. "
                      f"Raw reply ({len(reply)} chars):")
                print("        " + reply[:1500].replace("\n", "\n        "))
            return {}
        if data.pop("_partial", False) and self.verbose:
            print("  [llm] reply was malformed; salvaged "
                  f"{len(data)} field(s) from it")
        data["_elapsed"] = round(elapsed, 2)
        data["_model"] = self.repo
        if key is not None:
            self._load_cache()[key] = data
            self._save_cache()
        return data
