"""Rule layer: deterministic, offline, bilingual entity extraction.

This is the *baseline* NER system - a lexicon-driven, layout-aware extractor.
It exists for three reasons:

* it is instant, free and reproducible, so it can be reported as a genuine
  baseline to compare the LLM against;
* it is exact where an LLM is merely probable (a regex never hallucinates a
  total that is not on the paper);
* it gives the hybrid stage a second opinion, and agreement between two
  independent methods is a far better confidence signal than either alone.

Every entity produced here carries character offsets into the OCR text.
"""

from __future__ import annotations

import re

from dms.lexicon import (
    ADDRESS_MARKERS,
    CASHIER_LABELS,
    COMPANY_MARKERS,
    DATE_LABELS,
    DAY_NAMES,
    INVOICE_LABELS,
    MALAYSIAN_PLACES,
    NOISE_LINES,
    SUMMARY_HEADERS,
    PAYMENT_METHODS,
    PHONE_LABELS,
    TAX_ID_LABELS,
    TIME_LABELS,
    all_money_labels,
)
from dms.schema import Entity
from dms.textutils import (
    find_money_values,
    format_money,
    normalize_date,
    normalize_spaces,
    normalize_time,
    repair_digits,
)

_MONEY_LABELS = all_money_labels()

# Labels that unambiguously mark the final payable amount.
_SPECIFIC_TOTAL = {
    "jumlah besar", "jumlah keseluruhan", "jumlah akhir", "jumlah bayaran",
    "grand total", "total amount", "nett total", "net total", "total due",
    "amount due",
    # A post-rounding line is the definitive payable amount on Malaysian
    # receipts, so it outranks an earlier plain "Total".
    "total after adjustment", "total after adj", "total after rounding",
    "jumlah selepas pelarasan", "jumlah selepas adj",
    "total inclusive of gst", "total incl gst", "total incl. gst",
}

_QTY_LINE = re.compile(r"\b(barang|items?|qty|kuantiti|unit|pcs|keping|helai)\b", re.I)
# A company registration number printed after the trading name, e.g.
# "KMF FOODICIOUS SDN BHD (1132106-H)" - not part of the merchant's name.
_REG_SUFFIX = re.compile(r"\s*[\(\[]\s*\d{4,}\s*[-\s]?\s*[A-Za-z]?\s*[\)\]]?\s*$")
# The same number printed *before* the address, because line reconstruction put
# the registration and the street on one visual line:
#   "Co REG No 210038-K, 42-46, JLN SULTAN AZLAN SHAH"  ->  "42-46, JLN ..."
_REG_PREFIX = re.compile(
    r"^\s*(?:co\.?\s*reg\.?\s*(?:no\.?)?\s*|reg\.?\s*no\.?\s*)?"
    r"[\(\[]?\d{5,}\s*-\s*[A-Za-z][\)\]]?\s*[,.]?\s*", re.I)
_POSTCODE = re.compile(r"\b\d{5}\b")
_PHONE_GENERIC = re.compile(r"\b(?:\+?60|0)\d{1,2}[-\s]?\d{3,4}[-\s]?\d{3,4}\b")
_TIME_STRICT = re.compile(r"\b([01]?\d|2[0-3])\s?[:;]\s?([0-5]\d)")
# An identifier must contain at least one digit, otherwise the line
# "RESIT JUALAN" yields the invoice number "JUALAN".
_ID_VALUE = r"((?=[A-Za-z0-9\-/]*\d)[A-Za-z0-9][A-Za-z0-9\-/]{3,24})"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def line_offsets(text: str) -> list[tuple[str, int]]:
    """Split ``text`` into lines, keeping each line's offset in the original."""
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((line, pos))
        pos += len(line) + 1
    return out


def _word_bounded(s: str, start: int, length: int) -> bool:
    before = s[start - 1] if start > 0 else " "
    after_i = start + length
    after = s[after_i] if after_i < len(s) else " "
    return not before.isalnum() and not after.isalnum()


def _contains_any(low: str, needles) -> str | None:
    for n in needles:
        i = low.find(n)
        if i != -1 and _word_bounded(low, i, len(n)):
            return n
    return None


def _is_noise(low: str) -> bool:
    return any(n in low for n in NOISE_LINES)


def _mk(etype, value, text, start, end, conf=0.8, source="rule", **meta) -> Entity:
    return Entity(
        type=etype,
        value=str(value),
        text=normalize_spaces(text),
        start=start,
        end=end,
        confidence=conf,
        source=source,
        meta=meta,
    )


# --------------------------------------------------------------------------
# money fields
# --------------------------------------------------------------------------

def _money_candidates(lines: list[tuple[str, int]]) -> list[dict]:
    """Find every ``<label> ... <amount>`` pair, in reading order."""
    cands: list[dict] = []
    in_summary = False
    for i, (line, off) in enumerate(lines):
        low = line.lower()
        # Once a tax-summary header appears, the rest of the receipt is a
        # breakdown table. Its "Total" row totals the *summary*, and its last
        # column is tax - so nothing below may supply the grand total.
        if _contains_any(low, SUMMARY_HEADERS):
            in_summary = True
        monies = find_money_values(line)
        for label, etype in _MONEY_LABELS:          # longest label first
            p = low.find(label)
            if p == -1 or not _word_bounded(low, p, len(label)):
                continue
            if etype == "TOTAL" and in_summary:
                break                               # a summary row, not the bill
            # "JUMLAH BARANG 3.00" is a count, not an amount.
            if etype == "TOTAL" and label not in _SPECIFIC_TOTAL and _QTY_LINE.search(low):
                break
            after = [m for m in monies if m[1] >= p + len(label)]
            if after:
                val, s, e = after[-1]                # label left, amount right
                base = off
            else:
                # Some layouts put the amount on the following line alone.
                nxt = lines[i + 1] if i + 1 < len(lines) else None
                if not nxt:
                    break
                nline, noff = nxt
                nmon = find_money_values(nline)
                if not nmon or len(nline.strip()) > 24:
                    break
                val, s, e = nmon[0]
                base = noff
            cands.append({
                "type": etype, "label": label, "value": val,
                "start": base + s, "end": base + e,
                "text": line[s:e] if after else nline[s:e], "line": i,
            })
            break                                    # one field per line
    return cands


def _pick_money(cands: list[dict]) -> list[Entity]:
    """Resolve competing candidates into at most one entity per money type."""
    out: list[Entity] = []
    by_type: dict[str, list[dict]] = {}
    for c in cands:
        by_type.setdefault(c["type"], []).append(c)

    for etype, group in by_type.items():
        group.sort(key=lambda c: c["line"])
        if etype == "TOTAL":
            # A receipt total is never 0.00. Malaysian receipts routinely carry
            # a "ROUNDING 0.00" or "GST 0.00" line that also matches a total
            # label, so zero candidates are used only as a last resort.
            pool = [c for c in group if c["value"] > 0] or group
            specific = [c for c in pool if c["label"] in _SPECIFIC_TOTAL]
            chosen = specific[-1] if specific else pool[-1]
        elif etype in ("PAID", "CHANGE"):
            chosen = group[-1]
        else:
            chosen = group[0]
        out.append(_mk(etype, format_money(chosen["value"]), chosen["text"],
                       chosen["start"], chosen["end"], conf=0.85,
                       label=chosen["label"], amount=chosen["value"]))
    return out


# --------------------------------------------------------------------------
# header fields
# --------------------------------------------------------------------------

def _extract_merchant(lines: list[tuple[str, int]]) -> Entity | None:
    """The merchant is the branded line at the very top of the receipt."""
    head = lines[:7]
    best = None
    for idx, (line, off) in enumerate(head):
        low = line.lower().strip()
        if len(low) < 3 or _is_noise(low):
            continue
        if _contains_any(low, COMPANY_MARKERS):
            best = (idx, line, off)
            break
    if best is None:
        for idx, (line, off) in enumerate(head):
            stripped = line.strip()
            low = stripped.lower()
            letters = sum(ch.isalpha() for ch in stripped)
            if letters >= 4 and not _is_noise(low) and not _POSTCODE.search(stripped):
                best = (idx, line, off)
                break
    if best is None:
        return None

    idx, line, off = best
    text = line.strip()
    start = off + line.index(text) if text in line else off
    end = start + len(text)

    # "KEDAI RUNCIT MAKMUR" / "SDN BHD" split across two printed lines.
    if idx + 1 < len(lines):
        nline, noff = lines[idx + 1]
        nstr = nline.strip()
        if 0 < len(nstr) <= 18 and _contains_any(nstr.lower(), COMPANY_MARKERS):
            text = f"{text} {nstr}"
            end = noff + nline.index(nstr) + len(nstr)

    cleaned = _REG_SUFFIX.sub("", normalize_spaces(text)).strip(" ,.-")
    return _mk("MERCHANT", cleaned or normalize_spaces(text), text,
               start, end, conf=0.72)


def _extract_address(lines: list[tuple[str, int]], merchant_end: int) -> Entity | None:
    """Address lines sit directly under the merchant name."""
    picked: list[tuple[str, int, int]] = []
    for line, off in lines[:12]:
        if off < merchant_end:
            continue
        stripped = line.strip()
        low = stripped.lower()
        if not stripped or _is_noise(low):
            continue
        hit = (
            _contains_any(low, ADDRESS_MARKERS)
            or _contains_any(low, MALAYSIAN_PLACES)
            or _POSTCODE.search(stripped)
        )
        if hit:
            s = off + line.index(stripped)
            picked.append((stripped, s, s + len(stripped)))
        elif picked:
            break                                   # address block has ended
    if not picked:
        return None
    text = ", ".join(p[0] for p in picked)
    cleaned = _REG_PREFIX.sub("", normalize_spaces(text)).strip(" ,.-")
    return _mk("ADDRESS", cleaned or normalize_spaces(text), text,
               picked[0][1], picked[-1][2], conf=0.7)


def _labelled_value(lines, labels, pattern: str, etype: str,
                    conf: float = 0.8) -> Entity | None:
    """Generic ``<label>: <value>`` extractor used for ids and phone numbers."""
    for line, off in lines:
        low = line.lower()
        for label in sorted(labels, key=len, reverse=True):
            p = low.find(label)
            if p == -1 or not _word_bounded(low, p, len(label)):
                continue
            tail = line[p + len(label):]
            # ';' is included deliberately: OCR reads the colon in "Inv:R0001"
            # as a semicolon often enough that omitting it loses the number.
            m = re.search(r"^\s*(?:no\.?|id|reg\.?|#)?\s*[:;\-.]?\s*" + pattern, tail)
            if m:
                val = m.group(1).strip(" .:-/")
                if len(val) < 3:
                    continue
                s = off + p + len(label) + m.start(1)
                return _mk(etype, val, val, s, s + len(m.group(1)), conf=conf,
                           label=label)
    return None


def _extract_date(lines) -> Entity | None:
    """Prefer a date that is labelled or sits beside a Malay day name."""
    fallback = None
    for line, off in lines:
        low = line.lower()
        found = normalize_date(line)
        if not found:
            continue
        iso, s, e = found
        anchored = bool(_contains_any(low, DATE_LABELS) or _contains_any(low, DAY_NAMES))
        ent = _mk("DATE", iso, line[s:e], off + s, off + e,
                  conf=0.9 if anchored else 0.75)
        if anchored:
            return ent
        fallback = fallback or ent
    return fallback


def _extract_time(lines) -> Entity | None:
    """Only accept a '.'-separated time when the line says it is a time.

    Without this guard the price ``21.32`` reads as 21:32.
    """
    fallback = None
    for line, off in lines:
        low = line.lower()
        anchored = bool(_contains_any(low, TIME_LABELS)) or bool(
            re.search(r"\b(pagi|petang|malam|am|pm)\b", low)
        )
        m = _TIME_STRICT.search(line)
        if m:
            found = normalize_time(line[m.start():])
            if found:
                iso, s, e = found
                ent = _mk("TIME", iso, line[m.start() + s:m.start() + e],
                          off + m.start() + s, off + m.start() + e,
                          conf=0.9 if anchored else 0.8)
                if anchored:
                    return ent
                fallback = fallback or ent
        elif anchored:
            found = normalize_time(line)
            if found:
                iso, s, e = found
                fallback = fallback or _mk("TIME", iso, line[s:e],
                                           off + s, off + e, conf=0.7)
    return fallback


def _extract_phone(lines) -> Entity | None:
    ent = _labelled_value(lines, PHONE_LABELS, r"([\d][\d\-\s()]{6,16})", "PHONE", 0.85)
    if ent:
        ent.value = normalize_spaces(ent.value)
        return ent
    for line, off in lines[:12]:
        m = _PHONE_GENERIC.search(line)
        if m:
            return _mk("PHONE", normalize_spaces(m.group(0)), m.group(0),
                       off + m.start(), off + m.end(), conf=0.7)
    return None


def _extract_payment(lines) -> Entity | None:
    for line, off in lines:
        low = line.lower()
        for method, forms in PAYMENT_METHODS.items():
            hit = _contains_any(low, forms)
            if hit:
                p = low.find(hit)
                return _mk("PAYMENT_METHOD", method, line[p:p + len(hit)],
                           off + p, off + p + len(hit), conf=0.8)
    return None


def _extract_items(lines) -> list[Entity]:
    """Fallback line-item reader: ``<name> <qty> x <unit> <amount>``."""
    out: list[Entity] = []
    rx = re.compile(r"(\d+(?:[.,]\d+)?)\s*[xX×*]\s*([\d.,\-–]+)")
    for line, off in lines:
        low = line.lower()
        if _contains_any(low, [lbl for lbl, _ in _MONEY_LABELS]):
            continue
        m = rx.search(line)
        if not m:
            continue
        name = normalize_spaces(re.sub(r"[\d.,]+$", "", line[:m.start()])).strip(" -|:")
        if len(name) < 2:
            continue
        monies = find_money_values(line)
        amount = monies[-1][0] if monies else None
        qty = repair_digits(m.group(1)).replace(",", ".")
        unit = find_money_values(m.group(2))
        # `name` was whitespace-normalised, so it may not occur verbatim.
        local = line.find(name)
        s = off + (local if local != -1 else len(line) - len(line.lstrip()))
        out.append(_mk(
            "ITEM", name, name, s, s + len(name), conf=0.6,
            qty=float(qty) if qty.replace(".", "").isdigit() else None,
            unit_price=unit[0][0] if unit else None,
            amount=amount,
        ))
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def looks_like_address(value: str) -> bool:
    """Is this string plausibly a street address?

    Used to veto the LLM when it answers the "address" field with whatever
    happened to sit near the top of the receipt - on a receipt with no printed
    address it will happily offer the GST registration number instead.
    """
    if not value:
        return False
    low = value.lower()
    if len(re.findall(r"[a-z]{3,}", low)) < 2:
        return False
    return bool(
        _contains_any(low, ADDRESS_MARKERS)
        or _contains_any(low, MALAYSIAN_PLACES)
        or _POSTCODE.search(value)
    )


def extract_rules(ocr_text: str) -> list[Entity]:
    """Run the full rule-based extractor over line-reconstructed OCR text."""
    if not ocr_text or not ocr_text.strip():
        return []
    lines = line_offsets(ocr_text)
    entities: list[Entity] = []

    entities.extend(_pick_money(_money_candidates(lines)))

    merchant = _extract_merchant(lines)
    if merchant:
        entities.append(merchant)

    address = _extract_address(lines, merchant.end if merchant else 0)
    if address:
        entities.append(address)

    for ent in (
        _extract_date(lines),
        _extract_time(lines),
        _extract_phone(lines),
        _extract_payment(lines),
        _labelled_value(lines, TAX_ID_LABELS, _ID_VALUE, "TAX_ID", 0.85),
        _labelled_value(lines, INVOICE_LABELS, _ID_VALUE, "INVOICE_NO", 0.8),
        _labelled_value(lines, CASHIER_LABELS, r"([A-Za-z][A-Za-z0-9 .'-]{1,24})",
                        "CASHIER", 0.7),
    ):
        if ent:
            entities.append(ent)

    if re.search(r"\b(rm|myr)\b", ocr_text, re.I):
        m = re.search(r"\b(RM|MYR)\b", ocr_text, re.I)
        entities.append(_mk("CURRENCY", "MYR", m.group(0), m.start(), m.end(), conf=0.9))

    entities.extend(_extract_items(lines))
    return entities
