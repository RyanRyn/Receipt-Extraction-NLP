"""Text normalisation, OCR-noise repair, money/date parsing and span alignment.

The span-alignment helpers are the bridge to the search half of the DMS: the
LLM returns *values*, but highlighting a hit inside a document needs *offsets*,
so every extracted value is re-anchored onto the OCR text here.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from dms.lexicon import MONTHS, DAY_PERIODS

# --------------------------------------------------------------------------
# Basic normalisation
# --------------------------------------------------------------------------

def normalize_spaces(s: str) -> str:
    """Collapse runs of whitespace, strip the ends."""
    return re.sub(r"\s+", " ", s or "").strip()


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_key(s: str) -> str:
    """Aggressive normalisation used for *comparison only* (search, dedup)."""
    s = strip_accents((s or "").lower())
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return normalize_spaces(s)


# --------------------------------------------------------------------------
# OCR digit repair
# --------------------------------------------------------------------------
# Glyph confusions seen constantly in receipt OCR. Applied ONLY when the token
# is already mostly numeric and the repair makes it fully numeric - this keeps
# the heuristic from mangling real words.
_DIGIT_FIXES = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "|": "1", "i": "1",
    "S": "5", "s": "5",
    "Z": "2", "z": "2",
    "B": "8",
    "G": "6", "g": "6",
    "T": "7",
    "b": "6",
})


def repair_digits(token: str) -> str:
    """Best-effort repair of a numeric token corrupted by OCR.

    ``"Z1.32" -> "21.32"``, ``"1.2g" -> "1.26"``, ``"3O.OO" -> "30.00"``.
    Returns the token unchanged when the repair is not clearly safe.
    """
    if not token:
        return token
    core = token.strip()
    digits = sum(ch.isdigit() for ch in core)
    alpha = sum(ch.isalpha() for ch in core)
    # Needs at least one real digit, and must not be mostly letters - that is
    # what separates "3O.OO" (an amount) from "TOTAL" (a word).
    if digits == 0 or alpha > 0.6 * len(core):
        return token
    fixed = core.translate(_DIGIT_FIXES)
    if re.fullmatch(r"[\d.,\-–—=\s]+", fixed):
        return fixed
    return token


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------
# A decimal amount, tolerating the separator being OCR'd as '-', ',' or a dash
# and tolerating stray spaces around it ("22-60", "7 - 40", "1,28").
_SEP = r"[.,\-–—=]"          # '=' and '-' are routine OCR misreads of '.'
_DECIMAL_MONEY = rf"\d{{1,3}}(?:[,\s]?\d{{3}})*\s?{_SEP}\s?\d{{2}}(?!\d)"
_MONEY_RE = re.compile(_DECIMAL_MONEY)

# Patterns masked out before money hunting, so a date or a phone number is
# never mistaken for an amount.
_DATE_LIKE = re.compile(r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
# Deliberately ':' and ';' only. Allowing '.' here would mask the price
# "21.32" as if it were the time 21:32 and delete it from the money hunt.
_TIME_LIKE = re.compile(r"\b\d{1,2}\s?[:;]\s?\d{2}(?:\s?[:;]\s?\d{2})?\b")
_PHONE_LIKE = re.compile(r"\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b")
_LONG_ID = re.compile(r"\b\d{9,}\b")          # GST numbers, barcodes


def mask_non_money(text: str) -> str:
    """Replace dates/times/phones/long ids with '#' so money regexes skip them."""
    out = text
    for rx in (_DATE_LIKE, _TIME_LIKE, _PHONE_LIKE, _LONG_ID):
        out = rx.sub(lambda m: "#" * len(m.group(0)), out)
    return out


def parse_money(raw) -> float | None:
    """Parse a money-ish value into a float, or ``None``.

    Handles both well-formed numbers (the LLM returns JSON numbers) and
    OCR-damaged strings where the decimal point was read as '-', ',' or '='.
    The order of the checks matters: a clean decimal must be recognised
    *before* the damaged-separator branch, otherwise ``"22.6"`` loses its point
    and becomes ``226``.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    s = repair_digits(str(raw).strip())
    s = re.sub(r"(?i)\b(rm|myr)\b", " ", s)
    s = re.sub(r"[^\d.,\-–—=]", "", s).strip()
    if not s:
        return None

    # 1. an ordinary decimal number: "22.6", "22.60", "1234.5"
    if re.fullmatch(r"\d+\.\d+", s):
        return float(s)

    # 2. thousands separators plus a decimal: "1,234.56"
    m = re.fullmatch(r"(\d{1,3}(?:,\d{3})+)\.(\d{1,2})", s)
    if m:
        return float(f"{m.group(1).replace(',', '')}.{m.group(2)}")

    # 3. a 2-digit tail whose separator was misread: "22-60", "1,28", "2=50"
    m = re.search(rf"^(.*?)({_SEP})(\d{{2}})$", s)
    if m:
        head, _, cents = m.groups()
        head = re.sub(r"[^\d]", "", head) or "0"
        try:
            return float(f"{head}.{cents}")
        except ValueError:
            return None

    # 4. bare digits
    digits = re.sub(r"[^\d]", "", s)
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def find_money_values(line: str) -> list[tuple[float, int, int]]:
    """All amounts in ``line`` as ``(value, start, end)`` char offsets."""
    masked = mask_non_money(line)
    out: list[tuple[float, int, int]] = []
    for m in _MONEY_RE.finditer(masked):
        val = parse_money(line[m.start():m.end()])
        if val is not None:
            out.append((val, m.start(), m.end()))
    return out


def format_money(v: float | None) -> str | None:
    return None if v is None else f"{v:.2f}"


# --------------------------------------------------------------------------
# Dates & times
# --------------------------------------------------------------------------
_NUM_DATE = re.compile(r"\b(\d{1,4})\s?[/\-.]\s?(\d{1,2})\s?[/\-.]\s?(\d{2,4})\b")
# Separators around a spelled-out month may be spaces, hyphens, dots or
# slashes - Malaysian receipts print "28-FEB-2018" as readily as "28 Feb 2018".
_WORD_DATE = re.compile(
    r"\b(\d{1,2})[\s\-./]+([A-Za-z]{3,12})\.?[\s\-./]+(\d{2,4})\b|"
    r"\b([A-Za-z]{3,12})\.?[\s\-./]+(\d{1,2}),?[\s\-./]+(\d{4})\b"
)
_TIME_RE = re.compile(
    r"\b([01]?\d|2[0-3])\s?[:;.]\s?([0-5]\d)(?:\s?[:;.]\s?([0-5]\d))?\s*"
    r"(pagi|tengahari|petang|malam|am|pm|a\.m\.|p\.m\.)?",
    re.IGNORECASE,
)


def _four_digit_year(y: int) -> int:
    if y < 100:
        return 2000 + y if y <= 69 else 1900 + y
    return y


def normalize_date(text: str) -> tuple[str, int, int] | None:
    """Find the first date in ``text`` and return ``(iso, start, end)``.

    Numeric dates are read day-first (``dd/mm/yyyy``), the Malaysian
    convention, falling back to month-first only when day-first is impossible
    (e.g. ``03/25/2024``). ``yyyy-mm-dd`` input is detected by a 4-digit head.
    """
    m = _NUM_DATE.search(text)
    if m:
        a, b, c = (int(g) for g in m.groups())
        if len(m.group(1)) == 4:                 # yyyy-mm-dd
            year, month, day = a, b, c
        else:
            year = _four_digit_year(c)
            day, month = a, b
            if month > 12 and day <= 12:         # clearly mm/dd
                day, month = month, day
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}", m.start(), m.end()

    m = _WORD_DATE.search(text)
    if m:
        if m.group(1):
            day, name, year = m.group(1), m.group(2), m.group(3)
        else:
            name, day, year = m.group(4), m.group(5), m.group(6)
        month = MONTHS.get(name.lower().rstrip("."))
        if month:
            y = _four_digit_year(int(year))
            d = int(day)
            if 1 <= d <= 31:
                return f"{y:04d}-{month:02d}-{d:02d}", m.start(), m.end()
    return None


def normalize_time(text: str) -> tuple[str, int, int] | None:
    """Find the first time and return ``(HH:MM, start, end)`` in 24h form."""
    m = _TIME_RE.search(text)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    period = (m.group(4) or "").lower().replace(".", "")
    if period in ("pm", "petang", "malam") and hh < 12:
        hh += 12
    elif period in ("am", "pagi") and hh == 12:
        hh = 0
    if hh > 23 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}", m.start(), m.end()


# --------------------------------------------------------------------------
# Span alignment  (value -> offsets in the OCR text)
# --------------------------------------------------------------------------

def _norm_with_map(s: str) -> tuple[str, list[int]]:
    """Lowercase + collapse whitespace, keeping a map back to original indices."""
    chars: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" ")
            idx.append(i)
            prev_space = True
        else:
            chars.append(ch.lower())
            idx.append(i)
            prev_space = False
    return "".join(chars), idx


def locate_span(text: str, value: str, min_ratio: float = 0.72) -> tuple[int, int] | None:
    """Locate ``value`` inside ``text``, tolerating OCR noise and rewrites.

    Tries an exact (normalised) match first, then a sliding fuzzy window.
    Returns ``(start, end)`` offsets into the *original* ``text``, or ``None``.
    """
    if not text or not value:
        return None
    hay, hmap = _norm_with_map(text)
    needle, _ = _norm_with_map(str(value))
    if not needle:
        return None

    pos = hay.find(needle)
    if pos != -1:
        return hmap[pos], hmap[pos + len(needle) - 1] + 1

    # Fuzzy: slide a window the length of the needle across the haystack.
    n = len(needle)
    if n < 3 or n > len(hay):
        return None
    step = max(1, n // 6)
    best_ratio, best_pos = 0.0, -1
    sm = SequenceMatcher(autojunk=False)
    sm.set_seq2(needle)
    for start in range(0, len(hay) - n + 1, step):
        window = hay[start:start + n]
        sm.set_seq1(window)
        if sm.real_quick_ratio() < best_ratio or sm.quick_ratio() < best_ratio:
            continue
        r = sm.ratio()
        if r > best_ratio:
            best_ratio, best_pos = r, start
    if best_pos >= 0 and best_ratio >= min_ratio:
        end = min(best_pos + n - 1, len(hmap) - 1)
        return hmap[best_pos], hmap[end] + 1
    return None


def similarity(a: str, b: str) -> float:
    """Normalised whole-string similarity in ``[0, 1]``."""
    na, nb = normalize_key(a), normalize_key(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb, autojunk=False).ratio()


def partial_similarity(query: str, value: str) -> float:
    """Best similarity between ``query`` and any word-window of ``value``.

    Whole-string similarity is the wrong measure when searching for a place
    inside an address: "Kuala Lumpor" against
    "No 88, Jalan Tuanku Abdul Rahman, 50100 Kuala Lumpur" scores only 0.36,
    because most of the address is irrelevant to the query. Comparing the query
    against same-length windows instead lifts that to 0.92, while genuinely
    unrelated pairs stay near 0.5 - a wide enough margin to threshold on.
    """
    best = similarity(query, value)
    q_words = normalize_key(query).split()
    v_words = normalize_key(value).split()
    if not q_words or not v_words:
        return best

    q_joined, n = " ".join(q_words), len(q_words)
    for size in {max(1, n - 1), n, n + 1}:
        if size > len(v_words):
            continue
        for i in range(len(v_words) - size + 1):
            window = " ".join(v_words[i:i + size])
            ratio = SequenceMatcher(None, q_joined, window, autojunk=False).ratio()
            if ratio > best:
                best = ratio
    return best


# --------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------
_MS_MARKERS = {
    "jumlah", "tunai", "baki", "cukai", "tarikh", "masa", "resit", "harga",
    "kedai", "jalan", "terima", "kasih", "bayaran", "barang", "kuantiti",
    "sila", "datang", "lagi", "syarikat", "perniagaan", "besar", "kecil",
    "dan", "yang", "untuk", "dengan", "pada", "juruwang", "invois", "wang",
}
_EN_MARKERS = {
    "total", "cash", "change", "tax", "date", "time", "receipt", "price",
    "shop", "street", "thank", "you", "payment", "item", "items", "quantity",
    "please", "come", "again", "company", "invoice", "amount", "subtotal",
    "the", "and", "for", "with", "cashier",
}


def detect_language(text: str) -> str:
    """Return ``"ms"``, ``"en"`` or ``"mixed"`` for a receipt's text."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    ms = len(words & _MS_MARKERS)
    en = len(words & _EN_MARKERS)
    if ms == 0 and en == 0:
        return "unknown"
    if ms and en and min(ms, en) / max(ms, en) >= 0.4:
        return "mixed"
    return "ms" if ms > en else "en"
