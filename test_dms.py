"""
Self-contained test suite for the Receipt DMS.

    python test_dms.py

Runs without pytest and without downloading any model - the LLM layer is
disabled throughout, so this exercises OCR-independent logic: parsing, the rule
layer, span alignment, the search cascade and highlighting.

Each test doubles as a regression guard for a bug found during development;
the comment on each says which.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_FAILED: list[str] = []
_PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED.append(name)
        print(f"  FAIL  {name}   {detail}")


def section(title: str) -> None:
    print(f"\n{'-' * 68}\n{title}\n{'-' * 68}")


# ==========================================================================
def test_money() -> None:
    from dms.textutils import parse_money, find_money_values

    section("money parsing (OCR separator damage)")
    cases = {
        "22.60": 22.60,
        "22-60": 22.60,       # '-' misread for '.'
        "7 - 40": 7.40,
        "1,28": 1.28,
        "2=50": 2.50,         # '=' misread for '.'
        "RM 60.31": 60.31,
        "1,234.56": 1234.56,
        "3O.OO": 30.00,       # letter O for zero
        "Z1.32": 21.32,       # letter Z for two
        "226": 226.0,
    }
    for raw, expect in cases.items():
        got = parse_money(raw)
        check(f"parse_money({raw!r}) == {expect}", got == expect, f"got {got}")

    # Regression: a JSON number from the LLM must not lose its decimal point.
    # "22.6" was being stripped to "226" - a 10x error in every total.
    check("parse_money(22.6) == 22.6  [regression]", parse_money(22.6) == 22.6,
          f"got {parse_money(22.6)}")
    check("parse_money('22.6') == 22.6  [regression]", parse_money("22.6") == 22.6,
          f"got {parse_money('22.6')}")

    # A date must never be harvested as an amount.
    vals = [v for v, _, _ in find_money_values("Rabu, 10-05-2017 Time 21:51")]
    check("date line yields no amounts", vals == [], f"got {vals}")

    # ...but a price on the same kind of line must survive.
    vals = [v for v, _, _ in find_money_values("1.00 X 21.32 21.32")]
    check("price line yields amounts", 21.32 in vals, f"got {vals}")


def test_dates_times() -> None:
    from dms.textutils import normalize_date, normalize_time

    section("bilingual dates and times")
    for text, expect in [
        ("Tarikh: 14/03/2024", "2024-03-14"),
        ("Rabu, 10-05-2017", "2017-05-10"),
        ("12 Disember 2023", "2023-12-12"),      # Malay month
        ("5 Ogos 2021", "2021-08-05"),           # Malay month
        ("3 Mac 2020", "2020-03-03"),            # Malay month
    ]:
        got = normalize_date(text)
        check(f"date {text!r} -> {expect}", got and got[0] == expect, f"got {got}")

    got = normalize_time("Masa: 10:25 pagi")
    check("'10:25 pagi' -> 10:25", got and got[0] == "10:25", f"got {got}")
    got = normalize_time("Masa: 7:30 malam")
    check("'7:30 malam' -> 19:30", got and got[0] == "19:30", f"got {got}")
    got = normalize_time("Tiae 21;51")            # ';' misread for ':'
    check("'21;51' -> 21:51", got and got[0] == "21:51", f"got {got}")


def test_rules_malay() -> None:
    from dms.rules import extract_rules

    section("rule layer on a Bahasa Melayu receipt")
    text = """KEDAI RUNCIT MAKMUR SDN BHD
No 12, Jalan Bunga Raya, 43650 Bandar Baru Bangi, Selangor
Tel: 03-8926 1234
No. Cukai: W10-1808-31000123
RESIT JUALAN
No. Resit: RJ-2024-00871
Tarikh: 14/03/2024   Masa: 10:25 pagi
Beras Wangi 5kg      2  x  18.90   37.80
Jumlah Kecil                       56.90
Cukai SST 6%                        3.41
JUMLAH BESAR                       60.31
Tunai                              70.00
Baki                                9.69"""

    by = {e.type: e.value for e in extract_rules(text)}
    expected = {
        "MERCHANT": "KEDAI RUNCIT MAKMUR SDN BHD",
        "SUBTOTAL": "56.90",     # JUMLAH KECIL, not swallowed by JUMLAH
        "TAX": "3.41",           # CUKAI SST
        "TOTAL": "60.31",        # JUMLAH BESAR beats JUMLAH KECIL
        "PAID": "70.00",         # TUNAI
        "CHANGE": "9.69",        # BAKI
        "DATE": "2024-03-14",    # TARIKH
        "TIME": "10:25",         # MASA ... pagi
        "TAX_ID": "W10-1808-31000123",
        "INVOICE_NO": "RJ-2024-00871",
        "PAYMENT_METHOD": "CASH",
        "PHONE": "03-8926 1234",
    }
    for etype, want in expected.items():
        check(f"{etype} == {want!r}", by.get(etype) == want, f"got {by.get(etype)!r}")

    check("ADDRESS contains Bangi", "Bangi" in (by.get("ADDRESS") or ""),
          f"got {by.get('ADDRESS')!r}")

    # Regression: "RESIT JUALAN" was yielding the invoice number "JUALAN".
    check("INVOICE_NO is not 'JUALAN'  [regression]", by.get("INVOICE_NO") != "JUALAN")


def test_rules_english_noisy() -> None:
    from dms.rules import extract_rules, looks_like_address

    section("rule layer on OCR-damaged English")
    text = """PERNIAGAAN RIANG
GST: 001662431232
Tax INVOICE
1.00 X 21.32   21.32
No. Items: 1
TOTAL 22-60
CASH 30.00
CHANGE 7-40
Rabu, 10-05-2017 Time 21:51
Inv: R000039737"""

    by = {e.type: e.value for e in extract_rules(text)}
    for etype, want in [("TOTAL", "22.60"), ("PAID", "30.00"), ("CHANGE", "7.40"),
                        ("DATE", "2017-05-10"), ("TIME", "21:51"),
                        ("TAX_ID", "001662431232"), ("INVOICE_NO", "R000039737"),
                        ("MERCHANT", "PERNIAGAAN RIANG")]:
        check(f"{etype} == {want!r}", by.get(etype) == want, f"got {by.get(etype)!r}")

    # Regression: "No. Items: 1" was being classified as the address.
    check("no bogus ADDRESS  [regression]", by.get("ADDRESS") is None,
          f"got {by.get('ADDRESS')!r}")

    # Regression: OCR reads the colon after "Inv" as a semicolon, which used to
    # lose the invoice number entirely.
    noisy = extract_rules("SFULL Inv;R00oo39737\nToTaL 22-60")
    ids = {e.type: e.value for e in noisy}
    check("invoice number survives 'Inv;' separator  [regression]",
          ids.get("INVOICE_NO", "").upper().endswith("39737"),
          f"got {ids.get('INVOICE_NO')!r}")

    check("address veto rejects a tax number",
          not looks_like_address("95t,001662431232 1210644T"))
    check("address veto accepts a real address",
          looks_like_address("No 12, Jalan Bunga Raya, 43650 Bangi, Selangor"))


def test_gst_summary_and_rounding() -> None:
    from dms.ner import HybridNER
    from dms.rules import extract_rules

    section("GST summary tables and rounding adjustments")

    # Regression (test_00028): a Malaysian receipt ends with a tax-summary
    # table whose "Total" row lists amount and tax. The rule layer took the
    # rightmost figure, 0.52, as the grand total.
    text = """AEON CO. (M) BHD
Sub-total 24.11
Total Sales Incl GST 24.11
Rounding Adj -0.01
Total After Adj Incl GST 24.10
CASH 25.10
Item Count 6 Change Amt 1.00
GST Summary Amount Tax
Total 23.09 0.52"""
    by = {e.type: e.value for e in extract_rules(text)}
    check("post-rounding line wins as the total  [regression]",
          by.get("TOTAL") == "24.10", f"got {by.get('TOTAL')!r}")
    check("the GST summary row is not the total  [regression]",
          by.get("TOTAL") != "0.52", f"got {by.get('TOTAL')!r}")
    check("sub-total still read", by.get("SUBTOTAL") == "24.11",
          f"got {by.get('SUBTOTAL')!r}")

    # Regression: when OCR misreads the total (24.10 -> 94.10) the figure
    # exceeds the cash tendered, which is impossible, so the tendered
    # arithmetic must override even though 24.10 is not printed anywhere.
    misread = text.replace("Total After Adj Incl GST 24.10",
                           "Total After Adj Incl GST 94.10")
    _, fields, _, warns = HybridNER(use_llm=False, verbose=False).extract(misread)
    check("total above the cash tendered is corrected  [regression]",
          fields.get("total") == "24.10", f"got {fields.get('total')!r}")
    check("the correction is explained",
          any("cannot happen" in w for w in warns), f"got {warns}")

    # A legitimate receipt where the total is simply below the cash paid must
    # not be touched.
    ok = "KEDAI A\nJUMLAH BESAR 20.00\nTUNAI 50.00\nBAKI 30.00"
    _, fields, _, warns = HybridNER(use_llm=False, verbose=False).extract(ok)
    check("a consistent receipt is left alone", fields.get("total") == "20.00",
          f"got {fields.get('total')!r}")


def test_span_alignment() -> None:
    from dms.textutils import locate_span

    section("span alignment (value -> offsets for highlighting)")
    text = "ToTaL 22-60\nCAsh 30.00\nInv;R00oo39737"

    span = locate_span(text, "30.00")
    check("exact value aligns", span is not None and text[span[0]:span[1]] == "30.00",
          f"got {span}")

    # The LLM returns the *repaired* value; it must still find the damaged text.
    span = locate_span(text, "22.60")
    check("repaired '22.60' aligns onto '22-60'",
          span is not None and text[span[0]:span[1]] == "22-60", f"got {span}")

    span = locate_span(text, "R000039737")
    check("repaired id aligns onto OCR-damaged id",
          span is not None and "39737" in text[span[0]:span[1]], f"got {span}")

    check("absent value returns None", locate_span(text, "ZZZZ nonexistent") is None)


def test_database_and_search() -> None:
    from dms.database import ReceiptDB
    from dms.rules import extract_rules
    from dms.schema import ReceiptDocument

    section("database, search cascade and highlighting")
    tmp = os.path.join(tempfile.mkdtemp(), "test.sqlite3")
    db = ReceiptDB(tmp)

    docs = {
        "bangi.jpg": """KEDAI RUNCIT MAKMUR SDN BHD
No 12, Jalan Bunga Raya, 43650 Bandar Baru Bangi, Selangor
JUMLAH BESAR 60.31""",
        "kl.jpg": """RESTORAN NASI KANDAR SEJAHTERA
No 88, Jalan Tuanku Abdul Rahman, 50100 Kuala Lumpur
JUMLAH BESAR 23.32""",
    }
    for name, text in docs.items():
        ents = extract_rules(text)
        db.add_document(ReceiptDocument(filename=name, path=name, ocr_text=text,
                                        entities=ents))

    check("two documents stored", db.stats()["documents"] == 2)

    r = db.search_entities("Kuala Lumpur")
    check("'Kuala Lumpur' found", r["mode"] in ("exact", "partial") and r["hits"],
          f"mode={r['mode']}")

    r = db.search_entities("60.31")
    check("amount searchable", bool(r["hits"]), f"mode={r['mode']}")

    # The assignment's headline requirement. This asserts the OUTCOME - that a
    # document with a different Malaysian city comes back - not the mechanism,
    # so it stays valid whether the answer arrives via semantic similarity or
    # the place-name fallback.
    r = db.search_entities("Johor Bahru")
    check("'Johor Bahru' absent -> still returns something",
          r["mode"] in ("similar", "type_fallback") and r["hits"],
          f"mode={r['mode']} hits={len(r['hits'])}")
    values = " ".join(h["value"] for h in r["hits"])
    check("fallback surfaces the Kuala Lumpur document", "Kuala Lumpur" in values,
          f"got {values[:80]!r}")

    r = db.search_entities("Kuala Lumpor")          # misspelling
    check("misspelling still resolves", bool(r["hits"]), f"mode={r['mode']}")

    # Lexical-only must still behave exactly as before semantic search existed.
    r = db.search_entities("zzzznotathing", semantic=False)
    check("nonsense query is empty (lexical only)", r["mode"] == "empty",
          f"mode={r['mode']}")
    r = db.search_entities("Kuala Lumpur", semantic=False)
    check("lexical-only still finds a real city", bool(r["hits"]), f"mode={r['mode']}")

    hit = db.search_entities("Kuala Lumpur")["hits"][0]
    marked = db.highlight_document(hit["doc_id"], [hit["entity_id"]])
    check("highlight wraps the hit", "[[" in marked and "Kuala Lumpur" in marked)
    html = db.highlight_html(hit["doc_id"], [hit["entity_id"]])
    check("html highlight emits <mark>", "<mark" in html)

    db.close()


def test_semantic_types() -> None:
    from dms.schema import SEMANTIC_TYPES

    section("semantic indexing scope")
    # Regression: embedding money/date/enum values polluted retrieval - a
    # search for "Kota Kinabalu" came back with "PAID 30.00" ranked first.
    for etype in ("MERCHANT", "ADDRESS", "ITEM"):
        check(f"{etype} is embedded", etype in SEMANTIC_TYPES)
    for etype in ("TOTAL", "PAID", "CHANGE", "DATE", "TIME",
                  "CURRENCY", "PAYMENT_METHOD", "TAX_ID", "INVOICE_NO"):
        check(f"{etype} is NOT embedded  [regression]", etype not in SEMANTIC_TYPES)


def test_embedding_registry() -> None:
    from dms.embeddings import EMBEDDING_MODELS, resolve_embedder

    section("embedding model registry")
    check("default resolves", bool(resolve_embedder()[0]))
    for alias, cfg in EMBEDDING_MODELS.items():
        check(f"{alias} has a measured min_score",
              isinstance(cfg.get("min_score"), (int, float)))
    # E5 needs asymmetric prefixes; omitting them degrades retrieval.
    check("e5 models carry query/passage prefixes",
          EMBEDDING_MODELS["e5-base"]["query_prefix"] == "query: "
          and EMBEDDING_MODELS["e5-base"]["passage_prefix"] == "passage: ")
    # Thresholds must differ per model - the scales are not comparable.
    scores = {a: c["min_score"] for a, c in EMBEDDING_MODELS.items()}
    check("thresholds are per-model, not global", len(set(scores.values())) > 1,
          f"got {scores}")


def test_llm_json_recovery() -> None:
    from dms.llm import parse_json_object

    section("LLM JSON recovery (no model needed)")
    check("plain json", parse_json_object('{"total": 22.6}') == {"total": 22.6})
    check("fenced json",
          parse_json_object('```json\n{"total": 1}\n```') == {"total": 1})
    check("trailing prose",
          parse_json_object('Here you go: {"total": 2} hope that helps')
          == {"total": 2})
    check("trailing comma repaired",
          parse_json_object('{"a": 1, "b": 2,}') == {"a": 1, "b": 2})
    check("python dict literal",
          parse_json_object("{'total': '8.00'}") == {"total": "8.00"})
    check("garbage returns None", parse_json_object("no json at all") is None)

    # Regression: the model answered with a calculation instead of a value.
    # `"unit_price": 54.40 / 3` is not JSON, and it used to discard the entire
    # extraction - merchant, date, total and all.
    got = parse_json_object('{"total": 12.20, "unit_price": 54.40 / 3}')
    check("arithmetic expression evaluated  [regression]",
          got is not None and got.get("total") == 12.20
          and got.get("unit_price") == 18.13, f"got {got}")
    check("division by zero becomes null",
          (parse_json_object('{"x": 5 / 0}') or {}).get("x", "missing") is None,
          f"got {parse_json_object('{\"x\": 5 / 0}')}")

    # A reply cut off by the token limit should still yield its header fields.
    truncated = ('{"merchant": "KEDAI MAKMUR", "total": 60.31, "items": [{"name": "Ber')
    got = parse_json_object(truncated)
    check("truncated reply salvages header fields  [regression]",
          got is not None and got.get("merchant") == "KEDAI MAKMUR"
          and got.get("total") == 60.31, f"got {got}")

    # Salvage must not let a nested key shadow the real top-level one.
    got = parse_json_object(
        '{"total": 12.20, "items": [{"name": "x", "amount": 99.99} }}BROKEN')
    check("top-level value wins over nested  [regression]",
          got is not None and got.get("total") == 12.20, f"got {got}")


def test_implausible_tax() -> None:
    from dms.ner import _fix_implausible_tax
    from dms.schema import Entity

    section("implausible tax detection")

    # Regression: the LLM labelled the subtotal 21.32 as the tax on a total of
    # 22.60 (94%). It is moved to the empty subtotal slot instead.
    ents = [Entity(type="TAX", value="21.32", confidence=0.7, source="llm"),
            Entity(type="TOTAL", value="22.60", confidence=0.9, source="rule")]
    warns: list[str] = []
    _fix_implausible_tax(ents, warns)
    types = {e.type: e.value for e in ents}
    check("94% 'tax' reclassified as subtotal  [regression]",
          types.get("SUBTOTAL") == "21.32" and "TAX" not in types, f"got {types}")

    # A genuine 6% GST must be left completely alone.
    ents = [Entity(type="TAX", value="1.28", confidence=0.7, source="llm"),
            Entity(type="TOTAL", value="22.60", confidence=0.9, source="rule")]
    warns = []
    _fix_implausible_tax(ents, warns)
    check("6% tax untouched",
          {e.type for e in ents} == {"TAX", "TOTAL"} and not warns, f"got {warns}")

    # 10% SST plus a 10% service charge is still legitimate.
    ents = [Entity(type="TAX", value="4.50", confidence=0.7, source="llm"),
            Entity(type="TOTAL", value="24.50", confidence=0.9, source="rule")]
    _fix_implausible_tax(ents, [])
    check("18% tax untouched", {e.type for e in ents} == {"TAX", "TOTAL"})

    # Implausible, but the subtotal slot is taken, so it is dropped not moved.
    ents = [Entity(type="TAX", value="20.00", confidence=0.7, source="llm"),
            Entity(type="SUBTOTAL", value="21.32", confidence=0.8, source="rule"),
            Entity(type="TOTAL", value="22.60", confidence=0.9, source="rule")]
    _fix_implausible_tax(ents, [])
    check("implausible tax dropped when subtotal exists",
          {e.type for e in ents} == {"SUBTOTAL", "TOTAL"},
          f"got {[e.type for e in ents]}")


def test_null_words() -> None:
    from dms.ner import _llm_to_entities

    section("models writing 'None' instead of null")
    text = "KEDAI MAKMUR\nJUMLAH BESAR 60.31\nBeras Wangi 5kg 37.80"

    # Regression: the model returned {"name": "None"} and it was stored as a
    # product called "None", which then polluted the semantic index too.
    ents = _llm_to_entities(
        {"merchant": "None", "phone": "N/A", "invoice_no": "-",
         "items": [{"name": "None"}, {"name": "null"},
                   {"name": "Beras Wangi 5kg"}]},
        text, verify_with_rules=False)
    values = {(e.type, e.value) for e in ents}
    check("item named 'None' is dropped  [regression]",
          not any(v.lower() == "none" for _, v in values), f"got {values}")
    check("merchant 'None' is dropped", ("MERCHANT", "None") not in values)
    check("phone 'N/A' is dropped", not any(t == "PHONE" for t, _ in values))
    check("a real item still survives",
          ("ITEM", "Beras Wangi 5kg") in values, f"got {values}")


def test_tax_inclusive() -> None:
    from dms.ner import is_tax_inclusive

    section("tax-inclusive receipts")
    check("English marker detected",
          is_tax_inclusive("Thank You\n(Price Inclusive Of GST)\nCome Again"))
    check("Malay marker detected",
          is_tax_inclusive("Harga termasuk GST\nTerima kasih"))
    check("ordinary receipt is not flagged",
          not is_tax_inclusive("JUMLAH BESAR 60.31\nCukai SST 6% 3.41"))


def test_language_detection() -> None:
    from dms.textutils import detect_language

    section("language identification")
    check("Malay detected",
          detect_language("Jumlah Besar Tunai Baki Cukai Tarikh Resit") == "ms")
    check("English detected",
          detect_language("Total Cash Change Tax Date Receipt") == "en")
    check("mixed detected",
          detect_language("Jumlah Tunai Baki Cukai Total Cash Change Tax") == "mixed")


# ==========================================================================
def main() -> int:
    tests = [
        test_money,
        test_dates_times,
        test_rules_malay,
        test_rules_english_noisy,
        test_gst_summary_and_rounding,
        test_span_alignment,
        test_database_and_search,
        test_semantic_types,
        test_embedding_registry,
        test_llm_json_recovery,
        test_implausible_tax,
        test_null_words,
        test_tax_inclusive,
        test_language_detection,
    ]
    print("=" * 68)
    print("Receipt DMS - test suite (no model download, no GPU required)")
    print("=" * 68)
    for test in tests:
        try:
            test()
        except Exception:
            _FAILED.append(test.__name__)
            print(f"  ERROR in {test.__name__}:")
            traceback.print_exc()

    print(f"\n{'=' * 68}")
    if _FAILED:
        print(f"{_PASSED} passed, {len(_FAILED)} FAILED")
        for name in _FAILED:
            print(f"   - {name}")
        return 1
    print(f"ALL {_PASSED} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

