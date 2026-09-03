"""Hybrid NER: fuse the rule layer and the local-LLM layer, then validate.

Neither layer is trusted blindly:

* the rule layer is precise on *printed patterns* (amounts, ids, dates) but
  brittle on anything that needs reading comprehension (which line is the shop
  name? where does the address stop?);
* the LLM is strong on exactly that, but can drift on digits.

So each type is resolved by the layer that is actually good at it, agreement
between the two is used as a confidence signal, and the result is then checked
against receipt arithmetic (``subtotal + tax == total``,
``paid - change == total``). Running the layers independently also means the
system degrades gracefully: with no model downloaded it still works, just with
rule-only accuracy.
"""

from __future__ import annotations

from dms.config import (
    ARITHMETIC_TOLERANCE,
    CONF_AGREE,
    CONF_LLM_ONLY,
    CONF_RULE_ONLY,
    SUBTOTAL_LIKE_FRACTION,
    TAX_MAX_FRACTION,
)
from dms.lexicon import TAX_INCLUSIVE_MARKERS
from dms.llm import LocalLLMExtractor
from dms.rules import extract_rules, line_offsets, looks_like_address
from dms.textutils import find_money_values
from dms.schema import ENTITY_TYPES, MONEY_TYPES, Entity
from dms.textutils import (
    format_money,
    strip_graphic_noise,
    locate_span,
    normalize_spaces,
    parse_money,
    similarity,
)

# Which layer wins when the two disagree.
#
# MERCHANT sits with the rules rather than the LLM, which is not what was
# assumed at first. Measured over 30 receipts the rule layer scored 46.7% exact
# on the merchant name against the LLM's 36.7%, and the failure mode is
# systematic: the model drops leading words, answering "RIANG" for "PERNIAGAAN
# RIANG" and "RUN CIT MAKMUR SDN BHD" for "KEDAI RUNCIT MAKMUR SDN BHD".
# The merchant is almost always simply the top line, which is a positional rule
# the regex layer expresses exactly.
#
# ADDRESS stays with the LLM: there the rules score 33.3% and the LLM's broader
# reading of where the address block ends still adds value once implausible
# answers are vetoed (see rules.looks_like_address).
PREFER_LLM = {"ADDRESS", "ITEM"}
PREFER_RULE = {
    "MERCHANT",
    "SUBTOTAL", "TAX", "TOTAL", "PAID", "CHANGE",
    "DATE", "TIME", "PHONE", "TAX_ID", "INVOICE_NO",
    "PAYMENT_METHOD", "CURRENCY", "CASHIER",
}

# Strings a model emits when it means "nothing", instead of a JSON null.
_NULL_WORDS = {"none", "null", "nil", "n/a", "na", "-", "unknown", "undefined",
               "nan", "not available", "tiada", "tidak diketahui"}

# LLM JSON key -> entity type
LLM_KEY_TO_TYPE = {
    "merchant": "MERCHANT",
    "address": "ADDRESS",
    "phone": "PHONE",
    "tax_id": "TAX_ID",
    "invoice_no": "INVOICE_NO",
    "date": "DATE",
    "time": "TIME",
    "currency": "CURRENCY",
    "subtotal": "SUBTOTAL",
    "tax": "TAX",
    "total": "TOTAL",
    "paid": "PAID",
    "change": "CHANGE",
    "payment_method": "PAYMENT_METHOD",
}

FIELD_ORDER = list(LLM_KEY_TO_TYPE)

# Every scalar entity type appears in the extracted record, whether or not it was
# found, with null standing for "looked and it is not on this receipt".
#
# A record that silently omits what it could not extract is not a record, it is a
# summary: a consumer cannot tell "no cashier printed" from "cashier never
# considered", and the two mean very different things. Deriving the set from
# ENTITY_TYPES rather than from the LLM's key list also stops the record drifting
# when a type is added - CASHIER was recognised, stored and searchable, yet
# missing from every extracted record precisely because it was absent from that
# list.
#
# ITEM is excluded because it repeats rather than being scalar. It has its own
# `items` list, which is empty rather than null when nothing was found.
SCALAR_FIELDS = {etype: etype.lower()
                 for etype in ENTITY_TYPES if etype != "ITEM"}


def _agree(etype: str, a: str, b: str) -> bool:
    """Do two values for the same entity type mean the same thing?"""
    if a is None or b is None:
        return False
    if etype in MONEY_TYPES:
        va, vb = parse_money(a), parse_money(b)
        return va is not None and vb is not None and abs(va - vb) < 0.005
    if etype in ("DATE", "TIME", "TAX_ID", "INVOICE_NO", "CURRENCY", "PAYMENT_METHOD"):
        return str(a).strip().lower() == str(b).strip().lower()
    return similarity(str(a), str(b)) >= 0.85


def _llm_to_entities(data: dict, ocr_text: str,
                     verify_with_rules: bool = True) -> list[Entity]:
    """Turn the LLM's JSON into span-aligned entities.

    ``verify_with_rules`` applies two sanity checks that use the rule layer's
    knowledge, so it is switched off in the LLM-only ablation to keep that
    baseline honest:

    * an "address" that does not look like an address is dropped (on a receipt
      with no printed address the model offers the GST number instead);
    * an amount that appears nowhere on the receipt is dropped. The check is
      numeric and OCR-tolerant, so a repaired ``22.60`` still matches the
      printed ``22-60`` - but an invented total matches nothing.
    """
    out: list[Entity] = []
    for key, etype in LLM_KEY_TO_TYPE.items():
        raw = data.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        if etype in MONEY_TYPES:
            value = format_money(parse_money(raw))
        elif etype in ("MERCHANT", "ADDRESS"):
            # The model reads the same logo artefacts the rules do.
            value = strip_graphic_noise(str(raw))
        else:
            value = normalize_spaces(str(raw))
        if not value:
            continue
        if etype not in MONEY_TYPES and value.strip(" -.").lower() in _NULL_WORDS:
            continue        # the word "None", not an actual value
        if verify_with_rules:
            if etype == "ADDRESS" and not looks_like_address(value):
                continue    # the model invented an address; the rules veto it
            if etype in MONEY_TYPES and _value_printed(ocr_text, parse_money(value)) is None:
                continue    # that amount is not printed anywhere on the receipt
        span = locate_span(ocr_text, str(raw))
        start, end = span if span else (-1, -1)
        out.append(Entity(
            type=etype, value=value,
            text=ocr_text[start:end] if span else str(raw),
            start=start, end=end,
            confidence=CONF_LLM_ONLY, source="llm",
        ))

    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = normalize_spaces(str(item.get("name") or ""))
        # A model asked for JSON sometimes writes the *word* "None" (or "null",
        # "N/A") instead of emitting an actual null. Those are not product
        # names, and stored verbatim they pollute both the item list and the
        # semantic index.
        if not name or name.strip(" -.").lower() in _NULL_WORDS:
            continue
        span = locate_span(ocr_text, name)
        start, end = span if span else (-1, -1)
        out.append(Entity(
            type="ITEM", value=name,
            text=ocr_text[start:end] if span else name,
            start=start, end=end,
            confidence=CONF_LLM_ONLY, source="llm",
            meta={"qty": item.get("qty"),
                  "unit_price": item.get("unit_price"),
                  "amount": item.get("amount")},
        ))
    return out


def _merge_single(etype: str, rule: Entity | None, llm: Entity | None,
                  warnings: list[str],
                  prefer_llm: set[str] | None = None) -> Entity | None:
    """Reconcile the two layers' opinions about one entity type.

    ``prefer_llm`` overrides which types the LLM owns on a disagreement. It is
    a parameter rather than a constant so the routing can be *tuned on the
    training split* and only then frozen, instead of being fitted to the test
    data it is later reported on (see tune_on_train.py).
    """
    if prefer_llm is None:
        prefer_llm = PREFER_LLM
    if rule is None and llm is None:
        return None
    if rule is not None and llm is None:
        rule.confidence = max(rule.confidence, CONF_RULE_ONLY)
        rule.source = "rule"
        return rule
    if rule is None and llm is not None:
        llm.confidence = CONF_LLM_ONLY
        llm.source = "llm"
        return llm

    if _agree(etype, rule.value, llm.value):
        winner = rule if rule.is_aligned else llm
        winner.confidence = CONF_AGREE
        winner.source = "rule+llm"
        return winner

    # Disagreement: hand the field to whichever layer owns it.
    if etype in prefer_llm:
        winner, loser = llm, rule
    else:
        winner, loser = rule, llm
    warnings.append(
        f"{etype}: rule={rule.value!r} vs llm={llm.value!r} -> kept {winner.source}"
    )
    winner.confidence = 0.55
    winner.meta = dict(winner.meta or {}, alternative=loser.value,
                       alternative_source=loser.source,
                       alternative_start=loser.start, alternative_end=loser.end,
                       alternative_text=loser.text)
    return winner


def _arbitrate_money(merged: list[Entity], warnings: list[str]) -> None:
    """Let receipt arithmetic settle a disagreement the layers could not.

    When the rule layer and the LLM disagree on subtotal/tax/total, exactly one
    combination normally satisfies ``subtotal + tax == total``. Picking that one
    turns an arbitrary tie-break into a checkable decision - and it is what
    recovers a tax of 1.28 when a smudged "1.2g" made the regex read 0.00.
    """
    by = {e.type: e for e in merged if e.type in ("SUBTOTAL", "TAX", "TOTAL")}
    if len(by) < 3:
        return

    options: dict[str, list[tuple[float, Entity | None]]] = {}
    for etype, ent in by.items():
        seen, choices = set(), []
        primary = parse_money(ent.value)
        if primary is not None:
            seen.add(round(primary, 2))
            choices.append((primary, None))
        alt = (ent.meta or {}).get("alternative")
        alt_value = parse_money(alt) if alt is not None else None
        if alt_value is not None and round(alt_value, 2) not in seen:
            choices.append((alt_value, ent))
        options[etype] = choices
    if not all(options.values()):
        return

    current = (options["SUBTOTAL"][0][0], options["TAX"][0][0], options["TOTAL"][0][0])
    if abs(current[0] + current[1] - current[2]) <= ARITHMETIC_TOLERANCE:
        return                                    # already consistent

    # Several combinations may balance. Prefer the one that disturbs the least
    # confident readings: overwriting a value both layers agreed on in order to
    # satisfy a figure only one layer proposed makes the result worse, not
    # better. Disruption is the summed confidence of the entities that change.
    best: tuple[float, list[tuple[str, float]]] | None = None
    for sub, _ in options["SUBTOTAL"]:
        for tax, _ in options["TAX"]:
            for total, _ in options["TOTAL"]:
                if abs(sub + tax - total) > ARITHMETIC_TOLERANCE:
                    continue
                changed, disruption = [], 0.0
                for etype, chosen in (("SUBTOTAL", sub), ("TAX", tax), ("TOTAL", total)):
                    ent = by[etype]
                    current = parse_money(ent.value)
                    if current is None or abs(current - chosen) > 0.005:
                        changed.append((etype, chosen))
                        disruption += ent.confidence
                if changed and (best is None or disruption < best[0]):
                    best = (disruption, changed)

    if best is None:
        return
    for etype, chosen in best[1]:
        ent = by[etype]
        meta = ent.meta or {}
        warnings.append(
            f"{etype}: {ent.value} -> {format_money(chosen)} "
            "(chosen because subtotal + tax = total)"
        )
        ent.value = format_money(chosen)
        ent.source = f"{meta.get('alternative_source', 'llm')}+arithmetic"
        ent.confidence = 0.90
        if meta.get("alternative_start", -1) >= 0:
            ent.start = meta["alternative_start"]
            ent.end = meta["alternative_end"]
            ent.text = meta.get("alternative_text") or ent.text
        ent.meta = dict(meta, arithmetic_ok=True)


def is_tax_inclusive(ocr_text: str) -> bool:
    """Does the receipt say its prices already include tax?"""
    low = (ocr_text or "").lower()
    return any(marker in low for marker in TAX_INCLUSIVE_MARKERS)


def _validate_arithmetic(fields: dict, entities: list[Entity],
                         warnings: list[str], tax_inclusive: bool = False) -> None:
    """Cross-check the money fields; raise or lower confidence accordingly."""
    def num(key):
        return parse_money(fields.get(key)) if fields.get(key) is not None else None

    sub, tax, total = num("subtotal"), num("tax"), num("total")
    paid, change = num("paid"), num("change")
    by_type = {e.type: e for e in entities}

    if sub is not None and tax is not None and total is not None and not tax_inclusive:
        if abs(sub + tax - total) <= ARITHMETIC_TOLERANCE:
            for t in ("SUBTOTAL", "TAX", "TOTAL"):
                if t in by_type:
                    by_type[t].confidence = min(0.99, by_type[t].confidence + 0.1)
                    by_type[t].meta = dict(by_type[t].meta or {}, arithmetic_ok=True)
        else:
            warnings.append(
                f"arithmetic: subtotal {sub:.2f} + tax {tax:.2f} != total {total:.2f}"
            )

    if paid is not None and change is not None:
        implied = round(paid - change, 2)
        if total is None:
            # A safe inference, and clearly flagged as derived rather than read.
            fields["total"] = format_money(implied)
            entities.append(Entity(
                type="TOTAL", value=format_money(implied), text="",
                confidence=0.60, source="derived",
                meta={"derived_from": "paid - change"},
            ))
            warnings.append("total absent; derived from paid - change")
        elif abs(implied - total) > ARITHMETIC_TOLERANCE:
            warnings.append(
                f"arithmetic: paid {paid:.2f} - change {change:.2f} != total {total:.2f}"
            )


def _value_printed(ocr_text: str, value: float) -> tuple[str, int, int] | None:
    """Locate ``value`` among the amounts actually printed on the receipt."""
    for line, off in line_offsets(ocr_text):
        for found, s, e in find_money_values(line):
            if abs(found - value) <= 0.005:
                return line[s:e], off + s, off + e
    return None


def _fix_implausible_tax(merged: list[Entity], warnings: list[str]) -> None:
    """Catch an amount labelled as tax that cannot possibly be one.

    Malaysian GST/SST is 6-10%, so tax is a small fraction of the total. When a
    layer reports a "tax" close to the total it has almost always picked up the
    *subtotal* - the two sit next to each other on the receipt and, once OCR
    destroys the captions, they are only distinguishable by magnitude.

    If the subtotal slot is empty, the value is moved there rather than
    discarded; the real tax is then re-derived from ``total - subtotal`` and
    accepted only if that figure is actually printed
    (:func:`_recover_tax_from_text`). Otherwise the implausible value is dropped,
    because reporting no tax is better than reporting a wrong one.
    """
    by = {e.type: e for e in merged}
    tax_ent, total_ent = by.get("TAX"), by.get("TOTAL")
    if tax_ent is None or total_ent is None:
        return

    tax, total = parse_money(tax_ent.value), parse_money(total_ent.value)
    if tax is None or total is None or total <= 0 or tax <= total * TAX_MAX_FRACTION:
        return                                      # plausible, leave it alone
    if (tax_ent.meta or {}).get("arithmetic_ok"):
        return                                      # corroborated by arithmetic

    if "SUBTOTAL" not in by and total * SUBTOTAL_LIKE_FRACTION <= tax < total:
        warnings.append(
            f"TAX {tax_ent.value} is {tax / total:.0%} of the total - "
            "reinterpreted as the subtotal"
        )
        tax_ent.type = "SUBTOTAL"
        tax_ent.confidence = min(tax_ent.confidence, 0.65)
        tax_ent.meta = dict(tax_ent.meta or {}, reclassified_from="TAX")
        return

    # Before discarding it, check what the layer that *lost* the merge proposed.
    # A money value from the LLM was already tested against the printed text when
    # it was created, so promoting it here adopts evidence that was gathered and
    # then thrown away by the routing table - it does not invent anything.
    alt = parse_money((tax_ent.meta or {}).get("alternative"))
    if alt is not None and 0 < alt <= total * TAX_MAX_FRACTION:
        meta = tax_ent.meta or {}
        warnings.append(
            f"TAX {tax_ent.value} is {tax / total:.0%} of the total - implausible; "
            f"using {format_money(alt)} from the {meta.get('alternative_source', 'other')} "
            "layer instead"
        )
        tax_ent.value = format_money(alt)
        tax_ent.source = f"{meta.get('alternative_source', 'llm')}+plausibility"
        tax_ent.confidence = 0.65
        if meta.get("alternative_start", -1) >= 0:
            tax_ent.start = meta["alternative_start"]
            tax_ent.end = meta["alternative_end"]
            tax_ent.text = meta.get("alternative_text") or tax_ent.text
        tax_ent.meta = dict(meta, reinstated_from="alternative")
        return

    warnings.append(
        f"TAX {tax_ent.value} is {tax / total:.0%} of the total - implausible, dropped"
    )
    merged.remove(tax_ent)


def _repair_total_from_payment(merged: list[Entity], ocr_text: str,
                               warnings: list[str]) -> None:
    """Correct a total that contradicts the cash tendered.

    ``paid - change`` is an independent statement of the amount due, computed
    by the till itself. When it disagrees with the total we read, and the
    figure it implies is printed on the receipt, the tendered arithmetic is the
    better evidence - this recovers totals whose leading digit OCR dropped
    (12.72 read as 0.72). A total that already satisfies ``subtotal + tax`` is
    left alone, and nothing is adopted that is not printed.
    """
    by = {e.type: e for e in merged}
    paid = parse_money(by["PAID"].value) if "PAID" in by else None
    change = parse_money(by["CHANGE"].value) if "CHANGE" in by else None
    if paid is None or change is None:
        return

    implied = round(paid - change, 2)
    if implied <= 0:
        return

    total_ent = by.get("TOTAL")
    total = parse_money(total_ent.value) if total_ent else None
    if total is not None and abs(total - implied) <= ARITHMETIC_TOLERANCE:
        return                                     # already agrees

    sub = parse_money(by["SUBTOTAL"].value) if "SUBTOTAL" in by else None
    tax = parse_money(by["TAX"].value) if "TAX" in by else None
    if (total is not None and sub is not None and tax is not None
            and abs(sub + tax - total) <= ARITHMETIC_TOLERANCE):
        return                                     # corroborated another way

    # Normally the implied figure must be printed on the receipt before it is
    # adopted. There is one exception: a total larger than the cash tendered is
    # not merely doubtful, it is impossible - nobody settles a 94.10 bill with
    # 25.10 - so an OCR misreading is the only explanation and the tendered
    # arithmetic wins even without a printed counterpart.
    impossible = total is not None and total > paid + ARITHMETIC_TOLERANCE
    printed = _value_printed(ocr_text, implied)
    if printed is None and not impossible:
        return

    if printed is not None:
        text, start, end = printed
        why = "paid - change, and printed on the receipt"
        confidence = 0.85
    else:
        text, start, end = "", -1, -1
        why = (f"paid - change; the stored total {format_money(total)} exceeds "
               f"the {format_money(paid)} tendered, which cannot happen")
        confidence = 0.70
    warnings.append(
        f"TOTAL: {total_ent.value if total_ent else 'absent'} -> "
        f"{format_money(implied)} ({why})"
    )
    if total_ent is None:
        total_ent = Entity(type="TOTAL", value="", source="")
        merged.append(total_ent)
    total_ent.value = format_money(implied)
    total_ent.text, total_ent.start, total_ent.end = text, start, end
    total_ent.confidence = confidence
    total_ent.source = "arithmetic+text" if printed else "arithmetic"
    total_ent.meta = dict(total_ent.meta or {}, derived_from="paid - change")


def _recover_tax_from_text(merged: list[Entity], ocr_text: str,
                           warnings: list[str]) -> None:
    """Recover a tax amount that OCR stripped of its label.

    Thermal receipts often lose the "GST 6%" caption while keeping the figure,
    leaving an orphan number no layer can attribute. If ``total - subtotal``
    equals an amount actually printed on the receipt, that is strong evidence -
    so the value is adopted, but only when it is really there. Nothing is
    invented: a figure with no counterpart in the text is left alone.
    """
    by = {e.type: e for e in merged}
    sub = parse_money(by["SUBTOTAL"].value) if "SUBTOTAL" in by else None
    total = parse_money(by["TOTAL"].value) if "TOTAL" in by else None
    if sub is None or total is None:
        return

    tax_ent = by.get("TAX")
    tax = parse_money(tax_ent.value) if tax_ent else None
    if tax is not None and abs(sub + tax - total) <= ARITHMETIC_TOLERANCE:
        return                                     # already consistent

    implied = round(total - sub, 2)
    if implied <= 0:
        return

    printed = _value_printed(ocr_text, implied)
    if printed is None:
        return

    text, start, end = printed
    warnings.append(
        f"TAX: {tax_ent.value if tax_ent else 'absent'} -> {format_money(implied)} "
        "(total - subtotal, and printed on the receipt)"
    )
    if tax_ent is None:
        tax_ent = Entity(type="TAX", value="", source="")
        merged.append(tax_ent)
    tax_ent.value = format_money(implied)
    tax_ent.text, tax_ent.start, tax_ent.end = text, start, end
    tax_ent.confidence = 0.85
    tax_ent.source = "arithmetic+text"
    tax_ent.meta = dict(tax_ent.meta or {}, arithmetic_ok=True,
                        derived_from="total - subtotal")


class HybridNER:
    """Rule layer + local LLM layer, merged and validated.

    ``use_rules`` / ``use_llm`` can be toggled independently, which is what the
    ablation study in the report uses to compare the three configurations.
    """

    def __init__(self, use_rules: bool = True, use_llm: bool = True,
                 model: str | None = None, mode: str = "full",
                 verbose: bool = True, cache: bool = False,
                 prefer_llm: set[str] | None = None):
        self.use_rules = use_rules
        self.use_llm = use_llm
        self.mode = mode
        self.verbose = verbose
        self.prefer_llm = set(prefer_llm) if prefer_llm is not None else set(PREFER_LLM)
        self._llm = (LocalLLMExtractor(model=model, verbose=verbose, cache=cache)
                     if use_llm else None)

    def extract(self, ocr_text: str):
        """Return ``(entities, fields, items, warnings)`` for one receipt."""
        warnings: list[str] = []

        rule_entities = extract_rules(ocr_text) if self.use_rules else []
        llm_data: dict = {}
        if self.use_llm and self._llm is not None:
            llm_data = self._llm.extract(ocr_text, mode=self.mode)
            if not llm_data:
                warnings.append("LLM layer returned nothing; using rules only")
        llm_entities = (
            _llm_to_entities(llm_data, ocr_text, verify_with_rules=self.use_rules)
            if llm_data else []
        )

        def first(entities, etype):
            found = [e for e in entities if e.type == etype]
            return found[0] if found else None

        merged: list[Entity] = []
        for etype in dict.fromkeys(LLM_KEY_TO_TYPE.values()):   # stable order
            ent = _merge_single(etype, first(rule_entities, etype),
                                first(llm_entities, etype), warnings,
                                prefer_llm=self.prefer_llm)
            if ent is not None:
                merged.append(ent)

        # Line items: the LLM's reading beats the regex fallback when present.
        llm_items = [e for e in llm_entities if e.type == "ITEM"]
        rule_items = [e for e in rule_entities if e.type == "ITEM"]
        merged.extend(llm_items or rule_items)

        cashier = first(rule_entities, "CASHIER")
        if cashier:
            merged.append(cashier)

        # On a tax-inclusive receipt `subtotal + tax == total` is deliberately
        # false, so the checks built on it must not fire. The payment
        # cross-check (paid - change == total) is unaffected and still runs.
        tax_inclusive = is_tax_inclusive(ocr_text)
        if not tax_inclusive:
            _arbitrate_money(merged, warnings)
        _repair_total_from_payment(merged, ocr_text, warnings)
        # The plausibility bound is a *magnitude* test - Malaysian GST/SST is
        # 6-10% of the bill whether or not the printed prices include it - so
        # unlike the identity checks above it remains valid on a tax-inclusive
        # receipt. Suppressing it there let a "tax" equal to the entire subtotal
        # survive, overriding the correct figure the model had proposed.
        _fix_implausible_tax(merged, warnings)
        if not tax_inclusive:
            _recover_tax_from_text(merged, ocr_text, warnings)

        fields = {}
        by_type = {e.type: e for e in merged}
        for etype, key in SCALAR_FIELDS.items():
            ent = by_type.get(etype)
            fields[key] = ent.value if ent else None

        _validate_arithmetic(fields, merged, warnings, tax_inclusive=tax_inclusive)
        # _validate_arithmetic may have appended a derived TOTAL.
        by_type = {e.type: e for e in merged}
        fields["total"] = by_type["TOTAL"].value if "TOTAL" in by_type else fields.get("total")

        items = [
            {"name": e.value, **{k: e.meta.get(k) for k in ("qty", "unit_price", "amount")}}
            for e in merged if e.type == "ITEM"
        ]

        # Report what was looked for and not found, not only what was found.
        #
        # These are produced here rather than when the document is saved so that
        # every consumer sees the same set: the command line, the interface, a
        # JSON export and the database cannot disagree about how many types a
        # receipt carries, because there is one place that decides.
        #
        # ITEM is excluded: it repeats, so "no line items" is already said
        # unambiguously by an empty list and does not need a null standing in
        # for it.
        present = {e.type for e in merged}
        for etype in SCALAR_FIELDS:
            if etype not in present:
                merged.append(Entity(type=etype, value=None, text="",
                                     start=-1, end=-1, confidence=0.0,
                                     source="absent"))

        # start=-1 sorts last, so what was found comes before what was not.
        merged.sort(key=lambda e: (e.start if e.start >= 0 else 10**9, e.type))
        return merged, fields, items, warnings
