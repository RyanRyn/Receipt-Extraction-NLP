"""Bilingual (Bahasa Melayu / English) lexicon for Malaysian receipts.

Everything language-specific lives here so the rule layer stays readable and a
new language can be added without touching the extraction logic.

The Malay terms matter: a receipt printed by a Malaysian SME routinely labels
the grand total "JUMLAH BESAR", the tax "CUKAI", cash "TUNAI" and change
"BAKI" - none of which an English-only extractor recognises.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Money field labels, most specific first (order is significant).
# --------------------------------------------------------------------------
# Each entry: (ENTITY_TYPE, [surface forms]).  Longer / more specific labels are
# listed before their prefixes so "JUMLAH KECIL" is not swallowed by "JUMLAH".
MONEY_LABELS: list[tuple[str, list[str]]] = [
    ("SUBTOTAL", [
        "jumlah kecil", "jumlah sebelum cukai", "sub jumlah", "subjumlah",
        "sub total", "subtotal", "sub-total", "amount before tax",
    ]),
    ("TAX", [
        "cukai sst", "cukai gst", "cukai jualan", "cukai perkhidmatan", "cukai",
        "sst 6%", "gst 6%", "sst", "gst", "service tax", "sales tax", "tax",
        "vat",
    ]),
    ("TOTAL", [
        # Malaysian receipts frequently print a rounding adjustment and then a
        # final line such as "TOTAL AFTER ADJ INCL GST". That line, not the
        # earlier "Total Sales", is the amount actually payable, so its labels
        # are listed first and marked specific (see rules._SPECIFIC_TOTAL).
        "total after adjustment", "total after adj", "total after rounding",
        "jumlah selepas pelarasan", "jumlah selepas adj",
        "total inclusive of gst", "total incl gst", "total incl. gst",
        "total sales incl gst", "total sales inclusive of gst",
        "jumlah besar", "jumlah keseluruhan", "jumlah bayaran", "jumlah akhir",
        "grand total", "total amount", "nett total", "net total", "total due",
        "amount due", "jumlah", "amaun", "total", "netto",
    ]),
    ("PAID", [
        "tunai", "wang tunai", "bayaran tunai", "dibayar", "jumlah dibayar",
        "cash", "amount paid", "paid", "payment", "kad kredit", "kad debit",
    ]),
    ("CHANGE", [
        "baki", "wang baki", "duit baki", "change", "changes", "balance",
    ]),
]

# --------------------------------------------------------------------------
# Other field labels
# --------------------------------------------------------------------------
# A tax-summary block near the foot of a Malaysian receipt, e.g.
#
#     GST Summary   Amount    Tax
#     Total          23.09    0.52
#
# The word "Total" there belongs to the summary table, and its rightmost figure
# is the *tax*, not the amount payable. Everything after such a header is
# therefore barred from supplying the grand total.
SUMMARY_HEADERS = [
    "gst summary", "sst summary", "tax summary", "summary of gst",
    "ringkasan cukai", "ringkasan gst", "gst analysis", "tax analysis",
    "gst breakdown",
]

DATE_LABELS = ["tarikh", "tkh", "date", "dated", "tarikh resit"]
TIME_LABELS = ["masa", "waktu", "time", "jam"]
INVOICE_LABELS = [
    "no. resit", "no resit", "nombor resit", "resit no", "resit",
    "no. invois", "no invois", "invois", "invoice no", "invoice",
    "inv no", "inv", "bill no", "bil no", "receipt no", "receipt",
    "no. transaksi", "trans no", "slip no", "doc no", "struk",
]
TAX_ID_LABELS = [
    "no. cukai", "no cukai", "nombor cukai", "gst id", "gst no", "gst reg",
    "sst id", "sst no", "sst reg", "tax id", "tax no", "gst", "sst", "roc",
    "no. pendaftaran", "reg no",
]
PHONE_LABELS = ["tel", "telefon", "no. tel", "no tel", "phone", "hp", "h/p", "fax", "faks"]
CASHIER_LABELS = ["juruwang", "kasir", "cashier", "served by", "dilayan oleh", "operator"]

# --------------------------------------------------------------------------
# Payment methods
# --------------------------------------------------------------------------
PAYMENT_METHODS: dict[str, list[str]] = {
    "CASH":        ["tunai", "cash", "wang tunai"],
    "CREDIT_CARD": ["kad kredit", "credit card", "visa", "mastercard", "master card", "amex"],
    "DEBIT_CARD":  ["kad debit", "debit card", "debit"],
    "EWALLET":     ["touch n go", "touch 'n go", "tng", "grabpay", "boost",
                    "shopeepay", "e-wallet", "ewallet", "dompet digital"],
    "QR":          ["duitnow", "duit now", "qr pay", "qr code", "scan qr"],
    "ONLINE":      ["online banking", "fpx", "bank transfer", "pindahan bank"],
}

# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
# Malay month names (and their common 3-letter abbreviations) mapped to numbers.
MONTHS: dict[str, int] = {
    # Bahasa Melayu
    "januari": 1, "februari": 2, "mac": 3, "april": 4, "mei": 5, "jun": 6,
    "julai": 7, "ogos": 8, "september": 9, "oktober": 10, "november": 11,
    "disember": 12,
    "jan": 1, "feb": 2, "apr": 4, "jul": 7, "ogo": 8, "sep": 9, "okt": 10,
    "nov": 11, "dis": 12,
    # English
    "january": 1, "february": 2, "march": 3, "may": 5, "june": 6, "july": 7,
    "august": 8, "october": 10, "december": 12,
    "mar": 3, "aug": 8, "oct": 10, "dec": 12,
}

# Malay day names - useful as a *date anchor*: the OCR of the sample receipt
# reads "Rabu, 10-05-2017", so a day name signals a nearby date.
DAY_NAMES = [
    "isnin", "selasa", "rabu", "khamis", "jumaat", "sabtu", "ahad",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
]

# Malay time-of-day qualifiers.
DAY_PERIODS = ["pagi", "tengahari", "petang", "malam", "am", "pm", "a.m.", "p.m."]

# --------------------------------------------------------------------------
# Business / address markers
# --------------------------------------------------------------------------
COMPANY_MARKERS = [
    "sdn bhd", "sdn. bhd", "bhd", "enterprise", "enterprises", "trading",
    "perniagaan", "kedai", "restoran", "restaurant", "cafe", "kafe", "gerai",
    "mini market", "minimarket", "supermarket", "pasar raya", "pasaraya",
    "hardware", "pharmacy", "farmasi", "klinik", "clinic", "berhad",
    "holdings", "services", "servis", "motor", "auto", "stationery",
    "bakery", "catering", "resources", "marketing", "industries", "plt",
]

# Deliberately excludes bare "no." / "no": on its own it matches "No. Items: 1"
# and turns a totals line into an address. A real address line carries a street
# word, a place name or a postcode as well.
ADDRESS_MARKERS = [
    "jalan", "jln", "lorong", "lrg", "taman", "tmn", "kampung", "kg",
    "bandar", "seksyen", "sekysen", "persiaran", "lebuh", "lebuhraya",
    "bangunan", "blok", "tingkat", "wisma", "plaza",
    "batu", "mukim", "daerah", "pekan", "kompleks",
]

# Malaysian states + major cities. Two jobs:
#   1. recognising an address line,
#   2. powering the "similar entity" fallback - if a user searches for
#      "Johor Bahru" and it is absent, the system knows the query is a *place*
#      and can return the other places it does hold (e.g. "Kuala Lumpur").
MALAYSIAN_PLACES = [
    # states / federal territories
    "johor", "kedah", "kelantan", "melaka", "malacca", "negeri sembilan",
    "pahang", "perak", "perlis", "pulau pinang", "penang", "sabah", "sarawak",
    "selangor", "terengganu", "kuala lumpur", "labuan", "putrajaya",
    # major towns / cities
    "johor bahru", "johor baru", "iskandar puteri", "batu pahat", "muar",
    "kluang", "kulai", "pontian", "segamat", "alor setar", "sungai petani",
    "kulim", "langkawi", "kota bharu", "ayer keroh", "seremban", "nilai",
    "port dickson", "kuantan", "temerloh", "bentong", "ipoh", "taiping",
    "teluk intan", "sitiawan", "kangar", "george town", "butterworth",
    "bayan lepas", "bukit mertajam", "kota kinabalu", "sandakan", "tawau",
    "kuching", "miri", "sibu", "bintulu", "shah alam", "petaling jaya",
    "subang jaya", "klang", "kajang", "bangi", "cyberjaya", "rawang",
    "puchong", "cheras", "ampang", "setapak", "wangsa maju", "sepang",
    "kuala terengganu", "kemaman", "dungun", "batang kali", "bandar utama",
    "damansara", "mont kiara", "sri hartamas", "bukit bintang", "sentul",
]

# --------------------------------------------------------------------------
# Noise lines worth ignoring when guessing the merchant name
# --------------------------------------------------------------------------
NOISE_LINES = [
    "tax invoice", "invois cukai", "resit jualan", "sales receipt",
    "simplified tax invoice", "cash bill", "bil tunai", "official receipt",
    "terima kasih", "terima kaseh", "thank you", "thank u", "thanks",
    # "TQ" is near-universal Malaysian shorthand for "thank you" and appears in
    # receipt footers ("TQ FOR SHOPPING WITH ..."), which were being mistaken
    # for the merchant name when the real name carried no company marker.
    "tq for", "tq!", "sila datang lagi",
    "please come again", "barang yang dijual", "goods sold",
    "tidak boleh dipulangkan", "not refundable", "no refund",
    "selamat datang", "welcome", "customer copy", "salinan pelanggan",
]


# Malaysian receipts often quote prices with the tax already included, and then
# print the tax separately for information. On such a receipt
# `subtotal + tax == total` is deliberately false, so the arithmetic check must
# be suppressed rather than "corrected".
TAX_INCLUSIVE_MARKERS = [
    "inclusive of gst", "inclusive of sst", "inclusive of tax",
    "gst inclusive", "sst inclusive", "tax inclusive", "incl. gst", "incl gst",
    "price inclusive", "prices inclusive", "inclusive gst",
    "termasuk gst", "termasuk sst", "termasuk cukai", "harga termasuk",
    "termasuk dalam harga",
]


def all_money_labels() -> list[tuple[str, str]]:
    """Flatten :data:`MONEY_LABELS` to ``(label, ENTITY_TYPE)``, longest first.

    Sorting by descending length is what stops the substring "jumlah" from
    matching inside "jumlah kecil".
    """
    pairs = [(lbl, etype) for etype, labels in MONEY_LABELS for lbl in labels]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs
