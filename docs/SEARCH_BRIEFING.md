# Search briefing — for the demo

**Read this once before Saturday. Twenty minutes is enough.**

You are presenting the search half of the system. You did not write this code,
so this document covers what it does, what to click, and what to say if you are
asked about it. Everything below has been **run against the actual database** —
the results shown are what you will see.

---

## 1. What the search half is for

The assignment requires a Document Management System that can retrieve stored
documents by their named entities, and — this is the part that matters —

> *return **related** documents when the exact query is not present.*

That single requirement is why search is a cascade rather than one lookup.

---

## 2. The four strategies, in order

The system tries each in turn and stops at the first that finds something. It
always reports **which one answered**, so a result set can explain itself.

| # | Strategy | Plain meaning | Example |
|---|---|---|---|
| 1 | **exact** | stored value is identical | `60.31` |
| 2 | **partial** | query appears inside a stored value | `Kuala Lumpur` inside a full address |
| 3 | **similar** | nothing matched — find the closest, two different ways | `Kuala Lumpor`, `nasi ayam` |
| 4 | **type_fallback** | the query is a Malaysian place we don't hold — return the places we *do* hold | `Kota Kinabalu` |
| — | *empty* | nothing was close enough, plus did-you-mean suggestions | |

### Stage 3 is the interesting one

It runs **two independent measures of closeness** and merges the results:

- **Lexical** — spelling distance. Rescues a typo. `Kuala Lumpor` → `Kuala
  Lumpur` scores **0.917**; the threshold is **0.72**.
- **Semantic** — meaning distance, using sentence embeddings. Rescues a
  *different word for a related thing*. `nasi ayam` → `Chicken` — no shared
  letters at all.

Each result shows both scores and a `matched_by` column saying `lexical`,
`semantic`, or `both`. **That column is your best visual aid** — it makes the
difference between the two mechanisms concrete on screen.

---

## 3. What to click — verified output

Search page, semantic toggle **on**. Type these four, in this order. The scores
below are real.

### `Kuala Lumpur` → mode **partial**
Five hits. The query sits inside longer stored addresses.
Say: *"ordinary substring matching — the baseline case."*

### `Kuala Lumpor` → mode **similar**
Deliberately misspelled. Still finds the right addresses, `lex=0.917`,
`sem=None`, `matched_by=lexical`.
Say: *"exact and substring both failed, so it fell through to similarity. The
spelling measure rescued it — 0.917 against a 0.72 threshold."*

### `nasi ayam` → mode **similar** ← **the best moment in the demo**

| result | lexical | semantic | matched by |
|---|---|---|---|
| `Chicken` | — | **0.592** | semantic |
| `Add Chicken` | — | **0.583** | semantic |
| `L AyamGoreng` | — | **0.570** | semantic |

Say: *"A Malay query returning English results. The lexical score is **null** —
'nasi ayam' and 'Chicken' share no letters, so string matching finds nothing at
all. Only the embedding model can connect them, because it places them close
together by meaning. This is what semantic search is for."*

**Pause here.** This is the single clearest evidence in the whole presentation
that real NLP is happening rather than pattern matching.

### `Kota Kinabalu` → mode **type_fallback**
A city not in the database. Returns other stored locations instead.
Say: *"This is the requirement the assignment names directly — when the query
isn't present, return related documents rather than an empty page. The system
recognises it as a Malaysian place name, so it returns the locations we do hold."*

### Optional, if you have time — `chicken rice` → mode **similar**
Matches `matched_by=both`: `lex=0.737`, `sem=0.769`. Shows the two mechanisms
agreeing and the union ranking them together.

---

## 4. Questions you might get

**Q: What is an embedding?**
> A model turns a piece of text into a list of 1024 numbers, positioned so that
> texts meaning similar things end up near each other. Comparing two texts is
> then just measuring the angle between their two lists — cosine similarity.

**Q: Which model?**
> BGE-M3. It's multilingual, which is why a Malay query can match English text.
> It runs locally on the CPU; nothing is sent anywhere.

**Q: Why not just use semantic search for everything?**
> Because it's slower and less precise where exact matching already works. You
> don't want a search for the total `60.31` returning "amounts that feel similar".
> We only embed the four entity types whose values are actually language —
> merchant, address, item, cashier. Money, dates and IDs are served better by
> exact matching.

**Q: Why does the cascade go in that order?**
> Cheapest and most precise first. Exact matching costs nothing and is never
> wrong; embeddings cost a model inference. There's no reason to reach for the
> expensive method when an exact match exists.

**Q: What is the threshold and how was it chosen?**
> 0.72 for lexical, 0.53 for semantic. Both were measured, not guessed — real
> typos score 0.87–1.00 while unrelated pairs peak at 0.55, so 0.72 sits in the
> gap. The figures come from `tools/benchmark_embeddings.py`.

**Q (harder): Your benchmark shows plain string matching beat the embeddings.
So why use them?**
> Because that benchmark's queries were spelled similarly to the stored text,
> which is where string matching is strongest. Embeddings earn their place where
> lexical similarity is *zero* — the `nasi ayam` case. That's why the system runs
> both and merges the results instead of choosing one.

---

## 5. Two things not to say

1. **Don't claim you wrote the extraction pipeline.** If asked about OCR, the
   rule layer, the language model or the merge logic, hand it over:
   *"that's [teammate]'s half — he can take that."* The rubric marks you
   individually on what you can explain accurately; a confident wrong answer
   costs more than a handover.

2. **Don't quote an accuracy figure for search.** The 63.6% headline is the
   *extraction* accuracy. Search was measured separately with Recall@1, Recall@5
   and MRR. If pressed: *"Recall@1 of 86.4% for the embedding model on our
   retrieval probes."*

---

## 6. Before you present

- [ ] Open the app once yourself and run all four searches
- [ ] Check the Database page shows **40 documents**
- [ ] Know where the semantic toggle is — turn it **off** and re-run `nasi ayam`
      to show it returns nothing. That contrast proves the embeddings are doing
      the work, and it takes five seconds.
