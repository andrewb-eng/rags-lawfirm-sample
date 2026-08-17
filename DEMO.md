# 5-minute demo walkthrough

The pitch in one sentence: *a local hybrid-retrieval RAG that stays precise
over 500+ real contracts — proven by an adversarial eval, measured by a
ground-truth scale eval, and honest because every generated claim carries an
API-verified quote.*

## 1. The problem worth showing (30s)

Legal corpora are hostile to naive RAG: near-identical documents that differ
only by party name, keyword collisions ("relinquish control" means different
things in a merger vs. an indemnification carve-out), and deliberate decoys.
This repo bakes those failure modes into a 10-document needle set with a
ground-truth answer key — then hides it inside 510 real commercial contracts
(CUAD v1, ~27 MB) to prove precision survives scale.

## 2. Show the scale (30s)

```bash
.venv/bin/python fetch_corpus.py   # idempotent — reports corpus already present
.venv/bin/python main.py index     # 520 docs -> 50,397 chunks in ~81s
```

Talking points: section-aware chunking tuned to the embedder's 256-token
window; batched embedding + batched Chroma writes; manifest so a stale or
crashed index is detected; BM25 cache for fast startup.

## 3. Run the eval — the centerpiece (90s)

```bash
.venv/bin/python main.py eval
```

What to narrate while it prints:

- **Needle 5/5 at 500× noise.** Q1's top-5 documents are exactly the five
  near-identical amendments — three affirming relinquishment, two negating
  it — with 510 real contracts pushed below. Q2 puts David Okafor's amendment
  at #1 among four look-alikes differing only by name.
- **Scale eval: recall@5 0.92, MRR 0.85, p50 88 ms** over 100 queries
  synthesized from CUAD's expert annotations (each query names a
  corpus-unique party; exactly one contract is the right answer).

## 4. Ask a live question (90s)

```bash
.venv/bin/python main.py ask "Which parties relinquished control of their shares?"
```

The answer names Margaret Ellison, Thomas Vance, and David Okafor; excludes
Susan Cho and Priya Raman *because it read the negation* ("shall **not** be
deemed to have relinquished..."); and cites every claim with a quote the
Claude citations API guarantees is a verbatim span of the source. Point out
the trap it dodged: all five documents contain the phrase "relinquished
control of the shares."

Party-scoped follow-up: `--party "Anita Desai"` restricts both retrievers
via extracted metadata.

## 5. The engineering story for interviews (60s)

1. **Measured, not vibed.** Rank fusion (RRF) is the textbook hybrid choice —
   and it demonstrably failed here: five near-twins held BM25 scores 50%
   above every competitor, but rank compression erased the margin and two of
   them fell out of the top-5. Replacing it with per-query min-max score
   fusion fixed the needle eval (3/5 → 5/5) *and* raised scale recall
   (0.86 → 0.92). One measured decision, two metrics improved.
2. **Near-duplicates break assumptions.** MiniLM spreads "identical except
   the name" chunks across dense ranks 2–56 (exact, not an ANN artifact —
   verified by raising HNSW search_ef). Doc-level ranking therefore pools
   200 candidates deep.
3. **Ground truth or it didn't happen.** Both evals are derived from data
   (answer key / CUAD annotations), not hand-picked outputs, and `eval`'s
   exit code gates the build.
4. **Trust boundaries.** Retrieval is fully local; only generation touches
   the network; citations are API-verified so the model cannot fabricate a
   quote.
