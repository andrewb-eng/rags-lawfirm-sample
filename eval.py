"""Stage 6 — eval: score retrieval.

Two sections:

1. NEEDLE EVAL — the suggested test queries from README.txt, scored against
   answer_key.json. Expected-document sets are DERIVED from the key's fields
   (type / party / has_relinquish_control_carveout / note) rather than
   hardcoded filenames, so the harness follows the key if it changes. This is
   the adversarial precision test: five near-identical amendments must stay
   distinguishable — even when they hide among 500+ real contracts.

2. SCALE EVAL (runs when the CUAD corpus is indexed) — ~100 retrieval queries
   synthesized from CUAD's expert annotations: each targets a provision
   (governing law, change of control, ...) of a contract identified by a
   party name that is unique in the corpus, so exactly one document is the
   right answer. Reports recall@k, MRR, and retrieval latency percentiles.

Per query, documents are classified three ways:
  expected — must appear in the top-k document ranking
  allowed  — reasonable to retrieve, neither required nor penalized (e.g. the
             non-carve-out amendments for Q1: retrieval hands them to the
             generator, which reads the negation and excludes them)
  decoys   — must NOT outrank any expected document (README: the NDA and
             lease "should rank LOW" on relinquish-control queries)

PASS = every expected doc in top-k, no decoy above any expected doc, and — for
queries marked exact — the expected doc at rank 1.

Also audits the party metadata extracted at chunk time against the answer
key's party field, since the whole precision story rests on it.
"""

import json
import sys

import config
from retrieve import Retriever


def _tokens(name: str) -> set[str]:
    return set(name.lower().replace("_", " ").split())


def party_matches(key_party: str, extracted: set[str]) -> bool:
    key = key_party.lower()
    for name in extracted:
        candidate = name.lower()
        if key in candidate or candidate in key:
            return True
        if len(_tokens(key) & _tokens(candidate)) >= 2:
            return True
    return False


def build_cases(key: dict) -> list[dict]:
    def files(pred):
        return sorted(f for f, meta in key.items() if pred(meta))

    indemnifications = files(lambda m: m["type"] == "indemnification_agreement")
    carveouts = files(
        lambda m: m["type"] == "indemnification_agreement"
        and m["has_relinquish_control_carveout"]
    )
    decoys = files(lambda m: "Decoy" in m.get("note", ""))
    okafor = files(lambda m: m["party"] == "David Okafor")
    nda = files(lambda m: m["type"] == "nda")
    merger = files(lambda m: m["type"] == "merger_agreement")
    employment = files(lambda m: m["type"] == "employment_agreement")
    lease = files(lambda m: m["type"] == "lease")

    return [
        {
            "name": "Q1  relinquished control of shares",
            "query": "Which parties relinquished control of their shares?",
            "expected": carveouts,
            # Non-carve-out amendments mention the phrase in a negation and the
            # merger uses it in a different sense — the generator sorts those
            # out, so retrieving them is not an error.
            "allowed": sorted(set(indemnifications) - set(carveouts)) + merger,
            "decoys": decoys,
        },
        {
            "name": "Q2  Okafor carve-out (unfiltered)",
            "query": "Find the change-in-control carve-out for David Okafor.",
            "expected": okafor,
            "allowed": sorted(set(indemnifications) - set(okafor)),
            "decoys": [],
            "exact": True,  # the core precision test: party must win over near-dupes
        },
        {
            "name": "Q2b Okafor carve-out (party filter)",
            "query": "Find the change-in-control carve-out.",
            "party": "David Okafor",
            "expected": okafor,
            "allowed": [],
            "decoys": [],
            "exact": True,
        },
        {
            "name": "Q3  NDA change-of-control provision",
            # At 10 documents "the NDA" was unambiguous; among 500+ real
            # contracts a definite-article query is underspecified, so the
            # realistic form names its target document.
            "query": "Does the Vantage Analytics NDA contain a change of control provision?",
            "expected": nda,
            # Per README the full answer is "no — only lease/employment do",
            # so surfacing those alongside the NDA is fine.
            "allowed": employment + lease,
            "decoys": [],
        },
        {
            "name": "Q4  Coastal Devices merger",
            "query": "Which documents involve the Coastal Devices merger?",
            "expected": merger,
            # The amendments are effective at that merger's Effective Time and
            # cite the merger agreement by date.
            "allowed": indemnifications,
            "decoys": [],
        },
    ]


def run_case(retriever: Retriever, case: dict, k: int) -> dict:
    ranking = retriever.retrieve_docs(case["query"], k_docs=k, party=case.get("party"))
    rank_of = {doc: i + 1 for i, doc in enumerate(ranking)}

    expected_ranks = {doc: rank_of.get(doc) for doc in case["expected"]}
    found = [doc for doc, rank in expected_ranks.items() if rank is not None]
    worst_expected = max((r for r in expected_ranks.values() if r), default=None)
    bad_decoys = [
        doc
        for doc in case["decoys"]
        if rank_of.get(doc) is not None
        and (worst_expected is None or rank_of[doc] < worst_expected)
    ]

    passed = (
        len(found) == len(case["expected"])
        and not bad_decoys
        and (not case.get("exact") or expected_ranks[case["expected"][0]] == 1)
    )
    return {
        "ranking": ranking,
        "expected_ranks": expected_ranks,
        "recall": len(found) / len(case["expected"]),
        "bad_decoys": bad_decoys,
        "passed": passed,
    }


def print_case(case: dict, result: dict, k: int) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] {case['name']}")
    print(f"       query: {case['query']!r}"
          + (f"  (party filter: {case['party']!r})" if case.get("party") else ""))
    for i, doc in enumerate(result["ranking"], 1):
        if doc in case["expected"]:
            marker = "expected"
        elif doc in case["allowed"]:
            marker = "allowed"
        elif doc in case["decoys"]:
            marker = "DECOY"
        else:
            marker = ""
        print(f"       {i}. {doc:42s} {marker}")
    missing = [d for d, r in result["expected_ranks"].items() if r is None]
    print(f"       recall@{k}: {result['recall']:.2f}"
          + (f"  MISSING: {missing}" if missing else "")
          + (f"  DECOY ABOVE EXPECTED: {result['bad_decoys']}" if result["bad_decoys"] else ""))
    print()


# --- scale eval --------------------------------------------------------------

# Parties answers that are role words rather than names never identify a
# single contract and are skipped.
_GENERIC_PARTIES = {
    "company", "distributor", "customer", "supplier", "buyer", "seller",
    "licensee", "licensor", "purchaser", "vendor", "client", "borrower",
    "lender", "contractor", "consultant", "parent", "subsidiary", "agent",
    "franchisee", "franchisor", "reseller", "provider", "recipient",
    "sponsor", "developer", "manufacturer", "owner", "operator", "party",
    "parties", "the company", "employee", "executive", "investor", "member",
    "partner", "trustee", "guarantor",
}

_SCALE_TEMPLATES = [
    ("Governing Law", "What law governs the agreement involving {party}?"),
    ("Change Of Control",
     "Does the agreement involving {party} contain a change of control provision?"),
    ("Anti-Assignment",
     "Can {party} assign the agreement to a third party without consent?"),
    ("Expiration Date", "When does the agreement involving {party} expire?"),
    ("Agreement Date", "When was the agreement with {party} entered into?"),
]


def build_scale_cases(
    annotations: dict, indexed_sources: set[str], n: int = config.SCALE_EVAL_N
) -> list[dict]:
    """Synthesize retrieval queries with exactly one correct document.

    A contract is eligible when one of its annotated party names is unique
    across the whole corpus; each of its annotated-present provisions yields
    one query. Deterministically seeded so runs are comparable.
    """
    import random

    party_docs: dict[str, set[str]] = {}
    for doc, meta in annotations.items():
        for p in meta["categories"].get("Parties", {}).get("answers", []):
            party_docs.setdefault(p.lower(), set()).add(doc)

    cases = []
    for doc, meta in sorted(annotations.items()):
        if doc not in indexed_sources:
            continue
        cats = meta["categories"]
        candidates = [
            p
            for p in cats.get("Parties", {}).get("answers", [])
            if 4 <= len(p) <= 60
            and p.lower() not in _GENERIC_PARTIES
            and len(party_docs[p.lower()]) == 1
        ]
        if not candidates:
            continue
        party = max(candidates, key=len)
        for category, template in _SCALE_TEMPLATES:
            info = cats.get(category)
            if info and info["present"] and info["answers"]:
                cases.append(
                    {
                        "query": template.format(party=party),
                        "expected": doc,
                        "category": category,
                    }
                )
    random.Random(13).shuffle(cases)
    return cases[:n]


def run_scale_eval(retriever: Retriever, cases: list[dict], k: int) -> dict:
    """Grade document-level retrieval; annotate each case with its rank."""
    import time

    latencies, hits, reciprocal = [], 0, 0.0
    for case in cases:
        t0 = time.perf_counter()
        ranking = retriever.retrieve_docs(case["query"], k_docs=k)
        latencies.append((time.perf_counter() - t0) * 1000)
        rank = ranking.index(case["expected"]) + 1 if case["expected"] in ranking else None
        case["rank"] = rank
        if rank is not None:
            hits += 1
            reciprocal += 1.0 / rank
    latencies.sort()
    return {
        "n": len(cases),
        "recall_at_k": hits / len(cases),
        "mrr": reciprocal / len(cases),
        "latency_ms": {
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
        },
    }


def print_scale_results(cases: list[dict], result: dict, k: int) -> None:
    by_cat: dict[str, list[dict]] = {}
    for case in cases:
        by_cat.setdefault(case["category"], []).append(case)
    print(f"Scale eval — {result['n']} CUAD-annotation queries, one correct "
          f"contract each (recall@{k}):")
    for category, cat_cases in sorted(by_cat.items()):
        found = sum(1 for c in cat_cases if c["rank"] is not None)
        print(f"  {category:22s} {found}/{len(cat_cases)}")
    print(f"  recall@{k}: {result['recall_at_k']:.2f}   "
          f"MRR: {result['mrr']:.2f}   "
          f"latency p50/p95: {result['latency_ms']['p50']:.0f}/"
          f"{result['latency_ms']['p95']:.0f} ms")
    misses = [c for c in cases if c["rank"] is None][:5]
    if misses:
        print("  sample misses:")
        for c in misses:
            print(f"    {c['query'][:66]!r} -> {c['expected'][:48]}")
    print()


def check_party_metadata(key: dict) -> bool:
    with config.CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    extracted: dict[str, set[str]] = {}
    for c in chunks:
        names = set(c["parties"]) | ({c["party"]} if c["party"] else set())
        extracted.setdefault(c["source"], set()).update(names)

    print("Party metadata vs answer key:")
    all_ok = True
    for source, meta in key.items():
        names = extracted.get(source, set())
        ok = party_matches(meta["party"], names)
        all_ok &= ok
        print(f"  [{'ok' if ok else 'MISS'}] {source:42s} "
              f"key={meta['party']!r}  extracted={sorted(names)}")
    print()
    return all_ok


def main(k: int = config.DEFAULT_TOP_K_DOCS, skip_scale: bool = False) -> int:
    key = json.loads(config.ANSWER_KEY_FILE.read_text(encoding="utf-8"))
    retriever = Retriever()

    parties_ok = check_party_metadata(key)

    results = []
    for case in build_cases(key):
        result = run_case(retriever, case, k)
        print_case(case, result, k)
        results.append(result["passed"])

    # Scale section: only meaningful when the CUAD corpus is in the index and
    # its annotations are on disk.
    scale_ok, scale_note = True, "skipped (no CUAD corpus indexed)"
    indexed_sources = {c["source"] for c in retriever.chunks}
    has_cuad = any(s.startswith("cuad/") for s in indexed_sources)
    if not skip_scale and has_cuad and config.CUAD_ANNOTATIONS_FILE.exists():
        annotations = json.loads(
            config.CUAD_ANNOTATIONS_FILE.read_text(encoding="utf-8")
        )
        cases = build_scale_cases(annotations, indexed_sources)
        scale = run_scale_eval(retriever, cases, k)
        print_scale_results(cases, scale, k)
        scale_ok = scale["recall_at_k"] >= config.SCALE_RECALL_TARGET
        scale_note = (
            f"recall@{k} {scale['recall_at_k']:.2f} "
            f"{'>=' if scale_ok else '< TARGET'} {config.SCALE_RECALL_TARGET}"
        )
    elif skip_scale:
        scale_note = "skipped (--skip-scale)"

    passed = sum(results)
    print(f"{passed}/{len(results)} needle queries passed"
          f" | party metadata {'ok' if parties_ok else 'HAS MISSES'}"
          f" | scale: {scale_note}")
    return 0 if passed == len(results) and parties_ok and scale_ok else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score retrieval vs answer_key.json")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K_DOCS,
                        help="document-level cutoff (default %(default)s)")
    parser.add_argument("--skip-scale", action="store_true",
                        help="run only the needle eval")
    args = parser.parse_args()
    sys.exit(main(args.k, skip_scale=args.skip_scale))
