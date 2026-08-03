"""Stage 6 — eval: score retrieval against answer_key.json.

Runs the suggested test queries from README.txt and reports which of them
returned the correct documents. Expected-document sets are DERIVED from
answer_key.json fields (type / party / has_relinquish_control_carveout / note)
rather than hardcoded filenames, so the harness follows the key if it changes.

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
            "query": "Does the NDA contain a change of control provision?",
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


def main(k: int = config.DEFAULT_TOP_K_DOCS) -> int:
    key = json.loads(config.ANSWER_KEY_FILE.read_text(encoding="utf-8"))
    retriever = Retriever()

    parties_ok = check_party_metadata(key)

    results = []
    for case in build_cases(key):
        result = run_case(retriever, case, k)
        print_case(case, result, k)
        results.append(result["passed"])

    passed = sum(results)
    print(f"{passed}/{len(results)} retrieval queries passed"
          f" | party metadata {'ok' if parties_ok else 'HAS MISSES'}")
    return 0 if passed == len(results) and parties_ok else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score retrieval vs answer_key.json")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K_DOCS,
                        help="document-level cutoff (default %(default)s)")
    sys.exit(main(parser.parse_args().k))
