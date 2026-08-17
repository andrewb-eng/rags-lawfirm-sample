"""Stage 4 — retrieve: hybrid search over the local index.

Two retrievers run per query and their scores are fused:

  * dense   — cosine similarity against the Chroma vectors, which catches
              conceptual matches ("change-in-control carve-out").
  * lexical — BM25 over chunk tokens, which keeps exact legal phrases like
              "relinquish control" from being lost to semantic drift.

Fusion is relative-score fusion: each retriever's scores are min-max
normalized over the query's candidate pool, then combined as
`HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * bm25`. The obvious alternative,
reciprocal-rank fusion, was measured to fail on this corpus: when five
near-identical amendments share decisive lexical evidence, their BM25 scores
sit 50%+ above every other chunk, but rank-based fusion compresses that
cliff into adjacent ranks — letting chunks that are merely mediocre in both
retrievers outrank documents with one emphatic signal. Score-relative fusion
preserves the margin.

Party filtering resolves a party string against the parties parsed at chunk
time, then constrains BOTH retrievers to that party's documents.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

import config
import embed

_SUFFIXES = ("ing", "ed", "s")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with crude suffix stripping.

    The stripping is not real stemming — it only needs to be *consistent*
    between queries and documents so that "relinquished" matches "relinquish"
    and "shares" matches "share". Tokens are interned: BM25 holds millions of
    them across a 50k-chunk corpus, and interning collapses the duplicates.
    """
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) > 3:
                token = token[: -len(suffix)]
                break
        tokens.append(sys.intern(token))
    return tokens


class Retriever:
    def __init__(self):
        manifest = embed.read_manifest()
        if manifest is None or not config.CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"no complete index in {config.INDEX_DIR} — run "
                "`python main.py index` first"
            )
        if manifest["embed_model"] != config.EMBED_MODEL:
            raise RuntimeError(
                f"index was built with {manifest['embed_model']!r} but config "
                f"wants {config.EMBED_MODEL!r} — mixing embedding spaces would "
                "silently break retrieval; run `python main.py index`"
            )
        from load import corpus_fingerprint

        if manifest["corpus_fingerprint"] != corpus_fingerprint():
            print(
                "warning: corpus has changed since the index was built — "
                "results reflect the OLD corpus; run `python main.py index`",
                file=sys.stderr,
            )

        with config.CHUNKS_FILE.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        self.bm25 = self._load_or_build_bm25(manifest)
        self.collection = embed.get_collection()

        self.parties_by_source: dict[str, set[str]] = {}
        for c in self.chunks:
            names = set(c["parties"]) | ({c["party"]} if c["party"] else set())
            self.parties_by_source.setdefault(c["source"], set()).update(names)

    def _load_or_build_bm25(self, manifest: dict) -> BM25Okapi:
        """Load the pickled BM25 index, rebuilding (and re-caching) on miss.

        BM25 tokens include the party and the filename stem so lexical
        queries naming a party ("David Okafor") hit every chunk of that
        party's document, not just the chunk containing the signature line.
        """
        cache_key = (manifest["corpus_fingerprint"], manifest["n_chunks"])
        if config.BM25_CACHE_FILE.exists():
            try:
                with config.BM25_CACHE_FILE.open("rb") as f:
                    payload = pickle.load(f)
                if payload.get("key") == cache_key:
                    return payload["bm25"]
            except Exception:
                pass  # corrupt/old cache: rebuild below

        bm25 = BM25Okapi(
            [
                tokenize(c["text"])
                + tokenize(c["party"])
                + tokenize(Path(c["source"]).stem)
                for c in self.chunks
            ]
        )
        with config.BM25_CACHE_FILE.open("wb") as f:
            pickle.dump({"key": cache_key, "bm25": bm25}, f, protocol=5)
        return bm25

    def resolve_party(self, party: str) -> list[str]:
        """Map a party string to source files by token containment.

        The user's tokens must all appear in a document's party name (partial
        names like "Vantage" work), or a multi-token party name may be a
        subset of the user's longer string. Plain substring matching is too
        loose at scale — a contract party named just "David" must not claim
        the "David Okafor" filter.
        """
        needle = set(tokenize(party))
        if not needle:
            raise ValueError(f"party {party!r} has no searchable tokens")

        def matches(name: str) -> bool:
            name_tokens = set(tokenize(name))
            return needle <= name_tokens or (
                len(name_tokens) >= 2 and name_tokens <= needle
            )

        sources = [
            source
            for source, names in self.parties_by_source.items()
            if any(matches(name) for name in names)
        ]
        if not sources:
            known = sorted({n for names in self.parties_by_source.values() for n in names})
            raise ValueError(f"no documents match party {party!r}; known parties: {known}")
        return sorted(sources)

    def _dense_scores(
        self, query: str, allowed: list[str] | None, depth: int
    ) -> dict[str, float]:
        where = {"source": {"$in": allowed}} if allowed else None
        result = self.collection.query(
            query_embeddings=embed.embed_texts([query]),
            n_results=min(depth, len(self.chunks)),
            where=where,
        )
        return {
            cid: 1.0 - distance  # cosine distance -> similarity
            for cid, distance in zip(result["ids"][0], result["distances"][0])
        }

    def _bm25_scores(
        self, query: str, allowed: list[str] | None, depth: int
    ) -> dict[str, float]:
        scores = self.bm25.get_scores(tokenize(query))
        candidates = [
            (score, chunk["id"])
            for score, chunk in zip(scores, self.chunks)
            if score > 0 and (allowed is None or chunk["source"] in allowed)
        ]
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        return {cid: score for score, cid in candidates[:depth]}

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalize over the query's candidate pool.

        Chunks absent from a retriever's pool get 0 — being unranked is
        indistinguishable from being at the pool floor."""
        if not scores:
            return {}
        low, high = min(scores.values()), max(scores.values())
        if high - low < 1e-9:
            return {cid: 1.0 for cid in scores}
        return {cid: (v - low) / (high - low) for cid, v in scores.items()}

    def retrieve(
        self,
        query: str,
        k: int = config.DEFAULT_TOP_K_CHUNKS,
        party: str | None = None,
        depth: int | None = None,
    ) -> list[dict]:
        """Top-k chunks by relative-score fusion, optionally party-filtered."""
        depth = depth or config.CANDIDATE_DEPTH
        allowed = self.resolve_party(party) if party else None
        dense = self._dense_scores(query, allowed, depth)
        lexical = self._bm25_scores(query, allowed, depth)
        dense_n, lexical_n = self._normalize(dense), self._normalize(lexical)

        alpha = config.HYBRID_ALPHA
        fused = sorted(
            (
                (
                    alpha * dense_n.get(cid, 0.0)
                    + (1 - alpha) * lexical_n.get(cid, 0.0),
                    cid,
                )
                for cid in dense.keys() | lexical.keys()
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )

        dense_rank = {
            cid: i + 1
            for i, cid in enumerate(sorted(dense, key=lambda c: -dense[c]))
        }
        bm25_rank = {
            cid: i + 1
            for i, cid in enumerate(sorted(lexical, key=lambda c: -lexical[c]))
        }
        by_id = {c["id"]: c for c in self.chunks}
        return [
            {
                **by_id[cid],
                "score": round(score, 5),
                "dense_rank": dense_rank.get(cid),
                "bm25_rank": bm25_rank.get(cid),
            }
            for score, cid in fused[:k]
        ]

    def retrieve_docs(
        self,
        query: str,
        k_docs: int = config.DEFAULT_TOP_K_DOCS,
        party: str | None = None,
    ) -> list[str]:
        """Document-level ranking: each document scores as its best fused
        chunk (first appearance), over a candidate pool much deeper than the
        chunk-level default.

        The deep pool matters at corpus scale: near-identical documents
        interleave many chunks near the top of the fused list, so a shallow
        pool runs out of unique-document slots before every relevant document
        has surfaced. Best-chunk scoring (rather than summing a document's
        chunks) avoids the length bias where a 300-chunk contract outscores a
        short document by accumulating dozens of weak hits."""
        chunks = self.retrieve(
            query,
            k=config.DOC_CANDIDATE_DEPTH,
            party=party,
            depth=config.DOC_CANDIDATE_DEPTH,
        )
        return list(dict.fromkeys(c["source"] for c in chunks))[:k_docs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the hybrid retriever")
    parser.add_argument("query")
    parser.add_argument("--party", help="restrict to documents naming this party")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K_CHUNKS)
    args = parser.parse_args()

    for i, c in enumerate(Retriever().retrieve(args.query, k=args.k, party=args.party), 1):
        print(
            f"{i}. {c['id']}  (party: {c['party'] or '-'}, score={c['score']}, "
            f"dense#{c['dense_rank']}, bm25#{c['bm25_rank']})"
        )
        print(f"   {' '.join(c['text'].split())[:150]}...")
