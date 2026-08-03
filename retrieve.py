"""Stage 4 — retrieve: hybrid search over the local index.

Two retrievers run per query and their rankings are fused:

  * dense   — cosine similarity against the Chroma vectors, which catches
              conceptual matches ("change-in-control carve-out").
  * lexical — BM25 over chunk tokens, which keeps exact legal phrases like
              "relinquish control" from being lost to semantic drift.

Fusion is reciprocal-rank fusion (RRF): score = sum(1 / (RRF_K + rank)).
Rank-based fusion sidesteps the fact that cosine similarities and BM25 scores
live on incomparable scales.

Party filtering resolves a party string against the parties parsed at chunk
time, then constrains BOTH retrievers to that party's documents.
"""

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

import config
import embed

_SUFFIXES = ("ing", "ed", "s")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with crude suffix stripping.

    The stripping is not real stemming — it only needs to be *consistent*
    between queries and documents so that "relinquished" matches "relinquish"
    and "shares" matches "share".
    """
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) > 3:
                token = token[: -len(suffix)]
                break
        tokens.append(token)
    return tokens


class Retriever:
    def __init__(self):
        if not config.CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"{config.CHUNKS_FILE} not found — run `python main.py index` first"
            )
        with config.CHUNKS_FILE.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        # BM25 tokens include the party and the filename stem so lexical
        # queries naming a party ("David Okafor") hit every chunk of that
        # party's document, not just the chunk containing the signature line.
        self.bm25 = BM25Okapi(
            [
                tokenize(c["text"])
                + tokenize(c["party"])
                + tokenize(Path(c["source"]).stem)
                for c in self.chunks
            ]
        )
        self.collection = embed.get_collection()

        self.parties_by_source: dict[str, set[str]] = {}
        for c in self.chunks:
            names = set(c["parties"]) | ({c["party"]} if c["party"] else set())
            self.parties_by_source.setdefault(c["source"], set()).update(names)

    def resolve_party(self, party: str) -> list[str]:
        """Map a party string (case-insensitive substring) to source files."""
        needle = party.lower().strip()
        sources = [
            source
            for source, names in self.parties_by_source.items()
            if any(needle in name.lower() or name.lower() in needle for name in names)
        ]
        if not sources:
            known = sorted({n for names in self.parties_by_source.values() for n in names})
            raise ValueError(f"no documents match party {party!r}; known parties: {known}")
        return sorted(sources)

    def _dense_ranks(self, query: str, allowed: list[str] | None) -> dict[str, int]:
        where = {"source": {"$in": allowed}} if allowed else None
        result = self.collection.query(
            query_embeddings=embed.embed_texts([query]),
            n_results=min(config.CANDIDATE_DEPTH, len(self.chunks)),
            where=where,
        )
        return {cid: rank + 1 for rank, cid in enumerate(result["ids"][0])}

    def _bm25_ranks(self, query: str, allowed: list[str] | None) -> dict[str, int]:
        scores = self.bm25.get_scores(tokenize(query))
        candidates = [
            (score, chunk["id"])
            for score, chunk in zip(scores, self.chunks)
            if score > 0 and (allowed is None or chunk["source"] in allowed)
        ]
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        return {
            cid: rank + 1
            for rank, (_, cid) in enumerate(candidates[: config.CANDIDATE_DEPTH])
        }

    def retrieve(
        self, query: str, k: int = config.DEFAULT_TOP_K_CHUNKS, party: str | None = None
    ) -> list[dict]:
        """Return the top-k chunks by RRF-fused rank, optionally party-filtered."""
        allowed = self.resolve_party(party) if party else None
        dense = self._dense_ranks(query, allowed)
        lexical = self._bm25_ranks(query, allowed)

        fused = []
        by_id = {c["id"]: c for c in self.chunks}
        for cid in dense.keys() | lexical.keys():
            score = sum(
                1.0 / (config.RRF_K + rank)
                for rank in (dense.get(cid), lexical.get(cid))
                if rank is not None
            )
            fused.append((score, cid))
        fused.sort(key=lambda pair: (-pair[0], pair[1]))

        return [
            {
                **by_id[cid],
                "score": round(score, 5),
                "dense_rank": dense.get(cid),
                "bm25_rank": lexical.get(cid),
            }
            for score, cid in fused[:k]
        ]

    def retrieve_docs(
        self,
        query: str,
        k_docs: int = config.DEFAULT_TOP_K_DOCS,
        party: str | None = None,
    ) -> list[str]:
        """Document-level ranking: collapse the fused chunk ranking to unique
        source files in first-appearance order (used by the eval)."""
        chunks = self.retrieve(query, k=config.CANDIDATE_DEPTH, party=party)
        return list(dict.fromkeys(c["source"] for c in chunks))[:k_docs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the hybrid retriever")
    parser.add_argument("query")
    parser.add_argument("--party", help="restrict to documents naming this party")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K_CHUNKS)
    args = parser.parse_args()

    for i, c in enumerate(Retriever().retrieve(args.query, k=args.k, party=args.party), 1):
        print(
            f"{i}. {c['id']}  (party: {c['party'] or '-'}, rrf={c['score']}, "
            f"dense#{c['dense_rank']}, bm25#{c['bm25_rank']})"
        )
        print(f"   {' '.join(c['text'].split())[:150]}...")
