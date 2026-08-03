"""Stage 3 — embed: encode chunks with sentence-transformers into local Chroma.

Everything here runs on-machine: the MiniLM model is downloaded once from the
Hugging Face hub and inference is local; Chroma persists to ./index_store with
telemetry disabled. The raw chunk text is stored as the Chroma document so
generation can quote it verbatim, while the *embedded* text is prefixed with a
`[source | party]` header — that header is what separates the near-identical
indemnification chunks in vector space, since their bodies differ by only a
name.
"""

import json

import chromadb
from chromadb.config import Settings

import config
from chunk import chunk_documents
from load import load_documents

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embedding_input(chunk: dict) -> str:
    party = chunk["party"] or "unknown"
    return f"[{chunk['source']} | party: {party}] {chunk['text']}"


def get_chroma_client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection():
    return get_chroma_client().get_collection(config.COLLECTION_NAME)


def build_index(chunks: list[dict]) -> int:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # chunks.jsonl is the manifest: retrieve.py rebuilds BM25 from it and the
    # eval reads party metadata from it, so it always mirrors the vector store.
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    client = get_chroma_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embed_texts([embedding_input(c) for c in chunks]),
        documents=[c["text"] for c in chunks],
        # Chroma metadata values must be scalars, so the party list is JSON.
        metadatas=[
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "party": c["party"],
                "parties": json.dumps(c["parties"]),
            }
            for c in chunks
        ],
    )
    return collection.count()


if __name__ == "__main__":
    all_chunks = chunk_documents(load_documents())
    count = build_index(all_chunks)
    print(f"Indexed {count} chunks into {config.CHROMA_DIR} "
          f"(collection '{config.COLLECTION_NAME}', model {config.EMBED_MODEL})")
