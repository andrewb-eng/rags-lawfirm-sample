"""Stage 5 — generate: answer from retrieved chunks via the Claude API.

This is the only stage that leaves the machine. Each retrieved chunk is passed
as a document content block with API-native citations enabled, so every claim
in the answer carries a machine-verified quote from a chunk (`cited_text` is
guaranteed by the API to be a span of the source document — the model cannot
fabricate a quote). The system prompt additionally forbids answering from
outside the provided chunks and requires naming the source file per claim.
"""

import argparse
import sys

import anthropic

import config

class MissingCredentials(RuntimeError):
    """Raised before any API call when no Anthropic credentials resolve."""


NO_CREDENTIALS_MSG = (
    "No Anthropic credentials found. Set ANTHROPIC_API_KEY (or "
    "ANTHROPIC_AUTH_TOKEN, or log in with `ant auth login`), then retry. "
    "Retrieval works without credentials: `python main.py ask ... --retrieve-only`."
)

SYSTEM_PROMPT = """\
You are a legal research assistant. You answer questions using ONLY the
documents provided in the conversation. Rules:

1. Every factual claim must cite the source filename and quote the exact
   supporting passage from that document. Prefer verbatim quotes over
   paraphrase.
2. The corpus contains near-identical documents that differ only by party.
   Never attribute one party's provision to another; check the party named in
   each document before relying on it.
3. Read carefully for negations: a document may mention a phrase ("relinquish
   control", "change in control") while negating or disclaiming it, or use it
   in a different sense than the question intends.
4. If the provided documents do not contain the answer, say so plainly
   ("The retrieved documents do not answer this") rather than guessing or
   using outside knowledge.
"""


def _document_block(chunk: dict) -> dict:
    return {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": chunk["text"],
        },
        "title": chunk["source"],
        "context": (
            f"Source file: {chunk['source']} (chunk {chunk['chunk_index']}). "
            f"Party: {chunk['party'] or 'unknown'}. "
            f"All parties: {', '.join(chunk['parties']) or 'unknown'}."
        ),
        "citations": {"enabled": True},
    }


def _render(response) -> str:
    """Flatten the response into text with numbered citation footnotes."""
    body, footnotes = [], []
    for block in response.content:
        if block.type != "text":
            continue
        body.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            footnotes.append((citation.document_title, citation.cited_text))
            body.append(f" [{len(footnotes)}]")

    out = "".join(body)
    if footnotes:
        out += "\n\nSources:\n"
        for i, (title, quote) in enumerate(footnotes, 1):
            out += f"  [{i}] {title} — \"{' '.join(quote.split())}\"\n"
    return out


def answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No chunks were retrieved, so there is nothing to answer from."

    client = anthropic.Anthropic()
    # The SDK resolves credentials from env vars or an `ant auth login`
    # profile; fail with a clear message before building a request if none did.
    if not (client.api_key or client.auth_token or getattr(client, "credentials", None)):
        raise MissingCredentials(NO_CREDENTIALS_MSG)

    content = [_document_block(c) for c in chunks]
    content.append({"type": "text", "text": question})

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this request (stop_reason: refusal)."
    return _render(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and answer a question")
    parser.add_argument("question")
    parser.add_argument("--party", help="restrict retrieval to this party's documents")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K_CHUNKS)
    args = parser.parse_args()

    from retrieve import Retriever

    retrieved = Retriever().retrieve(args.question, k=args.k, party=args.party)
    print("Retrieved:", ", ".join(c["id"] for c in retrieved), "\n", file=sys.stderr)
    try:
        print(answer(args.question, retrieved))
    except MissingCredentials as err:
        sys.exit(str(err))
    except anthropic.AuthenticationError:
        sys.exit("Anthropic credentials were found but rejected (invalid/expired key).")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Claude API (network error).")
