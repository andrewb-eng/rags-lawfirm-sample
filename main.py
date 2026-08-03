"""CLI entry point wiring the pipeline stages together.

    python main.py index                # load -> chunk -> embed into Chroma
    python main.py ask "question"       # retrieve -> generate (Claude API)
    python main.py ask "q" --party X    # restrict retrieval to party X
    python main.py ask "q" --retrieve-only   # no API call, show chunks only
    python main.py eval                 # score retrieval vs answer_key.json
"""

import argparse
import sys

import config


def cmd_index(_args) -> int:
    from chunk import chunk_documents
    from embed import build_index
    from load import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    count = build_index(chunks)
    print(f"Indexed {count} chunks from {len(docs)} documents into {config.INDEX_DIR}")
    return 0


def cmd_ask(args) -> int:
    from retrieve import Retriever

    try:
        chunks = Retriever().retrieve(args.question, k=args.k, party=args.party)
    except ValueError as err:  # unknown party
        print(err, file=sys.stderr)
        return 2

    print("Retrieved chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c['id']}  (party: {c['party'] or '-'}, rrf={c['score']})")
        if args.show_chunks:
            print("     " + " ".join(c["text"].split())[:200])
    print()

    if args.retrieve_only:
        return 0

    import anthropic

    from generate import MissingCredentials, answer

    try:
        print(answer(args.question, chunks))
    except MissingCredentials as err:
        print(err, file=sys.stderr)
        return 2
    except anthropic.AuthenticationError:
        print("Anthropic credentials were found but rejected (invalid/expired key).",
              file=sys.stderr)
        return 2
    except anthropic.APIConnectionError:
        print("Could not reach the Claude API (network error).", file=sys.stderr)
        return 2
    return 0


def cmd_eval(args) -> int:
    from eval import main as eval_main

    return eval_main(k=args.k)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local retrieve-and-cite legal RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="build the local vector + BM25 index")

    ask = sub.add_parser("ask", help="answer a question from the corpus")
    ask.add_argument("question")
    ask.add_argument("--party", help="restrict retrieval to this party's documents")
    ask.add_argument("-k", type=int, default=config.DEFAULT_TOP_K_CHUNKS,
                     help="chunks to retrieve (default %(default)s)")
    ask.add_argument("--retrieve-only", action="store_true",
                     help="skip the Claude API call; just show retrieval")
    ask.add_argument("--show-chunks", action="store_true",
                     help="print a preview of each retrieved chunk")

    ev = sub.add_parser("eval", help="score retrieval against answer_key.json")
    ev.add_argument("-k", type=int, default=config.DEFAULT_TOP_K_DOCS,
                    help="document-level cutoff (default %(default)s)")

    args = parser.parse_args()
    return {"index": cmd_index, "ask": cmd_ask, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
