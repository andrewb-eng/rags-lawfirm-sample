"""Unit tests for tokenization and RRF fusion math (no index required)."""

import config
from retrieve import tokenize


def test_tokenize_is_consistent_between_query_and_document_forms():
    # crude suffix stripping only needs to be consistent, not linguistic:
    # the query-side form and the document-side form must produce the same
    # token so BM25 can match them
    assert tokenize("relinquished") == tokenize("relinquish")
    assert tokenize("shares") == tokenize("share")
    assert tokenize("change of control") == ["change", "of", "control"]
    assert tokenize("controls changed") == ["control", "chang"]


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("Okafor's carve-out (Section 2.1)") == [
        "okafor", "s", "carve", "out", "section", "2", "1",
    ]


def test_tokenize_short_words_not_overstripped():
    # words at or below the length guard keep their suffix
    assert tokenize("dies") == ["dies"]  # len-1 == 3, not > 3
    assert tokenize("used") == ["used"]


def test_relative_score_fusion_preserves_score_cliffs():
    """A chunk with emphatic lexical evidence must outrank chunks that are
    merely mediocre in both retrievers — the failure mode of rank fusion."""
    from retrieve import Retriever

    # measured shape from the Q1 investigation: the twin's dense similarity
    # sits mid-pool (real pools have a floor well below it), while its BM25
    # score is 50%+ above everything else
    dense = {"twin": 0.50, "generic": 0.58, "filler": 0.60, "floor": 0.40}
    lexical = {"twin": 25.3, "generic": 16.3, "filler": 12.0, "floor": 10.0}
    dense_n = Retriever._normalize(dense)
    lexical_n = Retriever._normalize(lexical)
    a = config.HYBRID_ALPHA

    def fused(cid):
        return a * dense_n.get(cid, 0.0) + (1 - a) * lexical_n.get(cid, 0.0)

    assert fused("twin") > fused("generic") > fused("filler")


def test_normalize_handles_uniform_and_empty_pools():
    from retrieve import Retriever

    assert Retriever._normalize({}) == {}
    assert Retriever._normalize({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}
    normalized = Retriever._normalize({"a": 1.0, "b": 3.0})
    assert normalized == {"a": 0.0, "b": 1.0}
