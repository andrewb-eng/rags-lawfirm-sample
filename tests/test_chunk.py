"""Unit tests for section-aware chunking and party extraction."""

import config
from chunk import chunk_text, extract_parties

# CUAD-style text: PDF extraction collapses newlines into runs of spaces, so
# section markers appear mid-line after 3+ spaces.
CUAD_STYLE = (
    "agree as follows:     1. DEFINITIONS. Capitalized terms used herein "
    'have the following meaning:     1.1. "Region" means the sales region. '
    "    2. DISTRIBUTORSHIP     2.1. Appointment. Accuray hereby appoints "
    "Distributor as a non-exclusive, worldwide distributor of Products "
    "pursuant to Section 2.3 of the Strategic Alliance Agreement."
)


def _sections(chunks):
    return [c["section"] for c in chunks]


def test_page_number_lines_dropped():
    text = "First paragraph of the agreement.\n\n     7\n\nSecond paragraph."
    chunks = chunk_text(text)
    joined = " ".join(c["text"] for c in chunks)
    assert "7" not in joined.split()
    assert "First paragraph" in joined and "Second paragraph" in joined


def test_caps_heading_starts_new_chunk_and_labels_section():
    text = (
        "INTRO PARAGRAPH TEXT HERE\n\nRECITALS\n\nThe parties wish to "
        "cooperate.\n\nNOW, THEREFORE, the parties agree."
    )
    chunks = chunk_text(text)
    labels = _sections(chunks)
    assert any(label == "RECITALS" for label in labels)


def test_numbered_heading_at_line_start_becomes_section():
    text = (
        "Preamble text.\n\n1. Continuation of Indemnification. The Company "
        "shall continue to indemnify the Indemnitee.\n\n2. Change in "
        "Control. The parties acknowledge a change in control."
    )
    chunks = chunk_text(text)
    labels = " | ".join(_sections(chunks))
    assert "1. Continuation of Indemnification" in labels
    assert "2. Change in Control" in labels


def test_inline_section_markers_split_cuad_blobs():
    chunks = chunk_text(CUAD_STYLE)
    labels = _sections(chunks)
    assert any(label.startswith("2. DISTRIBUTORSHIP") for label in labels)
    # "pursuant to Section 2.3 of" is mid-sentence (single space before it)
    # and must NOT create a section boundary.
    assert not any("Section 2.3" in label for label in labels)


def test_never_packs_across_section_boundary():
    text = "1. First. Tiny section one.\n\n2. Second. Tiny section two."
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert "section one" in chunks[0]["text"]
    assert "section two" in chunks[1]["text"]


def test_hard_split_respects_max_and_prefers_sentence_boundaries():
    blob = ("This is a sentence about obligations of the parties. " * 200).strip()
    chunks = chunk_text(blob)
    assert all(len(c["text"]) <= config.CHUNK_MAX_CHARS for c in chunks)
    # most split points should land on sentence ends, not mid-word
    ends_on_sentence = sum(
        1 for c in chunks[:-1] if c["text"].rstrip().endswith(".")
    )
    assert ends_on_sentence >= (len(chunks) - 1) * 0.8


def test_extract_parties_defined_terms_needle_style():
    text = (
        'entered into by and between Coastal Devices Inc. (the "Company"), '
        'a Delaware corporation, and David Okafor (the "Indemnitee").'
    )
    primary, parties = extract_parties(text)
    assert primary == "David Okafor"
    assert "Coastal Devices Inc." in parties


def test_extract_parties_cuad_recital_with_address_blob():
    text = (
        "THIS DISTRIBUTOR AGREEMENT is entered into by and between ACCURAY "
        "INCORPORATED, a Delaware corporation with its executive offices "
        "located at 1310 Chesapeake Terrace, Sunnyvale, California 94089, "
        "USA, and SIEMENS AKTIENGESELLSCHAFT, a corporation formed under "
        "the laws of the Federal Republic of Germany."
    )
    primary, parties = extract_parties(text)
    assert "ACCURAY INCORPORATED" in parties
    # the address blob must not leak into the party name
    assert all("Chesapeake" not in p for p in parties)


def test_extract_parties_quoted_alias_but_not_boilerplate_terms():
    text = (
        'This Agreement ("Agreement") is made by Electric City Corp., a '
        'Delaware corporation ("Company"), and Accuray Incorporated, a '
        'Delaware corporation ("Accuray"), as of June 8, 2010 ("Effective '
        'Date").'
    )
    primary, parties = extract_parties(text)
    assert "Accuray" in parties
    assert "Agreement" not in parties
    assert "Effective Date" not in parties
    assert "Company" not in parties  # generic role word, not a name


def test_recital_fallback_keeps_last_party_primary():
    text = (
        "This Mutual Nondisclosure Agreement is entered into by and "
        "between Northgate Instruments Inc. and Vantage Analytics, a "
        "Washington corporation."
    )
    primary, parties = extract_parties(text)
    assert primary == "Vantage Analytics"
    assert "Northgate Instruments Inc." in parties


def test_defined_party_requires_a_party_role():
    text = (
        'for sale of Products in Japan (the "Territory") entered by '
        'Electric City Corp. (the "Company") on the date hereof.'
    )
    primary, parties = extract_parties(text)
    assert "Japan" not in parties
    assert primary == "Electric City Corp."


def test_boilerplate_name_never_becomes_party():
    text = 'pursuant to this Distribution Agreement (the "Company") dated.'
    primary, parties = extract_parties(text)
    assert "Distribution Agreement" not in parties


def test_all_caps_name_containing_and_survives_recital():
    text = (
        "COLLABORATION AGREEMENT by and between INSTITUTE OF GRASSLAND AND "
        "ENVIRONMENTAL RESEARCH and Ceres Inc., a Delaware corporation."
    )
    primary, parties = extract_parties(text)
    assert "INSTITUTE OF GRASSLAND AND ENVIRONMENTAL RESEARCH" in parties
    # trailing corporate-suffix punctuation may be trimmed; substring match
    # is what party resolution uses anyway
    assert any(p.startswith("Ceres Inc") for p in parties)
