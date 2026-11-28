from __future__ import annotations

import pytest

from lankatax_ai.backend.rag.query_preprocessor import QueryPreprocessor

pytestmark = pytest.mark.unit


def test_ordinary_words_are_not_treated_as_acronyms():
    result = QueryPreprocessor().process("What is the standard VAT rate?")

    assert "WHT" not in result.expanded
    assert "Withholding" not in result.expanded
    assert result.expanded == "What is the standard VAT Value Added Tax rate?"


def test_uppercase_acronyms_are_expanded():
    assert QueryPreprocessor().process("WHT rate").expanded == "WHT Withholding Tax rate"


def test_lowercase_words_are_never_expanded():
    # "it" must not become Income Tax, and lower-case "vat" stays as typed.
    assert QueryPreprocessor().process("vat registration").expanded == "vat registration"
    assert QueryPreprocessor().process("who pays it?").expanded == "who pays it?"


def test_inflections_are_not_rewritten():
    result = QueryPreprocessor().process("How is PAYE withheld?")

    assert "withheld" in result.expanded
    assert "withholding" not in result.expanded
    assert result.expanded.endswith("?")


def test_punctuation_is_preserved_around_expansions():
    result = QueryPreprocessor().process("What is VAT, and who pays it?")

    assert result.expanded == "What is VAT Value Added Tax, and who pays it?"


def test_close_typos_are_corrected():
    assert QueryPreprocessor().process("witholding tax").expanded == "withholding tax"


def test_unknown_words_keep_their_case():
    assert QueryPreprocessor().process("Income Tax overview").expanded == ("Income Tax overview")
