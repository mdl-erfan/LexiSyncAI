import pytest

from retrieval.tokenizer import normalize_numbers, tokenize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10 000 000 EUR", "10000000 EUR"),
        ("EUR 2,000,000", "EUR 2000000"),
        ("1.500.000", "1500000"),
        ("25 May 2018", "25 May 2018"),     # a date is not a grouped number
        ("clause 9.3", "clause 9.3"),       # a clause number is not a grouped number
    ],
)
def test_normalize_numbers(text, expected):
    assert normalize_numbers(text) == expected


def test_space_and_comma_amounts_give_the_same_token():
    assert "10000000" in tokenize("fines up to 10 000 000 EUR")
    assert "10000000" in tokenize("fines up to 10,000,000 EUR")


@pytest.mark.parametrize("ref", ["83(4)", "6(1)(a)", "9(2)(j)"])
def test_legal_references_stay_whole_and_add_their_base_number(ref):
    tokens = tokenize(f"Article {ref}")
    assert ref in tokens
    assert ref.split("(")[0] in tokens


@pytest.mark.parametrize("clause", ["9.3", "7.2.1"])
def test_clause_numbers_stay_whole(clause):
    assert tokenize(f"Clause {clause} applies") == ["clause", clause, "applies"]


def test_redactions_become_one_token():
    assert tokenize("The cap is [***] and [*****].") == ["cap", "[redacted]", "[redacted]"]


def test_lowercase_and_stopwords_removed():
    assert tokenize("The Supplier IS liable for the Damages") == ["supplier", "liable", "damages"]


def test_legal_words_are_not_stopwords():
    assert tokenize("shall not may") == ["shall", "not", "may"]


def test_empty_text_gives_no_tokens():
    assert tokenize("") == []
    assert tokenize("  ... ,, ") == []