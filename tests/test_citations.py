import pytest

from generation.citations import extract_citations


@pytest.mark.parametrize(
    "answer, n_passages, expected",
    [
        ("Fine is 10M [1].", 3, ([1], [])),
        ("A [1] and B [2][3].", 3, ([1, 2, 3], [])),
        ("Fine [1] and again [1].", 3, ([1], [])),      # duplicates collapse
        ("Wrong [7].", 3, ([], [7])),                   # passage 7 was never in the prompt
        ("Zero [0].", 3, ([], [0])),                    # numbering starts at 1
        ("No citations here.", 3, ([], [])),
    ],
)
def test_extract_citations(answer, n_passages, expected):
    assert extract_citations(answer, n_passages) == expected