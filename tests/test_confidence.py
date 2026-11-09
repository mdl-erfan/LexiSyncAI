import pytest

from generation.confidence import Band, score_to_band
from generation.thresholds import get_thresholds

MINILM = get_thresholds("Xenova/ms-marco-MiniLM-L-6-v2")


@pytest.mark.parametrize(
    "score, expected",
    [
        (5.0, Band.HIGH),
        (3.0, Band.HIGH),             # boundary: exactly at "high"
        (2.99, Band.MEDIUM),
        (0.0, Band.MEDIUM),           # boundary: exactly at "medium"
        (-0.01, Band.LOW),
        (-1.62, Band.LOW),            # highest unanswerable score seen
        (-3.22, Band.LOW),            # lowest answerable score seen
        (-4.0, Band.LOW),             # boundary: exactly at "insufficient"
        (-4.01, Band.INSUFFICIENT),
        (None, Band.INSUFFICIENT),    # retrieval returned nothing
    ],
)
def test_score_to_band(score, expected):
    assert score_to_band(score, MINILM) == expected