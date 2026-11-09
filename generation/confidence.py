from enum import Enum
from generation.thresholds import Thresholds

class Band(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"

def score_to_band(score: float | None, thresholds: Thresholds) -> Band:
    """Map the best reranker score to a confidence band (Layer 1 of abstention)."""
    if score is None:  # retrieval returned nothing
        return Band.INSUFFICIENT
    if score >= thresholds.high:
        return Band.HIGH
    if score >= thresholds.medium:
        return Band.MEDIUM
    if score >= thresholds.insufficient:
        return Band.LOW
    return Band.INSUFFICIENT