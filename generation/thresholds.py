from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    high: float
    medium: float
    insufficient: float

# Calibrated on the 20-question evaluation set (see evaluation/questions.json).
# Lowest answerable best-score: -3.22; highest unanswerable best-score: -1.62.
# These overlap, so scores alone cannot separate the two cases. That is why
# the LOW band hands the decision to the LLM instead of refusing outright.
THRESHOLDS: dict[str, Thresholds] = {
    "Xenova/ms-marco-MiniLM-L-6-v2": Thresholds(
        high=3.0,
        medium=0.0,
        insufficient=-4.0,
    ),
}


def get_thresholds(model_name: str) -> Thresholds:
    """Return the calibrated band cut-offs for a reranker model."""
    try:
        return THRESHOLDS[model_name]
    except KeyError:
        raise ValueError(
            f"No calibrated thresholds for reranker '{model_name}'. "
            "Each reranker scores on its own scale, so reusing another "
            "model's cut-offs would give wrong confidence badges. "
            "Calibrate it on the evaluation questions and add an entry "
            "to THRESHOLDS."
        ) from None