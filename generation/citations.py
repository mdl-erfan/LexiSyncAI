import re

_MARKER = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str, n_passages: int) -> tuple[list[int], list[int]]:
    """Find [n] markers in an answer.

    Returns (valid, invalid): passage numbers in order of first appearance, no
    duplicates. 'valid' are in 1..n_passages; 'invalid' point to passages that
    were never in the prompt (invented citations).
    """
    valid: list[int] = []
    invalid: list[int] = []
    for match in _MARKER.finditer(answer):
        number = int(match.group(1))
        bucket = valid if 1 <= number <= n_passages else invalid
        if number not in bucket:
            bucket.append(number)
    return valid, invalid