"""Step 5A: turn legal text into search tokens, keeping references and amounts intact."""
import re

# "10 000 000", "2,000,000", "1.500.000" -> one number
GROUPED_NUMBER = re.compile(r"\b\d{1,3}(?:[ ,.\u00a0]\d{3})+\b")

TOKEN = re.compile(
    r"\[\*+\]"                                  # redaction marker [***]
    r"|\d+(?:\.\d+)+"                           # clause numbers 9.3, 7.2.1
    r"|\d+[a-z]?(?:\((?:\d+[a-z]?|[a-z])\))+"   # references 83(4), 6(1)(a)
    r"|[a-z]+\d*|\d+"                           # words, plain numbers
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with",
}

def normalize_numbers(text: str) -> str:
    """Join digit groups: '10 000 000 EUR' -> '10000000 EUR'."""
    return GROUPED_NUMBER.sub(lambda m: re.sub(r"\D", "", m.group()), text)

def tokenize(text: str) -> list[str]:
    """Lowercase, normalise numbers, and split into search tokens."""
    text = normalize_numbers(text.lower())
    tokens = []
    for tok in TOKEN.findall(text):
        if tok in STOPWORDS:
            continue
        if tok.startswith("["):
            tokens.append("[redacted]")
            continue
        tokens.append(tok)
        if "(" in tok:
            tokens.append(tok.split("(")[0])   # 83(4) also indexed as 83
    return tokens