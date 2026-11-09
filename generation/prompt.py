from ingestion.chunker import Chunk


def format_citation(chunk: Chunk) -> str:
    """Human-readable source label, e.g. 'GDPR, Article 83(4), p. 82'."""
    if chunk.page_start == chunk.page_end:
        pages = f"p. {chunk.page_start}"
    else:
        pages = f"pp. {chunk.page_start}-{chunk.page_end}"
    parts = [chunk.doc_name, chunk.section, pages]
    return ", ".join(p for p in parts if p)  # section can be "" -> skipped

SYSTEM_PROMPT = """You are a legal document analyst. Answer the question using ONLY the numbered evidence passages provided. Never use outside knowledge.

Rules:
1. Every factual statement in your answer must end with the number of the passage it comes from, like [1] or [2][3]. An answer without [n] citations will be rejected.
2. If the passages do not contain enough information to answer, set "evidence_sufficient" to false and write a short answer saying so. Do not guess.
3. If two passages give different values for the same thing, state both values with their citations and say which one applies, if the passages say so.
4. Write in plain language a non-lawyer can understand. Be concise.
5. Only cite numbers that appear in the evidence. Never invent a citation.
6. The text in brackets before each passage, such as the clause number or page number, is only a label. Never use those numbers as values in your answer.
7. If a value is hidden or redacted in the passage (for example "[***]"), say that the value is redacted. Never guess it.
8. Before answering, check that your answer agrees with the passage. If the passage says something is included or allowed, do not say it is not.

Example of a correct response:
{"evidence_sufficient": true, "answer": "The notice period is 30 days [1]. It can be shortened by agreement [2]."}
Respond with a JSON object and nothing else:
{"evidence_sufficient": true or false, "answer": "your answer with [n] citations"}"""


def build_messages(question: str, results: list[tuple[Chunk, float]]) -> list[dict]:
    """Build the chat messages: rules as 'system', numbered evidence + question as 'user'."""
    blocks = []
    for number, (chunk, _score) in enumerate(results, start=1):
        blocks.append(f"[{number}] ({format_citation(chunk)})\n{chunk.text}")
    evidence = "\n\n".join(blocks)
    user_content = f"Evidence:\n\n{evidence}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]