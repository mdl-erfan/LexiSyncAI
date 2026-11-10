import json
import time
from dataclasses import dataclass, field

from generation.citations import extract_citations
from generation.confidence import Band, score_to_band
from generation.llm import generate
from generation.prompt import build_messages, format_citation
from generation.thresholds import get_thresholds


@dataclass
class Citation:
    number: int        # the [n] used in the answer text
    label: str         # "gdpr_2016_679.pdf, Article 83(4), p. 82"
    text: str          # the cited passage, for the side-by-side view
    page_start: int
    page_end: int
    doc_name: str = "" # which PDF the passage comes from, to render its page



@dataclass
class Answer:
    question: str
    text: str
    badge: Band
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    documents_searched: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    top_score: float | None = None
    latency_s: float = 0.0
    reason: str = ""       # why it abstained, for debugging and the evaluation tables
    raw_answer: str = ""   # the LLM's text before the grounding check, for debugging

def _abstain(question, docs, reason, score, start , raw_answer="") -> Answer:
    text = ("The documents do not contain enough evidence to answer this question. "
            f"Documents searched: {', '.join(docs)}.")
    return Answer(question=question, text=text, badge=Band.INSUFFICIENT, abstained=True,
                  documents_searched=docs, top_score=score, reason=reason,
                  raw_answer=raw_answer,latency_s=time.time() - start)


def answer_question(question, index, documents_searched, k=5, llm=generate) -> Answer:
    start = time.time()
    results = index.search(question, k=k)
    score = results[0][1] if results else None
    band = score_to_band(score, get_thresholds(index.reranker.model_name))
    # show passages in document order, which is easier for the model to read
    results = sorted(results, key=lambda r: (r[0].doc_name, r[0].page_start, r[0].section))
    # Layer 1: the reranker score alone is enough to refuse.
    if band is Band.INSUFFICIENT:
        return _abstain(question, documents_searched, "reranker score below threshold", score, start)

    # Layer 2: the LLM must read the evidence and agree it answers the question.
    messages = build_messages(question, results)
    try:
        output = llm(messages)
    except ValueError:
        return _abstain(question, documents_searched, "LLM output was not valid JSON", score, start)
    if not output.get("evidence_sufficient"):
        return _abstain(question, documents_searched, "LLM judged the evidence insufficient", score, start)

    # Grounding check: an answer with no verifiable citation gets one retry.
    valid, invalid = extract_citations(output["answer"], len(results))
    if not valid:
        retry = messages + [
            {"role": "assistant", "content": json.dumps(output)},
            {"role": "user", "content": "Your answer has no [n] citations. Rewrite it so every "
                                        "statement ends with the number of the passage it comes "
                                        "from, like [1]. Same JSON format."},
        ]
        try:
            output = llm(retry)
        except ValueError:
            return _abstain(question, documents_searched, "LLM output was not valid JSON", score, start)
        valid, invalid = extract_citations(output["answer"], len(results))
    if not valid:
        return _abstain(question, documents_searched, "answer had no valid citation",
                        score, start, raw_answer=output["answer"])

    citations = []
    for n in valid:
        chunk = results[n - 1][0]
        citations.append(Citation(n, format_citation(chunk), chunk.text,
                                  chunk.page_start, chunk.page_end, chunk.doc_name))
    return Answer(question=question, text=output["answer"], badge=band, abstained=False,
                  citations=citations, documents_searched=documents_searched,
                  invalid_citations=invalid, top_score=score,
                  latency_s=time.time() - start)