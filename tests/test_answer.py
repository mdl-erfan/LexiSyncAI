from types import SimpleNamespace

from generation.answer import answer_question
from generation.confidence import Band
from ingestion.chunker import Chunk

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # must exist in THRESHOLDS
DOCS = ["contract.pdf"]


def make_chunk(section="Article 1", text="The fine is 10 million EUR.", page=5):
    return Chunk(doc_name="contract.pdf", section=section, heading=section,
                 lines=[text], line_pages=[page], chunk_id=f"contract.pdf-{section}")


class FakeIndex:
    """Stands in for RerankedIndex: returns prepared (chunk, score) pairs."""

    def __init__(self, results):
        self.results = results
        self.reranker = SimpleNamespace(model_name=MODEL)

    def search(self, query, k=5):
        return self.results[:k]


class FakeLLM:
    """Stands in for generate(): returns prepared replies in order and counts calls."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_very_low_score_refuses_without_calling_llm():
    index = FakeIndex([(make_chunk(), -10.0)])
    llm = FakeLLM()          # no replies prepared: any call would fail

    answer = answer_question("anything", index, DOCS, llm=llm)

    assert answer.abstained is True
    assert answer.badge is Band.INSUFFICIENT
    assert answer.reason == "reranker score below threshold"
    assert llm.calls == 0

def test_no_results_refuses_without_calling_llm():
    llm = FakeLLM()
    answer = answer_question("anything", FakeIndex([]), DOCS, llm=llm)
    assert answer.abstained is True
    assert answer.top_score is None
    assert llm.calls == 0


def test_llm_says_insufficient_refuses():
    llm = FakeLLM({"evidence_sufficient": False, "answer": "No."})
    answer = answer_question("anything", FakeIndex([(make_chunk(), 5.0)]), DOCS, llm=llm)
    assert answer.abstained is True
    assert answer.reason == "LLM judged the evidence insufficient"
    assert llm.calls == 1


def test_good_answer_is_returned_with_citation():
    llm = FakeLLM({"evidence_sufficient": True, "answer": "The fine is 10 million [1]."})
    answer = answer_question("anything", FakeIndex([(make_chunk(page=7), 5.0)]), DOCS, llm=llm)
    assert answer.abstained is False
    assert answer.badge is Band.HIGH
    assert len(answer.citations) == 1
    assert answer.citations[0].page_start == 7
    assert "Article 1" in answer.citations[0].label


def test_missing_citation_gets_one_retry():
    llm = FakeLLM({"evidence_sufficient": True, "answer": "The fine is 10 million."},
                  {"evidence_sufficient": True, "answer": "The fine is 10 million [1]."})
    answer = answer_question("anything", FakeIndex([(make_chunk(), 5.0)]), DOCS, llm=llm)
    assert answer.abstained is False
    assert llm.calls == 2


def test_still_no_citation_after_retry_refuses():
    no_cite = {"evidence_sufficient": True, "answer": "The fine is 10 million."}
    llm = FakeLLM(no_cite, no_cite)
    answer = answer_question("anything", FakeIndex([(make_chunk(), 5.0)]), DOCS, llm=llm)
    assert answer.abstained is True
    assert answer.reason == "answer had no valid citation"
    assert answer.raw_answer == "The fine is 10 million."
    assert llm.calls == 2


def test_invented_citation_is_recorded():
    llm = FakeLLM({"evidence_sufficient": True, "answer": "Fine [1], and also [7]."})
    answer = answer_question("anything", FakeIndex([(make_chunk(), 5.0)]), DOCS, llm=llm)
    assert answer.abstained is False
    assert answer.invalid_citations == [7]


def test_broken_json_refuses_without_crashing():
    llm = FakeLLM(ValueError("not JSON"))
    answer = answer_question("anything", FakeIndex([(make_chunk(), 5.0)]), DOCS, llm=llm)
    assert answer.abstained is True
    assert answer.reason == "LLM output was not valid JSON"