.PHONY: help setup test eval score app

help:
	@echo "LexiSyncAI - run these inside the activated conda env (conda activate lexiSyncAI)"
	@echo ""
	@echo "  make setup   install pinned Python packages and download Llama 3.2 (Ollama)"
	@echo "  make test    run the automated tests"
	@echo "  make eval    run retrieval and answer evaluation (several minutes)"
	@echo "  make score   rebuild the results table from the labelled final run"
	@echo "  make app     start the Streamlit UI"

setup:
	python -m pip install -r requirements.txt
	ollama pull llama3.2

test:
	python -m pytest

eval:
	python -m evaluation.evaluate_retrieval
	python -m evaluation.evaluate_answers

score:
	python -m evaluation.score_answers

app:
	python -m streamlit run app/main.py