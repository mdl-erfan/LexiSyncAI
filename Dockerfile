FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/home/user/.ollama/models \
    HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && useradd --create-home --uid 1000 user

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p /app/data/raw/uploads /home/user/.ollama/models \
    && chown -R user:user /app /home/user

USER user

# Bundle the LLM into the image so first-use requests do not download it.
RUN ollama serve > /tmp/ollama-build.log 2>&1 & \
    OLLAMA_PID=$!; \
    for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done; \
    ollama list >/dev/null 2>&1; \
    ollama pull llama3.2; \
    kill "$OLLAMA_PID"

EXPOSE 7860
CMD ["./start.sh"]
