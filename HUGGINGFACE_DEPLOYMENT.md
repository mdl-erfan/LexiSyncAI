# Hugging Face Spaces deployment

This repository includes a Docker deployment that runs the complete LexiSyncAI
pipeline on one Hugging Face Space:

- Streamlit serves the UI on port 7860.
- Ollama runs privately inside the same container on `127.0.0.1:11434`.
- `llama3.2` is pulled during the Docker image build, so it is available when
the Space starts.
- FastEmbed downloads the embedding and reranking models on the first app start.

## Prerequisite

Docker Spaces require a Hugging Face **PRO** subscription for a personal
account (or Team/Enterprise for an organization), even when using the free
CPU Basic hardware. CPU Basic provides 2 vCPU, 16 GB RAM and 50 GB ephemeral
disk. It is sufficient to start this model, although answers will be slow.
CPU Upgrade (8 vCPU, 32 GB RAM, $0.03/hour) is recommended for a live demo.

## Create and publish the Space

1. In Hugging Face, choose **New Space**.
2. Set an owner, name it `LexiSyncAI`, choose **Docker**, and choose a paid CPU
   upgrade with at least 16 GB RAM.
3. Make it **Public** if the professor must access it without an account.
4. Add the Space remote shown by Hugging Face, then push this branch:

   ```bash
   git remote add huggingface https://huggingface.co/spaces/<your-username>/LexiSyncAI
   git push huggingface deploy/huggingface-space:main
   ```

5. Wait for the Docker build to finish. The initial build is large because it
   downloads the Llama 3.2 model. Later rebuilds reuse Docker layers unless the
   model-pull layer changes.
6. Open the Space URL and upload a PDF to verify indexing, then ask a cited
   question.

## Operational notes

- This public demo accepts uploaded PDFs. Hugging Face ephemeral disk is not
  suitable for confidential documents; uploads disappear when the container
  restarts but may exist while it is running.
- The app uses CPU inference. Expect first-answer latency to be noticeably
  higher than on a local machine.
- The model and answer settings remain unchanged from local evaluation:
  `llama3.2`, temperature `0`, seed `42`, and structured JSON output.
