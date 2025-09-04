# CUDAgent CLI/API

FastAPI backend exposing Coach Mode (LLM) and run endpoints.

## Run locally

- Ensure you have Python 3.11+ and dependencies available (uv or pip).
- Set env: `OPENAI_API_KEY` (see `.env.example`).
- Install local packages (from repo root):
  - `pip install -e pkgs/core`
  - `pip install -e apps/cli`
- Start server:
  - `uv run uvicorn cuda_agent_cli.main:app --reload --port 8000`
  - Or `python -m uvicorn cuda_agent_cli.main:app --reload --port 8000`

With the Next.js proxy, the web app calls `/api/...` and forwards to `http://localhost:8000`.

## Endpoints

- `GET /health` → `{ status: "ok" }`
- `POST /v1/coach/plan` (streaming SSE) → streams LLM tokens as `data: <chunk>` lines.
- `POST /v1/coach/execute` (streaming) → simulates compile/validate/benchmark logs.
- `POST /v1/run` → stub for direct run (to be implemented with runtime hooks).
