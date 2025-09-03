# Repository Guidelines

## Project Structure & Module Organization
- Apps: `apps/web` (Next.js UI), `apps/cli` (CLI + FastAPI).
- Packages: `pkgs/` (Python mono-repo) — `core`, `kernels/{triton,cuda}`, `runtime`, `search`, `profile`, `reports`, `exporters`.
- Workflows: `examples/`, `datasets/`, `runners/{local,kubernetes}`.
- Environments: `docker/dev.Dockerfile`, `docker/runtime.Dockerfile`.
- Tooling: `tools/precommit` (Ruff/Black), `tools/scripts`.
- Quality & Docs: `tests/`, `docs/{rfcs,concepts,tutorials}`.
- Root: `pyproject.toml` (uv workspace), `Makefile`, CONTRIBUTING, LICENSE, README.

## Build, Test, and Development Commands
- `make setup`: bootstrap dev (uv sync, pre-commit hooks).
- `make dev`: run local dev (backend/API and DX helpers).
- `make test`: run unit + integration + golden perf tests with coverage.
- `make bench`: execute performance benchmarks and emit reports.
- `make lint`: run Ruff/Black and any web linters.
Alternatives: `uv run pytest -q`, `ruff check`, `black .`. Docker: `docker build -f docker/dev.Dockerfile .`.

## Coding Style & Naming Conventions
- Python: 4 spaces, `snake_case` funcs/files, `PascalCase` classes. Keep modules cohesive under `pkgs/<pkg>/`.
- Web (Next.js): `camelCase` variables, `PascalCase` components. Use project ESLint/Prettier settings.
- Kernels: place templates under `pkgs/kernels/{triton,cuda}`; include op/arch in names (e.g., `softmax_sm90.cu`).
- Run `make lint` before pushing; commit only formatted code.

## Testing Guidelines
- Framework: `pytest` (+ coverage). Mirror package layout under `tests/<pkg>/test_*.py`.
- For kernels: assert numerical correctness vs PyTorch baselines; add perf assertions when stable.
- Target ≥85% coverage on touched lines. Run locally via `make test`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat: add triton softmax tuner`). Reference issues (`Closes #123`).
- PRs must describe approach, correctness validation, and perf deltas (attach `reports/` tables/plots). Link related RFCs in `docs/rfcs`.

## Security & Configuration Tips
- Never commit secrets; provide `.env.example`. Prefer Docker for reproducibility.
- Document GPU/driver/toolkit requirements in PRs affecting kernels. Note CUDA arch support where relevant.
