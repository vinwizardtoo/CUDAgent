# cuda-agent

Agentic playground for GPU kernel optimization using Triton and CUDA. cuda-agent supports two modes:
- Coach Mode: describe goals in natural language; the system proposes kernels, compiles, validates against PyTorch, and benchmarks.
- Pro Mode: edit kernel templates and tuning parameters directly; the system compiles, profiles, and logs results for regression and rollback.

It measures latency/throughput/occupancy, compares against baselines, and can export optimized kernels as wheels, shared libraries, TensorRT/ONNX plugins, or Docker images.

## Repository Structure
```
cuda-agent/
├─ apps/                       # User-facing entry points
│  ├─ web/                     # Next.js UI (playground frontend)
│  └─ cli/                     # CLI & FastAPI service
├─ pkgs/                       # Core Python packages (mono-repo)
│  ├─ core/                    # IR, config, logging, env fingerprint
│  ├─ kernels/                 # Authoritative kernel sources + templates
│  │  ├─ triton/               # Triton kernels
│  │  └─ cuda/                 # CUDA / CUTLASS templates
│  ├─ runtime/                 # Compile, load, run, validate
│  ├─ search/                  # Grid/Bayes/RL strategies
│  ├─ profile/                 # Timing + CUPTI/Nsight adapters
│  ├─ reports/                 # Perf tables, roofline plots, export
│  └─ exporters/               # Wheels, .so/.cubin, TRT/ONNX, Docker
├─ examples/                   # Demos of tuning/replacing ops
├─ datasets/                   # Input generators + fixtures
├─ runners/                    # Execution backends (local, kubernetes)
├─ docker/                     # Reproducible environments
├─ tools/                      # Dev utilities (precommit, scripts)
├─ tests/                      # Unit + integration + golden perf tests
├─ docs/                       # Docs, RFCS, concepts, tutorials
└─ pyproject.toml, Makefile, CONTRIBUTING.md, LICENSE, README.md
```

## Quick Start
Prerequisites
- NVIDIA GPU + drivers, recent CUDA toolkit matching your PyTorch/Triton.
- Python 3.11+ and uv (workspace manager).
- Node.js >= 18.17.0 for the web UI (see `.nvmrc`).
- Optional: Docker with `--gpus all` runtime.

LLM (Coach Mode)
- Uses LiteLLM to call ChatGPT-compatible endpoints with retries/backoff.
- Set `OPENAI_API_KEY` in your environment (see `.env.example`).

Setup
- make setup

Run locally
- make dev
  - Starts FastAPI/CLI backend and dev helpers.
  - UI: run web commands from inside `apps/web` (do not run `npm install` from repo root):
    - `cd apps/web && npm install`
    - `npm run dev`

Try examples
- uv run python examples/tune_softmax_4090.py
- uv run python examples/tune_layernorm_a100.py

Testing and linting
- make test           # pytest + coverage
- make bench          # run performance suites, emit reports
- make lint           # ruff/black, plus web linters if configured

Docker (reproducible env)
- docker build -f docker/dev.Dockerfile -t cuda-agent-dev .
- docker run --gpus all -it -v $PWD:/workspace cuda-agent-dev

## Web UI
- Location: `apps/web` (Next.js App Router)
- Development:
  - `cd apps/web && npm install`
  - `npm run dev` then open http://localhost:3000 (playground at `/playground`)

Note: npm commands must be run from `apps/web`, not the repository root.

Node version: Next.js requires Node >= 18.17.0. Use nvm/asdf/volta:
- nvm: `nvm install` (reads `.nvmrc`) then `nvm use`
- asdf: respects `.node-version`
- Volta: `volta pin node@18.17.0` (optional)

### Deploy to Vercel
- Import the repo into Vercel and set Root Directory to `apps/web`.
- Keep default Next.js build settings; add env `NEXT_PUBLIC_API_BASE_URL` if calling an external backend.
- Production deploys from `main`; feature branches (e.g., `feat/webdev`) get Preview URLs.

### Planned Makefile shortcuts (future)
To simplify web workflows from the repo root, we plan to add:
- `make web-install` → `cd apps/web && npm install`
- `make web-dev` → `cd apps/web && npm run dev`
- `make web-build` → `cd apps/web && npm run build`


## Concepts
- IR and Runtime: canonical op descriptions feed compilation/execution.
- Search: pluggable strategies explore kernels and configs.
- Profiling: standardized timers + CUPTI/Nsight hooks.
- Reports/Exporters: persist results, plots, and distributable artifacts.

## Contributing
See CONTRIBUTING.md, CODE_OF_CONDUCT.md, and AGENTS.md for style, commands, tests, and PR expectations. Proposals go in `docs/rfcs`.

## License
Apache-2.0 (see LICENSE).
