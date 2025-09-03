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
- Optional: Docker with `--gpus all` runtime.

Setup
- make setup

Run locally
- make dev
  - Starts FastAPI/CLI backend and dev helpers.
  - Start the UI from `apps/web` if not wired: `npm install && npm run dev`.

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

## Concepts
- IR and Runtime: canonical op descriptions feed compilation/execution.
- Search: pluggable strategies explore kernels and configs.
- Profiling: standardized timers + CUPTI/Nsight hooks.
- Reports/Exporters: persist results, plots, and distributable artifacts.

## Contributing
See CONTRIBUTING.md, CODE_OF_CONDUCT.md, and AGENTS.md for style, commands, tests, and PR expectations. Proposals go in `docs/rfcs`.

## License
Apache-2.0 (see LICENSE).
