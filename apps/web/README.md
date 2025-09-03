## Web (Next.js UI)

Playground UI for CUDAgent. This is a minimal scaffold using Next.js App Router.

### Run locally

1. Install deps
   - npm install
2. Start dev server
   - npm run dev
3. Open http://localhost:3000

The playground is at `/playground` and currently uses a fake run that simulates compile/validate timings. Backend wiring will hook this to the CLI/FastAPI service in `apps/cli`.

### Modes
- Coach Mode (`/coach`): Describe goals in natural language; the system proposes kernels, compiles, validates, and benchmarks. The current UI has a stubbed chat; wire it to the backend for real runs.
- Pro Mode (`/playground`): Edit kernel/config directly and run. Currently simulates output.

### Intended Backend API (sketch)
- POST `/v1/coach/plan` → { messages: Message[] } → returns a structured plan and next actions.
- POST `/v1/coach/execute` → { action } → triggers compile/validate/benchmark; streams progress.
- POST `/v1/run` → { kernel, config } → runs a single compile/validate/benchmark.

Prefer server-sent events (SSE) or chunked responses for progress streaming; add CORS or use Next rewrites.

### Deploy to Vercel (Monorepo)

When importing this repo in Vercel:
- Root Directory: set to `apps/web`.
- Framework Preset: Next.js (auto-detected).
- Install Command: leave default (npm ci / npm install).
- Build Command: leave default (`next build`).
- Environment Variables: add `NEXT_PUBLIC_API_BASE_URL` (see `.env.example`).
  - Prefer `API_BASE_URL` for server-side proxy (see Proxy below).

Branch behavior:
- Production: set to your `main` branch.
- Preview: every PR/branch (e.g., `feat/webdev`) will auto-deploy with a unique URL.

Notes:
- Next.js requires Node >= 18.17.0 (covered by `.nvmrc`). Vercel uses a compatible version automatically.
- If your backend isn’t on Vercel, either call `/api/...` (proxy) or use the full `NEXT_PUBLIC_API_BASE_URL`.

### Proxy backend (recommended)
We proxy browser calls to the backend via Next rewrites to avoid CORS:

- Config: set `API_BASE_URL` in env (defaults to `http://localhost:8000`).
- Client code: call relative paths like `/api/v1/run` instead of hardcoding the backend host.
- Vercel: set `API_BASE_URL` to your backend’s public URL in Project Settings.

### Structure

- `app/` App Router pages (`/` and `/playground`).
- `components/` UI components (e.g., `PlaygroundEditor`).
- `next.config.ts`, `tsconfig.json`, `package.json` standard Next.js setup.

### Next steps

- Add API routes or configure the base URL to talk to the backend.
- Replace the fake run with calls to the tuning/compile endpoints.
- Add an editor (Monaco/Codemirror) and basic syntax highlighting.
- Persist sessions/results and render perf tables/plots.
