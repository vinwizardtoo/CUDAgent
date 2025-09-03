## Web (Next.js UI)

Playground UI for CUDAgent. This is a minimal scaffold using Next.js App Router.

### Run locally

1. Install deps
   - npm install
2. Start dev server
   - npm run dev
3. Open http://localhost:3000

The playground is at `/playground` and currently uses a fake run that simulates compile/validate timings. Backend wiring will hook this to the CLI/FastAPI service in `apps/cli`.

### Structure

- `app/` App Router pages (`/` and `/playground`).
- `components/` UI components (e.g., `PlaygroundEditor`).
- `next.config.ts`, `tsconfig.json`, `package.json` standard Next.js setup.

### Next steps

- Add API routes or configure the base URL to talk to the backend.
- Replace the fake run with calls to the tuning/compile endpoints.
- Add an editor (Monaco/Codemirror) and basic syntax highlighting.
- Persist sessions/results and render perf tables/plots.

