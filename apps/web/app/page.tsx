import Link from 'next/link';

export default function Home() {
  return (
    <main className="container">
      <h1>CUDAgent</h1>
      <p className="muted">Agentic playground for GPU kernel optimization.</p>
      <div style={{ height: 20 }} />
      <div style={{ display: 'flex', gap: 12 }}>
        <Link href="/playground">
          <button>Open Playground (Pro)</button>
        </Link>
        <Link href="/coach">
          <button>Open Coach Mode</button>
        </Link>
      </div>
      <div style={{ height: 24 }} />
      <p className="muted">
        Backend/API wiring comes later. For now, experiment locally in the UI.
      </p>
    </main>
  );
}
