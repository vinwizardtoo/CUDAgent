import Link from 'next/link';

export default function Home() {
  return (
    <main className="container">
      <h1>CUDAgent</h1>
      <p className="muted">Agentic playground for GPU kernel optimization.</p>
      <div style={{ height: 20 }} />
      <Link href="/playground">
        <button>Open Playground</button>
      </Link>
      <div style={{ height: 24 }} />
      <p className="muted">
        Backend/API wiring comes later. For now, experiment locally in the UI.
      </p>
    </main>
  );
}

