import PlaygroundEditor from '@/components/PlaygroundEditor';

export default function PlaygroundPage() {
  return (
    <main className="container">
      <h2>Playground</h2>
      <p className="muted">Edit kernels and tuning configs, then run.</p>
      <div style={{ height: 12 }} />
      <PlaygroundEditor />
    </main>
  );
}

