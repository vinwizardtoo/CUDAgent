import CoachChat from '@/components/CoachChat';

export const metadata = {
  title: 'CUDAgent — Coach Mode',
};

export default function CoachPage() {
  return (
    <main className="container">
      <h2>Coach Mode</h2>
      <p className="muted">Describe goals in natural language; the system proposes kernels, validates, and benchmarks.</p>
      <div style={{ height: 12 }} />
      <CoachChat />
    </main>
  );
}

