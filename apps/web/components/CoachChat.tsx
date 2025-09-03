"use client";

import { useEffect, useRef, useState } from 'react';

type Role = 'user' | 'assistant' | 'system';
interface Message {
  id: string;
  role: Role;
  content: string;
}

const seed: Message[] = [
  {
    id: 'sys-1',
    role: 'system',
    content:
      'You are in Coach Mode. Ask for an optimization goal (op, tensor shapes, constraints). I will propose kernels, compile, validate vs PyTorch, and benchmark.',
  },
];

export default function CoachChat() {
  const [messages, setMessages] = useState<Message[]>(seed);
  const [input, setInput] = useState('Optimize softmax for 4090, batch=64, seq=4096.');
  const [sending, setSending] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  async function onSend() {
    if (!input.trim()) return;
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: input.trim() };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setSending(true);

    // Stubbed assistant flow; replace with backend call (streaming preferred)
    const fakeAssistant: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: [
        'Plan:\n1) Gather shapes/dtype and constraints',
        '2) Propose kernels (Triton, fallback CUDA)',
        '3) Compile and validate vs PyTorch baseline',
        '4) Search tuning space (BLOCK_SIZE, num_warps, etc.)',
        '5) Benchmark and report latency/throughput; export artifacts',
        '',
        'First proposal: Triton softmax with BLOCK_SIZE=128, num_warps=4. Want me to compile and run 1000 reps?',
      ].join('\n'),
    };

    // Simulate latency
    await new Promise((r) => setTimeout(r, 500));
    setMessages((m) => [...m, fakeAssistant]);
    setSending(false);
  }

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '75vh' }}>
      <div ref={scroller} style={{ overflowY: 'auto', flex: 1, paddingRight: 6 }}>
        {messages.map((msg) => (
          <div key={msg.id} style={{ marginBottom: 10 }}>
            <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
              {msg.role.toUpperCase()}
            </div>
            <div
              style={{
                whiteSpace: 'pre-wrap',
                background: msg.role === 'user' ? '#0f172a' : '#121826',
                border: '1px solid #1f2937',
                borderRadius: 8,
                padding: 10,
              }}
            >
              {msg.content}
            </div>
          </div>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSend();
        }}
        style={{ display: 'flex', gap: 8 }}
      >
        <input
          aria-label="Message"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe your goal in natural language..."
          style={{ flex: 1 }}
        />
        <button disabled={sending}>{sending ? 'Sending…' : 'Send'}</button>
      </form>
    </div>
  );
}

