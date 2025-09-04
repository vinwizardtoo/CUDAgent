"use client";

import { useEffect, useRef, useState } from 'react';
import Markdown from '@/components/Markdown';

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

    // Start an assistant message that we'll fill as we stream
    const assistantId = crypto.randomUUID();
    setMessages((m) => [...m, { id: assistantId, role: 'assistant', content: '' }]);

    try {
      const res = await fetch('/api/v1/coach/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            ...messages.map(({ role, content }) => ({ role, content })),
            { role: 'user', content: userMsg.content },
          ],
          temperature: 0.2,
        }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        for (const evt of events) {
          const lines = evt.split('\n');
          for (const ln of lines) {
            if (!ln.startsWith('data:')) continue;
            // Preserve spaces in streamed deltas; remove only the first space after 'data:' if present
            let raw = ln.slice(5); // keep leading space if present
            if (raw.startsWith(' ')) raw = raw.slice(1);
            if (raw.trim() === '[DONE]') continue;

            // Try to parse JSON events (status/error), but do NOT trim normal text chunks
            let delta = '';
            const jsonProbe = raw.trimStart();
            if (jsonProbe.startsWith('{') || jsonProbe.startsWith('[')) {
              try {
                const obj = JSON.parse(jsonProbe);
                if (obj?.error) delta = `\n[error] ${obj.error}`;
                // ignore status pings
              } catch {
                delta = raw; // fall back to raw text
              }
            } else {
              delta = raw;
            }

            if (delta) {
              setMessages((m) =>
                m.map((msg) => (msg.id === assistantId ? { ...msg, content: msg.content + delta } : msg))
              );
            }
          }
        }
      }
    } catch (err: any) {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: msg.content + `\n[error] ${err?.message || 'stream failed'}` }
            : msg
        )
      );
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '75vh' }}>
      <div ref={scroller} style={{ overflowY: 'auto', flex: 1, paddingRight: 6 }} aria-live="polite">
        {messages.map((msg) => (
          <div key={msg.id} style={{ marginBottom: 10 }}>
            <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
              {msg.role.toUpperCase()}
            </div>
            <div className={`chat-bubble ${msg.role === 'user' ? 'user' : 'assistant'}`}>
              <Markdown>{msg.content}</Markdown>
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
