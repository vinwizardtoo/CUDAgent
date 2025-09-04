"use client";

import { useState } from 'react';

export default function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function onCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {}
  }
  return (
    <button type="button" className="copy-btn" onClick={onCopy} aria-label="Copy code">
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

