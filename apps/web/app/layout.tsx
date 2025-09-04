import './globals.css';
import 'github-markdown-css/github-markdown-dark.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'CUDAgent Playground',
  description: 'Play with Triton/CUDA kernels and tuning configs.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
