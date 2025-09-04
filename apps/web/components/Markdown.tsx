"use client";

import React from 'react';
import dynamic from 'next/dynamic';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import CopyButton from '@/components/CopyButton';

// Dynamically import react-markdown to avoid any SSR/ESM quirks
const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false });

type Props = {
  children: string;
};

export default function Markdown({ children }: Props) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        code({ inline, className, children, ...props }) {
          const lang = /language-(\w+)/.exec(className || "")?.[1];
          const text = String(children ?? "");
          const trimmed = text.replace(/\n$/, "");
          if (inline) {
            return (
              <code className="md-inline-code" {...props}>
                {trimmed}
              </code>
            );
          }
          return (
            <div className="md-code-wrap">
              <div className="md-code-header">
                <span className="md-code-lang">{lang || 'code'}</span>
                <CopyButton text={trimmed} />
              </div>
              <pre className="md-code-block">
                <code className={className || (lang ? `language-${lang}` : undefined)} {...props}>
                  {trimmed}
                </code>
              </pre>
            </div>
          );
        },
        a({ href, children, ...props }) {
          return (
            <a href={href} target="_blank" rel="noreferrer" {...props}>
              {children}
            </a>
          );
        },
        table({ children }) {
          return <div className="md-table-wrap"><table>{children}</table></div>;
        },
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
