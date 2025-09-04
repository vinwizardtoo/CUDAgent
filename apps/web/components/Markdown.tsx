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
  // Work around type incompatibilities between rehype-highlight and unified/vfile types
  // seen in some builds (nested vfile versions). Keep runtime behavior identical.
  const rehypePluginsSafe = [rehypeHighlight as unknown as any];
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={rehypePluginsSafe as unknown as any}
      components={{
        // Version-agnostic code renderer:
        // - react-markdown v8: receives `inline` on `code`
        // - react-markdown v9: inline code uses `inlineCode`; `code` is block-only
        // Use `any` to avoid TS type churn across versions in CI.
        code(props: any) {
          const { className, children, ...rest } = props || {};
          const lang = /language-(\w+)/.exec(className || "")?.[1];
          const text = String(children ?? "");
          const trimmed = text.replace(/\n$/, "");
          const isInline = Boolean((props as any)?.inline) || (!className && !/\n/.test(text));
          if (isInline) {
            return (
              <code className="md-inline-code" {...rest}>
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
                <code className={className || (lang ? `language-${lang}` : undefined)} {...rest}>
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
