"use client";

import { useMemo, useState } from 'react';

const defaultKernel = `// Triton example: vector add
// Adjust BLOCK_SIZE to explore perf space
#define BLOCK_SIZE 128

// Pseudo-code placeholder (UI only)
// Implement real kernels in pkgs/kernels and wire via API
`;

const defaultConfig = `{
  "dtype": "float32",
  "block_size": 128,
  "grid": [256, 1, 1],
  "repetitions": 1000
}`;

export default function PlaygroundEditor() {
  const [kernel, setKernel] = useState(defaultKernel);
  const [config, setConfig] = useState(defaultConfig);
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState<string>("Ready. No run yet.");

  const parsedConfig = useMemo(() => {
    try {
      return JSON.parse(config);
    } catch (e) {
      return null;
    }
  }, [config]);

  function fakeRun() {
    setRunning(true);
    setOutput("Compiling kernel ...");
    // Simulate work; in real app, call backend API
    setTimeout(() => {
      const bs = parsedConfig?.block_size ?? 'n/a';
      setOutput(
        [
          "Compile: success",
          `Config: block_size=${bs}`,
          "Validate: OK vs PyTorch baseline",
          "Latency: 12.34 us  |  Throughput: 89.0 GB/s",
        ].join("\n")
      );
      setRunning(false);
    }, 600);
  }

  function clearOutput() {
    setOutput("");
  }

  return (
    <div className="row">
      <section className="col panel">
        <h3>Kernel</h3>
        <textarea
          aria-label="Kernel source"
          value={kernel}
          onChange={(e) => setKernel(e.target.value)}
        />
      </section>
      <section className="col panel">
        <h3>Config</h3>
        <textarea
          aria-label="Tuning config JSON"
          value={config}
          onChange={(e) => setConfig(e.target.value)}
        />
        <div style={{ height: 8 }} />
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={fakeRun} disabled={running || !parsedConfig}>
            {running ? 'Running…' : 'Run'}
          </button>
          <button onClick={clearOutput}>Clear</button>
        </div>
        <div style={{ height: 12 }} />
        <h4>Output</h4>
        <pre>{output}</pre>
      </section>
    </div>
  );
}

