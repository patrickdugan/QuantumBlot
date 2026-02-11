#!/usr/bin/env node

/**
 * TypeScript entrypoint wrapper.
 * Delegates to qft-runner.js so CLI flags stay exactly aligned.
 */

import { spawn } from 'child_process';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main(): Promise<void> {
  const jsRunner = resolve(__dirname, 'qft-runner.js');
  const args = process.argv.slice(2);

  await new Promise<void>((resolvePromise, rejectPromise) => {
    const proc = spawn(process.execPath, [jsRunner, ...args], {
      stdio: 'inherit',
      env: process.env,
    });

    proc.on('error', rejectPromise);
    proc.on('close', (code) => {
      if (code && code !== 0) {
        process.exitCode = code;
      }
      resolvePromise();
    });
  });
}

if (process.argv[1] && resolve(process.argv[1]) === __filename) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  });
}

export { main };
