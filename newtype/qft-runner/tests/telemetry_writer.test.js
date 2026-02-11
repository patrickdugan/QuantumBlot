import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

import { JsonlTelemetryWriter } from '../telemetry_runtime.js';

test('JsonlTelemetryWriter flushes records to disk', () => {
  const dir = mkdtempSync(join(tmpdir(), 'sc1028-test-'));
  const file = join(dir, 'telemetry.jsonl');
  const writer = new JsonlTelemetryWriter(file, { flushEvery: 1 });
  try {
    writer.open();
    writer.write({ run_id: 'r1', chunk_id: 1, ok: true });
    writer.flush();
    const body = readFileSync(file, 'utf8').trim();
    assert.ok(body.includes('"run_id":"r1"'));
    assert.ok(body.includes('"chunk_id":1'));
  } finally {
    writer.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

