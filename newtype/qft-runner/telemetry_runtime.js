import { openSync, writeSync, fsyncSync, closeSync, existsSync, readFileSync, unlinkSync } from 'fs';
import { dirname, resolve } from 'path';
import { mkdirSync } from 'fs';
import os from 'os';

export class JsonlTelemetryWriter {
  constructor(filePath, { flushEvery = 1 } = {}) {
    this.filePath = filePath;
    this.flushEvery = Math.max(1, Number(flushEvery) || 1);
    this.fd = null;
    this.count = 0;
  }

  open() {
    mkdirSync(dirname(resolve(this.filePath)), { recursive: true });
    this.fd = openSync(this.filePath, 'a');
  }

  write(record) {
    if (this.fd === null) {
      throw new Error('Telemetry writer is not open');
    }
    const line = `${JSON.stringify(record)}\n`;
    writeSync(this.fd, line, null, 'utf8');
    this.count += 1;
    if (this.count % this.flushEvery === 0) {
      this.flush();
    }
  }

  flush() {
    if (this.fd !== null) {
      fsyncSync(this.fd);
    }
  }

  close() {
    if (this.fd !== null) {
      this.flush();
      closeSync(this.fd);
      this.fd = null;
    }
  }
}

export class RunLock {
  constructor(lockfilePath, runId) {
    this.lockfilePath = lockfilePath;
    this.runId = runId;
    this.lockHeld = false;
  }

  pidAlive(pid) {
    if (!Number.isInteger(pid) || pid <= 0) {
      return false;
    }
    try {
      process.kill(pid, 0);
      return true;
    } catch (_err) {
      return false;
    }
  }

  acquire() {
    if (existsSync(this.lockfilePath)) {
      let stale = false;
      try {
        const payload = JSON.parse(readFileSync(this.lockfilePath, 'utf8'));
        stale = !this.pidAlive(Number(payload.pid));
      } catch (_err) {
        stale = false;
      }
      if (stale) {
        unlinkSync(this.lockfilePath);
      } else {
        throw new Error(`Lockfile already exists: ${this.lockfilePath}`);
      }
    }

    const payload = {
      run_id: this.runId,
      pid: process.pid,
      created_unix: Date.now() / 1000,
    };
    const fd = openSync(this.lockfilePath, 'wx');
    try {
      writeSync(fd, JSON.stringify(payload), null, 'utf8');
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    this.lockHeld = true;
  }

  release() {
    if (this.lockHeld && existsSync(this.lockfilePath)) {
      unlinkSync(this.lockfilePath);
    }
    this.lockHeld = false;
  }
}

export class RSSWatchdog {
  constructor(threshold = 0.85) {
    this.threshold = threshold;
    this.totalBytes = os.totalmem();
  }

  sample() {
    const rss = process.memoryUsage().rss;
    const ratio = this.totalBytes > 0 ? rss / this.totalBytes : null;
    const tripped = ratio !== null && ratio >= this.threshold;
    return {
      rssBytes: rss,
      rssMb: rss / (1024 * 1024),
      totalBytes: this.totalBytes,
      totalMb: this.totalBytes / (1024 * 1024),
      ratio,
      tripped,
    };
  }
}

export class ChunkCycleTracker {
  constructor(windowSize = 256) {
    this.windowSize = windowSize;
    this.obsWindow = [];
    this.actionWindow = [];
  }

  update(obsHash, actionHash) {
    const repeatObs = this.obsWindow.includes(obsHash);
    const repeatAction = this.actionWindow.includes(actionHash);
    const shortCycle = this.actionWindow.length >= 2 && this.actionWindow[this.actionWindow.length - 2] === actionHash;

    this.obsWindow.push(obsHash);
    this.actionWindow.push(actionHash);
    if (this.obsWindow.length > this.windowSize) {
      this.obsWindow.shift();
    }
    if (this.actionWindow.length > this.windowSize) {
      this.actionWindow.shift();
    }

    return { repeatObs, repeatAction, shortCycle };
  }
}

