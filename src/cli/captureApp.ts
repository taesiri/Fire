import './capture.css';
import type { StreamTargetChunk } from 'mediabunny';
import {
  getOfflineVideoPreset,
  renderOfflineVideoToStream,
  type OfflineVideoProgress,
} from '../export/offlineVideo';
import { parseFireCliCaptureConfig, type FireCliCaptureConfig } from './config';
import { FireXEngine, resolveFireXOfflineDimensions } from '../sim/firex/FireXEngine';
import { getFireXFieldView, getFireXScene } from '../sim/firex/scenes';

interface FireReplicaCaptureBridge {
  readonly version: 1;
  start(): Promise<void>;
}

declare global {
  interface Window {
    fireReplicaCapture: FireReplicaCaptureBridge;
  }
}

const canvas = required<HTMLCanvasElement>('#capture-canvas');
const title = required<HTMLElement>('#capture-title');
const detail = required<HTMLElement>('#capture-detail');
const progressElement = required<HTMLProgressElement>('#capture-progress');
const framesElement = required<HTMLElement>('#capture-frames');
const bytesElement = required<HTMLElement>('#capture-bytes');
const token = new URLSearchParams(window.location.search).get('fire-cli');

let startPromise: Promise<void> | null = null;
let streamedBytes = 0;

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing capture element: ${selector}`);
  return element;
}

function endpoint(path: string, parameters: Record<string, string | number> = {}): string {
  if (!token) throw new Error('No local CLI token was provided.');
  const url = new URL(path, window.location.origin);
  url.searchParams.set('token', token);
  for (const [name, value] of Object.entries(parameters)) url.searchParams.set(name, String(value));
  return url.toString();
}

async function loadConfig(): Promise<FireCliCaptureConfig> {
  const response = await fetch(endpoint('/__fire_cli/config'), { cache: 'no-store' });
  if (!response.ok) throw new Error(`The local CLI rejected the capture configuration (${response.status}).`);
  return parseFireCliCaptureConfig(await response.json());
}

async function postEvent(type: string, payload: Record<string, unknown>): Promise<void> {
  const response = await fetch(endpoint('/__fire_cli/event'), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ type, ...payload }),
  });
  if (!response.ok) throw new Error(`The local CLI rejected a ${type} event (${response.status}).`);
}

function createUploadStream(): WritableStream<StreamTargetChunk> {
  let nextPosition = 0;
  return new WritableStream<StreamTargetChunk>({
    async write(chunk) {
      if (chunk.type !== 'write') throw new Error(`Unsupported stream operation: ${String(chunk.type)}.`);
      if (chunk.position !== nextPosition) {
        throw new Error(`Non-monotonic MP4 write: received ${chunk.position}, expected ${nextPosition}.`);
      }
      const response = await fetch(endpoint('/__fire_cli/chunk', { position: chunk.position }), {
        method: 'POST',
        body: chunk.data,
      });
      if (!response.ok) {
        const message = await response.text().catch(() => '');
        throw new Error(`Local file sink rejected bytes at ${chunk.position}: ${message || response.status}.`);
      }
      nextPosition += chunk.data.byteLength;
      streamedBytes = nextPosition;
      bytesElement.textContent = `${formatBytes(streamedBytes)} streamed`;
    },
  });
}

function sameDimensions(
  left: readonly [number, number, number],
  right: readonly [number, number, number],
): boolean {
  return left.every((value, index) => value === right[index]);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes / 1024;
  let unit = units[0]!;
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index]!;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`;
}

async function runCapture(): Promise<void> {
  if (!token) {
    title.textContent = 'No local recording job';
    detail.textContent = 'Launch this page through npm run record. It will not accept a filesystem destination from a URL.';
    return;
  }

  const config = await loadConfig();
  const preset = getOfflineVideoPreset(config.outputPreset);
  const scene = getFireXScene(config.scene);
  const field = getFireXFieldView(config.fieldView);
  const engine = new FireXEngine(canvas);
  const startedAt = performance.now();
  let captureStartedAt = startedAt;
  let lastProgressEventAt = Number.NEGATIVE_INFINITY;

  title.textContent = `${scene.label} · ${field.label}`;
  detail.textContent = 'Configuring the exact Fire-X solver before GPU allocation…';

  try {
    engine.setGridMemoryBudget(config.gridMemoryBudgetBytes);
    await engine.setQuality('cinematic');
    await engine.setGridDimensions(config.gridDimensions);
    await engine.setOfflineRenderTier(config.solverTier);
    engine.setOfflineOpticalDetailTarget(config.opticalDetailTarget);
    for (const [name, value] of Object.entries(scene.values)) engine.setParameter(name, value);
    engine.setParameter('viewMode', field.value);
    engine.setParameter('cameraYaw', config.camera.yaw);
    engine.setParameter('cameraPitch', config.camera.pitch);
    engine.setParameter('cameraDistance', config.camera.distance);

    detail.textContent = 'Requesting the high-performance WebGPU adapter and allocating the exact solver…';
    await engine.initialize();
    const expectedDimensions = resolveFireXOfflineDimensions(config.solverTier, config.gridDimensions);
    const gridInfo = engine.getGridInfo();
    if (!sameDimensions(gridInfo.dimensions, expectedDimensions)) {
      throw new Error(
        `Fire-X allocated ${gridInfo.dimensions.join('×')}, expected exact ${expectedDimensions.join('×')}.`,
      );
    }
    await engine.resizeOutput(preset.width, preset.height);
    if (canvas.width !== preset.width || canvas.height !== preset.height) {
      throw new Error(`Capture canvas is ${canvas.width}×${canvas.height}, expected ${preset.width}×${preset.height}.`);
    }
    const warmupFrames = Math.round(config.warmupSeconds * preset.frameRate);
    detail.textContent = `Warming the deterministic solver for ${warmupFrames.toLocaleString('en-US')} frames…`;
    for (let index = 0; index < warmupFrames; index += 1) {
      await engine.renderOfflineFrame(1 / preset.frameRate, false);
      progressElement.value = warmupFrames > 0 ? ((index + 1) / warmupFrames) * 0.01 : 0.01;
      framesElement.textContent = `${(index + 1).toLocaleString('en-US')} warmup frames`;
    }
    captureStartedAt = performance.now();

    const uploadStream = createUploadStream();
    const onProgress = (progress: OfflineVideoProgress): void => {
      progressElement.value = Math.min(1, 0.01 + progress.fraction * 0.99);
      framesElement.textContent = `${progress.completedFrames.toLocaleString('en-US')} / ${progress.totalFrames.toLocaleString('en-US')} frames`;
      detail.textContent = progress.phase === 'capture'
        ? `Rendering ${preset.label} at an exact ${preset.frameRate} fps timeline…`
        : progress.phase === 'finalizing'
          ? 'Finalizing streamed MP4 fragments…'
          : 'Probing the exact H.264 encoder…';
      const now = performance.now();
      if (now - lastProgressEventAt >= 1000 || progress.fraction >= 1) {
        lastProgressEventAt = now;
        void postEvent('progress', {
          phase: progress.phase,
          completedFrames: progress.completedFrames,
          totalFrames: progress.totalFrames,
          fraction: progress.fraction,
          streamedBytes,
          wallSeconds: (now - captureStartedAt) / 1000,
          totalWallSeconds: (now - startedAt) / 1000,
        }).catch(() => undefined);
      }
    };

    const result = await renderOfflineVideoToStream({
      canvas,
      preset,
      durationSeconds: config.durationSeconds,
      writable: uploadStream,
      renderFrame: (_index, timing) => engine.renderOfflineVideoFrame(
        1 / preset.frameRate,
        timing.timestampSeconds,
        timing.durationSeconds,
      ),
      onProgress,
    });

    progressElement.value = 1;
    title.textContent = 'Capture stream finalized';
    detail.textContent = 'Handing the file to ffprobe/ffmpeg for independent disk verification…';
    await postEvent('complete', {
      frameCount: result.frameCount,
      codec: result.codec,
      hardwareAcceleration: result.hardwareAcceleration,
      streamedBytes,
      wallSeconds: (performance.now() - captureStartedAt) / 1000,
      totalWallSeconds: (performance.now() - startedAt) / 1000,
      gridInfo,
      gridCapabilities: engine.getGridCapabilities(),
      engineDetail: engine.detail,
      performance: engine.getOfflinePerformanceInfo(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    title.textContent = 'Capture failed';
    detail.textContent = message;
    await postEvent('error', {
      message,
      stack: error instanceof Error ? error.stack : undefined,
      streamedBytes,
      wallSeconds: (performance.now() - startedAt) / 1000,
    }).catch(() => undefined);
    throw error;
  } finally {
    engine.dispose();
  }
}

window.fireReplicaCapture = Object.freeze({
  version: 1,
  start(): Promise<void> {
    startPromise ??= runCapture();
    return startPromise;
  },
});

void window.fireReplicaCapture.start().catch((error) => {
  console.error('Fire Replica CLI capture failed.', error);
});
