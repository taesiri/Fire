import {
  BufferSource,
  BufferTarget,
  EncodedPacketSink,
  Input,
  Mp4InputFormat,
  Mp4OutputFormat,
  Output,
  StreamTarget,
  VideoSample,
  VideoSampleSink,
  VideoSampleSource,
} from 'mediabunny';
import type { InputVideoTrack, StreamTargetChunk } from 'mediabunny';
import type { OfflineRenderTier } from '../sim/types';

export const OFFLINE_VIDEO_WARMUP_SECONDS = 1;
export const OFFLINE_VIDEO_MAX_DURATION_SECONDS = 10;

export const OFFLINE_VIDEO_ENCODING_POLICY = Object.freeze({
  bitrateMode: 'variable' as const,
  latencyMode: 'realtime' as const,
  keyFrameIntervalSeconds: 1,
});

interface OfflineVideoEncodingProfile {
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly frameRate: number;
  readonly bitrate: number;
  readonly codecString: string;
}

export const OFFLINE_VIDEO_PREVIEW_PROFILE = Object.freeze({
  label: 'Playback proof · 720p60',
  width: 1280,
  height: 720,
  frameRate: 60,
  bitrate: 12_000_000,
  codecString: 'avc1.640029',
} as const satisfies OfflineVideoEncodingProfile);

export const OFFLINE_VIDEO_PRESETS = Object.freeze([
  Object.freeze({
    id: 'hd',
    label: 'Full HD · 1080p60',
    width: 1920,
    height: 1080,
    frameRate: 60,
    bitrate: 24_000_000,
    maxDurationSeconds: 10,
    offlineTier: 'hd' as const,
    codecString: 'avc1.64002a',
    codecLabel: 'H.264 High L4.2',
  }),
  Object.freeze({
    id: 'qhd',
    label: 'QHD · 1440p60',
    width: 2560,
    height: 1440,
    frameRate: 60,
    bitrate: 40_000_000,
    maxDurationSeconds: 7,
    offlineTier: 'qhd' as const,
    codecString: 'avc1.640033',
    codecLabel: 'H.264 High L5.1',
  }),
  Object.freeze({
    id: 'uhd',
    label: '4K UHD · 2160p60',
    width: 3840,
    height: 2160,
    frameRate: 60,
    bitrate: 65_000_000,
    maxDurationSeconds: 5,
    offlineTier: 'uhd' as const,
    codecString: 'avc1.640034',
    codecLabel: 'H.264 High L5.2',
  }),
] as const satisfies readonly {
  readonly id: string;
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly frameRate: 60;
  readonly bitrate: number;
  readonly maxDurationSeconds: number;
  readonly offlineTier: OfflineRenderTier;
  readonly codecString: string;
  readonly codecLabel: string;
}[]);

export type OfflineVideoPreset = (typeof OFFLINE_VIDEO_PRESETS)[number];
export type OfflineVideoPresetId = OfflineVideoPreset['id'];
export type OfflineVideoRenderPhase = 'warmup' | 'capture';
export type OfflineVideoProgressPhase = 'probing' | 'allocating' | OfflineVideoRenderPhase | 'finalizing' | 'verifying' | 'proxying';
export type OfflineVideoHardwareAcceleration = 'prefer-hardware' | 'no-preference';

export interface OfflineVideoProgress {
  phase: OfflineVideoProgressPhase;
  completedFrames: number;
  totalFrames: number;
  fraction: number;
}

export interface OfflineVideoPlaybackCapability {
  supported: boolean;
  smooth: boolean;
  powerEfficient: boolean;
}

export interface RenderOfflineVideoOptions {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  preset: OfflineVideoPreset;
  durationSeconds: number;
  signal?: AbortSignal;
  prepare?: () => Promise<void>;
  renderFrame: (
    phase: OfflineVideoRenderPhase,
    index: number,
    timing?: Readonly<{ timestampSeconds: number; durationSeconds: number }>,
  ) => Promise<VideoFrame | void>;
  onProgress?: (progress: OfflineVideoProgress) => void;
}

export interface OfflineVideoResult {
  blob: Blob;
  playbackPreviewBlob: Blob;
  codec: 'avc';
  frameCount: number;
  hardwareAcceleration: OfflineVideoHardwareAcceleration;
  playbackCapability: OfflineVideoPlaybackCapability | null;
  playbackPreviewCapability: OfflineVideoPlaybackCapability | null;
  verification: OfflineVideoVerification;
  playbackPreviewVerification: OfflineVideoVerification;
}

export interface RenderOfflineVideoStreamOptions {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  preset: OfflineVideoPreset;
  durationSeconds: number;
  writable: WritableStream<StreamTargetChunk>;
  signal?: AbortSignal;
  prepare?: () => Promise<void>;
  renderFrame: (
    index: number,
    timing: Readonly<{ timestampSeconds: number; durationSeconds: number }>,
  ) => Promise<VideoFrame>;
  onProgress?: (progress: OfflineVideoProgress) => void;
}

export interface OfflineVideoStreamResult {
  codec: 'avc';
  frameCount: number;
  hardwareAcceleration: OfflineVideoHardwareAcceleration;
}

export interface OfflineVideoVerification {
  codedWidth: number;
  codedHeight: number;
  frameCount: number;
  frameRate: number;
  durationSeconds: number;
  averageBitrate: number;
  codec: 'avc';
  codecString: string;
  cadenceFrameCount: number;
  maximumTimestampErrorSeconds: number;
  maximumDurationErrorSeconds: number;
  keyFrameCount: number;
  maximumKeyFrameGapSeconds: number;
  decodeOrderTimestampRegressionCount: number;
  decodedFrameCount: number;
  maximumDecodedTimestampErrorSeconds: number;
  maximumDecodedDurationErrorSeconds: number;
  exactDuplicateAdjacentFrameCount: number;
  nearDuplicateAdjacentFrameCount: number;
  longestNearDuplicateTransitionRun: number;
  minimumAdjacentLumaMad: number;
  medianAdjacentLumaMad: number;
  maximumAdjacentLumaMad: number;
  discontinuousAdjacentFrameCount: number;
}

export function getOfflineVideoPreset(id: string): OfflineVideoPreset {
  const preset = OFFLINE_VIDEO_PRESETS.find((candidate) => candidate.id === id);
  if (!preset) throw new RangeError(`Unknown offline video preset: ${id}`);
  return preset;
}

export function getOfflineVideoFrameCount(
  durationSeconds: number,
  presetOrFps: OfflineVideoPreset | number,
): number {
  assertPositiveFinite(durationSeconds, 'durationSeconds');
  const fps = typeof presetOrFps === 'number' ? presetOrFps : presetOrFps.frameRate;
  assertPositiveFinite(fps, 'fps');
  const exactFrameCount = durationSeconds * fps;
  const frameCount = Math.round(exactFrameCount);
  if (Math.abs(exactFrameCount - frameCount) > 1e-7) {
    throw new RangeError('durationSeconds must end on an exact output-frame boundary.');
  }
  return Math.max(1, frameCount);
}

export function estimateOfflineVideoBytes(
  preset: OfflineVideoPreset,
  durationSeconds: number,
): number {
  assertPositiveFinite(durationSeconds, 'durationSeconds');
  return Math.ceil((preset.bitrate * durationSeconds) / 8);
}

export function isOfflineVideoKeyFrame(index: number, preset: OfflineVideoPreset): boolean {
  if (!Number.isInteger(index) || index < 0) {
    throw new RangeError('index must be a non-negative integer.');
  }
  const keyFrameStride = preset.frameRate * OFFLINE_VIDEO_ENCODING_POLICY.keyFrameIntervalSeconds;
  return index % keyFrameStride === 0;
}

export function isOfflineVideoSupported(): boolean {
  return typeof globalThis.VideoEncoder === 'function'
    && typeof globalThis.VideoFrame === 'function';
}

export async function renderOfflineVideo(
  options: RenderOfflineVideoOptions,
): Promise<OfflineVideoResult> {
  const { canvas, preset, durationSeconds, signal, prepare, renderFrame, onProgress } = options;
  validateRenderOptions(options);
  throwIfAborted(signal);

  const frameCount = getOfflineVideoFrameCount(durationSeconds, preset);
  const warmupFrameCount = getOfflineVideoFrameCount(OFFLINE_VIDEO_WARMUP_SECONDS, preset);
  const totalWorkFrames = warmupFrameCount + frameCount;
  const reportProgress = (
    phase: OfflineVideoProgressPhase,
    completedFrames: number,
    fraction = (completedFrames / totalWorkFrames) * 0.97,
  ): void => {
    onProgress?.({ phase, completedFrames, totalFrames: totalWorkFrames, fraction });
  };

  reportProgress('probing', 0);
  const hardwareAcceleration = await probeExactAvcEncoder(preset);
  const playbackCapability = await probePlaybackCapability(preset);
  throwIfAborted(signal);
  reportProgress('allocating', 0, 0.005);
  await prepare?.();
  throwIfAborted(signal);
  validatePreparedCanvas(canvas, preset);

  const target = new BufferTarget();
  const output = new Output({
    format: new Mp4OutputFormat({ fastStart: 'reserve' }),
    target,
  });
  const source = new VideoSampleSource({
    codec: 'avc',
    fullCodecString: preset.codecString,
    bitrate: preset.bitrate,
    bitrateMode: OFFLINE_VIDEO_ENCODING_POLICY.bitrateMode,
    latencyMode: OFFLINE_VIDEO_ENCODING_POLICY.latencyMode,
    hardwareAcceleration,
    contentHint: 'motion',
    sizeChangeBehavior: 'deny',
    alpha: 'discard',
    keyFrameInterval: OFFLINE_VIDEO_ENCODING_POLICY.keyFrameIntervalSeconds,
  });
  output.addVideoTrack(source, {
    frameRate: preset.frameRate,
    maximumPacketCount: frameCount,
  });

  let cancelPromise: Promise<void> | null = null;
  const cancelOutput = (): Promise<void> => {
    cancelPromise ??= output.cancel();
    return cancelPromise;
  };
  const handleAbort = (): void => {
    void cancelOutput().catch(() => undefined);
  };
  signal?.addEventListener('abort', handleAbort, { once: true });

  try {
    await output.start();
    throwIfAborted(signal);

    for (let index = 0; index < warmupFrameCount; index += 1) {
      throwIfAborted(signal);
      const unusedFrame = await renderFrame('warmup', index);
      unusedFrame?.close();
      reportProgress('warmup', index + 1);
      if ((index + 1) % 2 === 0) await yieldToEventLoop();
    }

    for (let index = 0; index < frameCount; index += 1) {
      throwIfAborted(signal);
      const timestamp = index / preset.frameRate;
      const nextTimestamp = (index + 1) / preset.frameRate;
      const duration = nextTimestamp - timestamp;
      const frame = await renderFrame('capture', index, {
        timestampSeconds: timestamp,
        durationSeconds: duration,
      });
      if (!(frame instanceof VideoFrame)) {
        throw new Error('Offline renderer did not return an immutable VideoFrame for capture.');
      }
      try {
        throwIfAborted(signal);
        if (frame.codedWidth !== preset.width || frame.codedHeight !== preset.height) {
          throw new Error(
            `Offline renderer returned ${frame.codedWidth}×${frame.codedHeight}, expected ${preset.width}×${preset.height}.`,
          );
        }
        const sample = new VideoSample(frame);
        try {
          await source.add(sample, {
            keyFrame: isOfflineVideoKeyFrame(index, preset),
          });
        } finally {
          sample.close();
        }
      } finally {
        frame.close();
      }
      reportProgress('capture', warmupFrameCount + index + 1);
      if ((index + 1) % 2 === 0) await yieldToEventLoop();
    }

    source.close();
    reportProgress('finalizing', totalWorkFrames, 0.98);
    await output.finalize();
    throwIfAborted(signal);
    if (!target.buffer) throw new Error('The MP4 encoder finalized without producing an output buffer.');

    reportProgress('verifying', totalWorkFrames, 0.985);
    const verification = await verifyOfflineVideo(
      target.buffer,
      preset,
      frameCount,
      durationSeconds,
      signal,
    );
    throwIfAborted(signal);
    reportProgress('proxying', totalWorkFrames, 0.995);
    const playbackPreview = await createOfflinePlaybackPreview(
      target.buffer,
      signal,
    );
    throwIfAborted(signal);

    return {
      blob: new Blob([target.buffer], { type: 'video/mp4' }),
      playbackPreviewBlob: playbackPreview.blob,
      codec: 'avc',
      frameCount,
      hardwareAcceleration,
      playbackCapability,
      playbackPreviewCapability: playbackPreview.playbackCapability,
      verification,
      playbackPreviewVerification: playbackPreview.verification,
    };
  } catch (error) {
    await cancelOutput().catch(() => undefined);
    if (signal?.aborted) throw createAbortError();
    throw error;
  } finally {
    signal?.removeEventListener('abort', handleAbort);
  }
}

/**
 * Encodes a long deterministic recording into a writable stream. Unlike
 * renderOfflineVideo(), this path never retains encoded payloads, decodes, or
 * proxies the finished file in the browser. The fMP4 muxer retains lightweight
 * index metadata until finalization; the CLI sink verifies payloads from disk.
 */
export async function renderOfflineVideoToStream(
  options: RenderOfflineVideoStreamOptions,
): Promise<OfflineVideoStreamResult> {
  if (!options || typeof options !== 'object') throw new TypeError('Offline stream options are required.');
  if (!isOfflineVideoSupported()) {
    throw new Error('Streamed H.264 MP4 rendering requires the browser WebCodecs API.');
  }
  if (!(options.writable instanceof WritableStream)) {
    throw new TypeError('writable must be a WritableStream.');
  }
  if (typeof options.renderFrame !== 'function') throw new TypeError('renderFrame must be a function.');
  assertPositiveFinite(options.durationSeconds, 'durationSeconds');

  const {
    canvas,
    preset,
    durationSeconds,
    writable,
    signal,
    prepare,
    renderFrame,
    onProgress,
  } = options;
  throwIfAborted(signal);
  const frameCount = getOfflineVideoFrameCount(durationSeconds, preset);
  const reportProgress = (
    phase: OfflineVideoProgressPhase,
    completedFrames: number,
    fraction: number,
  ): void => onProgress?.({ phase, completedFrames, totalFrames: frameCount, fraction });

  reportProgress('probing', 0, 0);
  const hardwareAcceleration = await probeExactAvcEncoder(preset);
  throwIfAborted(signal);
  reportProgress('allocating', 0, 0.002);
  await prepare?.();
  throwIfAborted(signal);
  validatePreparedCanvas(canvas, preset);

  const target = new StreamTarget(writable, {
    chunked: true,
    chunkSize: 16 * 1024 * 1024,
  });
  const output = new Output({
    format: new Mp4OutputFormat({
      fastStart: 'fragmented',
      minimumFragmentDuration: OFFLINE_VIDEO_ENCODING_POLICY.keyFrameIntervalSeconds,
    }),
    target,
  });
  const source = new VideoSampleSource({
    codec: 'avc',
    fullCodecString: preset.codecString,
    bitrate: preset.bitrate,
    bitrateMode: OFFLINE_VIDEO_ENCODING_POLICY.bitrateMode,
    latencyMode: OFFLINE_VIDEO_ENCODING_POLICY.latencyMode,
    hardwareAcceleration,
    contentHint: 'motion',
    sizeChangeBehavior: 'deny',
    alpha: 'discard',
    keyFrameInterval: OFFLINE_VIDEO_ENCODING_POLICY.keyFrameIntervalSeconds,
  });
  output.addVideoTrack(source, { frameRate: preset.frameRate });

  let cancelPromise: Promise<void> | null = null;
  const cancelOutput = (): Promise<void> => {
    cancelPromise ??= output.cancel();
    return cancelPromise;
  };
  const handleAbort = (): void => {
    void cancelOutput().catch(() => undefined);
  };
  signal?.addEventListener('abort', handleAbort, { once: true });

  try {
    await output.start();
    for (let index = 0; index < frameCount; index += 1) {
      throwIfAborted(signal);
      const timestampSeconds = index / preset.frameRate;
      const duration = (index + 1) / preset.frameRate - timestampSeconds;
      const frame = await renderFrame(index, { timestampSeconds, durationSeconds: duration });
      if (!(frame instanceof VideoFrame)) {
        throw new Error('Streamed offline renderer did not return an immutable VideoFrame.');
      }
      try {
        if (frame.codedWidth !== preset.width || frame.codedHeight !== preset.height) {
          throw new Error(
            `Streamed renderer returned ${frame.codedWidth}×${frame.codedHeight}, expected ${preset.width}×${preset.height}.`,
          );
        }
        const sample = new VideoSample(frame);
        try {
          await source.add(sample, { keyFrame: isOfflineVideoKeyFrame(index, preset) });
        } finally {
          sample.close();
        }
      } finally {
        frame.close();
      }
      reportProgress('capture', index + 1, ((index + 1) / frameCount) * 0.995);
      if ((index + 1) % 2 === 0) await yieldToEventLoop();
    }

    source.close();
    reportProgress('finalizing', frameCount, 0.997);
    await output.finalize();
    throwIfAborted(signal);
    reportProgress('finalizing', frameCount, 1);
    return { codec: 'avc', frameCount, hardwareAcceleration };
  } catch (error) {
    await cancelOutput().catch(() => undefined);
    if (signal?.aborted) throw createAbortError();
    throw error;
  } finally {
    signal?.removeEventListener('abort', handleAbort);
  }
}

async function probeExactAvcEncoder(
  preset: OfflineVideoEncodingProfile,
): Promise<OfflineVideoHardwareAcceleration> {
  for (const hardwareAcceleration of ['prefer-hardware', 'no-preference'] as const) {
    const config: VideoEncoderConfig = {
      codec: preset.codecString,
      width: preset.width,
      height: preset.height,
      bitrate: preset.bitrate,
      framerate: preset.frameRate,
      bitrateMode: OFFLINE_VIDEO_ENCODING_POLICY.bitrateMode,
      latencyMode: OFFLINE_VIDEO_ENCODING_POLICY.latencyMode,
      hardwareAcceleration,
      alpha: 'discard',
      avc: { format: 'avc' },
      contentHint: 'motion',
    };
    try {
      const support = await VideoEncoder.isConfigSupported(config);
      if (support.supported) return hardwareAcceleration;
    } catch {
      // Try the same exact codec, size, cadence, and bitrate without a hardware preference.
    }
  }
  throw new Error(
    `This browser cannot encode exact ${preset.width}×${preset.height} H.264 High Profile at ${preset.frameRate} fps. The render was not silently downscaled or slowed.`,
  );
}

async function probePlaybackCapability(
  preset: OfflineVideoEncodingProfile,
): Promise<OfflineVideoPlaybackCapability | null> {
  if (typeof navigator === 'undefined' || !navigator.mediaCapabilities?.decodingInfo) return null;
  try {
    const result = await navigator.mediaCapabilities.decodingInfo({
      type: 'file',
      video: {
        contentType: `video/mp4; codecs="${preset.codecString}"`,
        width: preset.width,
        height: preset.height,
        bitrate: preset.bitrate,
        framerate: preset.frameRate,
      },
    });
    return {
      supported: result.supported,
      smooth: result.smooth,
      powerEfficient: result.powerEfficient,
    };
  } catch {
    return null;
  }
}

function validateRenderOptions(options: RenderOfflineVideoOptions): void {
  if (!options || typeof options !== 'object') throw new TypeError('Offline video render options are required.');
  if (!isOfflineVideoSupported()) {
    throw new Error('Offline H.264 MP4 rendering requires the browser WebCodecs API.');
  }
  if (typeof options.renderFrame !== 'function') throw new TypeError('renderFrame must be a function.');
  assertPositiveFinite(options.durationSeconds, 'durationSeconds');
  if (options.durationSeconds > options.preset.maxDurationSeconds) {
    throw new RangeError(
      `durationSeconds cannot exceed ${options.preset.maxDurationSeconds} seconds for the ${options.preset.label} in-memory export.`,
    );
  }
}

function validatePreparedCanvas(
  canvas: HTMLCanvasElement | OffscreenCanvas,
  preset: OfflineVideoPreset,
): void {
  if (canvas.width !== preset.width || canvas.height !== preset.height) {
    throw new RangeError(
      `Canvas must be exactly ${preset.width}×${preset.height} after offline solver preparation.`,
    );
  }
}

interface OfflinePlaybackPreviewResult {
  blob: Blob;
  playbackCapability: OfflineVideoPlaybackCapability | null;
  verification: OfflineVideoVerification;
}

async function createOfflinePlaybackPreview(
  masterBuffer: ArrayBuffer,
  signal?: AbortSignal,
): Promise<OfflinePlaybackPreviewResult> {
  const input = new Input({
    formats: [new Mp4InputFormat()],
    source: new BufferSource(masterBuffer),
  });
  const target = new BufferTarget();
  const output = new Output({
    format: new Mp4OutputFormat({ fastStart: 'reserve' }),
    target,
  });
  let outputStarted = false;

  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error('Playback preview creation failed: the verified master has no video track.');
    const masterDuration = await track.computeDuration();
    const expectedPreviewFrameCount = getOfflineVideoFrameCount(
      masterDuration,
      OFFLINE_VIDEO_PREVIEW_PROFILE.frameRate,
    );
    const hardwareAcceleration = await probeExactAvcEncoder(OFFLINE_VIDEO_PREVIEW_PROFILE);
    const source = new VideoSampleSource({
      codec: 'avc',
      fullCodecString: OFFLINE_VIDEO_PREVIEW_PROFILE.codecString,
      bitrate: OFFLINE_VIDEO_PREVIEW_PROFILE.bitrate,
      bitrateMode: OFFLINE_VIDEO_ENCODING_POLICY.bitrateMode,
      latencyMode: OFFLINE_VIDEO_ENCODING_POLICY.latencyMode,
      hardwareAcceleration,
      contentHint: 'motion',
      sizeChangeBehavior: 'deny',
      alpha: 'discard',
      keyFrameInterval: OFFLINE_VIDEO_ENCODING_POLICY.keyFrameIntervalSeconds,
      transform: {
        width: OFFLINE_VIDEO_PREVIEW_PROFILE.width,
        height: OFFLINE_VIDEO_PREVIEW_PROFILE.height,
        fit: 'contain',
      },
    });
    output.addVideoTrack(source, {
      frameRate: OFFLINE_VIDEO_PREVIEW_PROFILE.frameRate,
      maximumPacketCount: expectedPreviewFrameCount,
    });
    await output.start();
    outputStarted = true;
    throwIfAborted(signal);

    const sink = new VideoSampleSink(track, {
      hardwareAcceleration: 'no-preference',
      optimizeForLatency: true,
    });
    let previewFrameIndex = 0;
    for await (const sample of sink.samples()) {
      try {
        throwIfAborted(signal);
        sample.setTimestamp(previewFrameIndex / OFFLINE_VIDEO_PREVIEW_PROFILE.frameRate);
        sample.setDuration(1 / OFFLINE_VIDEO_PREVIEW_PROFILE.frameRate);
        await source.add(sample, {
          keyFrame: previewFrameIndex % OFFLINE_VIDEO_PREVIEW_PROFILE.frameRate === 0,
        });
        previewFrameIndex += 1;
      } finally {
        sample.close();
      }
    }
    if (previewFrameIndex !== expectedPreviewFrameCount) {
      throw new Error(
        `Playback preview creation failed: encoded ${previewFrameIndex} frames, expected ${expectedPreviewFrameCount}.`,
      );
    }

    source.close();
    await output.finalize();
    if (!target.buffer) throw new Error('Playback preview creation failed without an output buffer.');
    const previewDuration = previewFrameIndex / OFFLINE_VIDEO_PREVIEW_PROFILE.frameRate;
    const verification = await verifyOfflineVideo(
      target.buffer,
      OFFLINE_VIDEO_PREVIEW_PROFILE,
      previewFrameIndex,
      previewDuration,
      signal,
    );
    const playbackCapability = await probePlaybackCapability(OFFLINE_VIDEO_PREVIEW_PROFILE);
    return {
      blob: new Blob([target.buffer], { type: 'video/mp4' }),
      playbackCapability,
      verification,
    };
  } catch (error) {
    if (outputStarted) await output.cancel().catch(() => undefined);
    if (signal?.aborted) throw createAbortError();
    throw error;
  } finally {
    input.dispose();
  }
}

async function verifyOfflineVideo(
  buffer: ArrayBuffer,
  preset: OfflineVideoEncodingProfile,
  expectedFrameCount: number,
  expectedDurationSeconds: number,
  signal?: AbortSignal,
): Promise<OfflineVideoVerification> {
  const input = new Input({
    formats: [new Mp4InputFormat()],
    source: new BufferSource(buffer),
  });

  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error('The finalized MP4 does not contain a video track.');
    const [codedWidth, codedHeight, codec, codecString, durationSeconds, packetStats] = await Promise.all([
      track.getCodedWidth(),
      track.getCodedHeight(),
      track.getCodec(),
      track.getCodecParameterString(),
      track.computeDuration(),
      track.computePacketStats(),
    ]);
    if (codec !== 'avc') throw new Error(`The finalized MP4 reports an unexpected ${String(codec)} video codec.`);
    if (!codecString) throw new Error('The finalized MP4 does not report an H.264 profile and level string.');

    const packetSink = new EncodedPacketSink(track);
    const timings: { timestamp: number; duration: number; type: 'key' | 'delta' }[] = [];
    let previousDecodeTimestamp = Number.NEGATIVE_INFINITY;
    let decodeOrderTimestampRegressionCount = 0;
    for await (const packet of packetSink.packets(undefined, undefined, { verifyKeyPackets: true })) {
      throwIfAborted(signal);
      if (packet.timestamp < previousDecodeTimestamp) decodeOrderTimestampRegressionCount += 1;
      previousDecodeTimestamp = packet.timestamp;
      timings.push({ timestamp: packet.timestamp, duration: packet.duration, type: packet.type });
    }
    const frameDuration = 1 / preset.frameRate;
    let maximumTimestampErrorSeconds = 0;
    let maximumDurationErrorSeconds = 0;
    timings.forEach((timing, index) => {
      maximumTimestampErrorSeconds = Math.max(
        maximumTimestampErrorSeconds,
        Math.abs(timing.timestamp - index * frameDuration),
      );
      maximumDurationErrorSeconds = Math.max(
        maximumDurationErrorSeconds,
        Math.abs(timing.duration - frameDuration),
      );
    });
    const keyFrameTimestamps = timings
      .filter((timing) => timing.type === 'key')
      .map((timing) => timing.timestamp);
    let maximumKeyFrameGapSeconds = 0;
    for (let index = 1; index < keyFrameTimestamps.length; index += 1) {
      maximumKeyFrameGapSeconds = Math.max(
        maximumKeyFrameGapSeconds,
        keyFrameTimestamps[index]! - keyFrameTimestamps[index - 1]!,
      );
    }
    const decoded = await verifyDecodedVideo(track, preset.frameRate, signal);

    const verification: OfflineVideoVerification = {
      codedWidth,
      codedHeight,
      frameCount: packetStats.packetCount,
      frameRate: packetStats.averagePacketRate,
      durationSeconds,
      averageBitrate: packetStats.averageBitrate,
      codec,
      codecString,
      cadenceFrameCount: timings.length,
      maximumTimestampErrorSeconds,
      maximumDurationErrorSeconds,
      keyFrameCount: keyFrameTimestamps.length,
      maximumKeyFrameGapSeconds,
      decodeOrderTimestampRegressionCount,
      ...decoded,
    };
    validateOfflineVideoVerification(
      verification,
      preset,
      expectedFrameCount,
      expectedDurationSeconds,
    );
    return verification;
  } finally {
    input.dispose();
  }
}

export function validateOfflineVideoVerification(
  verification: OfflineVideoVerification,
  preset: Pick<OfflineVideoEncodingProfile, 'width' | 'height' | 'frameRate' | 'codecString'>,
  expectedFrameCount: number,
  expectedDurationSeconds: number,
): void {
  if (verification.codedWidth !== preset.width || verification.codedHeight !== preset.height) {
    throw new Error(
      `MP4 verification failed: encoded ${verification.codedWidth}×${verification.codedHeight}, expected ${preset.width}×${preset.height}.`,
    );
  }
  if (verification.frameCount !== expectedFrameCount) {
    throw new Error(
      `MP4 verification failed: encoded ${verification.frameCount} frames, expected ${expectedFrameCount}.`,
    );
  }
  if (verification.cadenceFrameCount !== expectedFrameCount) {
    throw new Error(
      `MP4 verification failed: inspected cadence for ${verification.cadenceFrameCount} frames, expected ${expectedFrameCount}.`,
    );
  }
  if (verification.decodeOrderTimestampRegressionCount !== 0) {
    throw new Error(
      `MP4 verification failed: ${verification.decodeOrderTimestampRegressionCount} packet timestamps regress in decode order.`,
    );
  }
  if (verification.codec !== 'avc') {
    throw new Error(`MP4 verification failed: encoded ${verification.codec}, expected H.264/AVC.`);
  }
  if (!isCompatibleAvcEncoderOutput(verification.codecString, preset.codecString)) {
    throw new Error(
      `MP4 verification failed: encoded ${verification.codecString}, expected ${preset.codecString}.`,
    );
  }
  if (!Number.isFinite(verification.frameRate)
    || Math.abs(verification.frameRate - preset.frameRate) > 0.5) {
    throw new Error(
      `MP4 verification failed: measured ${verification.frameRate.toFixed(2)} fps, expected ${preset.frameRate} fps.`,
    );
  }
  const durationTolerance = 1 / preset.frameRate;
  if (!Number.isFinite(verification.durationSeconds)
    || Math.abs(verification.durationSeconds - expectedDurationSeconds) > durationTolerance) {
    throw new Error(
      `MP4 verification failed: measured ${verification.durationSeconds.toFixed(3)} seconds, expected ${expectedDurationSeconds.toFixed(3)} seconds.`,
    );
  }
  const cadenceTolerance = 1 / (preset.frameRate * 100);
  if (!Number.isFinite(verification.maximumTimestampErrorSeconds)
    || verification.maximumTimestampErrorSeconds > cadenceTolerance) {
    throw new Error(
      `MP4 verification failed: individual frame timestamps drift by up to ${(verification.maximumTimestampErrorSeconds * 1000).toFixed(3)} ms.`,
    );
  }
  if (!Number.isFinite(verification.maximumDurationErrorSeconds)
    || verification.maximumDurationErrorSeconds > cadenceTolerance) {
    throw new Error(
      `MP4 verification failed: individual frame durations drift by up to ${(verification.maximumDurationErrorSeconds * 1000).toFixed(3)} ms.`,
    );
  }
  const minimumKeyFrameCount = Math.max(1, Math.ceil(expectedDurationSeconds));
  if (verification.keyFrameCount < minimumKeyFrameCount) {
    throw new Error(
      `MP4 verification failed: found ${verification.keyFrameCount} verified keyframes, expected at least ${minimumKeyFrameCount}.`,
    );
  }
  if (!Number.isFinite(verification.maximumKeyFrameGapSeconds)
    || verification.maximumKeyFrameGapSeconds > 1 + durationTolerance) {
    throw new Error(
      `MP4 verification failed: keyframes are up to ${verification.maximumKeyFrameGapSeconds.toFixed(3)} seconds apart.`,
    );
  }
  if (verification.decodedFrameCount !== expectedFrameCount) {
    throw new Error(
      `MP4 verification failed: decoded ${verification.decodedFrameCount} frames, expected ${expectedFrameCount}.`,
    );
  }
  if (!Number.isFinite(verification.maximumDecodedTimestampErrorSeconds)
    || verification.maximumDecodedTimestampErrorSeconds > cadenceTolerance) {
    throw new Error(
      `MP4 verification failed: decoded frame timestamps drift by up to ${(verification.maximumDecodedTimestampErrorSeconds * 1000).toFixed(3)} ms.`,
    );
  }
  if (!Number.isFinite(verification.maximumDecodedDurationErrorSeconds)
    || verification.maximumDecodedDurationErrorSeconds > cadenceTolerance) {
    throw new Error(
      `MP4 verification failed: decoded frame durations drift by up to ${(verification.maximumDecodedDurationErrorSeconds * 1000).toFixed(3)} ms.`,
    );
  }
  const transitionCount = Math.max(1, verification.decodedFrameCount - 1);
  const exactDuplicateRatio = verification.exactDuplicateAdjacentFrameCount / transitionCount;
  if (!Number.isFinite(exactDuplicateRatio) || exactDuplicateRatio > 0.05) {
    throw new Error(
      `MP4 verification failed: ${verification.exactDuplicateAdjacentFrameCount} adjacent decoded frames are exact visual duplicates.`,
    );
  }
  const nearDuplicateRatio = verification.nearDuplicateAdjacentFrameCount / transitionCount;
  const maximumNearDuplicateRun = Math.max(2, Math.ceil(preset.frameRate * 0.05));
  if (!Number.isFinite(nearDuplicateRatio)
    || nearDuplicateRatio > 0.20
    || !Number.isFinite(verification.longestNearDuplicateTransitionRun)
    || verification.longestNearDuplicateTransitionRun > maximumNearDuplicateRun) {
    throw new Error(
      `MP4 verification failed: near-duplicate motion affects ${verification.nearDuplicateAdjacentFrameCount} decoded transitions with a longest run of ${verification.longestNearDuplicateTransitionRun} frames.`,
    );
  }
  if (!Number.isFinite(verification.discontinuousAdjacentFrameCount)
    || verification.discontinuousAdjacentFrameCount > 0) {
    throw new Error(
      `MP4 verification failed: ${verification.discontinuousAdjacentFrameCount} decoded transitions contain abrupt full-frame discontinuities.`,
    );
  }
  if (!Number.isFinite(verification.averageBitrate) || verification.averageBitrate <= 0) {
    throw new Error('MP4 verification failed: the encoded video has no measurable bitrate.');
  }
}

interface ParsedAvcCodecString {
  sampleEntry: 'avc1' | 'avc3';
  profileIdc: number;
  constraintByte: number;
  levelIdc: number;
}

function parseAvcCodecString(value: string): ParsedAvcCodecString | null {
  const match = /^(avc1|avc3)\.([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(value);
  if (!match) return null;
  return {
    sampleEntry: match[1]!.toLowerCase() as 'avc1' | 'avc3',
    profileIdc: Number.parseInt(match[2]!, 16),
    constraintByte: Number.parseInt(match[3]!, 16),
    levelIdc: Number.parseInt(match[4]!, 16),
  };
}

function isCompatibleAvcEncoderOutput(actualValue: string, requestedValue: string): boolean {
  const actual = parseAvcCodecString(actualValue);
  const requested = parseAvcCodecString(requestedValue);
  if (!actual || !requested) return false;
  if (actual.sampleEntry !== requested.sampleEntry
    || actual.profileIdc !== requested.profileIdc
    || actual.levelIdc !== requested.levelIdc) {
    return false;
  }

  // RFC 6381's middle byte contains AVC constraint flags. WebCodecs encoders
  // may add constraints to the requested profile/level, but must not emit a
  // less-constrained stream. Every preset requests the unconstrained 0x00
  // form, so Chromium's valid High L5.2 output (for example 640c34) is the
  // same requested profile and level, not a quality or cadence downgrade.
  if ((actual.constraintByte & 0x03) !== 0) return false;
  return (actual.constraintByte & requested.constraintByte) === requested.constraintByte;
}

interface DecodedVideoVerification {
  decodedFrameCount: number;
  maximumDecodedTimestampErrorSeconds: number;
  maximumDecodedDurationErrorSeconds: number;
  exactDuplicateAdjacentFrameCount: number;
  nearDuplicateAdjacentFrameCount: number;
  longestNearDuplicateTransitionRun: number;
  minimumAdjacentLumaMad: number;
  medianAdjacentLumaMad: number;
  maximumAdjacentLumaMad: number;
  discontinuousAdjacentFrameCount: number;
}

type FingerprintContext = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

function createFingerprintContext(width: number, height: number): FingerprintContext {
  if (typeof OffscreenCanvas === 'function') {
    const context = new OffscreenCanvas(width, height).getContext('2d');
    if (context) return context;
  }
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (context) return context;
  }
  throw new Error('MP4 verification failed: no 2D canvas is available for decoded-frame fingerprinting.');
}

async function verifyDecodedVideo(
  track: InputVideoTrack,
  frameRate: number,
  signal?: AbortSignal,
): Promise<DecodedVideoVerification> {
  if (!(await track.canDecode())) {
    throw new Error('MP4 verification failed: this browser cannot decode the finalized H.264 track.');
  }

  const width = 128;
  const height = 72;
  const pixelCount = width * height;
  const context = createFingerprintContext(width, height);
  const sink = new VideoSampleSink(track, {
    hardwareAcceleration: 'no-preference',
    optimizeForLatency: true,
  });
  const expectedFrameDuration = 1 / frameRate;
  let previous = new Uint8Array(pixelCount);
  let current = new Uint8Array(pixelCount);
  let hasPrevious = false;
  let previousTimestamp = Number.NEGATIVE_INFINITY;
  let decodedFrameCount = 0;
  let maximumDecodedTimestampErrorSeconds = 0;
  let maximumDecodedDurationErrorSeconds = 0;
  let exactDuplicateAdjacentFrameCount = 0;
  let nearDuplicateAdjacentFrameCount = 0;
  let currentNearDuplicateRun = 0;
  let longestNearDuplicateTransitionRun = 0;
  let minimumAdjacentLumaMad = Number.POSITIVE_INFINITY;
  const adjacentLumaMads: number[] = [];

  for await (const sample of sink.samples()) {
    try {
      throwIfAborted(signal);
      if (sample.timestamp <= previousTimestamp) {
        throw new Error(
          `MP4 verification failed: decoded frame ${decodedFrameCount} is not in increasing presentation order.`,
        );
      }
      maximumDecodedTimestampErrorSeconds = Math.max(
        maximumDecodedTimestampErrorSeconds,
        Math.abs(sample.timestamp - decodedFrameCount * expectedFrameDuration),
      );
      maximumDecodedDurationErrorSeconds = Math.max(
        maximumDecodedDurationErrorSeconds,
        Math.abs(sample.duration - expectedFrameDuration),
      );

      context.clearRect(0, 0, width, height);
      sample.draw(context, 0, 0, width, height);
      const rgba = context.getImageData(0, 0, width, height).data;
      let totalDifference = 0;
      for (let pixel = 0, offset = 0; pixel < pixelCount; pixel += 1, offset += 4) {
        const luma = (
          77 * rgba[offset]!
          + 150 * rgba[offset + 1]!
          + 29 * rgba[offset + 2]!
          + 128
        ) >> 8;
        current[pixel] = luma;
        if (hasPrevious) totalDifference += Math.abs(luma - previous[pixel]!);
      }
      if (hasPrevious) {
        const mad = totalDifference / pixelCount;
        adjacentLumaMads.push(mad);
        minimumAdjacentLumaMad = Math.min(minimumAdjacentLumaMad, mad);
        if (totalDifference === 0) exactDuplicateAdjacentFrameCount += 1;
      }
      [previous, current] = [current, previous];
      hasPrevious = true;

      previousTimestamp = sample.timestamp;
      decodedFrameCount += 1;
    } finally {
      sample.close();
    }
  }

  const sortedMads = [...adjacentLumaMads].sort((a, b) => a - b);
  const medianAdjacentLumaMad = sortedMads.length === 0
    ? 0
    : sortedMads[Math.floor(sortedMads.length / 2)]!;
  const maximumAdjacentLumaMad = sortedMads.at(-1) ?? 0;
  const nearDuplicateThreshold = Math.max(0.01, medianAdjacentLumaMad * 0.10);
  for (const mad of adjacentLumaMads) {
    if (mad <= nearDuplicateThreshold) {
      nearDuplicateAdjacentFrameCount += 1;
      currentNearDuplicateRun += 1;
      longestNearDuplicateTransitionRun = Math.max(
        longestNearDuplicateTransitionRun,
        currentNearDuplicateRun,
      );
    } else {
      currentNearDuplicateRun = 0;
    }
  }
  const discontinuityThreshold = Math.max(8, medianAdjacentLumaMad * 20);
  const discontinuousAdjacentFrameCount = adjacentLumaMads
    .filter((mad) => mad > discontinuityThreshold)
    .length;

  return {
    decodedFrameCount,
    maximumDecodedTimestampErrorSeconds,
    maximumDecodedDurationErrorSeconds,
    exactDuplicateAdjacentFrameCount,
    nearDuplicateAdjacentFrameCount,
    longestNearDuplicateTransitionRun,
    minimumAdjacentLumaMad: Number.isFinite(minimumAdjacentLumaMad) ? minimumAdjacentLumaMad : 0,
    medianAdjacentLumaMad,
    maximumAdjacentLumaMad,
    discontinuousAdjacentFrameCount,
  };
}

function assertPositiveFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be a positive finite number.`);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw createAbortError();
}

function createAbortError(): Error {
  if (typeof DOMException === 'function') {
    return new DOMException('Offline MP4 render canceled.', 'AbortError');
  }
  const error = new Error('Offline MP4 render canceled.');
  error.name = 'AbortError';
  return error;
}

function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
