import { describe, expect, it } from 'vitest';
import {
  OFFLINE_VIDEO_ENCODING_POLICY,
  OFFLINE_VIDEO_PREVIEW_PROFILE,
  OFFLINE_VIDEO_PRESETS,
  estimateOfflineVideoBytes,
  getOfflineVideoFrameCount,
  getOfflineVideoPreset,
  isOfflineVideoKeyFrame,
  isOfflineVideoSupported,
  validateOfflineVideoVerification,
  type OfflineVideoVerification,
} from './offlineVideo';

describe('offline H.264 MP4 presets', () => {
  it('defines immutable Full HD, QHD, and 4K budgets at true 60 fps', () => {
    expect(OFFLINE_VIDEO_PRESETS).toEqual([
      {
        id: 'hd',
        label: 'Full HD · 1080p60',
        width: 1920,
        height: 1080,
        frameRate: 60,
        bitrate: 24_000_000,
        maxDurationSeconds: 10,
        offlineTier: 'hd',
        codecString: 'avc1.64002a',
        codecLabel: 'H.264 High L4.2',
      },
      {
        id: 'qhd',
        label: 'QHD · 1440p60',
        width: 2560,
        height: 1440,
        frameRate: 60,
        bitrate: 40_000_000,
        maxDurationSeconds: 7,
        offlineTier: 'qhd',
        codecString: 'avc1.640033',
        codecLabel: 'H.264 High L5.1',
      },
      {
        id: 'uhd',
        label: '4K UHD · 2160p60',
        width: 3840,
        height: 2160,
        frameRate: 60,
        bitrate: 65_000_000,
        maxDurationSeconds: 5,
        offlineTier: 'uhd',
        codecString: 'avc1.640034',
        codecLabel: 'H.264 High L5.2',
      },
    ]);
    expect(Object.isFrozen(OFFLINE_VIDEO_PRESETS)).toBe(true);
    expect(OFFLINE_VIDEO_PRESETS.every(Object.isFrozen)).toBe(true);
    expect(OFFLINE_VIDEO_PRESETS.every((preset) => preset.frameRate === 60)).toBe(true);
  });

  it('looks up presets without creating mutable copies', () => {
    expect(getOfflineVideoPreset('qhd')).toBe(OFFLINE_VIDEO_PRESETS[1]);
    expect(() => getOfflineVideoPreset('unknown')).toThrow(RangeError);
  });
});

describe('offline MP4 frame and memory budgets', () => {
  it('derives deterministic 60 fps frame counts', () => {
    expect(getOfflineVideoFrameCount(1, OFFLINE_VIDEO_PRESETS[0])).toBe(60);
    expect(getOfflineVideoFrameCount(5, OFFLINE_VIDEO_PRESETS[2])).toBe(300);
    expect(getOfflineVideoFrameCount(2.5, 60)).toBe(150);
  });

  it('rejects invalid frame-count inputs', () => {
    expect(() => getOfflineVideoFrameCount(0, OFFLINE_VIDEO_PRESETS[0])).toThrow(RangeError);
    expect(() => getOfflineVideoFrameCount(1, Number.NaN)).toThrow(RangeError);
    expect(() => getOfflineVideoFrameCount(0.101, 60)).toThrow(/frame boundary/);
  });

  it('uses decoder-friendly bitrate caps and keeps each target below 100 MB', () => {
    expect(estimateOfflineVideoBytes(OFFLINE_VIDEO_PRESETS[0], 10)).toBe(30_000_000);
    expect(estimateOfflineVideoBytes(OFFLINE_VIDEO_PRESETS[1], 7)).toBe(35_000_000);
    expect(estimateOfflineVideoBytes(OFFLINE_VIDEO_PRESETS[2], 5)).toBe(40_625_000);
    for (const preset of OFFLINE_VIDEO_PRESETS) {
      expect(estimateOfflineVideoBytes(preset, preset.maxDurationSeconds)).toBeLessThan(100_000_000);
    }
  });

  it('can be queried in a non-browser test environment', () => {
    expect(typeof isOfflineVideoSupported()).toBe('boolean');
  });
});

describe('offline MP4 playback-safe encoder policy', () => {
  it('uses realtime variable-rate H.264 with one-second random access', () => {
    expect(OFFLINE_VIDEO_ENCODING_POLICY).toEqual({
      bitrateMode: 'variable',
      latencyMode: 'realtime',
      keyFrameIntervalSeconds: 1,
    });
    expect(Object.isFrozen(OFFLINE_VIDEO_ENCODING_POLICY)).toBe(true);
  });

  it('defines a decoder-light 720p60 proof without discarding master frames', () => {
    expect(OFFLINE_VIDEO_PREVIEW_PROFILE).toEqual({
      label: 'Playback proof · 720p60',
      width: 1280,
      height: 720,
      frameRate: 60,
      bitrate: 12_000_000,
      codecString: 'avc1.640029',
    });
    expect(Object.isFrozen(OFFLINE_VIDEO_PREVIEW_PROFILE)).toBe(true);
    expect(OFFLINE_VIDEO_PRESETS.every((preset) => preset.frameRate === 60)).toBe(true);
  });

  it('requests a keyframe at frame zero and once per second', () => {
    const hd = OFFLINE_VIDEO_PRESETS[0];
    expect(isOfflineVideoKeyFrame(0, hd)).toBe(true);
    expect(isOfflineVideoKeyFrame(59, hd)).toBe(false);
    expect(isOfflineVideoKeyFrame(60, hd)).toBe(true);
    expect(isOfflineVideoKeyFrame(120, hd)).toBe(true);
    expect(() => isOfflineVideoKeyFrame(-1, hd)).toThrow(RangeError);
  });
});

describe('offline MP4 verification', () => {
  const qhd = OFFLINE_VIDEO_PRESETS[1];
  const verified: OfflineVideoVerification = {
    codedWidth: 2560,
    codedHeight: 1440,
    frameCount: 300,
    frameRate: 60,
    durationSeconds: 5,
    averageBitrate: 38_000_000,
    codec: 'avc',
    codecString: 'avc1.640033',
    cadenceFrameCount: 300,
    maximumTimestampErrorSeconds: 0,
    maximumDurationErrorSeconds: 0,
    keyFrameCount: 5,
    maximumKeyFrameGapSeconds: 1,
    decodeOrderTimestampRegressionCount: 0,
    decodedFrameCount: 300,
    maximumDecodedTimestampErrorSeconds: 0,
    maximumDecodedDurationErrorSeconds: 0,
    exactDuplicateAdjacentFrameCount: 0,
    nearDuplicateAdjacentFrameCount: 0,
    longestNearDuplicateTransitionRun: 0,
    minimumAdjacentLumaMad: 1,
    medianAdjacentLumaMad: 1.4,
    maximumAdjacentLumaMad: 2,
    discontinuousAdjacentFrameCount: 0,
  };

  it('accepts the exact coded size, timeline, duration, codec, and per-frame cadence', () => {
    expect(() => validateOfflineVideoVerification(verified, qhd, 300, 5)).not.toThrow();
  });

  it('accepts additional AVC constraint flags at the requested High profile and level', () => {
    const uhd = OFFLINE_VIDEO_PRESETS[2];
    expect(() => validateOfflineVideoVerification(
      {
        ...verified,
        codedWidth: uhd.width,
        codedHeight: uhd.height,
        codecString: 'avc1.640c34',
      },
      uhd,
      300,
      5,
    )).not.toThrow();
    expect(() => validateOfflineVideoVerification(
      { ...verified, codecString: 'avc1.640c33' },
      { ...qhd, codecString: 'avc1.640833' },
      300,
      5,
    )).not.toThrow();
  });

  it('rejects a silently downscaled or incomplete file', () => {
    expect(() => validateOfflineVideoVerification(
      { ...verified, codedWidth: 1920 },
      qhd,
      300,
      5,
    )).toThrow(/encoded 1920×1440/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, frameCount: 299 },
      qhd,
      300,
      5,
    )).toThrow(/299 frames/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, cadenceFrameCount: 299 },
      qhd,
      300,
      5,
    )).toThrow(/cadence for 299 frames/);
  });

  it('rejects average cadence, duration, and individual packet drift', () => {
    expect(() => validateOfflineVideoVerification(
      { ...verified, codecString: 'avc1.4d0033' },
      qhd,
      300,
      5,
    )).toThrow(/avc1\.4d0033/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, codecString: 'avc1.640034' },
      qhd,
      300,
      5,
    )).toThrow(/avc1\.640034/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, codecString: 'avc3.640c33' },
      qhd,
      300,
      5,
    )).toThrow(/avc3\.640c33/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, codecString: 'avc1.640133' },
      qhd,
      300,
      5,
    )).toThrow(/avc1\.640133/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, codecString: 'avc1.640833' },
      { ...qhd, codecString: 'avc1.640c33' },
      300,
      5,
    )).toThrow(/avc1\.640833/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, frameRate: 30 },
      qhd,
      300,
      5,
    )).toThrow(/30\.00 fps/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, durationSeconds: 4.8 },
      qhd,
      300,
      5,
    )).toThrow(/4\.800 seconds/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, maximumTimestampErrorSeconds: 0.001 },
      qhd,
      300,
      5,
    )).toThrow(/timestamps drift/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, maximumDurationErrorSeconds: 0.001 },
      qhd,
      300,
      5,
    )).toThrow(/durations drift/);
  });

  it('rejects unsafe decode order, sparse keyframes, and incomplete decoding', () => {
    expect(() => validateOfflineVideoVerification(
      { ...verified, decodeOrderTimestampRegressionCount: 1 },
      qhd,
      300,
      5,
    )).toThrow(/regress in decode order/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, keyFrameCount: 4 },
      qhd,
      300,
      5,
    )).toThrow(/4 verified keyframes/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, decodedFrameCount: 299 },
      qhd,
      300,
      5,
    )).toThrow(/decoded 299 frames/);
  });

  it('rejects decoded cadence drift and excessive exact visual duplicates', () => {
    expect(() => validateOfflineVideoVerification(
      { ...verified, maximumDecodedTimestampErrorSeconds: 0.001 },
      qhd,
      300,
      5,
    )).toThrow(/decoded frame timestamps drift/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, maximumDecodedDurationErrorSeconds: 0.001 },
      qhd,
      300,
      5,
    )).toThrow(/decoded frame durations drift/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, exactDuplicateAdjacentFrameCount: 16 },
      qhd,
      300,
      5,
    )).toThrow(/exact visual duplicates/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, nearDuplicateAdjacentFrameCount: 61 },
      qhd,
      300,
      5,
    )).toThrow(/near-duplicate motion/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, nearDuplicateAdjacentFrameCount: 4, longestNearDuplicateTransitionRun: 4 },
      qhd,
      300,
      5,
    )).toThrow(/near-duplicate motion/);
    expect(() => validateOfflineVideoVerification(
      { ...verified, discontinuousAdjacentFrameCount: 1 },
      qhd,
      300,
      5,
    )).toThrow(/abrupt full-frame discontinuities/);
  });
});
