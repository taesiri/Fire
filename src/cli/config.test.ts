import { describe, expect, it } from 'vitest';
import { parseFireCliCaptureConfig } from './config';

const validConfig = {
  schemaVersion: 1,
  jobId: 'inferno-beauty-test',
  method: 'firex',
  scene: 'inferno',
  fieldView: 'beauty',
  outputPreset: 'hd',
  solverTier: 'hd',
  durationSeconds: 30,
  warmupSeconds: 1,
  gridDimensions: null,
  gridMemoryBudgetBytes: 2 * 1024 ** 3,
  opticalDetailTarget: 1024,
  camera: { yaw: -0.34, pitch: 0.03, distance: 4.9 },
};

describe('Fire CLI capture configuration', () => {
  it('accepts the deterministic 30 second smoke-test contract', () => {
    const parsed = parseFireCliCaptureConfig(validConfig);
    expect(parsed.scene).toBe('inferno');
    expect(parsed.fieldView).toBe('beauty');
    expect(parsed.durationSeconds * 60).toBe(1800);
    expect(parsed.gridDimensions).toBeNull();
  });

  it('accepts exact aligned Reference dimensions without rounding', () => {
    const parsed = parseFireCliCaptureConfig({
      ...validConfig,
      outputPreset: 'uhd',
      solverTier: 'uhd',
      durationSeconds: 3600,
      gridDimensions: [508, 508, 508],
      gridMemoryBudgetBytes: 32 * 1024 ** 3,
    });
    expect(parsed.gridDimensions).toEqual([508, 508, 508]);
    expect(parsed.durationSeconds * 60).toBe(216_000);
  });

  it('rejects unknown views and unaligned grid requests', () => {
    expect(() => parseFireCliCaptureConfig({ ...validConfig, fieldView: 'smoke' })).toThrow(/Unknown Fire-X field view/);
    expect(() => parseFireCliCaptureConfig({ ...validConfig, gridDimensions: [255, 256, 256] })).toThrow(/multiple of 4/);
  });

  it('rejects a method fallback or unsupported schema', () => {
    expect(() => parseFireCliCaptureConfig({ ...validConfig, method: 'horvath' })).toThrow(/firex/);
    expect(() => parseFireCliCaptureConfig({ ...validConfig, schemaVersion: 2 })).toThrow(/schema version/);
  });

  it('rejects non-frame-aligned capture and warmup durations', () => {
    expect(() => parseFireCliCaptureConfig({ ...validConfig, durationSeconds: 0.101 })).toThrow(/frame boundary/);
    expect(() => parseFireCliCaptureConfig({ ...validConfig, warmupSeconds: 0.101 })).toThrow(/frame boundary/);
    expect(() => parseFireCliCaptureConfig({ ...validConfig, durationSeconds: 3601 })).toThrow(/one hour/);
  });
});
