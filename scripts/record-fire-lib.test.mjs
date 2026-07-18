import { describe, expect, it } from 'vitest';
import {
  createDryRunPlan,
  estimateGridMemory,
  estimateOutputBytes,
  isHeavyJob,
  parseArguments,
  parseDuration,
  parseGrid,
} from './record-fire-lib.mjs';

describe('record-fire CLI planning', () => {
  it('plans exact 30 second and one hour frame counts', () => {
    expect(parseDuration('30s') * 60).toBe(1800);
    expect(parseDuration('1h') * 60).toBe(216_000);
    expect(estimateOutputBytes('uhd', 3600)).toBe(29_250_000_000);
  });

  it('rejects durations that cannot be represented by an exact 60 fps frame count', () => {
    expect(parseDuration('100ms')).toBe(0.1);
    expect(() => parseDuration('101ms')).toThrow(/frame boundary/i);
    expect(() => parseArguments(['--warmup', '17ms'])).toThrow(/frame boundary/i);
    expect(() => parseDuration('61m')).toThrow(/one hour/i);
    expect(() => parseDuration(3601)).toThrow(/one hour/i);
  });

  it('normalizes accepted fractional-frame text to one integral frame count', () => {
    const [plan] = createDryRunPlan(parseArguments(['--duration', '516.6666666666667ms']));
    expect(plan.frames).toBe(31);
    expect(plan.durationSeconds).toBe(31 / 60);
  });

  it('builds the explicit highest-quality plan without changing the requested grid', () => {
    const options = parseArguments(['--quality', 'highest', '--hour', '--field', 'beauty', '--confirm-heavy']);
    const [plan] = createDryRunPlan(options, 'D:\\capture');
    expect(options.effectiveDimensions).toEqual([508, 508, 508]);
    expect(options.gridMemoryBudgetBytes).toBe(32 * 1024 ** 3);
    expect(plan.frames).toBe(216_000);
    expect(plan.estimatedVideoBytes).toBe(29_250_000_000);
    expect(isHeavyJob(options)).toBe(true);
  });

  it('expands the multi-field 30 second gate', () => {
    const options = parseArguments(['--test-views']);
    expect(options.fields).toEqual(['beauty', 'temperature', 'reaction', 'vapor-soot']);
    expect(options.durationSeconds).toBe(30);
    expect(options.outputPreset).toBe('hd');
    expect(createDryRunPlan(options).every((job) => job.frames === 1800)).toBe(true);
  });

  it('rejects structurally impossible dense grids instead of reducing them', () => {
    expect(() => parseGrid('1024x1024x1024')).toThrow(/above.*2 GiB/i);
    expect(() => parseGrid('255x256x256')).toThrow(/multiple of four/i);
  });

  it('matches the engine memory formula for Reference 508 cubed', () => {
    const estimate = estimateGridMemory([508, 508, 508], 4096);
    expect(estimate.cellCount).toBe(131_096_512);
    expect(estimate.totalBytes / 1024 ** 3).toBeCloseTo(13.187, 2);
  });

  it('uses the full normal layout below 64 Mi cells and the compact heavy layout at it', () => {
    const belowThreshold = estimateGridMemory([1020, 256, 256], 0);
    const atThreshold = estimateGridMemory([1024, 256, 256], 0);
    expect(belowThreshold.gridBytes).toBe(belowThreshold.cellCount * 116);
    expect(atThreshold.cellCount).toBe(64 * 1024 * 1024);
    expect(atThreshold.gridBytes).toBe(atThreshold.cellCount * 108);
  });
});
