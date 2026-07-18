import { describe, expect, it } from 'vitest';
import { FixedStepClock } from './FixedStepClock';

describe('FixedStepClock', () => {
  it('emits deterministic paper-scale steps independent of render cadence', () => {
    const clock = new FixedStepClock(1 / 120, 8);
    let elapsed = 0;
    const step = (dt: number): void => {
      elapsed += dt;
    };
    clock.advance(1 / 60, step);
    clock.advance(1 / 60, step);
    expect(elapsed).toBeCloseTo(1 / 30, 8);
  });

  it('bounds catch-up work after a long suspended frame', () => {
    const clock = new FixedStepClock(1 / 120, 4);
    let count = 0;
    expect(clock.advance(10, () => { count += 1; })).toBe(4);
    expect(count).toBe(4);
  });

  it('rejects invalid clock configurations and ignores non-finite deltas', () => {
    expect(() => new FixedStepClock(Number.NaN, 4)).toThrow();
    expect(() => new FixedStepClock(1 / 120, 1.5)).toThrow();
    const clock = new FixedStepClock(1 / 120, 4);
    expect(clock.advance(Number.POSITIVE_INFINITY, () => undefined)).toBe(0);
  });
});
