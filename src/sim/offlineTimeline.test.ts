import { describe, expect, it } from 'vitest';
import { splitOfflineSimulationDelta } from './offlineTimeline';

describe('offline simulation timeline', () => {
  it('advances every 60 fps output sample with one real 1/60 solver step', () => {
    const samples = Array.from(
      { length: 60 },
      () => splitOfflineSimulationDelta(1 / 60, 1 / 30),
    );

    expect(samples.every((steps) => steps.length === 1 && steps[0] > 0)).toBe(true);
    expect(samples.flat().reduce((total, step) => total + step, 0)).toBeCloseTo(1, 12);
  });

  it('splits slower output intervals without exceeding the native stability step', () => {
    const steps = splitOfflineSimulationDelta(1 / 15, 1 / 30);
    expect(steps).toHaveLength(2);
    expect(Math.max(...steps)).toBeLessThanOrEqual(1 / 30);
    expect(steps.reduce((total, step) => total + step, 0)).toBeCloseTo(1 / 15, 12);
  });

  it('renders without advancing for a zero delta and rejects invalid inputs', () => {
    expect(splitOfflineSimulationDelta(0, 1 / 30)).toEqual([]);
    expect(() => splitOfflineSimulationDelta(-1, 1 / 30)).toThrow(RangeError);
    expect(() => splitOfflineSimulationDelta(1 / 60, 0)).toThrow(RangeError);
  });
});
