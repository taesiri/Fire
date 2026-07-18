/**
 * Splits one offline output-frame interval into exact, positive solver steps.
 * Interactive fixed-step clocks deliberately do not use this path.
 */
export function splitOfflineSimulationDelta(
  deltaSeconds: number,
  maximumStepSeconds: number,
): readonly number[] {
  if (!Number.isFinite(deltaSeconds) || deltaSeconds < 0) {
    throw new RangeError('Offline frame delta must be finite and non-negative.');
  }
  if (!Number.isFinite(maximumStepSeconds) || maximumStepSeconds <= 0) {
    throw new RangeError('Offline maximum step must be positive and finite.');
  }
  if (deltaSeconds === 0) return [];

  const stepCount = Math.max(1, Math.ceil(deltaSeconds / maximumStepSeconds - 1e-12));
  const stepSeconds = deltaSeconds / stepCount;
  return Object.freeze(Array.from({ length: stepCount }, () => stepSeconds));
}
