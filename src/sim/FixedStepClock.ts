export class FixedStepClock {
  private accumulator = 0;

  constructor(
    readonly stepSeconds: number,
    readonly maximumStepsPerFrame: number,
  ) {
    if (!Number.isFinite(stepSeconds) || !(stepSeconds > 0)) {
      throw new Error('Fixed step must be finite and positive.');
    }
    if (!Number.isInteger(maximumStepsPerFrame) || !(maximumStepsPerFrame > 0)) {
      throw new Error('Maximum step count must be a positive integer.');
    }
  }

  advance(deltaSeconds: number, step: (dt: number) => void): number {
    if (!Number.isFinite(deltaSeconds)) return 0;
    const boundedDelta = Math.max(
      0,
      Math.min(deltaSeconds, this.stepSeconds * this.maximumStepsPerFrame),
    );
    this.accumulator += boundedDelta;
    let count = 0;
    while (this.accumulator >= this.stepSeconds && count < this.maximumStepsPerFrame) {
      step(this.stepSeconds);
      this.accumulator -= this.stepSeconds;
      count += 1;
    }
    if (count === this.maximumStepsPerFrame && this.accumulator >= this.stepSeconds) {
      this.accumulator %= this.stepSeconds;
    }
    return count;
  }

  reset(): void {
    this.accumulator = 0;
  }
}
