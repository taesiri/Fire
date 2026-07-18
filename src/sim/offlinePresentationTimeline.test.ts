import { describe, expect, it } from 'vitest';
import { FireXEngine } from './firex/FireXEngine';
import { HorvathEngine } from './horvath/HorvathEngine';

describe('offline presentation timeline', () => {
  it('advances the Horvath solver before every 60 fps capture', async () => {
    const simulationSteps: number[] = [];
    const presentedTimes: number[] = [];
    const gl = {
      flush: () => undefined,
      finish: () => undefined,
      isContextLost: () => false,
    };
    const engine = Object.assign(Object.create(HorvathEngine.prototype), {
      accumulator: 0,
      simulationTime: 0,
      contextLost: false,
      requireOfflineGl: () => gl,
      requireResources: () => ({ fixedTimeStep: 1 / 120 }),
      stepSimulation(timeStep: number) {
        simulationSteps.push(timeStep);
      },
      renderVolume() {
        presentedTimes.push(this.simulationTime);
      },
    }) as HorvathEngine;

    // Warm-up must advance state without putting a stale frame on the canvas.
    await engine.renderOfflineFrame(1 / 60, false);
    for (let frame = 0; frame < 3; frame += 1) {
      await engine.renderOfflineFrame(1 / 60, true);
    }

    expect(simulationSteps).toHaveLength(8);
    expect(simulationSteps.every((step) => step === 1 / 120)).toBe(true);
    expect(presentedTimes).toHaveLength(3);
    expect(presentedTimes[0]).toBeCloseTo(2 / 60, 12);
    expect(presentedTimes[1]).toBeCloseTo(3 / 60, 12);
    expect(presentedTimes[2]).toBeCloseTo(4 / 60, 12);
    expect(new Set(presentedTimes.map((time) => time.toFixed(12))).size).toBe(3);
  });

  it('advances the Fire-X solver before every 60 fps capture', async () => {
    const simulationSteps: number[] = [];
    const presentedTimes: number[] = [];
    const submittedCommandBuffers: unknown[][] = [];
    const device = {
      createCommandEncoder: () => ({ finish: () => ({}) }),
      queue: {
        submit: (buffers: unknown[]) => submittedCommandBuffers.push(buffers),
        onSubmittedWorkDone: async () => undefined,
      },
    };
    const engine = Object.assign(Object.create(FireXEngine.prototype), {
      resourceMutationQueue: Promise.resolve(),
      device,
      deviceLost: false,
      offlineRenderTier: 'hd',
      simulationTime: 0,
      stateEpoch: 0,
      frameIndex: 0,
      clock: { reset: () => undefined },
      requireOfflineDevice: () => device,
      activeConfiguration: () => ({ timeStep: 1 / 60 }),
      throwIfOfflineAborted: () => undefined,
      writeUniforms: () => undefined,
      submitOfflineSimulationStep() {
        simulationSteps.push(1 / 60);
        return null;
      },
      encodeRender() {
        presentedTimes.push(this.simulationTime);
      },
    }) as FireXEngine;

    await engine.renderOfflineFrame(1 / 60, false);
    for (let frame = 0; frame < 3; frame += 1) {
      await engine.renderOfflineFrame(1 / 60, true);
    }

    expect(simulationSteps).toHaveLength(4);
    expect(presentedTimes).toHaveLength(3);
    expect(presentedTimes[0]).toBeCloseTo(2 / 60, 12);
    expect(presentedTimes[1]).toBeCloseTo(3 / 60, 12);
    expect(presentedTimes[2]).toBeCloseTo(4 / 60, 12);
    expect(new Set(presentedTimes.map((time) => time.toFixed(12))).size).toBe(3);
    expect(submittedCommandBuffers).toHaveLength(3);
  });

  it('prepares fused Beauty volumes only on the final requested offline-video substep', async () => {
    const fusionRequests: boolean[] = [];
    const resources = {};
    const device = {
      queue: { onSubmittedWorkDone: async () => undefined },
    };
    const engine = Object.assign(Object.create(FireXEngine.prototype), {
      resourceMutationQueue: Promise.resolve(),
      device,
      deviceLost: false,
      offlineRenderTier: 'uhd',
      resources,
      simulationTime: 0,
      stateEpoch: 0,
      speciesIndex: 0,
      frameIndex: 0,
      clock: { reset: () => undefined },
      requireOfflineDevice: () => device,
      activeConfiguration: () => ({ timeStep: 1 / 60 }),
      throwIfOfflineAborted: () => undefined,
      writeUniforms: () => undefined,
      submitOfflineSimulationStep(_signal?: AbortSignal, prepareBeautyVolumes = false) {
        fusionRequests.push(prepareBeautyVolumes);
        return prepareBeautyVolumes
          ? { resources, stateEpoch: this.stateEpoch, speciesIndex: this.speciesIndex }
          : null;
      },
    }) as FireXEngine;
    const internal = engine as unknown as {
      renderOfflineFrameInternal(
        deltaSeconds: number,
        present: boolean,
        signal: AbortSignal | undefined,
        prepareBeautyVolumes: boolean,
      ): Promise<{ resources: object; stateEpoch: number; speciesIndex: 0 | 1 } | null>;
    };

    const prepared = await internal.renderOfflineFrameInternal(2 / 60, false, undefined, true);
    expect(fusionRequests).toEqual([false, true]);
    expect(prepared).toEqual({ resources, stateEpoch: 2, speciesIndex: 0 });

    fusionRequests.length = 0;
    await engine.renderOfflineFrame(1 / 60, false);
    expect(fusionRequests).toEqual([false]);
  });

  it('validates packed-volume handoffs and refreshes divergence at presentation time', () => {
    const resources = {
      dimensions: [4, 4, 4],
      projectedDivergenceGroup: {},
      packGroups: [{}, {}],
      renderGroups: [{}, {}],
      particleCount: 0,
    };
    const computeLabels: string[] = [];
    const computePass = {
      setPipeline: () => undefined,
      setBindGroup: () => undefined,
      dispatchWorkgroups: () => undefined,
      end: () => undefined,
    };
    const renderPass = {
      setPipeline: () => undefined,
      setBindGroup: () => undefined,
      draw: () => undefined,
      end: () => undefined,
    };
    const encoder = {
      beginComputePass: ({ label }: { label: string }) => {
        computeLabels.push(label);
        return computePass;
      },
      beginRenderPass: () => renderPass,
    };
    const engine = Object.assign(Object.create(FireXEngine.prototype), {
      context: {},
      resources,
      pipelines: {
        divergence: {},
        pack: {},
        render: {},
        particleRender: {},
      },
      parameters: { viewMode: 0 },
      stateEpoch: 10,
      speciesIndex: 0,
      particleIndex: 0,
      particleFieldActive: false,
      requireResources: () => resources,
      requirePipelines() { return this.pipelines; },
      usesCinematicPresentation: () => false,
    }) as FireXEngine;
    const privateEngine = engine as unknown as {
      stateEpoch: number;
      parameters: { viewMode: number };
      encodeRender(
        encoder: object,
        target: { createView(): object },
        prepared?: { resources: object; stateEpoch: number; speciesIndex: 0 | 1 } | null,
      ): void;
    };
    const target = { createView: () => ({}) };

    privateEngine.encodeRender(encoder, target, null);
    expect(computeLabels).toEqual(['Fire-X pack filterable render fields']);
    expect(privateEngine.stateEpoch).toBe(11);

    computeLabels.length = 0;
    const current = { resources, stateEpoch: 11, speciesIndex: 0 as const };
    privateEngine.encodeRender(encoder, target, current);
    expect(computeLabels).toEqual([]);
    expect(privateEngine.stateEpoch).toBe(11);

    computeLabels.length = 0;
    privateEngine.parameters.viewMode = 7;
    privateEngine.encodeRender(encoder, target, current);
    expect(computeLabels).toEqual([
      'Fire-X presentation final divergence',
      'Fire-X pack filterable render fields',
    ]);
    expect(privateEngine.stateEpoch).toBe(12);
  });
});
