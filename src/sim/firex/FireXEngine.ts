import { FixedStepClock } from '../FixedStepClock';
import { splitOfflineSimulationDelta } from '../offlineTimeline';
import type {
  OfflineRenderTier,
  QualityLevel,
  SimulationEngine,
  SimulationGridCapabilities,
  SimulationGridInfo,
} from '../types';
import { fitCanvasSize } from '../canvasSizing';
import { FIREX_COMPUTE_WGSL, FIREX_PARTICLE_WGSL } from './shaders';
import { FIREX_PACK_WGSL, FIREX_POST_WGSL, FIREX_QUALITY_RENDER_WGSL } from './qualityShaders';
import {
  deriveFireXGridInfo,
  estimateFireXGridMemory,
  FIREX_MAX_GRID_MEMORY_BUDGET_BYTES,
  FIREX_HEAVY_GRID_CELL_THRESHOLD,
  formatFireXBytes,
  formatFireXGridInfo,
  maximumFireXDeviceDenseCubeAxis,
  preflightFireXGrid,
  selectFireXRequiredDeviceLimits,
  type FireXGridDimensions,
} from './gridConfiguration';
import { FIREX_F16_PRESSURE_WGSL } from './pressureShaders';

export interface QualityConfiguration {
  dimensions: readonly [number, number, number];
  pressureIterations: number;
  correctionIterations: number;
  maximumSubsteps: number;
  timeStep: number;
  raySteps: number;
  renderScale: number;
  particleCount: number;
}

export interface FireXOfflineConfiguration extends QualityConfiguration {
  minimumMemoryGuardBytes: number;
  opticalDetailTarget: number;
}

export interface FireXOfflinePerformanceInfo {
  readonly measuredFrames: number;
  readonly averageSimulationMilliseconds: number;
  readonly averagePresentationMilliseconds: number;
  readonly averageVideoFrameConstructionMilliseconds: number;
  readonly averageTotalMilliseconds: number;
}

interface FireXParameters {
  burnerSize: number;
  burnerDepth: number;
  sourceThickness: number;
  firePower: number;
  sourceLift: number;
  buoyancyScale: number;
  flamePersistence: number;
  airEntrainment: number;
  fuelRate: number;
  oxygenRate: number;
  heatEfficiency: number;
  vorticity: number;
  waterFlow: number;
  sprayAngle: number;
  waterEnabled: boolean;
  cameraYaw: number;
  cameraPitch: number;
  cameraDistance: number;
  viewMode: number;
  aimHeight: number;
  nozzleType: number;
  exposure: number;
  frontDefinition: number;
  domainWidth: number;
  domainHeight: number;
  domainDepth: number;
  viewZoom: number;
  opticalDetailTarget: number;
}

interface PipelineSet {
  initialize: GPUComputePipeline;
  simulate: GPUComputePipeline;
  divergence: GPUComputePipeline;
  pressureGlobal: GPUComputePipeline;
  pressureTiled: GPUComputePipeline;
  pressureF16Tiled: GPUComputePipeline | null;
  project: GPUComputePipeline;
  projectF16: GPUComputePipeline | null;
  copyFlow: GPUComputePipeline;
  particleDensity: GPUComputePipeline;
  particleIntegrate: GPUComputePipeline;
  particleDeposit: GPUComputePipeline;
  pack: GPUComputePipeline;
  copyAndPackBeauty: GPUComputePipeline;
  render: GPURenderPipeline;
  particleRender: GPURenderPipeline;
  cinematicRender: GPURenderPipeline;
  cinematicParticleRender: GPURenderPipeline;
  bloom: GPURenderPipeline;
  composite: GPURenderPipeline;
}

interface PresentationResources {
  width: number;
  height: number;
  sceneTexture: GPUTexture;
  sceneView: GPUTextureView;
  bloomTexture: GPUTexture;
  bloomView: GPUTextureView;
  bloomGroup: GPUBindGroup;
  compositeGroup: GPUBindGroup;
}

interface ResourceSet {
  dimensions: readonly [number, number, number];
  cellCount: number;
  flow: readonly [GPUBuffer, GPUBuffer];
  species: readonly [GPUBuffer, GPUBuffer];
  pressure: readonly [GPUBuffer, GPUBuffer];
  correctionPressure: readonly [GPUBuffer, GPUBuffer];
  divergence: GPUBuffer;
  particleCount: number;
  particles: readonly [GPUBuffer, GPUBuffer];
  particleDensity: GPUBuffer;
  liquid: GPUBuffer;
  reaction: GPUBuffer;
  flowTexture: GPUTexture;
  chemistryTexture: GPUTexture;
  mediaTexture: GPUTexture;
  detailTexture: GPUTexture;
  usesF16Pressure: boolean;
  initializationGroup: GPUBindGroup;
  simulationGroups: readonly [GPUBindGroup, GPUBindGroup];
  divergenceGroup: GPUBindGroup;
  projectedDivergenceGroup: GPUBindGroup;
  pressureGroups: readonly [GPUBindGroup, GPUBindGroup];
  correctionPressureGroups: readonly [GPUBindGroup, GPUBindGroup];
  projectionGroups: readonly [GPUBindGroup, GPUBindGroup];
  correctionProjectionGroups: readonly [GPUBindGroup, GPUBindGroup];
  sharedCorrectionProjectionGroups: readonly [GPUBindGroup, GPUBindGroup];
  f16PrimaryPressureGroups: readonly [GPUBindGroup, GPUBindGroup] | null;
  f16CorrectionPressureGroups: readonly [GPUBindGroup, GPUBindGroup] | null;
  copyFlowGroup: GPUBindGroup;
  particleGroups: readonly [GPUBindGroup, GPUBindGroup];
  packGroups: readonly [GPUBindGroup, GPUBindGroup];
  copyAndPackBeautyGroups: readonly [GPUBindGroup, GPUBindGroup];
  renderGroups: readonly [GPUBindGroup, GPUBindGroup];
}

interface PreparedBeautyVolumes {
  readonly resources: ResourceSet;
  readonly stateEpoch: number;
  readonly speciesIndex: 0 | 1;
}

interface BeautyVolumeHandoffResolution {
  readonly reusePackedVolumes: boolean;
  readonly refreshFinalDivergence: boolean;
}

const UNIFORM_FLOATS = 44;
const PARTICLE_FLOATS = 8;
const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;
const GPU_MAP_MODE = {
  READ: 0x0001,
} as const;
const GPU_SHADER_STAGE = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
} as const;
const GPU_TEXTURE_USAGE = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

export const FIREX_QUALITY: Record<QualityLevel, QualityConfiguration> = {
  low: {
    dimensions: [32, 48, 32],
    pressureIterations: 14,
    correctionIterations: 14,
    maximumSubsteps: 1,
    timeStep: 1 / 30,
    raySteps: 80,
    renderScale: 0.82,
    particleCount: 384,
  },
  balanced: {
    dimensions: [40, 64, 40],
    pressureIterations: 24,
    correctionIterations: 24,
    maximumSubsteps: 1,
    timeStep: 1 / 30,
    raySteps: 112,
    renderScale: 1,
    particleCount: 640,
  },
  high: {
    dimensions: [64, 96, 64],
    pressureIterations: 36,
    correctionIterations: 36,
    maximumSubsteps: 1,
    timeStep: 1 / 30,
    raySteps: 144,
    renderScale: 1,
    particleCount: 1024,
  },
  maximum: {
    dimensions: [80, 120, 80],
    pressureIterations: 48,
    correctionIterations: 48,
    maximumSubsteps: 1,
    timeStep: 1 / 30,
    raySteps: 192,
    renderScale: 1.25,
    particleCount: 1536,
  },
  cinematic: {
    dimensions: [80, 120, 80],
    pressureIterations: 48,
    correctionIterations: 48,
    maximumSubsteps: 1,
    timeStep: 1 / 30,
    raySteps: 192,
    renderScale: 1,
    particleCount: 1536,
  },
};

export const FIREX_OFFLINE_QUALITY: Readonly<Record<OfflineRenderTier, Readonly<FireXOfflineConfiguration>>> = Object.freeze({
  hd: Object.freeze({
    dimensions: [128, 192, 128] as const,
    pressureIterations: 64,
    correctionIterations: 64,
    maximumSubsteps: 1,
    timeStep: 1 / 60,
    raySteps: 256,
    renderScale: 1,
    particleCount: 2048,
    minimumMemoryGuardBytes: 512 * 1024 * 1024,
    opticalDetailTarget: 512,
  }),
  qhd: Object.freeze({
    dimensions: [160, 240, 160] as const,
    pressureIterations: 80,
    correctionIterations: 80,
    maximumSubsteps: 1,
    timeStep: 1 / 60,
    raySteps: 320,
    renderScale: 1,
    particleCount: 3072,
    minimumMemoryGuardBytes: 1024 * 1024 * 1024,
    opticalDetailTarget: 768,
  }),
  uhd: Object.freeze({
    dimensions: [192, 288, 192] as const,
    pressureIterations: 96,
    correctionIterations: 96,
    maximumSubsteps: 1,
    timeStep: 1 / 60,
    raySteps: 384,
    renderScale: 1,
    particleCount: 4096,
    minimumMemoryGuardBytes: 2 * 1024 * 1024 * 1024,
    opticalDetailTarget: 1024,
  }),
});

export function resolveFireXOfflineDimensions(
  tier: OfflineRenderTier,
  override: FireXGridDimensions | null,
): FireXGridDimensions {
  const preset = FIREX_OFFLINE_QUALITY[tier].dimensions;
  if (!override) return preset;
  return [
    Math.max(preset[0], override[0]),
    Math.max(preset[1], override[1]),
    Math.max(preset[2], override[2]),
  ];
}

export const FIREX_PIXEL_RATIO_CAP: Record<QualityLevel, number> = {
  low: 1.25,
  balanced: 1.5,
  high: 1.75,
  maximum: 2,
  cinematic: 1.75,
};

function finiteNumber(value: number | boolean, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function requireOutputDimension(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer number of pixels.`);
  }
  return value;
}

function createPeriodicDetailVolume(size: number): Uint8Array {
  const data = new Uint8Array(size * size * size * 4);
  const period = size - 1;
  const hash = (x: number, y: number, z: number, seed: number): number => {
    let value = Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791) ^ seed;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return (value >>> 24) & 255;
  };
  const fade = (value: number): number => value * value * (3 - 2 * value);
  const noise = (x: number, y: number, z: number, frequency: number, seed: number): number => {
    const px = (x / period) * frequency;
    const py = (y / period) * frequency;
    const pz = (z / period) * frequency;
    const x0 = Math.floor(px) % frequency;
    const y0 = Math.floor(py) % frequency;
    const z0 = Math.floor(pz) % frequency;
    const x1 = (x0 + 1) % frequency;
    const y1 = (y0 + 1) % frequency;
    const z1 = (z0 + 1) % frequency;
    const tx = fade(px - Math.floor(px));
    const ty = fade(py - Math.floor(py));
    const tz = fade(pz - Math.floor(pz));
    const mix = (a: number, b: number, amount: number): number => a + (b - a) * amount;
    const lower = mix(
      mix(hash(x0, y0, z0, seed), hash(x1, y0, z0, seed), tx),
      mix(hash(x0, y1, z0, seed), hash(x1, y1, z0, seed), tx),
      ty,
    );
    const upper = mix(
      mix(hash(x0, y0, z1, seed), hash(x1, y0, z1, seed), tx),
      mix(hash(x0, y1, z1, seed), hash(x1, y1, z1, seed), tx),
      ty,
    );
    return Math.round(mix(lower, upper, tz));
  };
  for (let z = 0; z < size; z += 1) {
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const offset = (x + size * (y + size * z)) * 4;
        data[offset] = noise(x, y, z, 4, 0x51f15e5d);
        data[offset + 1] = noise(x, y, z, 8, 0x6d2b79f5);
        data[offset + 2] = noise(x, y, z, 14, 0x1b873593);
        data[offset + 3] = 255;
      }
    }
  }
  return data;
}

function storageEntry(binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPU_SHADER_STAGE.COMPUTE,
    buffer: { type },
  };
}

export class FireXEngine implements SimulationEngine {
  readonly backend = 'WebGPU · 3D thermochemical grid';

  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private canvasFormat: GPUTextureFormat = 'bgra8unorm';
  private computeLayout: GPUBindGroupLayout | null = null;
  private particleLayout: GPUBindGroupLayout | null = null;
  private packLayout: GPUBindGroupLayout | null = null;
  private renderLayout: GPUBindGroupLayout | null = null;
  private postLayout: GPUBindGroupLayout | null = null;
  private f16PressureLayout: GPUBindGroupLayout | null = null;
  private volumeSampler: GPUSampler | null = null;
  private detailSampler: GPUSampler | null = null;
  private postSampler: GPUSampler | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private f16PressureUniformBuffer: GPUBuffer | null = null;
  private pipelines: PipelineSet | null = null;
  private resources: ResourceSet | null = null;
  private presentation: PresentationResources | null = null;
  private offlineReadbackBuffer: GPUBuffer | null = null;
  private offlineReadbackBufferSize = 0;
  private offlineReadbackWidth = 0;
  private offlineReadbackHeight = 0;
  private quality: QualityLevel = 'balanced';
  private clock = new FixedStepClock(FIREX_QUALITY.balanced.timeStep, FIREX_QUALITY.balanced.maximumSubsteps);
  private speciesIndex: 0 | 1 = 0;
  private pressureIndex: 0 | 1 = 0;
  private particleIndex: 0 | 1 = 0;
  private liquidFieldMayContainWater = false;
  private particleFieldActive = false;
  private simulationTime = 0;
  private stateEpoch = 0;
  private offlineRenderTier: OfflineRenderTier | null = null;
  private offlineOpticalDetailTarget: 0 | 256 | 512 | 1024 | null = null;
  private offlineMeasuredFrames = 0;
  private offlineSimulationMilliseconds = 0;
  private offlinePresentationMilliseconds = 0;
  private offlineVideoFrameConstructionMilliseconds = 0;
  private frameIndex = 0;
  private paused = false;
  private initialized = false;
  private disposed = false;
  private deviceLost = false;
  private rebuilding = false;
  private resourceMutationQueue: Promise<void> = Promise.resolve();
  private cssWidth = 1;
  private cssHeight = 1;
  private pixelRatio = 1;
  private displayImperial = false;
  private gridOverride: FireXGridDimensions | null = null;
  private gridMemoryBudgetBytes = 512 * 1024 * 1024;
  private parameters: FireXParameters = {
    burnerSize: 3.1,
    burnerDepth: 2.7,
    sourceThickness: 1.35,
    firePower: 2.5,
    sourceLift: 1.6,
    buoyancyScale: 2.15,
    flamePersistence: 1.45,
    airEntrainment: 1.65,
    fuelRate: 0.30,
    oxygenRate: 1,
    heatEfficiency: 0.86,
    vorticity: 3.6,
    waterFlow: 0,
    sprayAngle: 10,
    waterEnabled: false,
    cameraYaw: -0.34,
    cameraPitch: 0.03,
    cameraDistance: 4.9,
    viewMode: 0,
    aimHeight: 0.42,
    nozzleType: 0,
    exposure: 1.25,
    frontDefinition: 1.65,
    domainWidth: 1.15,
    domainHeight: 1.35,
    domainDepth: 1.15,
    viewZoom: 2.1,
    opticalDetailTarget: 1024,
  };

  private readonly handleUncapturedError = (event: GPUUncapturedErrorEvent): void => {
    console.error('Fire-X WebGPU validation error:', event.error.message);
  };

  constructor(private readonly canvas: HTMLCanvasElement) {}

  get detail(): string {
    const resources = this.resources;
    if (!resources) return '3D fields pending';
    const quality = this.activeConfiguration();
    const formatted = formatFireXGridInfo(
      deriveFireXGridInfo(resources.dimensions, this.domainExtent(), resources.particleCount),
      this.displayImperial ? 'imperial' : 'metric',
    );
    const presentation = this.usesCinematicPresentation() ? ' · HDR cinema + detail volume' : '';
    const opticalDetailTarget = this.activeOpticalDetailTarget();
    const opticalDetail = this.usesCinematicPresentation() && opticalDetailTarget > 0
      ? ` · ${opticalDetailTarget}³ render optical detail`
      : '';
    const pressureBackend = resources.usesF16Pressure
      ? ' · mixed f16 pressure storage + f32 math'
      : '';
    return `${formatted.summary} · ${quality.raySteps} rays · ${quality.pressureIterations}+${quality.correctionIterations} pressure${pressureBackend} · ${Math.round(1 / quality.timeStep)} Hz sim (≤${quality.maximumSubsteps}/frame) · ${resources.particleCount} droplets${presentation}${opticalDetail}`;
  }

  getGridInfo(): SimulationGridInfo {
    const dimensions = this.resources?.dimensions ?? this.effectiveGridDimensions();
    const particleCount = this.resources?.particleCount ?? this.activeConfiguration().particleCount;
    const info = deriveFireXGridInfo(dimensions, this.domainExtent(), particleCount);
    return {
      dimensions: info.dimensions,
      domainMeters: info.domainMeters,
      cellSizeMeters: info.voxelMeters,
      cellCount: info.cellCount,
      estimatedBytes: info.totalBytes,
      custom: this.offlineRenderTier === null && this.gridOverride !== null,
    };
  }

  getGridCapabilities(): SimulationGridCapabilities | null {
    const device = this.device;
    if (!device) return null;
    const limits = {
      maxTextureDimension3D: device.limits.maxTextureDimension3D,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
    };
    return {
      ...limits,
      maximumDenseCubeAxis: maximumFireXDeviceDenseCubeAxis(limits),
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    if (this.disposed) throw new Error('The Fire-X engine has already been disposed.');
    if (!navigator.gpu) {
      throw new Error('WebGPU is required for the Fire-X thermochemical solver.');
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No compatible WebGPU adapter was found.');
    const requiredLimits = selectFireXRequiredDeviceLimits(adapter.limits);
    const requiredFeatures: GPUFeatureName[] = adapter.features.has('shader-f16') ? ['shader-f16'] : [];
    const device = await adapter.requestDevice({
      label: 'Fire-X simulation device',
      requiredLimits,
      requiredFeatures,
    });
    const context = this.canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) {
      device.destroy();
      throw new Error('The canvas could not create a WebGPU context.');
    }

    this.device = device;
    this.context = context;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: this.canvasFormat,
      alphaMode: 'opaque',
      usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.COPY_SRC,
    });
    device.addEventListener('uncapturederror', this.handleUncapturedError);
    void device.lost.then((information) => {
      if (this.disposed || information.reason === 'destroyed') return;
      this.deviceLost = true;
      console.error(`Fire-X WebGPU device lost: ${information.message}`);
    });

    await this.createPipelines();
    await this.rebuildResourcesSafely(this.effectiveGridDimensions());
    this.initialized = true;
  }

  resize(width: number, height: number, pixelRatio: number): void {
    this.cssWidth = Math.max(width, 1);
    this.cssHeight = Math.max(height, 1);
    this.pixelRatio = Math.max(pixelRatio, 0.5);
    const configuration = this.activeConfiguration();
    const scale = Math.min(this.pixelRatio, FIREX_PIXEL_RATIO_CAP[this.quality]) * configuration.renderScale;
    const { width: nextWidth, height: nextHeight } = fitCanvasSize(
      this.cssWidth,
      this.cssHeight,
      scale,
      3072,
      1728,
    );
    if (this.canvas.width !== nextWidth || this.canvas.height !== nextHeight) {
      this.canvas.width = nextWidth;
      this.canvas.height = nextHeight;
    }
  }

  async resizeOutput(widthPx: number, heightPx: number): Promise<void> {
    const device = this.requireOfflineDevice();
    const width = requireOutputDimension(widthPx, 'Output width');
    const height = requireOutputDimension(heightPx, 'Output height');
    const maximumDimension = device.limits.maxTextureDimension2D;
    if (width > maximumDimension || height > maximumDimension) {
      throw new Error(
        `The requested ${width}\u00d7${height} output exceeds this WebGPU device limit of ${maximumDimension} pixels per axis.`,
      );
    }

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.destroyPresentationResources();
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (this.canvas.width !== width || this.canvas.height !== height) {
      throw new Error(`The browser could not allocate the requested ${width}\u00d7${height} output canvas.`);
    }

    if (this.offlineRenderTier !== null) {
      const configuration = FIREX_OFFLINE_QUALITY[this.offlineRenderTier];
      const bloomWidth = Math.max(Math.ceil(width / 4), 1);
      const bloomHeight = Math.max(Math.ceil(height / 4), 1);
      const presentationBytes = width * height * 8
        + bloomWidth * bloomHeight * 8
        + width * height * 4;
      const solverBytes = this.resources
        ? estimateFireXGridMemory(this.resources.dimensions, this.resources.particleCount).totalBytes
        : 0;
      const presentationGuardBytes = Math.max(
        this.gridMemoryBudgetBytes,
        configuration.minimumMemoryGuardBytes,
      );
      if (solverBytes + presentationBytes > presentationGuardBytes) {
        throw new Error(
          `The exact offline solver and ${width}\u00d7${height} presentation require at least ${formatFireXBytes(solverBytes + presentationBytes)}, above the ${formatFireXBytes(presentationGuardBytes)} tier guard.`,
        );
      }

      device.pushErrorScope('out-of-memory');
      device.pushErrorScope('validation');
      try {
        this.ensurePresentationResources();
      } catch (error) {
        await device.popErrorScope();
        await device.popErrorScope();
        throw error;
      }
      const validationError = await device.popErrorScope();
      const memoryError = await device.popErrorScope();
      if (validationError || memoryError) {
        this.destroyPresentationResources();
        const details = [validationError?.message, memoryError?.message].filter(Boolean).join(' ');
        throw new Error(
          details || `The exact ${width}\u00d7${height} HDR presentation targets could not be allocated.`,
        );
      }
    }
  }

  frame(timeSeconds: number, deltaSeconds: number): number {
    void timeSeconds;
    if (
      !this.device ||
      !this.context ||
      !this.pipelines ||
      !this.resources ||
      !this.uniformBuffer ||
      this.rebuilding ||
      this.deviceLost
    ) return 0;
    return this.submitReadyFrame(deltaSeconds, !this.paused, false);
  }

  async renderOfflineFrame(
    deltaSeconds: number,
    present = true,
    signal?: AbortSignal,
  ): Promise<void> {
    await this.renderOfflineFrameInternal(deltaSeconds, present, signal, false);
  }

  private async renderOfflineFrameInternal(
    deltaSeconds: number,
    present: boolean,
    signal: AbortSignal | undefined,
    prepareBeautyVolumes: boolean,
  ): Promise<PreparedBeautyVolumes | null> {
    if (!Number.isFinite(deltaSeconds) || deltaSeconds < 0) {
      throw new Error('Offline frame delta must be finite and non-negative.');
    }
    await this.resourceMutationQueue;
    const device = this.requireOfflineDevice();
    const configuration = this.activeConfiguration();
    this.clock.reset();
    const timeSteps = splitOfflineSimulationDelta(deltaSeconds, configuration.timeStep);
    let preparedBeautyVolumes: PreparedBeautyVolumes | null = null;
    for (let stepIndex = 0; stepIndex < timeSteps.length; stepIndex += 1) {
      const timeStep = timeSteps[stepIndex];
      this.throwIfOfflineAborted(signal);
      this.advanceStateEpoch();
      this.writeUniforms(timeStep);
      if (this.offlineRenderTier !== null) {
        preparedBeautyVolumes = this.submitOfflineSimulationStep(
          signal,
          prepareBeautyVolumes && stepIndex === timeSteps.length - 1,
        );
      } else {
        const simulationEncoder = device.createCommandEncoder({ label: 'Fire-X offline simulation step' });
        this.encodeSimulationStep(simulationEncoder);
        device.queue.submit([simulationEncoder.finish()]);
      }
      this.simulationTime += timeStep;
      this.frameIndex += 1;
    }

    if (present) {
      this.writeUniforms(0);
      const renderEncoder = device.createCommandEncoder({ label: 'Fire-X offline presentation' });
      this.encodeRender(renderEncoder, undefined, preparedBeautyVolumes);
      device.queue.submit([renderEncoder.finish()]);
      preparedBeautyVolumes = null;
    }
    await device.queue.onSubmittedWorkDone();
    if (this.deviceLost || this.device !== device) {
      throw new Error('The Fire-X WebGPU device was lost during offline rendering.');
    }
    return preparedBeautyVolumes;
  }

  async renderOfflineVideoFrame(
    deltaSeconds: number,
    timestampSeconds: number,
    durationSeconds: number,
    signal?: AbortSignal,
  ): Promise<VideoFrame> {
    if (!Number.isFinite(timestampSeconds) || timestampSeconds < 0) {
      throw new Error('Offline video frame timestamp must be finite and non-negative.');
    }
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      throw new Error('Offline video frame duration must be finite and positive.');
    }

    const totalStartedAt = performance.now();
    const simulationStartedAt = totalStartedAt;
    const preparedBeautyVolumes = await this.renderOfflineFrameInternal(
      deltaSeconds,
      false,
      signal,
      true,
    );
    const simulationFinishedAt = performance.now();
    this.throwIfOfflineAborted(signal);

    const device = this.requireOfflineDevice();
    const context = this.context;
    if (!context) throw new Error('The Fire-X WebGPU canvas context is unavailable.');
    const width = this.canvas.width;
    const height = this.canvas.height;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const readbackSize = bytesPerRow * height;

    this.writeUniforms(0);
    const outputTexture = context.getCurrentTexture();
    const renderEncoder = device.createCommandEncoder({ label: 'Fire-X offline video presentation and readback' });
    this.encodeRender(renderEncoder, outputTexture, preparedBeautyVolumes);
    const readbackBuffer = this.ensureOfflineReadbackBuffer(width, height, readbackSize);
    renderEncoder.copyTextureToBuffer(
      { texture: outputTexture },
      { buffer: readbackBuffer, bytesPerRow, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );
    device.queue.submit([renderEncoder.finish()]);
    await readbackBuffer.mapAsync(GPU_MAP_MODE.READ, 0, readbackSize);
    const presentationFinishedAt = performance.now();
    let frame: VideoFrame;
    try {
      this.throwIfOfflineAborted(signal);
      if (this.deviceLost || this.device !== device) {
        throw new Error('The Fire-X WebGPU device was lost during offline video readback.');
      }
      const format = this.canvasFormat === 'bgra8unorm'
        ? 'BGRA'
        : this.canvasFormat === 'rgba8unorm'
          ? 'RGBA'
          : null;
      if (!format) {
        throw new Error(`The ${this.canvasFormat} canvas format cannot be copied into a raw video frame.`);
      }
      const timestamp = Math.round(timestampSeconds * 1_000_000);
      const endTimestamp = Math.round((timestampSeconds + durationSeconds) * 1_000_000);
      const duration = endTimestamp - timestamp;
      if (!Number.isSafeInteger(timestamp) || !Number.isSafeInteger(endTimestamp) || duration <= 0) {
        throw new Error('The Fire-X offline video timestamp is outside the supported range.');
      }
      frame = new VideoFrame(readbackBuffer.getMappedRange(0, readbackSize), {
        format,
        codedWidth: width,
        codedHeight: height,
        timestamp,
        duration,
        layout: [{ offset: 0, stride: bytesPerRow }],
      });
    } finally {
      readbackBuffer.unmap();
    }
    const frameFinishedAt = performance.now();
    this.offlineMeasuredFrames += 1;
    this.offlineSimulationMilliseconds += simulationFinishedAt - simulationStartedAt;
    this.offlinePresentationMilliseconds += presentationFinishedAt - simulationFinishedAt;
    this.offlineVideoFrameConstructionMilliseconds += frameFinishedAt - presentationFinishedAt;
    return frame;
  }

  getOfflinePerformanceInfo(): FireXOfflinePerformanceInfo {
    const denominator = Math.max(this.offlineMeasuredFrames, 1);
    const averageSimulationMilliseconds = this.offlineSimulationMilliseconds / denominator;
    const averagePresentationMilliseconds = this.offlinePresentationMilliseconds / denominator;
    const averageVideoFrameConstructionMilliseconds = this.offlineVideoFrameConstructionMilliseconds / denominator;
    return Object.freeze({
      measuredFrames: this.offlineMeasuredFrames,
      averageSimulationMilliseconds,
      averagePresentationMilliseconds,
      averageVideoFrameConstructionMilliseconds,
      averageTotalMilliseconds: averageSimulationMilliseconds
        + averagePresentationMilliseconds
        + averageVideoFrameConstructionMilliseconds,
    });
  }

  async setOfflineRenderTier(tier: OfflineRenderTier | null): Promise<void> {
    await this.enqueueResourceMutation(async () => {
      if (tier === this.offlineRenderTier) return;
      const previousTier = this.offlineRenderTier;
      const transitionGuard = Math.max(
        this.gridMemoryBudgetBytes,
        previousTier ? FIREX_OFFLINE_QUALITY[previousTier].minimumMemoryGuardBytes : 0,
        tier ? FIREX_OFFLINE_QUALITY[tier].minimumMemoryGuardBytes : 0,
      );
      this.advanceStateEpoch();
      this.rebuilding = true;
      try {
        await this.device?.queue.onSubmittedWorkDone();
        this.destroyPresentationResources();
        this.offlineRenderTier = tier;
        if (this.device && this.pipelines) {
          const dimensions = this.effectiveGridDimensions();
          const configuration = this.activeConfiguration();
          const resourcesMatch = this.resources
            && this.resources.particleCount === configuration.particleCount
            && this.resources.dimensions.every((value, index) => value === dimensions[index]);
          if (!resourcesMatch) await this.rebuildResourcesSafely(dimensions, transitionGuard);
        }
      } catch (error) {
        this.offlineRenderTier = previousTier;
        throw error;
      } finally {
        this.rebuilding = false;
      }
      const configuration = this.activeConfiguration();
      this.clock = new FixedStepClock(configuration.timeStep, configuration.maximumSubsteps);
    });
  }

  /** Overrides only the render-frequency target used by a CLI offline job. */
  setOfflineOpticalDetailTarget(target: number | null): void {
    this.advanceStateEpoch();
    if (target === null) {
      this.offlineOpticalDetailTarget = null;
      return;
    }
    this.offlineOpticalDetailTarget = target >= 768 ? 1024 : target >= 384 ? 512 : target >= 192 ? 256 : 0;
  }

  reset(): void {
    this.clock.reset();
    this.simulationTime = 0;
    this.frameIndex = 0;
    this.speciesIndex = 0;
    this.pressureIndex = 0;
    this.particleIndex = 0;
    this.resetOfflinePerformanceInfo();
    if (this.device && this.resources) this.initializeFields(this.resources);
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    if (paused) this.clock.reset();
  }

  async setQuality(quality: QualityLevel): Promise<void> {
    await this.enqueueResourceMutation(async () => {
      if (this.offlineRenderTier !== null) {
        throw new Error('Compute quality cannot change while an exact offline solver bundle is active.');
      }
      if (quality === this.quality) return;
      const previous = this.quality;
      this.quality = quality;
      let rebuiltResources = false;
      this.rebuilding = true;
      try {
        if (this.device && this.pipelines) {
          const dimensions = this.effectiveGridDimensions();
          const resources = this.resources;
          const configuration = FIREX_QUALITY[quality];
          const resourcesMatch = resources
            && resources.particleCount === configuration.particleCount
            && resources.dimensions.every((value, index) => value === dimensions[index]);
          if (!resourcesMatch) {
            await this.rebuildResourcesSafely(dimensions);
            rebuiltResources = true;
          }
        }
      } catch (error) {
        this.quality = previous;
        throw error;
      } finally {
        this.rebuilding = false;
      }
      this.destroyPresentationResources();
      this.clock = new FixedStepClock(FIREX_QUALITY[quality].timeStep, FIREX_QUALITY[quality].maximumSubsteps);
      this.resize(this.cssWidth, this.cssHeight, this.pixelRatio);
      if (!rebuiltResources && this.resources) this.reset();
    });
  }

  async setGridDimensions(dimensions: readonly [number, number, number] | null): Promise<void> {
    const next: FireXGridDimensions | null = dimensions
      ? [dimensions[0], dimensions[1], dimensions[2]]
      : null;
    await this.enqueueResourceMutation(async () => {
      if (this.offlineRenderTier !== null) {
        throw new Error('Grid dimensions cannot change while an exact offline solver bundle is active.');
      }
      const previous = this.gridOverride;
      if (
        previous === next
        || (previous && next && previous.every((value, index) => value === next[index]))
      ) return;

      const targetDimensions = next ?? FIREX_QUALITY[this.quality].dimensions;
      if (this.resources?.dimensions.every((value, index) => value === targetDimensions[index])) {
        this.gridOverride = next;
        return;
      }
      if (this.device) this.assertGridSupported(targetDimensions);
      if (!this.device || !this.pipelines) {
        this.gridOverride = next;
        return;
      }
      this.rebuilding = true;
      try {
        await this.rebuildResourcesSafely(targetDimensions);
        this.gridOverride = next;
      } finally {
        this.rebuilding = false;
      }
    });
  }

  setGridMemoryBudget(bytes: number): void {
    if (!Number.isFinite(bytes) || bytes <= 0) return;
    this.gridMemoryBudgetBytes = Math.max(
      256 * 1024 * 1024,
      Math.min(FIREX_MAX_GRID_MEMORY_BUDGET_BYTES, Math.floor(bytes)),
    );
  }

  setParameter(name: string, value: number | boolean): void {
    this.advanceStateEpoch();
    switch (name) {
      case 'burnerSize':
        this.parameters.burnerSize = Math.max(0.35, Math.min(3.2, finiteNumber(value, this.parameters.burnerSize)));
        break;
      case 'burnerDepth':
        this.parameters.burnerDepth = Math.max(0.35, Math.min(3.2, finiteNumber(value, this.parameters.burnerDepth)));
        break;
      case 'sourceThickness':
        this.parameters.sourceThickness = Math.max(0.35, Math.min(2.5, finiteNumber(value, this.parameters.sourceThickness)));
        break;
      case 'firePower':
        this.parameters.firePower = Math.max(0.1, Math.min(4, finiteNumber(value, this.parameters.firePower)));
        break;
      case 'sourceLift':
        this.parameters.sourceLift = Math.max(0, Math.min(4, finiteNumber(value, this.parameters.sourceLift)));
        break;
      case 'buoyancyScale':
        this.parameters.buoyancyScale = Math.max(0, Math.min(4, finiteNumber(value, this.parameters.buoyancyScale)));
        break;
      case 'flamePersistence':
        this.parameters.flamePersistence = Math.max(0.4, Math.min(3, finiteNumber(value, this.parameters.flamePersistence)));
        break;
      case 'airEntrainment':
        this.parameters.airEntrainment = Math.max(0, Math.min(3, finiteNumber(value, this.parameters.airEntrainment)));
        break;
      case 'fuelRate':
        this.parameters.fuelRate = finiteNumber(value, this.parameters.fuelRate);
        break;
      case 'oxygenRate':
        this.parameters.oxygenRate = finiteNumber(value, this.parameters.oxygenRate);
        break;
      case 'heatEfficiency':
        this.parameters.heatEfficiency = finiteNumber(value, this.parameters.heatEfficiency);
        break;
      case 'vorticity':
        this.parameters.vorticity = finiteNumber(value, this.parameters.vorticity);
        break;
      case 'waterFlow':
        this.parameters.waterFlow = finiteNumber(value, this.parameters.waterFlow);
        break;
      case 'sprayAngle':
        this.parameters.sprayAngle = finiteNumber(value, this.parameters.sprayAngle);
        break;
      case 'waterEnabled':
        if (typeof value === 'boolean') this.parameters.waterEnabled = value;
        break;
      case 'cameraYaw':
        this.parameters.cameraYaw = finiteNumber(value, this.parameters.cameraYaw);
        break;
      case 'cameraPitch':
        this.parameters.cameraPitch = finiteNumber(value, this.parameters.cameraPitch);
        break;
      case 'cameraDistance':
        this.parameters.cameraDistance = finiteNumber(value, this.parameters.cameraDistance);
        break;
      case 'viewMode':
        this.parameters.viewMode = Math.max(0, Math.min(7, Math.round(finiteNumber(value, this.parameters.viewMode))));
        break;
      case 'aimHeight':
        this.parameters.aimHeight = Math.max(0.08, Math.min(0.82, finiteNumber(value, this.parameters.aimHeight)));
        break;
      case 'nozzleType':
        this.parameters.nozzleType = finiteNumber(value, this.parameters.nozzleType) >= 0.5 ? 1 : 0;
        break;
      case 'exposure':
        this.parameters.exposure = Math.max(0.25, Math.min(3, finiteNumber(value, this.parameters.exposure)));
        break;
      case 'frontDefinition':
        this.parameters.frontDefinition = Math.max(0.5, Math.min(2.5, finiteNumber(value, this.parameters.frontDefinition)));
        break;
      case 'opticalDetailTarget': {
        const target = finiteNumber(value, this.parameters.opticalDetailTarget);
        this.parameters.opticalDetailTarget = target >= 768 ? 1024 : target >= 384 ? 512 : target >= 192 ? 256 : 0;
        break;
      }
      case 'domainWidth':
        this.parameters.domainWidth = Math.max(0.5, Math.min(4, finiteNumber(value, this.parameters.domainWidth)));
        break;
      case 'domainHeight':
        this.parameters.domainHeight = Math.max(0.5, Math.min(6, finiteNumber(value, this.parameters.domainHeight)));
        break;
      case 'domainDepth':
        this.parameters.domainDepth = Math.max(0.5, Math.min(4, finiteNumber(value, this.parameters.domainDepth)));
        break;
      case 'displayImperial':
        this.displayImperial = value === true || finiteNumber(value, 0) >= 0.5;
        break;
      case 'viewZoom':
        this.parameters.viewZoom = Math.max(0.5, Math.min(5, finiteNumber(value, this.parameters.viewZoom)));
        break;
      default:
        break;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.destroyPresentationResources();
    this.destroyOfflineReadbackBuffer();
    this.destroyResources();
    if (this.uniformBuffer) this.uniformBuffer.destroy();
    this.uniformBuffer = null;
    if (this.f16PressureUniformBuffer) this.f16PressureUniformBuffer.destroy();
    this.f16PressureUniformBuffer = null;
    if (this.device) {
      this.device.removeEventListener('uncapturederror', this.handleUncapturedError);
      this.device.destroy();
    }
    this.context?.unconfigure();
    this.context = null;
    this.device = null;
    this.pipelines = null;
    this.computeLayout = null;
    this.particleLayout = null;
    this.packLayout = null;
    this.renderLayout = null;
    this.postLayout = null;
    this.f16PressureLayout = null;
    this.volumeSampler = null;
    this.detailSampler = null;
    this.postSampler = null;
    this.initialized = false;
  }

  private submitReadyFrame(
    deltaSeconds: number,
    advanceSimulation: boolean,
    consumeFullDelta: boolean,
  ): number {
    const device = this.requireDevice();
    const start = performance.now();
    let simulationSteps = 0;
    if (advanceSimulation) {
      const countStep = (): void => {
        simulationSteps += 1;
      };
      if (consumeFullDelta) {
        let remainingDelta = deltaSeconds;
        const maximumClockDelta = this.clock.stepSeconds * this.clock.maximumStepsPerFrame;
        while (remainingDelta > 0) {
          const chunk = Math.min(remainingDelta, maximumClockDelta);
          this.clock.advance(chunk, countStep);
          remainingDelta = Math.max(0, remainingDelta - chunk);
        }
      } else {
        this.clock.advance(deltaSeconds, countStep);
      }
    }

    const timeStep = this.activeConfiguration().timeStep;
    this.writeUniforms(simulationSteps > 0 ? timeStep : 0);
    const encoder = device.createCommandEncoder({ label: 'Fire-X frame graph' });
    for (let step = 0; step < simulationSteps; step += 1) {
      this.advanceStateEpoch();
      this.encodeSimulationStep(encoder);
      this.simulationTime += timeStep;
      this.frameIndex += 1;
    }
    this.encodeRender(encoder);
    device.queue.submit([encoder.finish()]);
    return performance.now() - start;
  }

  private async createPipelines(): Promise<void> {
    const device = this.requireDevice();
    this.uniformBuffer = device.createBuffer({
      label: 'Fire-X frame parameters',
      size: UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    const f16PressureLayout = device.features.has('shader-f16')
      ? device.createBindGroupLayout({
        label: 'Fire-X mixed-precision pressure resources',
        entries: [
          { binding: 0, visibility: GPU_SHADER_STAGE.COMPUTE, buffer: { type: 'uniform' } },
          storageEntry(1, 'read-only-storage'),
          storageEntry(2, 'read-only-storage'),
          storageEntry(3, 'storage'),
          storageEntry(4, 'read-only-storage'),
          storageEntry(5, 'storage'),
        ],
      })
      : null;
    this.f16PressureLayout = f16PressureLayout;
    this.f16PressureUniformBuffer = f16PressureLayout
      ? device.createBuffer({
        label: 'Fire-X mixed-precision pressure parameters',
        size: 8 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
      })
      : null;

    const computeLayout = device.createBindGroupLayout({
      label: 'Fire-X compute resources',
      entries: [
        {
          binding: 0,
          visibility: GPU_SHADER_STAGE.COMPUTE,
          buffer: { type: 'uniform' },
        },
        storageEntry(1, 'read-only-storage'),
        storageEntry(2, 'read-only-storage'),
        storageEntry(3, 'storage'),
        storageEntry(4, 'storage'),
        storageEntry(5, 'storage'),
        storageEntry(6, 'read-only-storage'),
        storageEntry(7, 'storage'),
        storageEntry(8, 'read-only-storage'),
      ],
    });
    const particleLayout = device.createBindGroupLayout({
      label: 'Fire-X SPH particle resources',
      entries: [
        {
          binding: 0,
          visibility: GPU_SHADER_STAGE.COMPUTE,
          buffer: { type: 'uniform' },
        },
        storageEntry(1, 'read-only-storage'),
        storageEntry(2, 'read-only-storage'),
        storageEntry(3, 'storage'),
        storageEntry(4, 'storage'),
        storageEntry(5, 'storage'),
      ],
    });
    const packLayout = device.createBindGroupLayout({
      label: 'Fire-X filtered-volume packing resources',
      entries: [
        {
          binding: 0,
          visibility: GPU_SHADER_STAGE.COMPUTE,
          buffer: { type: 'uniform' },
        },
        storageEntry(1, 'read-only-storage'),
        storageEntry(2, 'read-only-storage'),
        storageEntry(3, 'read-only-storage'),
        storageEntry(4, 'read-only-storage'),
        storageEntry(5, 'read-only-storage'),
        ...([6, 7, 8] as const).map((binding) => ({
          binding,
          visibility: GPU_SHADER_STAGE.COMPUTE,
          storageTexture: {
            access: 'write-only' as GPUStorageTextureAccess,
            format: 'rgba16float' as GPUTextureFormat,
            viewDimension: '3d' as GPUTextureViewDimension,
          },
        })),
        storageEntry(9, 'storage'),
      ],
    });
    const renderLayout = device.createBindGroupLayout({
      label: 'Fire-X render resources',
      entries: [
        {
          binding: 0,
          visibility: GPU_SHADER_STAGE.VERTEX | GPU_SHADER_STAGE.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPU_SHADER_STAGE.FRAGMENT,
          sampler: { type: 'filtering' },
        },
        ...([2, 3, 4] as const).map((binding) => ({
          binding,
          visibility: GPU_SHADER_STAGE.FRAGMENT,
          texture: {
            sampleType: 'float' as GPUTextureSampleType,
            viewDimension: '3d' as GPUTextureViewDimension,
          },
        })),
        {
          binding: 5,
          visibility: GPU_SHADER_STAGE.VERTEX,
          buffer: { type: 'read-only-storage' },
        },
        {
          binding: 6,
          visibility: GPU_SHADER_STAGE.FRAGMENT,
          texture: {
            sampleType: 'float',
            viewDimension: '3d',
          },
        },
        {
          binding: 7,
          visibility: GPU_SHADER_STAGE.FRAGMENT,
          sampler: { type: 'filtering' },
        },
      ],
    });
    const postLayout = device.createBindGroupLayout({
      label: 'Fire-X cinematic presentation resources',
      entries: [
        {
          binding: 0,
          visibility: GPU_SHADER_STAGE.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPU_SHADER_STAGE.FRAGMENT,
          sampler: { type: 'filtering' },
        },
        ...([2, 3] as const).map((binding) => ({
          binding,
          visibility: GPU_SHADER_STAGE.FRAGMENT,
          texture: { sampleType: 'float' as GPUTextureSampleType },
        })),
      ],
    });
    this.computeLayout = computeLayout;
    this.particleLayout = particleLayout;
    this.packLayout = packLayout;
    this.renderLayout = renderLayout;
    this.postLayout = postLayout;
    this.volumeSampler = device.createSampler({
      label: 'Fire-X trilinear volume sampler',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });
    this.detailSampler = device.createSampler({
      label: 'Fire-X periodic cinematic detail sampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      addressModeW: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });
    this.postSampler = device.createSampler({
      label: 'Fire-X cinematic presentation sampler',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    const computeModule = device.createShaderModule({
      label: 'Fire-X thermochemical compute shader',
      code: FIREX_COMPUTE_WGSL,
    });
    const renderModule = device.createShaderModule({
      label: 'Fire-X volume render shader',
      code: FIREX_QUALITY_RENDER_WGSL,
    });
    const packModule = device.createShaderModule({
      label: 'Fire-X filtered-volume packing shader',
      code: FIREX_PACK_WGSL,
    });
    const particleModule = device.createShaderModule({
      label: 'Fire-X SPH particle shader',
      code: FIREX_PARTICLE_WGSL,
    });
    const postModule = device.createShaderModule({
      label: 'Fire-X cinematic HDR presentation shader',
      code: FIREX_POST_WGSL,
    });
    const f16PressureModule = f16PressureLayout
      ? device.createShaderModule({
        label: 'Fire-X mixed-precision pressure shader',
        code: FIREX_F16_PRESSURE_WGSL,
      })
      : null;
    await Promise.all([
      this.assertShaderCompilation(computeModule, 'compute'),
      this.assertShaderCompilation(renderModule, 'render'),
      this.assertShaderCompilation(packModule, 'field packing'),
      this.assertShaderCompilation(particleModule, 'particle'),
      this.assertShaderCompilation(postModule, 'cinematic presentation'),
      f16PressureModule
        ? this.assertShaderCompilation(f16PressureModule, 'mixed-precision pressure')
        : Promise.resolve(),
    ]);

    const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [computeLayout] });
    const particlePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [particleLayout] });
    const packPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [packLayout] });
    const renderPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] });
    const postPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [postLayout] });
    const [
      initialize,
      simulate,
      divergence,
      pressureGlobal,
      pressureTiled,
      project,
      copyFlow,
      particleDensity,
      particleIntegrate,
      particleDeposit,
      pack,
      copyAndPackBeauty,
      render,
      particleRender,
      cinematicRender,
      cinematicParticleRender,
      bloom,
      composite,
    ] = await Promise.all([
      device.createComputePipelineAsync({
        label: 'Fire-X GPU field initialization',
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'initializeFields' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X advection and reaction',
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'simulate' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X divergence',
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'computeDivergence' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X Jacobi pressure (global)',
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'solvePressureGlobal' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X Jacobi pressure (shared tile)',
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'solvePressureTiled' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X velocity projection',
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'projectVelocity' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X corrected-flow copy',
        layout: computePipelineLayout,
        compute: { module: computeModule, entryPoint: 'copyFlow' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X SPH density and pressure',
        layout: particlePipelineLayout,
        compute: { module: particleModule, entryPoint: 'computeParticleDensity' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X SPH pressure, viscosity, and integration',
        layout: particlePipelineLayout,
        compute: { module: particleModule, entryPoint: 'integrateParticles' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X SPH liquid deposition',
        layout: particlePipelineLayout,
        compute: { module: particleModule, entryPoint: 'depositParticles' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X pack filterable volume',
        layout: packPipelineLayout,
        compute: { module: packModule, entryPoint: 'packFields' },
      }),
      device.createComputePipelineAsync({
        label: 'Fire-X copy corrected flow and pack Beauty volumes',
        layout: packPipelineLayout,
        compute: { module: packModule, entryPoint: 'copyAndPackBeautyFields' },
      }),
      device.createRenderPipelineAsync({
        label: 'Fire-X volume ray marcher',
        layout: renderPipelineLayout,
        vertex: { module: renderModule, entryPoint: 'fullscreenVertex' },
        fragment: {
          module: renderModule,
          entryPoint: 'volumeFragment',
          targets: [{ format: this.canvasFormat }],
        },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        label: 'Fire-X non-emissive droplet streaks',
        layout: renderPipelineLayout,
        vertex: { module: renderModule, entryPoint: 'dropletVertex' },
        fragment: {
          module: renderModule,
          entryPoint: 'dropletFragment',
          targets: [{
            format: this.canvasFormat,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        label: 'Fire-X cinematic HDR volume ray marcher',
        layout: renderPipelineLayout,
        vertex: { module: renderModule, entryPoint: 'fullscreenVertex' },
        fragment: {
          module: renderModule,
          entryPoint: 'volumeFragment',
          targets: [{ format: 'rgba16float' }],
        },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        label: 'Fire-X cinematic HDR droplet streaks',
        layout: renderPipelineLayout,
        vertex: { module: renderModule, entryPoint: 'dropletVertex' },
        fragment: {
          module: renderModule,
          entryPoint: 'dropletFragment',
          targets: [{
            format: 'rgba16float',
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        label: 'Fire-X quarter-resolution cinematic bloom',
        layout: postPipelineLayout,
        vertex: { module: postModule, entryPoint: 'postVertex' },
        fragment: {
          module: postModule,
          entryPoint: 'bloomFragment',
          targets: [{ format: 'rgba16float' }],
        },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        label: 'Fire-X cinematic tone-map and composite',
        layout: postPipelineLayout,
        vertex: { module: postModule, entryPoint: 'postVertex' },
        fragment: {
          module: postModule,
          entryPoint: 'compositeFragment',
          targets: [{ format: this.canvasFormat }],
        },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
    let pressureF16Tiled: GPUComputePipeline | null = null;
    let projectF16: GPUComputePipeline | null = null;
    if (f16PressureLayout && f16PressureModule) {
      const layout = device.createPipelineLayout({ bindGroupLayouts: [f16PressureLayout] });
      [pressureF16Tiled, projectF16] = await Promise.all([
        device.createComputePipelineAsync({
          label: 'Fire-X mixed-precision tiled pressure',
          layout,
          compute: { module: f16PressureModule, entryPoint: 'solvePressureF16Tiled' },
        }),
        device.createComputePipelineAsync({
          label: 'Fire-X mixed-precision velocity projection',
          layout,
          compute: { module: f16PressureModule, entryPoint: 'projectVelocityF16' },
        }),
      ]);
    }
    this.pipelines = {
      initialize,
      simulate,
      divergence,
      pressureGlobal,
      pressureTiled,
      pressureF16Tiled,
      project,
      projectF16,
      copyFlow,
      particleDensity,
      particleIntegrate,
      particleDeposit,
      pack,
      copyAndPackBeauty,
      render,
      particleRender,
      cinematicRender,
      cinematicParticleRender,
      bloom,
      composite,
    };
  }

  private async assertShaderCompilation(module: GPUShaderModule, label: string): Promise<void> {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === 'error');
    if (errors.length === 0) return;
    const details = errors
      .slice(0, 8)
      .map((message) => `line ${message.lineNum}: ${message.message}`)
      .join('\n');
    throw new Error(`Fire-X ${label} shader failed to compile:\n${details}`);
  }

  private createResourceSet(dimensions: FireXGridDimensions): ResourceSet {
    const device = this.requireDevice();
    const computeLayout = this.requireComputeLayout();
    const particleLayout = this.requireParticleLayout();
    const packLayout = this.requirePackLayout();
    const renderLayout = this.requireRenderLayout();
    const volumeSampler = this.requireVolumeSampler();
    const detailSampler = this.requireDetailSampler();
    const uniformBuffer = this.requireUniformBuffer();
    const pipelines = this.requirePipelines();
    const f16PressureLayout = this.f16PressureLayout;
    const f16PressureUniformBuffer = this.f16PressureUniformBuffer;
    const configuration = this.activeConfiguration();
    const allocatedBuffers: GPUBuffer[] = [];
    const allocatedTextures: GPUTexture[] = [];
    try {
    const cellCount = dimensions[0] * dimensions[1] * dimensions[2];
    const vectorBytes = cellCount * 4 * Float32Array.BYTES_PER_ELEMENT;
    const scalarBytes = cellCount * Float32Array.BYTES_PER_ELEMENT;
    const usesF16Pressure = cellCount >= FIREX_HEAVY_GRID_CELL_THRESHOLD
      && pipelines.pressureF16Tiled !== null
      && pipelines.projectF16 !== null
      && f16PressureLayout !== null
      && f16PressureUniformBuffer !== null;
    const pressureBytes = usesF16Pressure ? Math.ceil((cellCount * 2) / 4) * 4 : scalarBytes;
    const particleCount = configuration.particleCount;
    const particleBytes = particleCount * PARTICLE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
    const particleDensityBytes = particleCount * 4 * Float32Array.BYTES_PER_ELEMENT;
    const createStorageBuffer = (label: string, size: number): GPUBuffer => {
      const buffer = device.createBuffer({
        label,
        size,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
      });
      allocatedBuffers.push(buffer);
      return buffer;
    };
    const flow = [
      createStorageBuffer('Fire-X projected flow', vectorBytes),
      createStorageBuffer('Fire-X predicted flow', vectorBytes),
    ] as const;
    const species = [
      createStorageBuffer('Fire-X species A', vectorBytes),
      createStorageBuffer('Fire-X species B', vectorBytes),
    ] as const;
    const pressure = [
      createStorageBuffer(`Fire-X pressure A${usesF16Pressure ? ' (f16)' : ''}`, pressureBytes),
      createStorageBuffer(`Fire-X pressure B${usesF16Pressure ? ' (f16)' : ''}`, pressureBytes),
    ] as const;
    const compactHeavyFallback = cellCount >= FIREX_HEAVY_GRID_CELL_THRESHOLD && !usesF16Pressure;
    const correctionPressure = [
      createStorageBuffer(
        `Fire-X residual pressure A${usesF16Pressure ? ' (f16)' : ''}`,
        compactHeavyFallback ? 4 : pressureBytes,
      ),
      createStorageBuffer(
        `Fire-X residual pressure B${usesF16Pressure ? ' (f16)' : ''}`,
        compactHeavyFallback ? 4 : pressureBytes,
      ),
    ] as const;
    const divergence = createStorageBuffer('Fire-X divergence', scalarBytes);
    const particles = [
      createStorageBuffer('Fire-X SPH particles A', particleBytes),
      createStorageBuffer('Fire-X SPH particles B', particleBytes),
    ] as const;
    const particleDensity = createStorageBuffer('Fire-X SPH density and pressure', particleDensityBytes);
    const liquid = createStorageBuffer('Fire-X deposited liquid voxels', scalarBytes);
    const reaction = createStorageBuffer('Fire-X instantaneous reaction rate', scalarBytes);
    const createVolumeTexture = (label: string): GPUTexture => {
      const texture = device.createTexture({
        label,
        size: {
          width: dimensions[0],
          height: dimensions[1],
          depthOrArrayLayers: dimensions[2],
        },
        dimension: '3d',
        format: 'rgba16float',
        usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.STORAGE_BINDING,
      });
      allocatedTextures.push(texture);
      return texture;
    };
    const flowTexture = createVolumeTexture('Fire-X filterable velocity and temperature');
    const chemistryTexture = createVolumeTexture('Fire-X filterable species');
    const mediaTexture = createVolumeTexture('Fire-X filterable reaction and liquid media');
    const detailSize = 64;
    const detailTexture = device.createTexture({
      label: 'Fire-X periodic band-limited cinematic detail',
      size: {
        width: detailSize,
        height: detailSize,
        depthOrArrayLayers: detailSize,
      },
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    });
    allocatedTextures.push(detailTexture);
    device.queue.writeTexture(
      { texture: detailTexture },
      createPeriodicDetailVolume(detailSize),
      { bytesPerRow: detailSize * 4, rowsPerImage: detailSize },
      { width: detailSize, height: detailSize, depthOrArrayLayers: detailSize },
    );

    const computeGroup = (
      label: string,
      flowRead: GPUBuffer,
      speciesRead: GPUBuffer,
      flowWrite: GPUBuffer,
      speciesWrite: GPUBuffer,
      pressureRead: GPUBuffer,
      pressureWrite: GPUBuffer,
    ): GPUBindGroup =>
      device.createBindGroup({
        label,
        layout: computeLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: flowRead } },
          { binding: 2, resource: { buffer: speciesRead } },
          { binding: 3, resource: { buffer: flowWrite } },
          { binding: 4, resource: { buffer: speciesWrite } },
          { binding: 5, resource: { buffer: divergence } },
          { binding: 6, resource: { buffer: pressureRead } },
          { binding: 7, resource: { buffer: pressureWrite } },
          { binding: 8, resource: { buffer: liquid } },
        ],
      });

    const initializationGroup = computeGroup(
      'Fire-X GPU field initialization',
      flow[1],
      species[1],
      flow[0],
      species[0],
      pressure[0],
      pressure[1],
    );
    const simulationGroups = [
      computeGroup('Fire-X simulation A to B', flow[0], species[0], flow[1], species[1], pressure[0], reaction),
      computeGroup('Fire-X simulation B to A', flow[0], species[1], flow[1], species[0], pressure[0], reaction),
    ] as const;
    const divergenceGroup = computeGroup(
      'Fire-X divergence resources',
      flow[1],
      species[0],
      flow[0],
      species[1],
      pressure[0],
      pressure[1],
    );
    const projectedDivergenceGroup = computeGroup(
      'Fire-X projected divergence resources',
      flow[0],
      species[0],
      flow[1],
      species[1],
      pressure[0],
      pressure[1],
    );
    const pressureGroups = [
      computeGroup('Fire-X pressure A to B', flow[0], species[0], flow[1], species[1], pressure[0], pressure[1]),
      computeGroup('Fire-X pressure B to A', flow[0], species[0], flow[1], species[1], pressure[1], pressure[0]),
    ] as const;
    const correctionPressureGroups = [
      computeGroup(
        'Fire-X residual pressure A to B',
        flow[0], species[0], flow[1], species[1], correctionPressure[0], correctionPressure[1],
      ),
      computeGroup(
        'Fire-X residual pressure B to A',
        flow[0], species[0], flow[1], species[1], correctionPressure[1], correctionPressure[0],
      ),
    ] as const;
    const projectionGroups = [
      computeGroup('Fire-X project pressure A', flow[1], species[0], flow[0], species[1], pressure[0], pressure[1]),
      computeGroup('Fire-X project pressure B', flow[1], species[0], flow[0], species[1], pressure[1], pressure[0]),
    ] as const;
    const correctionProjectionGroups = [
      computeGroup(
        'Fire-X residual project pressure A',
        flow[0], species[0], flow[1], species[1], correctionPressure[0], pressure[0],
      ),
      computeGroup(
        'Fire-X residual project pressure B',
        flow[0], species[0], flow[1], species[1], correctionPressure[1], pressure[0],
      ),
    ] as const;
    const sharedCorrectionProjectionGroups = [
      computeGroup(
        'Fire-X shared-buffer residual project pressure A',
        flow[0], species[0], flow[1], species[1], pressure[0], correctionPressure[0],
      ),
      computeGroup(
        'Fire-X shared-buffer residual project pressure B',
        flow[0], species[0], flow[1], species[1], pressure[1], correctionPressure[0],
      ),
    ] as const;
    const createF16PressureGroup = (
      label: string,
      pressureRead: GPUBuffer,
      pressureWrite: GPUBuffer,
      flowRead: GPUBuffer,
      flowWrite: GPUBuffer,
    ): GPUBindGroup => device.createBindGroup({
      label,
      layout: f16PressureLayout!,
      entries: [
        { binding: 0, resource: { buffer: f16PressureUniformBuffer! } },
        { binding: 1, resource: { buffer: divergence } },
        { binding: 2, resource: { buffer: pressureRead } },
        { binding: 3, resource: { buffer: pressureWrite } },
        { binding: 4, resource: { buffer: flowRead } },
        { binding: 5, resource: { buffer: flowWrite } },
      ],
    });
    const f16PrimaryPressureGroups = usesF16Pressure ? [
      createF16PressureGroup('Fire-X f16 primary pressure A to B', pressure[0], pressure[1], flow[1], flow[0]),
      createF16PressureGroup('Fire-X f16 primary pressure B to A', pressure[1], pressure[0], flow[1], flow[0]),
    ] as const : null;
    const f16CorrectionPressureGroups = usesF16Pressure ? [
      createF16PressureGroup(
        'Fire-X f16 residual pressure A to B',
        correctionPressure[0], correctionPressure[1], flow[0], flow[1],
      ),
      createF16PressureGroup(
        'Fire-X f16 residual pressure B to A',
        correctionPressure[1], correctionPressure[0], flow[0], flow[1],
      ),
    ] as const : null;
    const copyFlowGroup = computeGroup(
      'Fire-X copy residual-corrected flow',
      flow[1], species[0], flow[0], species[1], pressure[0], correctionPressure[0],
    );
    const particleGroups = particles.map((particleRead, index) =>
      device.createBindGroup({
        label: `Fire-X SPH particle state ${index}`,
        layout: particleLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: flow[0] } },
          { binding: 2, resource: { buffer: particleRead } },
          { binding: 3, resource: { buffer: particles[index === 0 ? 1 : 0] } },
          { binding: 4, resource: { buffer: particleDensity } },
          { binding: 5, resource: { buffer: liquid } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const packGroups = species.map((speciesBuffer, index) =>
      device.createBindGroup({
        label: `Fire-X filtered-volume pack state ${index}`,
        layout: packLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: flow[0] } },
          { binding: 2, resource: { buffer: speciesBuffer } },
          { binding: 3, resource: { buffer: reaction } },
          { binding: 4, resource: { buffer: liquid } },
          { binding: 5, resource: { buffer: divergence } },
          { binding: 6, resource: flowTexture.createView({ dimension: '3d' }) },
          { binding: 7, resource: chemistryTexture.createView({ dimension: '3d' }) },
          { binding: 8, resource: mediaTexture.createView({ dimension: '3d' }) },
          { binding: 9, resource: { buffer: flow[1] } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const copyAndPackBeautyGroups = species.map((speciesBuffer, index) =>
      device.createBindGroup({
        label: `Fire-X corrected-flow copy and Beauty pack state ${index}`,
        layout: packLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: flow[1] } },
          { binding: 2, resource: { buffer: speciesBuffer } },
          { binding: 3, resource: { buffer: reaction } },
          { binding: 4, resource: { buffer: liquid } },
          { binding: 5, resource: { buffer: divergence } },
          { binding: 6, resource: flowTexture.createView({ dimension: '3d' }) },
          { binding: 7, resource: chemistryTexture.createView({ dimension: '3d' }) },
          { binding: 8, resource: mediaTexture.createView({ dimension: '3d' }) },
          { binding: 9, resource: { buffer: flow[0] } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const renderGroups = particles.map((particleBuffer, index) =>
      device.createBindGroup({
        label: `Fire-X trilinear volume and droplet state ${index}`,
        layout: renderLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: volumeSampler },
          { binding: 2, resource: flowTexture.createView({ dimension: '3d' }) },
          { binding: 3, resource: chemistryTexture.createView({ dimension: '3d' }) },
          { binding: 4, resource: mediaTexture.createView({ dimension: '3d' }) },
          { binding: 5, resource: { buffer: particleBuffer } },
          { binding: 6, resource: detailTexture.createView({ dimension: '3d' }) },
          { binding: 7, resource: detailSampler },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];

    const resources: ResourceSet = {
      dimensions,
      cellCount,
      flow,
      species,
      pressure,
      correctionPressure,
      divergence,
      particleCount,
      particles,
      particleDensity,
      liquid,
      reaction,
      flowTexture,
      chemistryTexture,
      mediaTexture,
      detailTexture,
      usesF16Pressure,
      initializationGroup,
      simulationGroups,
      divergenceGroup,
      projectedDivergenceGroup,
      pressureGroups,
      correctionPressureGroups,
      projectionGroups,
      correctionProjectionGroups,
      sharedCorrectionProjectionGroups,
      f16PrimaryPressureGroups,
      f16CorrectionPressureGroups,
      copyFlowGroup,
      particleGroups,
      packGroups,
      copyAndPackBeautyGroups,
      renderGroups,
    };
    return resources;
    } catch (error) {
      for (const buffer of allocatedBuffers) buffer.destroy();
      for (const texture of allocatedTextures) texture.destroy();
      throw error;
    }
  }

  private enqueueResourceMutation(operation: () => Promise<void>): Promise<void> {
    const queued = this.resourceMutationQueue.then(operation);
    this.resourceMutationQueue = queued.then(
      () => undefined,
      () => undefined,
    );
    return queued;
  }

  private async rebuildResourcesSafely(
    dimensions: FireXGridDimensions,
    memoryGuardBytes = this.gridMemoryBudgetBytes,
  ): Promise<void> {
    const device = this.requireDevice();
    this.assertGridSupported(dimensions, memoryGuardBytes);
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    let replacement: ResourceSet;
    try {
      replacement = this.createResourceSet(dimensions);
    } catch (error) {
      await device.popErrorScope();
      await device.popErrorScope();
      throw error;
    }
    const validationError = await device.popErrorScope();
    const memoryError = await device.popErrorScope();
    if (validationError || memoryError) {
      this.destroyResourceSet(replacement);
      const details = [validationError?.message, memoryError?.message].filter(Boolean).join(' ');
      throw new Error(details || 'The selected Fire-X grid could not be allocated.');
    }
    this.installResourceSet(replacement);
  }

  private installResourceSet(resources: ResourceSet): void {
    const previous = this.resources;
    this.resources = resources;
    this.clock.reset();
    this.simulationTime = 0;
    this.frameIndex = 0;
    this.speciesIndex = 0;
    this.pressureIndex = 0;
    this.particleIndex = 0;
    this.liquidFieldMayContainWater = false;
    this.particleFieldActive = false;
    this.resetOfflinePerformanceInfo();
    this.initializeFields(resources);
    if (previous) this.destroyResourceSet(previous);
  }

  private initializeFields(resources: ResourceSet): void {
    this.advanceStateEpoch();
    const device = this.requireDevice();
    const pipelines = this.requirePipelines();
    const initializationUniforms = new Float32Array(UNIFORM_FLOATS);
    initializationUniforms.set([
      resources.dimensions[0],
      resources.dimensions[1],
      resources.dimensions[2],
      resources.cellCount,
    ], 0);
    initializationUniforms[9] = this.parameters.oxygenRate;
    initializationUniforms[15] = 300;
    device.queue.writeBuffer(this.requireUniformBuffer(), 0, initializationUniforms);

    const clearEncoder = device.createCommandEncoder({ label: 'Fire-X GPU field reset' });
    for (const buffer of [
      ...resources.flow,
      ...resources.species,
      ...resources.pressure,
      ...resources.correctionPressure,
      resources.divergence,
      ...resources.particles,
      resources.particleDensity,
      resources.liquid,
      resources.reaction,
    ]) {
      clearEncoder.clearBuffer(buffer);
    }
    const initializePass = clearEncoder.beginComputePass({ label: 'Fire-X ambient field initialization' });
    initializePass.setPipeline(pipelines.initialize);
    initializePass.setBindGroup(0, resources.initializationGroup);
    const totalWorkgroups = Math.ceil(resources.cellCount / 256);
    const groupsX = Math.min(totalWorkgroups, device.limits.maxComputeWorkgroupsPerDimension);
    const groupsY = Math.ceil(totalWorkgroups / groupsX);
    initializePass.dispatchWorkgroups(groupsX, groupsY);
    initializePass.end();
    device.queue.submit([clearEncoder.finish()]);

    this.writeParticleInitialState(resources, this.waterSimulationActive());
    this.liquidFieldMayContainWater = false;
  }

  private writeParticleInitialState(resources: ResourceSet, particlesActive: boolean): void {
    const particleInitial = new Float32Array(resources.particleCount * PARTICLE_FLOATS);
    const domainWidth = this.parameters.domainWidth;
    const domainHeight = this.parameters.domainHeight;
    const domainDepth = this.parameters.domainDepth;
    const aimX = (0.52 - 0.035) * domainWidth;
    const aimY = (this.parameters.aimHeight - 0.5) * domainHeight;
    const aimLength = Math.hypot(aimX, aimY);
    const baseDirection = [aimX / aimLength, aimY / aimLength, 0] as const;
    const tangent = [baseDirection[1], -baseDirection[0], 0] as const;
    const hash = (value: number): number => {
      const hashed = Math.sin(value * 12.9898 + 1.317) * 43758.5453;
      return hashed - Math.floor(hashed);
    };
    for (let index = 0; index < resources.particleCount; index += 1) {
      const offset = index * PARTICLE_FLOATS;
      const phase = index / resources.particleCount;
      const angle = hash(index * 3.771 + 4.91) * Math.PI * 2;
      const radial = Math.sqrt(hash(index * 3.771 + 0.17));
      const spreadScale = this.parameters.nozzleType >= 0.5 ? 0.72 : 0.1;
      const cone = Math.tan((this.parameters.sprayAngle * Math.PI) / 180) * radial * spreadScale;
      const diskX = tangent[0] * Math.cos(angle);
      const diskY = tangent[1] * Math.cos(angle);
      const diskZ = -Math.sin(angle);
      let directionX = baseDirection[0] + diskX * cone;
      let directionY = baseDirection[1] + diskY * cone;
      let directionZ = diskZ * cone;
      const directionLength = Math.hypot(directionX, directionY, directionZ);
      directionX /= directionLength;
      directionY /= directionLength;
      directionZ /= directionLength;
      const speedVariation = this.parameters.nozzleType >= 0.5 ? 0.24 : 0.05;
      const speed = (0.82 + this.parameters.waterFlow * 0.82)
        * (1 - speedVariation * 0.5 + hash(index * 3.771 + 9.37) * speedVariation);
      const age = phase * 0.72;
      const velocityX = directionX * speed;
      const velocityY = directionY * speed - 0.58 * age;
      const velocityZ = directionZ * speed;
      const originRadius = this.parameters.nozzleType >= 0.5 ? 0.006 : 0.0028;
      particleInitial[offset] = 0.035
        + (diskX * radial * originRadius + directionX * speed * age) / domainWidth;
      particleInitial[offset + 1] = Math.max(
        0.012,
        0.5 + (diskY * radial * originRadius + directionY * speed * age - 0.29 * age * age) / domainHeight,
      );
      particleInitial[offset + 2] = 0.5
        + (diskZ * radial * originRadius + directionZ * speed * age) / domainDepth;
      particleInitial[offset + 3] = particlesActive ? age : -phase * 0.72;
      particleInitial[offset + 4] = particlesActive ? velocityX : 0;
      particleInitial[offset + 5] = particlesActive ? velocityY : 0;
      particleInitial[offset + 6] = particlesActive ? velocityZ : 0;
      particleInitial[offset + 7] = particlesActive ? 0.52 + this.parameters.waterFlow * 0.88 : 0;
    }
    const device = this.requireDevice();
    device.queue.writeBuffer(resources.particles[0], 0, particleInitial);
    device.queue.writeBuffer(resources.particles[1], 0, particleInitial);
    this.particleIndex = 0;
    this.particleFieldActive = particlesActive;
  }

  private resetOfflinePerformanceInfo(): void {
    this.offlineMeasuredFrames = 0;
    this.offlineSimulationMilliseconds = 0;
    this.offlinePresentationMilliseconds = 0;
    this.offlineVideoFrameConstructionMilliseconds = 0;
  }

  private writeUniforms(deltaTime: number): void {
    const device = this.requireDevice();
    const resources = this.requireResources();
    const configuration = this.activeConfiguration();
    const data = new Float32Array(UNIFORM_FLOATS);
    data.set([resources.dimensions[0], resources.dimensions[1], resources.dimensions[2], resources.cellCount], 0);
    data.set([deltaTime, this.simulationTime, this.frameIndex, configuration.timeStep], 4);
    data.set([
      this.parameters.fuelRate,
      this.parameters.oxygenRate,
      this.parameters.heatEfficiency,
      this.parameters.vorticity,
    ], 8);
    data.set([
      this.parameters.waterFlow,
      (this.parameters.sprayAngle * Math.PI) / 180,
      this.parameters.waterEnabled ? 1 : 0,
      300,
    ], 12);
    data.set([
      this.parameters.cameraYaw,
      this.parameters.cameraPitch,
      this.parameters.cameraDistance,
      this.canvas.width / Math.max(this.canvas.height, 1),
    ], 16);
    const raySteps = Math.min(384, configuration.raySteps);
    data.set([this.canvas.width, this.canvas.height, raySteps, resources.particleCount], 20);
    data.set([
      this.parameters.viewMode,
      this.parameters.aimHeight,
      this.parameters.nozzleType,
      this.parameters.exposure,
    ], 24);
    data.set([
      this.parameters.burnerSize,
      this.parameters.burnerDepth,
      this.parameters.sourceThickness,
      this.parameters.firePower,
    ], 28);
    data.set([
      this.parameters.domainWidth,
      this.parameters.domainHeight,
      this.parameters.domainDepth,
      0,
    ], 32);
    data.set([
      this.parameters.sourceLift,
      this.parameters.buoyancyScale,
      this.parameters.flamePersistence,
      this.parameters.viewZoom,
    ], 36);
    data.set([
      this.parameters.airEntrainment,
      this.parameters.frontDefinition,
      this.usesCinematicPresentation() ? 1 : 0,
      this.activeOpticalDetailTarget(),
    ], 40);
    device.queue.writeBuffer(this.requireUniformBuffer(), 0, data);
    if (resources.usesF16Pressure && this.f16PressureUniformBuffer) {
      device.queue.writeBuffer(this.f16PressureUniformBuffer, 0, new Float32Array([
        resources.dimensions[0],
        resources.dimensions[1],
        resources.dimensions[2],
        resources.cellCount,
        this.parameters.domainWidth,
        this.parameters.domainHeight,
        this.parameters.domainDepth,
        0,
      ]));
    }
  }

  private encodeSimulationStep(encoder: GPUCommandEncoder): void {
    const pipelines = this.requirePipelines();
    const resources = this.requireResources();
    const configuration = this.activeConfiguration();
    const [x, y, z] = resources.dimensions;
    const dispatch = (
      label: string,
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
    ): void => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
      pass.end();
    };

    const particleDispatch = (
      label: string,
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
    ): void => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(resources.particleCount / 64));
      pass.end();
    };
    const pressureDispatch = (
      label: string,
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      workgroups: readonly [number, number, number] = [
        Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4),
      ],
    ): void => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...workgroups);
      pass.end();
    };

    if (this.waterSimulationActive()) {
      if (!this.particleFieldActive) this.writeParticleInitialState(resources, true);
      encoder.clearBuffer(resources.liquid);
      const particleGroup = resources.particleGroups[this.particleIndex];
      particleDispatch('Fire-X SPH density', pipelines.particleDensity, particleGroup);
      particleDispatch('Fire-X SPH integration', pipelines.particleIntegrate, particleGroup);
      particleDispatch('Fire-X liquid voxel deposition', pipelines.particleDeposit, particleGroup);
      this.particleIndex = this.particleIndex === 0 ? 1 : 0;
      this.liquidFieldMayContainWater = true;
    } else if (this.liquidFieldMayContainWater || this.particleFieldActive) {
      encoder.clearBuffer(resources.liquid);
      this.writeParticleInitialState(resources, false);
      this.liquidFieldMayContainWater = false;
    }
    dispatch('Fire-X advect/react', pipelines.simulate, resources.simulationGroups[this.speciesIndex]);
    this.speciesIndex = this.speciesIndex === 0 ? 1 : 0;
    dispatch('Fire-X divergence', pipelines.divergence, resources.divergenceGroup);
    if (resources.cellCount >= FIREX_HEAVY_GRID_CELL_THRESHOLD) {
      const tiledDispatch = [Math.ceil(x / 8), Math.ceil(y / 8), Math.ceil(z / 4)] as const;
      if (resources.usesF16Pressure) {
        const pressurePipeline = pipelines.pressureF16Tiled!;
        const projectionPipeline = pipelines.projectF16!;
        const primaryGroups = resources.f16PrimaryPressureGroups!;
        const correctionGroups = resources.f16CorrectionPressureGroups!;
        for (let iteration = 0; iteration < configuration.pressureIterations; iteration += 1) {
          pressureDispatch(
            'Fire-X mixed-precision primary pressure iteration',
            pressurePipeline,
            primaryGroups[this.pressureIndex],
            tiledDispatch,
          );
          this.pressureIndex = this.pressureIndex === 0 ? 1 : 0;
        }
        dispatch(
          'Fire-X mixed-precision primary projection',
          projectionPipeline,
          primaryGroups[this.pressureIndex],
        );
        dispatch('Fire-X projected residual divergence', pipelines.divergence, resources.projectedDivergenceGroup);
        encoder.clearBuffer(resources.correctionPressure[0]);
        encoder.clearBuffer(resources.correctionPressure[1]);
        let correctionIndex: 0 | 1 = 0;
        for (let iteration = 0; iteration < configuration.correctionIterations; iteration += 1) {
          pressureDispatch(
            'Fire-X mixed-precision residual pressure iteration',
            pressurePipeline,
            correctionGroups[correctionIndex],
            tiledDispatch,
          );
          correctionIndex = correctionIndex === 0 ? 1 : 0;
        }
        dispatch(
          'Fire-X mixed-precision residual projection',
          projectionPipeline,
          correctionGroups[correctionIndex],
        );
      } else {
        encoder.clearBuffer(resources.pressure[0]);
        encoder.clearBuffer(resources.pressure[1]);
        this.pressureIndex = 0;
        for (let iteration = 0; iteration < configuration.pressureIterations; iteration += 1) {
          pressureDispatch(
            'Fire-X tiled primary pressure iteration',
            pipelines.pressureTiled,
            resources.pressureGroups[this.pressureIndex],
            tiledDispatch,
          );
          this.pressureIndex = this.pressureIndex === 0 ? 1 : 0;
        }
        dispatch('Fire-X project primary pressure', pipelines.project, resources.projectionGroups[this.pressureIndex]);
        dispatch('Fire-X projected residual divergence', pipelines.divergence, resources.projectedDivergenceGroup);
        encoder.clearBuffer(resources.pressure[0]);
        encoder.clearBuffer(resources.pressure[1]);
        let correctionIndex: 0 | 1 = 0;
        for (let iteration = 0; iteration < configuration.correctionIterations; iteration += 1) {
          pressureDispatch(
            'Fire-X tiled residual pressure iteration',
            pipelines.pressureTiled,
            resources.pressureGroups[correctionIndex],
            tiledDispatch,
          );
          correctionIndex = correctionIndex === 0 ? 1 : 0;
        }
        dispatch(
          'Fire-X residual velocity projection',
          pipelines.project,
          resources.sharedCorrectionProjectionGroups[correctionIndex],
        );
      }
    } else {
      for (let iteration = 0; iteration < configuration.pressureIterations; iteration += 1) {
        pressureDispatch(
          'Fire-X pressure iteration',
          pipelines.pressureGlobal,
          resources.pressureGroups[this.pressureIndex],
        );
        this.pressureIndex = this.pressureIndex === 0 ? 1 : 0;
      }
      dispatch('Fire-X project primary pressure', pipelines.project, resources.projectionGroups[this.pressureIndex]);
      dispatch('Fire-X projected residual divergence', pipelines.divergence, resources.projectedDivergenceGroup);
      encoder.clearBuffer(resources.correctionPressure[0]);
      encoder.clearBuffer(resources.correctionPressure[1]);
      let correctionIndex: 0 | 1 = 0;
      for (let iteration = 0; iteration < configuration.correctionIterations; iteration += 1) {
        pressureDispatch(
          'Fire-X residual pressure iteration',
          pipelines.pressureGlobal,
          resources.correctionPressureGroups[correctionIndex],
        );
        correctionIndex = correctionIndex === 0 ? 1 : 0;
      }
      dispatch(
        'Fire-X residual velocity projection',
        pipelines.project,
        resources.correctionProjectionGroups[correctionIndex],
      );
    }
    dispatch('Fire-X copy residual-corrected velocity', pipelines.copyFlow, resources.copyFlowGroup);
  }

  private submitOfflineSimulationStep(
    signal?: AbortSignal,
    prepareBeautyVolumes = false,
  ): PreparedBeautyVolumes | null {
    const device = this.requireDevice();
    const pipelines = this.requirePipelines();
    const resources = this.requireResources();
    const configuration = this.activeConfiguration();
    const [x, y, z] = resources.dimensions;
    let preparedBeautyVolumes: PreparedBeautyVolumes | null = null;
    const submit = (
      label: string,
      encode: (encoder: GPUCommandEncoder) => void,
    ): void => {
      const encoder = device.createCommandEncoder({ label });
      encode(encoder);
      device.queue.submit([encoder.finish()]);
      this.throwIfOfflineAborted(signal);
    };
    const dispatch = (
      encoder: GPUCommandEncoder,
      label: string,
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
    ): void => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
      pass.end();
    };
    const particleDispatch = (
      encoder: GPUCommandEncoder,
      label: string,
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
    ): void => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(resources.particleCount / 64));
      pass.end();
    };

    submit('Fire-X offline transport batch', (encoder) => {
      if (this.waterSimulationActive()) {
        if (!this.particleFieldActive) this.writeParticleInitialState(resources, true);
        encoder.clearBuffer(resources.liquid);
        const particleGroup = resources.particleGroups[this.particleIndex];
        particleDispatch(encoder, 'Fire-X offline SPH density', pipelines.particleDensity, particleGroup);
        particleDispatch(encoder, 'Fire-X offline SPH integration', pipelines.particleIntegrate, particleGroup);
        particleDispatch(encoder, 'Fire-X offline liquid deposition', pipelines.particleDeposit, particleGroup);
        this.particleIndex = this.particleIndex === 0 ? 1 : 0;
        this.liquidFieldMayContainWater = true;
      } else if (this.liquidFieldMayContainWater || this.particleFieldActive) {
        encoder.clearBuffer(resources.liquid);
        this.writeParticleInitialState(resources, false);
        this.liquidFieldMayContainWater = false;
      }
      dispatch(
        encoder,
        'Fire-X offline advect/react',
        pipelines.simulate,
        resources.simulationGroups[this.speciesIndex],
      );
      this.speciesIndex = this.speciesIndex === 0 ? 1 : 0;
      dispatch(encoder, 'Fire-X offline divergence', pipelines.divergence, resources.divergenceGroup);
    });

    const iterationBatchSize = 8;
    if (resources.cellCount >= FIREX_HEAVY_GRID_CELL_THRESHOLD) {
      const tiledDispatch = [Math.ceil(x / 8), Math.ceil(y / 8), Math.ceil(z / 4)] as const;
      const submitTiledIterations = (
        label: string,
        pipeline: GPUComputePipeline,
        iterations: number,
        nextBindGroup: () => GPUBindGroup,
        advance: () => void,
      ): void => {
        for (let first = 0; first < iterations; first += iterationBatchSize) {
          const count = Math.min(iterationBatchSize, iterations - first);
          submit(label, (encoder) => {
            const pass = encoder.beginComputePass({ label });
            pass.setPipeline(pipeline);
            for (let iteration = 0; iteration < count; iteration += 1) {
              pass.setBindGroup(0, nextBindGroup());
              pass.dispatchWorkgroups(...tiledDispatch);
              advance();
            }
            pass.end();
          });
        }
      };

      if (resources.usesF16Pressure) {
        const pressurePipeline = pipelines.pressureF16Tiled!;
        const projectionPipeline = pipelines.projectF16!;
        const primaryGroups = resources.f16PrimaryPressureGroups!;
        const correctionGroups = resources.f16CorrectionPressureGroups!;
        submitTiledIterations(
          'Fire-X offline mixed-precision primary pressure',
          pressurePipeline,
          configuration.pressureIterations,
          () => primaryGroups[this.pressureIndex],
          () => { this.pressureIndex = this.pressureIndex === 0 ? 1 : 0; },
        );
        submit('Fire-X offline mixed-precision primary projection', (encoder) => {
          dispatch(
            encoder,
            'Fire-X offline project mixed-precision primary pressure',
            projectionPipeline,
            primaryGroups[this.pressureIndex],
          );
          dispatch(
            encoder,
            'Fire-X offline projected residual divergence',
            pipelines.divergence,
            resources.projectedDivergenceGroup,
          );
          encoder.clearBuffer(resources.correctionPressure[0]);
          encoder.clearBuffer(resources.correctionPressure[1]);
        });
        let correctionIndex: 0 | 1 = 0;
        submitTiledIterations(
          'Fire-X offline mixed-precision residual pressure',
          pressurePipeline,
          configuration.correctionIterations,
          () => correctionGroups[correctionIndex],
          () => { correctionIndex = correctionIndex === 0 ? 1 : 0; },
        );
        submit('Fire-X offline mixed-precision corrected projection', (encoder) => {
          dispatch(
            encoder,
            'Fire-X offline mixed-precision residual velocity projection',
            projectionPipeline,
            correctionGroups[correctionIndex],
          );
          if (prepareBeautyVolumes && this.beautyViewActive()) {
            dispatch(
              encoder,
              'Fire-X offline corrected-flow copy and Beauty volume pack',
              pipelines.copyAndPackBeauty,
              resources.copyAndPackBeautyGroups[this.speciesIndex],
            );
            preparedBeautyVolumes = {
              resources,
              stateEpoch: this.stateEpoch,
              speciesIndex: this.speciesIndex,
            };
          } else {
            dispatch(encoder, 'Fire-X offline corrected-flow copy', pipelines.copyFlow, resources.copyFlowGroup);
          }
        });
      } else {
        submit('Fire-X offline clear primary pressure', (encoder) => {
          encoder.clearBuffer(resources.pressure[0]);
          encoder.clearBuffer(resources.pressure[1]);
        });
        this.pressureIndex = 0;
        submitTiledIterations(
          'Fire-X offline tiled primary pressure',
          pipelines.pressureTiled,
          configuration.pressureIterations,
          () => resources.pressureGroups[this.pressureIndex],
          () => { this.pressureIndex = this.pressureIndex === 0 ? 1 : 0; },
        );
        submit('Fire-X offline primary projection', (encoder) => {
          dispatch(
            encoder,
            'Fire-X offline project primary pressure',
            pipelines.project,
            resources.projectionGroups[this.pressureIndex],
          );
          dispatch(
            encoder,
            'Fire-X offline projected residual divergence',
            pipelines.divergence,
            resources.projectedDivergenceGroup,
          );
          encoder.clearBuffer(resources.pressure[0]);
          encoder.clearBuffer(resources.pressure[1]);
        });
        let correctionIndex: 0 | 1 = 0;
        submitTiledIterations(
          'Fire-X offline tiled residual pressure',
          pipelines.pressureTiled,
          configuration.correctionIterations,
          () => resources.pressureGroups[correctionIndex],
          () => { correctionIndex = correctionIndex === 0 ? 1 : 0; },
        );
        submit('Fire-X offline corrected projection', (encoder) => {
          dispatch(
            encoder,
            'Fire-X offline residual velocity projection',
            pipelines.project,
            resources.sharedCorrectionProjectionGroups[correctionIndex],
          );
          dispatch(encoder, 'Fire-X offline corrected-flow copy', pipelines.copyFlow, resources.copyFlowGroup);
        });
      }
    } else {
      const globalDispatch = [Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4)] as const;
      for (let first = 0; first < configuration.pressureIterations; first += iterationBatchSize) {
        const count = Math.min(iterationBatchSize, configuration.pressureIterations - first);
        submit('Fire-X offline primary pressure batch', (encoder) => {
          const pass = encoder.beginComputePass({ label: 'Fire-X offline primary pressure iterations' });
          pass.setPipeline(pipelines.pressureGlobal);
          for (let iteration = 0; iteration < count; iteration += 1) {
            pass.setBindGroup(0, resources.pressureGroups[this.pressureIndex]);
            pass.dispatchWorkgroups(...globalDispatch);
            this.pressureIndex = this.pressureIndex === 0 ? 1 : 0;
          }
          pass.end();
        });
      }
      submit('Fire-X offline primary projection', (encoder) => {
        dispatch(
          encoder,
          'Fire-X offline project primary pressure',
          pipelines.project,
          resources.projectionGroups[this.pressureIndex],
        );
        dispatch(
          encoder,
          'Fire-X offline projected residual divergence',
          pipelines.divergence,
          resources.projectedDivergenceGroup,
        );
        encoder.clearBuffer(resources.correctionPressure[0]);
        encoder.clearBuffer(resources.correctionPressure[1]);
      });

      let correctionIndex: 0 | 1 = 0;
      for (let first = 0; first < configuration.correctionIterations; first += iterationBatchSize) {
        const count = Math.min(iterationBatchSize, configuration.correctionIterations - first);
        submit('Fire-X offline residual pressure batch', (encoder) => {
          const pass = encoder.beginComputePass({ label: 'Fire-X offline residual pressure iterations' });
          pass.setPipeline(pipelines.pressureGlobal);
          for (let iteration = 0; iteration < count; iteration += 1) {
            pass.setBindGroup(0, resources.correctionPressureGroups[correctionIndex]);
            pass.dispatchWorkgroups(...globalDispatch);
            correctionIndex = correctionIndex === 0 ? 1 : 0;
          }
          pass.end();
        });
      }
      submit('Fire-X offline corrected projection', (encoder) => {
        dispatch(
          encoder,
          'Fire-X offline residual velocity projection',
          pipelines.project,
          resources.correctionProjectionGroups[correctionIndex],
        );
        dispatch(encoder, 'Fire-X offline corrected-flow copy', pipelines.copyFlow, resources.copyFlowGroup);
      });
    }
    return preparedBeautyVolumes;
  }

  private waterSimulationActive(): boolean {
    return this.parameters.waterEnabled && this.parameters.waterFlow > 0.001;
  }

  private divergenceDiagnosticActive(): boolean {
    return Math.round(this.parameters.viewMode) === 7;
  }

  private beautyViewActive(): boolean {
    return Math.round(this.parameters.viewMode) === 0;
  }

  private resolveBeautyVolumeHandoff(
    prepared: PreparedBeautyVolumes | null,
    resources: ResourceSet,
  ): BeautyVolumeHandoffResolution {
    const matchesCurrentState = prepared !== null
      && prepared.resources === resources
      && prepared.stateEpoch === this.stateEpoch
      && prepared.speciesIndex === this.speciesIndex;
    return {
      reusePackedVolumes: matchesCurrentState && this.beautyViewActive(),
      // Divergence is presentation-only. Compute it from canonical flow[0]
      // immediately before packing, including paused and zero-step renders.
      refreshFinalDivergence: this.divergenceDiagnosticActive(),
    };
  }

  private advanceStateEpoch(): void {
    this.stateEpoch = this.stateEpoch >= Number.MAX_SAFE_INTEGER
      ? 1
      : this.stateEpoch + 1;
  }

  private throwIfOfflineAborted(signal: AbortSignal | undefined): void {
    if (!signal?.aborted) return;
    throw new DOMException('Offline render canceled.', 'AbortError');
  }

  private encodeRender(
    encoder: GPUCommandEncoder,
    targetTexture?: GPUTexture,
    preparedBeautyVolumes: PreparedBeautyVolumes | null = null,
  ): void {
    const context = this.context;
    if (!context) return;
    const outputView = (targetTexture ?? context.getCurrentTexture()).createView();
    const resources = this.requireResources();
    const pipelines = this.requirePipelines();
    const [x, y, z] = resources.dimensions;
    const handoff = this.resolveBeautyVolumeHandoff(preparedBeautyVolumes, resources);
    if (handoff.refreshFinalDivergence) {
      const divergencePass = encoder.beginComputePass({
        label: 'Fire-X presentation final divergence',
      });
      divergencePass.setPipeline(pipelines.divergence);
      divergencePass.setBindGroup(0, resources.projectedDivergenceGroup);
      divergencePass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
      divergencePass.end();
    }
    if (!handoff.reusePackedVolumes) {
      const packPass = encoder.beginComputePass({ label: 'Fire-X pack filterable render fields' });
      packPass.setPipeline(pipelines.pack);
      packPass.setBindGroup(0, resources.packGroups[this.speciesIndex]);
      packPass.dispatchWorkgroups(Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4));
      packPass.end();
      // A standalone pack overwrites the shared filterable volumes. Invalidate
      // any fused handoff that may still be awaiting its presentation fence.
      this.advanceStateEpoch();
    }

    if (this.usesCinematicPresentation()) {
      const presentation = this.ensurePresentationResources();
      const scenePass = encoder.beginRenderPass({
        label: 'Fire-X cinematic HDR scene',
        colorAttachments: [{
          view: presentation.sceneView,
          clearValue: { r: 0.003, g: 0.005, b: 0.008, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      scenePass.setPipeline(pipelines.cinematicRender);
      scenePass.setBindGroup(0, resources.renderGroups[this.particleIndex]);
      if (this.offlineRenderTier !== null) {
        const stripeHeight = 128;
        for (let yOffset = 0; yOffset < presentation.height; yOffset += stripeHeight) {
          scenePass.setScissorRect(
            0,
            yOffset,
            presentation.width,
            Math.min(stripeHeight, presentation.height - yOffset),
          );
          scenePass.draw(3);
        }
        scenePass.setScissorRect(0, 0, presentation.width, presentation.height);
      } else {
        scenePass.draw(3);
      }
      if (this.particleFieldActive) {
        scenePass.setPipeline(pipelines.cinematicParticleRender);
        scenePass.draw(6, resources.particleCount);
      }
      scenePass.end();

      const bloomPass = encoder.beginRenderPass({
        label: 'Fire-X cinematic quarter-resolution bloom',
        colorAttachments: [{
          view: presentation.bloomView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      bloomPass.setPipeline(pipelines.bloom);
      bloomPass.setBindGroup(0, presentation.bloomGroup);
      bloomPass.draw(3);
      bloomPass.end();

      const compositePass = encoder.beginRenderPass({
        label: 'Fire-X cinematic tone-map and composite',
        colorAttachments: [{
          view: outputView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      compositePass.setPipeline(pipelines.composite);
      compositePass.setBindGroup(0, presentation.compositeGroup);
      compositePass.draw(3);
      compositePass.end();
      return;
    }

    const pass = encoder.beginRenderPass({
      label: 'Fire-X volume rendering',
      colorAttachments: [
        {
          view: outputView,
          clearValue: { r: 0.003, g: 0.005, b: 0.008, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(pipelines.render);
    pass.setBindGroup(0, resources.renderGroups[this.particleIndex]);
    pass.draw(3);
    if (this.particleFieldActive) {
      pass.setPipeline(pipelines.particleRender);
      pass.draw(6, resources.particleCount);
    }
    pass.end();
  }

  private ensurePresentationResources(): PresentationResources {
    const existing = this.presentation;
    if (existing && existing.width === this.canvas.width && existing.height === this.canvas.height) {
      return existing;
    }
    this.destroyPresentationResources();

    const device = this.requireDevice();
    const postLayout = this.requirePostLayout();
    const postSampler = this.requirePostSampler();
    const uniformBuffer = this.requireUniformBuffer();
    const width = Math.max(this.canvas.width, 1);
    const height = Math.max(this.canvas.height, 1);
    const bloomWidth = Math.max(Math.ceil(width / 4), 1);
    const bloomHeight = Math.max(Math.ceil(height / 4), 1);
    const sceneTexture = device.createTexture({
      label: 'Fire-X cinematic linear HDR scene',
      size: { width, height },
      format: 'rgba16float',
      usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
    });
    const bloomTexture = device.createTexture({
      label: 'Fire-X cinematic quarter-resolution bloom',
      size: { width: bloomWidth, height: bloomHeight },
      format: 'rgba16float',
      usage: GPU_TEXTURE_USAGE.RENDER_ATTACHMENT | GPU_TEXTURE_USAGE.TEXTURE_BINDING,
    });
    const sceneView = sceneTexture.createView();
    const bloomView = bloomTexture.createView();
    const bloomGroup = device.createBindGroup({
      label: 'Fire-X cinematic bloom input',
      layout: postLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: postSampler },
        { binding: 2, resource: sceneView },
        { binding: 3, resource: sceneView },
      ],
    });
    const compositeGroup = device.createBindGroup({
      label: 'Fire-X cinematic scene and bloom composite',
      layout: postLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: postSampler },
        { binding: 2, resource: sceneView },
        { binding: 3, resource: bloomView },
      ],
    });
    this.presentation = {
      width,
      height,
      sceneTexture,
      sceneView,
      bloomTexture,
      bloomView,
      bloomGroup,
      compositeGroup,
    };
    return this.presentation;
  }

  private destroyPresentationResources(): void {
    const presentation = this.presentation;
    if (!presentation) return;
    presentation.sceneTexture.destroy();
    presentation.bloomTexture.destroy();
    this.presentation = null;
  }

  private ensureOfflineReadbackBuffer(width: number, height: number, size: number): GPUBuffer {
    const existing = this.offlineReadbackBuffer;
    if (
      existing
      && this.offlineReadbackBufferSize === size
      && this.offlineReadbackWidth === width
      && this.offlineReadbackHeight === height
    ) return existing;
    this.destroyOfflineReadbackBuffer();
    const buffer = this.requireDevice().createBuffer({
      label: `Fire-X ${width}\u00d7${height} offline video readback`,
      size,
      usage: GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST,
    });
    this.offlineReadbackBuffer = buffer;
    this.offlineReadbackBufferSize = size;
    this.offlineReadbackWidth = width;
    this.offlineReadbackHeight = height;
    return buffer;
  }

  private destroyOfflineReadbackBuffer(): void {
    this.offlineReadbackBuffer?.destroy();
    this.offlineReadbackBuffer = null;
    this.offlineReadbackBufferSize = 0;
    this.offlineReadbackWidth = 0;
    this.offlineReadbackHeight = 0;
  }

  private domainExtent(): readonly [number, number, number] {
    return [
      this.parameters.domainWidth,
      this.parameters.domainHeight,
      this.parameters.domainDepth,
    ];
  }

  private effectiveGridDimensions(): FireXGridDimensions {
    if (this.offlineRenderTier !== null) {
      return resolveFireXOfflineDimensions(this.offlineRenderTier, this.gridOverride);
    }
    return this.gridOverride ?? FIREX_QUALITY[this.quality].dimensions;
  }

  private activeConfiguration(): QualityConfiguration {
    return this.offlineRenderTier !== null
      ? FIREX_OFFLINE_QUALITY[this.offlineRenderTier]
      : FIREX_QUALITY[this.quality];
  }

  private activeOpticalDetailTarget(): number {
    if (this.offlineRenderTier === null) return this.parameters.opticalDetailTarget;
    return this.offlineOpticalDetailTarget
      ?? FIREX_OFFLINE_QUALITY[this.offlineRenderTier].opticalDetailTarget;
  }

  private usesCinematicPresentation(): boolean {
    return this.offlineRenderTier !== null || this.quality === 'cinematic';
  }

  private assertGridSupported(
    dimensions: FireXGridDimensions,
    memoryGuardBytes = this.gridMemoryBudgetBytes,
  ): void {
    const device = this.requireDevice();
    const currentBytes = this.resources
      ? estimateFireXGridMemory(this.resources.dimensions, this.resources.particleCount).totalBytes
      : 0;
    const result = preflightFireXGrid(
      dimensions,
      {
        maxTextureDimension3D: device.limits.maxTextureDimension3D,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        maxBufferSize: device.limits.maxBufferSize,
      },
      {
        particleCount: this.activeConfiguration().particleCount,
        memoryBudgetBytes: null,
      },
    );
    const issues = result.issues.map((issue) => issue.message);
    const peakBytes = currentBytes + result.estimate.totalBytes;
    if (peakBytes > memoryGuardBytes) {
      issues.push(
        `Transactional rebuild peak requires about ${formatFireXBytes(peakBytes)} (${formatFireXBytes(currentBytes)} current + ${formatFireXBytes(result.estimate.totalBytes)} replacement), above the selected ${formatFireXBytes(memoryGuardBytes)} guard.`,
      );
    }
    if (issues.length > 0) {
      throw new Error(issues.join(' '));
    }
  }

  private destroyResourceSet(resources: ResourceSet): void {
    for (const buffer of [
      ...resources.flow,
      ...resources.species,
      ...resources.pressure,
      ...resources.correctionPressure,
      ...resources.particles,
    ]) {
      buffer.destroy();
    }
    resources.divergence.destroy();
    resources.particleDensity.destroy();
    resources.liquid.destroy();
    resources.reaction.destroy();
    resources.flowTexture.destroy();
    resources.chemistryTexture.destroy();
    resources.mediaTexture.destroy();
    resources.detailTexture.destroy();
  }

  private destroyResources(): void {
    const resources = this.resources;
    if (!resources) return;
    this.resources = null;
    this.destroyResourceSet(resources);
  }

  private requireDevice(): GPUDevice {
    if (!this.device) throw new Error('The Fire-X WebGPU device is unavailable.');
    return this.device;
  }

  private requireOfflineDevice(): GPUDevice {
    if (this.disposed) throw new Error('The Fire-X engine has been disposed and cannot render an export.');
    if (!this.initialized) throw new Error('The Fire-X engine must be initialized before rendering an export.');
    if (this.deviceLost) throw new Error('The Fire-X WebGPU device is lost and cannot render an export.');
    if (this.rebuilding) throw new Error('Fire-X resources are still rebuilding; wait before rendering an export.');
    if (!this.context) throw new Error('The Fire-X WebGPU canvas context is unavailable.');
    this.requirePipelines();
    this.requireResources();
    this.requireUniformBuffer();
    return this.requireDevice();
  }

  private requireUniformBuffer(): GPUBuffer {
    if (!this.uniformBuffer) throw new Error('The Fire-X uniform buffer is unavailable.');
    return this.uniformBuffer;
  }

  private requireComputeLayout(): GPUBindGroupLayout {
    if (!this.computeLayout) throw new Error('The Fire-X compute layout is unavailable.');
    return this.computeLayout;
  }

  private requireParticleLayout(): GPUBindGroupLayout {
    if (!this.particleLayout) throw new Error('The Fire-X particle layout is unavailable.');
    return this.particleLayout;
  }

  private requirePackLayout(): GPUBindGroupLayout {
    if (!this.packLayout) throw new Error('The Fire-X field-packing layout is unavailable.');
    return this.packLayout;
  }

  private requireVolumeSampler(): GPUSampler {
    if (!this.volumeSampler) throw new Error('The Fire-X trilinear volume sampler is unavailable.');
    return this.volumeSampler;
  }

  private requireDetailSampler(): GPUSampler {
    if (!this.detailSampler) throw new Error('The Fire-X cinematic detail sampler is unavailable.');
    return this.detailSampler;
  }

  private requirePostSampler(): GPUSampler {
    if (!this.postSampler) throw new Error('The Fire-X cinematic presentation sampler is unavailable.');
    return this.postSampler;
  }

  private requireRenderLayout(): GPUBindGroupLayout {
    if (!this.renderLayout) throw new Error('The Fire-X render layout is unavailable.');
    return this.renderLayout;
  }

  private requirePostLayout(): GPUBindGroupLayout {
    if (!this.postLayout) throw new Error('The Fire-X cinematic presentation layout is unavailable.');
    return this.postLayout;
  }

  private requirePipelines(): PipelineSet {
    if (!this.pipelines) throw new Error('The Fire-X GPU pipelines are unavailable.');
    return this.pipelines;
  }

  private requireResources(): ResourceSet {
    if (!this.resources) throw new Error('The Fire-X simulation resources are unavailable.');
    return this.resources;
  }
}
