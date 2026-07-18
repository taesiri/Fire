import type { OfflineRenderTier, QualityLevel, SimulationEngine } from '../types';
import { fitCanvasSize } from '../canvasSizing';
import { splitOfflineSimulationDelta } from '../offlineTimeline';
import {
  bindTextureUnit,
  createHalfFloatTarget,
  deleteTarget,
  GLProgram,
  PingPongTarget,
  type RenderTarget,
} from './gl';
import {
  ADVECT_STATE_FRAGMENT,
  ADVECT_VELOCITY_FRAGMENT,
  CURL_FRAGMENT,
  DETAIL_FRAGMENT,
  DIVERGENCE_FRAGMENT,
  FORCE_FRAGMENT,
  FULLSCREEN_VERTEX,
  JACOBI_FRAGMENT,
  PARTICLE_PROJECT_FRAGMENT,
  PARTICLE_PROJECT_VERTEX,
  PARTICLE_UPDATE_FRAGMENT,
  PARTICLE_UPDATE_VERTEX,
  PROJECT_FRAGMENT,
  RENDER_FRAGMENT,
  SPARK_FRAGMENT,
  SPARK_VERTEX,
  SOURCE_VELOCITY_FRAGMENT,
  THERMO_FRAGMENT,
} from './shaders';

export interface QualityConfiguration {
  tileHeight: number;
  slices: number;
  columns: number;
  jacobiIterations: number;
  particleCount: number;
  particleRadius: number;
  raySteps: number;
  renderScale: number;
  fixedTimeStep: number;
  maximumSubsteps: number;
}

export interface HorvathOfflineConfiguration extends QualityConfiguration {
  minimumTextureSize: number;
}

interface ParticleResources {
  count: number;
  buffers: [WebGLBuffer, WebGLBuffer];
  vertexArrays: [WebGLVertexArrayObject, WebGLVertexArrayObject];
  transformFeedbacks: [WebGLTransformFeedback, WebGLTransformFeedback];
  readIndex: 0 | 1;
}

interface SimulationResources {
  tileWidth: number;
  tileHeight: number;
  columns: number;
  rows: number;
  slices: number;
  atlasWidth: number;
  atlasHeight: number;
  jacobiIterations: number;
  particleRadius: number;
  raySteps: number;
  fixedTimeStep: number;
  maximumSubsteps: number;
  detailFrequencyScale: number;
  state: PingPongTarget;
  velocity: PingPongTarget;
  pressure: PingPongTarget;
  auxiliary: RenderTarget;
  particleMotion: RenderTarget;
  particleThermo: RenderTarget;
  particleFramebuffer: WebGLFramebuffer;
  particles: ParticleResources;
}

interface ProgramSet {
  thermo: GLProgram;
  sourceVelocity: GLProgram;
  advectState: GLProgram;
  advectVelocity: GLProgram;
  detail: GLProgram;
  curl: GLProgram;
  force: GLProgram;
  divergence: GLProgram;
  jacobi: GLProgram;
  project: GLProgram;
  render: GLProgram;
  updateParticles: GLProgram;
  projectParticles: GLProgram;
  sparks: GLProgram;
}

interface HorvathParameters {
  sourceMode: 0 | 1 | 2;
  sourceSize: number;
  flameHeight: number;
  emission: number;
  buoyancy: number;
  vorticity: number;
  cooling: number;
  density: number;
  detail: number;
  exposure: number;
  cameraYaw: number;
  cameraPitch: number;
  cameraDistance: number;
  viewZoom: number;
}

export const HORVATH_QUALITY: Record<QualityLevel, QualityConfiguration> = {
  low: {
    tileHeight: 48,
    slices: 12,
    columns: 4,
    jacobiIterations: 4,
    particleCount: 2048,
    particleRadius: 2.0,
    raySteps: 22,
    renderScale: 0.82,
    fixedTimeStep: 1 / 30,
    maximumSubsteps: 1,
  },
  balanced: {
    tileHeight: 64,
    slices: 18,
    columns: 6,
    jacobiIterations: 6,
    particleCount: 4096,
    particleRadius: 2.7,
    raySteps: 28,
    renderScale: 1,
    fixedTimeStep: 1 / 30,
    maximumSubsteps: 1,
  },
  high: {
    tileHeight: 88,
    slices: 28,
    columns: 7,
    jacobiIterations: 10,
    particleCount: 8192,
    particleRadius: 3.7,
    raySteps: 44,
    renderScale: 1,
    fixedTimeStep: 1 / 30,
    maximumSubsteps: 1,
  },
  maximum: {
    tileHeight: 144,
    slices: 48,
    columns: 8,
    jacobiIterations: 18,
    particleCount: 32768,
    particleRadius: 4.7,
    raySteps: 96,
    renderScale: 1.25,
    fixedTimeStep: 1 / 30,
    maximumSubsteps: 1,
  },
  cinematic: {
    tileHeight: 176,
    slices: 56,
    columns: 8,
    jacobiIterations: 24,
    particleCount: 65536,
    particleRadius: 3.0,
    raySteps: 128,
    renderScale: 1.15,
    fixedTimeStep: 1 / 30,
    maximumSubsteps: 1,
  },
};

export const HORVATH_OFFLINE_QUALITY: Readonly<Record<OfflineRenderTier, Readonly<HorvathOfflineConfiguration>>> = Object.freeze({
  hd: Object.freeze({
    tileHeight: 256,
    slices: 64,
    columns: 8,
    jacobiIterations: 32,
    particleCount: 131_072,
    particleRadius: 3,
    raySteps: 256,
    renderScale: 1,
    fixedTimeStep: 1 / 120,
    maximumSubsteps: 2,
    minimumTextureSize: 2496,
  }),
  qhd: Object.freeze({
    tileHeight: 352,
    slices: 80,
    columns: 8,
    jacobiIterations: 48,
    particleCount: 262_144,
    particleRadius: 3,
    raySteps: 352,
    renderScale: 1,
    fixedTimeStep: 1 / 120,
    maximumSubsteps: 2,
    minimumTextureSize: 3520,
  }),
  uhd: Object.freeze({
    tileHeight: 448,
    slices: 96,
    columns: 8,
    jacobiIterations: 64,
    particleCount: 524_288,
    particleRadius: 3,
    raySteps: 448,
    renderScale: 1,
    fixedTimeStep: 1 / 120,
    maximumSubsteps: 2,
    minimumTextureSize: 5376,
  }),
});

export interface HorvathMemoryEstimate {
  atlasWidth: number;
  atlasHeight: number;
  atlasBytes: number;
  particleBytes: number;
  totalBytes: number;
}

export function estimateHorvathMemory(configuration: QualityConfiguration): HorvathMemoryEstimate {
  const tileWidth = roundToMultiple(configuration.tileHeight * 1.22, 4);
  const rows = Math.ceil(configuration.slices / configuration.columns);
  const atlasWidth = tileWidth * configuration.columns;
  const atlasHeight = configuration.tileHeight * rows;
  const atlasBytes = atlasWidth * atlasHeight * 9 * 8;
  const particleBytes = configuration.particleCount * 2 * 8 * Float32Array.BYTES_PER_ELEMENT;
  return { atlasWidth, atlasHeight, atlasBytes, particleBytes, totalBytes: atlasBytes + particleBytes };
}

function finiteOr(value: number | boolean, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function roundToMultiple(value: number, multiple: number): number {
  return Math.max(multiple, Math.round(value / multiple) * multiple);
}

function requireOutputDimension(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer number of pixels.`);
  }
  return value;
}

function requireFrameTimeMicroseconds(valueSeconds: number, label: string): number {
  if (!Number.isFinite(valueSeconds) || valueSeconds < 0) {
    throw new Error(`${label} must be finite and non-negative.`);
  }
  const valueMicroseconds = Math.round(valueSeconds * 1_000_000);
  if (!Number.isSafeInteger(valueMicroseconds)) {
    throw new Error(`${label} is outside the supported timestamp range.`);
  }
  return valueMicroseconds;
}

export class HorvathEngine implements SimulationEngine {
  readonly backend = 'WebGL2 · slab atlas · TF + RK2 advection';

  private gl: WebGL2RenderingContext | null = null;
  private programs: ProgramSet | null = null;
  private resources: SimulationResources | null = null;
  private vertexArray: WebGLVertexArrayObject | null = null;
  private quality: QualityLevel = 'balanced';
  private paused = false;
  private initialized = false;
  private disposed = false;
  private contextLost = false;
  private simulationTime = 0;
  private accumulator = 0;
  private offlineRenderTier: OfflineRenderTier | null = null;
  private offlineReadbackPixels: Uint8Array<ArrayBuffer> | null = null;
  private readonly zeroColor = new Float32Array(4);
  private parameters: HorvathParameters = {
    sourceMode: 0,
    sourceSize: 1,
    flameHeight: 1,
    emission: 0.86,
    buoyancy: 0.92,
    vorticity: 1.45,
    cooling: 0.58,
    density: 1.36,
    detail: 0.64,
    exposure: 1.08,
    cameraYaw: -0.34,
    cameraPitch: 0.03,
    cameraDistance: 4.9,
    viewZoom: 1,
  };

  private readonly handleContextLost = (event: Event): void => {
    event.preventDefault();
    this.contextLost = true;
    this.accumulator = 0;
  };

  private readonly handleContextRestored = (): void => {
    if (this.disposed || !this.gl) return;
    try {
      this.programs = null;
      this.resources = null;
      this.vertexArray = null;
      this.setupGpuObjects();
      this.contextLost = false;
    } catch (error) {
      this.contextLost = true;
      console.error('Unable to restore the Horvath WebGL2 engine.', error);
    }
  };

  constructor(private readonly canvas: HTMLCanvasElement) {
    canvas.addEventListener('webglcontextlost', this.handleContextLost);
    canvas.addEventListener('webglcontextrestored', this.handleContextRestored);
  }

  get detail(): string {
    const resources = this.resources;
    return resources
      ? `${resources.tileWidth}×${resources.tileHeight}×${resources.slices} · 1×1×1 normalized domain · ${resources.particles.count.toLocaleString()} TF particles · ${resources.jacobiIterations} pressure · ${resources.raySteps} rays · ${Math.round(1 / resources.fixedTimeStep)} Hz sim (≤${resources.maximumSubsteps}/frame)`
      : '3D slab atlas pending';
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    if (this.disposed) throw new Error('The Horvath engine has already been disposed.');

    const gl = this.canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is required for the Horvath–Geiger slice solver.');
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float is required for half-float slab operations.');
    }
    if (!gl.getExtension('OES_texture_float_linear')) {
      throw new Error('OES_texture_float_linear is required for filtered half-float advection.');
    }

    this.gl = gl;
    this.setupGpuObjects();
    this.initialized = true;
  }

  resize(width: number, height: number, pixelRatio: number): void {
    const configuration = this.activeConfiguration();
    const renderScale = configuration.renderScale;
    const { width: nextWidth, height: nextHeight } = fitCanvasSize(
      width,
      height,
      pixelRatio * renderScale,
      4096,
      4096,
    );
    if (this.canvas.width === nextWidth && this.canvas.height === nextHeight) return;
    this.offlineReadbackPixels = null;
    this.canvas.width = nextWidth;
    this.canvas.height = nextHeight;

    if (this.gl && this.programs && !this.contextLost) {
      const desiredWidth = this.desiredTileWidth(configuration.tileHeight);
      if (!this.resources || Math.abs(desiredWidth - this.resources.tileWidth) >= 4) {
        this.rebuildResources();
      }
    }
  }

  resizeOutput(widthPx: number, heightPx: number): void {
    const gl = this.requireOfflineGl();
    const width = requireOutputDimension(widthPx, 'Output width');
    const height = requireOutputDimension(heightPx, 'Output height');
    const viewportLimits = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array;
    const maxRenderbufferSize = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number;
    const maximumWidth = Math.min(viewportLimits[0], maxRenderbufferSize);
    const maximumHeight = Math.min(viewportLimits[1], maxRenderbufferSize);
    if (width > maximumWidth || height > maximumHeight) {
      throw new Error(
        `The requested ${width}\u00d7${height} output exceeds this WebGL2 device limit of ${maximumWidth}\u00d7${maximumHeight} pixels.`,
      );
    }

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.offlineReadbackPixels = null;
      this.canvas.width = width;
      this.canvas.height = height;
    }
    if (this.canvas.width !== width || this.canvas.height !== height) {
      throw new Error(`The browser could not allocate the requested ${width}\u00d7${height} output canvas.`);
    }
    if (this.contextLost || gl.isContextLost()) {
      throw new Error('The Horvath WebGL2 context was lost while resizing the offline output.');
    }
  }

  frame(timeSeconds: number, deltaSeconds: number): number {
    void timeSeconds;
    const gl = this.gl;
    const resources = this.resources;
    if (!gl || !resources || !this.programs || this.contextLost || gl.isContextLost()) return 0;
    return this.renderReadyFrame(gl, resources, deltaSeconds, !this.paused, false);
  }

  async renderOfflineFrame(
    deltaSeconds: number,
    present = true,
    signal?: AbortSignal,
  ): Promise<void> {
    this.renderOfflineFrameSynchronously(deltaSeconds, present, signal);
  }

  async renderOfflineVideoFrame(
    deltaSeconds: number,
    timestampSeconds: number,
    durationSeconds: number,
    signal?: AbortSignal,
  ): Promise<VideoFrame> {
    const timestamp = requireFrameTimeMicroseconds(timestampSeconds, 'Video frame timestamp');
    const endTimestamp = requireFrameTimeMicroseconds(
      timestampSeconds + durationSeconds,
      'Video frame end timestamp',
    );
    const duration = endTimestamp - timestamp;
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0 || duration <= 0) {
      throw new Error('Video frame duration must be finite and positive.');
    }

    const initialGl = this.requireOfflineGl();
    while (initialGl.getError() !== initialGl.NO_ERROR) {
      // Clear any diagnostic error left by a prior capability/allocation probe;
      // only errors produced by this exact render/readback should fail export.
    }

    // This method deliberately performs the render, GPU fence, readback, and
    // VideoFrame construction without an await. The default framebuffer can be
    // recycled by the browser compositor as soon as control returns to it.
    const gl = this.renderOfflineFrameSynchronously(deltaSeconds, true, signal, true);
    const width = this.canvas.width;
    const height = this.canvas.height;
    if (gl.drawingBufferWidth !== width || gl.drawingBufferHeight !== height) {
      throw new Error(
        `The Horvath offline drawing buffer is ${gl.drawingBufferWidth}\u00d7${gl.drawingBufferHeight}, expected ${width}\u00d7${height}.`,
      );
    }

    const byteLength = width * height * 4;
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      throw new Error(`The Horvath ${width}\u00d7${height} RGBA readback size is invalid.`);
    }
    let readback = this.offlineReadbackPixels;
    if (!readback || readback.byteLength !== byteLength) {
      readback = new Uint8Array(byteLength);
      this.offlineReadbackPixels = readback;
    }

    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.readBuffer(gl.BACK);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, readback);
    const readbackError = gl.getError();
    if (signal?.aborted) throw new DOMException('Offline render canceled.', 'AbortError');
    if (this.contextLost || gl.isContextLost()) {
      throw new Error('The Horvath WebGL2 context was lost during offline frame readback.');
    }
    if (readbackError !== gl.NO_ERROR) {
      throw new Error(`Horvath offline frame readback failed with WebGL error ${readbackError}.`);
    }

    // WebGL's framebuffer origin is bottom-left. VideoFrame packed RGBA rows
    // are top-left, so copy into a fresh buffer in reverse row order. The
    // VideoFrame constructor synchronously copies BufferSource data, keeping
    // the returned frame independent from the reusable GPU readback buffer.
    const pixels = new Uint8Array(byteLength);
    const rowBytes = width * 4;
    for (let targetY = 0; targetY < height; targetY += 1) {
      const sourceOffset = (height - targetY - 1) * rowBytes;
      pixels.set(readback.subarray(sourceOffset, sourceOffset + rowBytes), targetY * rowBytes);
    }

    return new VideoFrame(pixels, {
      format: 'RGBA',
      codedWidth: width,
      codedHeight: height,
      displayWidth: width,
      displayHeight: height,
      timestamp,
      duration,
    });
  }

  private renderOfflineFrameSynchronously(
    deltaSeconds: number,
    present: boolean,
    signal?: AbortSignal,
    validateErrors = false,
  ): WebGL2RenderingContext {
    if (!Number.isFinite(deltaSeconds) || deltaSeconds < 0) {
      throw new Error('Offline frame delta must be finite and non-negative.');
    }
    const gl = this.requireOfflineGl();
    const resources = this.requireResources();
    this.accumulator = 0;
    for (const timeStep of splitOfflineSimulationDelta(deltaSeconds, resources.fixedTimeStep)) {
      if (signal?.aborted) throw new DOMException('Offline render canceled.', 'AbortError');
      this.stepSimulation(timeStep, validateErrors);
      this.simulationTime += timeStep;
      if (validateErrors) {
        const simulationError = gl.getError();
        if (simulationError !== gl.NO_ERROR) {
          throw new Error(`Horvath offline simulation failed with WebGL error ${simulationError}.`);
        }
      }
    }
    if (present) {
      this.renderVolume(signal);
      if (validateErrors) {
        const presentationError = gl.getError();
        if (presentationError !== gl.NO_ERROR) {
          throw new Error(`Horvath offline presentation failed with WebGL error ${presentationError}.`);
        }
      }
    }
    gl.flush();
    gl.finish();
    if (signal?.aborted) throw new DOMException('Offline render canceled.', 'AbortError');
    if (this.contextLost || gl.isContextLost()) {
      throw new Error('The Horvath WebGL2 context was lost during offline rendering.');
    }
    return gl;
  }

  setOfflineRenderTier(tier: OfflineRenderTier | null): void {
    if (tier === this.offlineRenderTier) return;
    const previousTier = this.offlineRenderTier;
    this.offlineRenderTier = tier;
    try {
      if (this.gl && this.programs && !this.contextLost) this.rebuildResources();
    } catch (error) {
      this.offlineRenderTier = previousTier;
      throw error;
    }
    if (tier === null) this.offlineReadbackPixels = null;
    this.accumulator = 0;
  }

  reset(): void {
    this.simulationTime = 0;
    this.accumulator = 0;
    if (!this.gl || !this.resources || this.contextLost) return;
    this.clearResources();
    this.resetParticles();
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    if (paused) this.accumulator = 0;
  }

  setQuality(quality: QualityLevel): void {
    if (this.offlineRenderTier !== null) {
      throw new Error('Compute quality cannot change while an exact offline solver bundle is active.');
    }
    if (quality === this.quality) return;
    this.quality = quality;
    this.accumulator = 0;
    if (this.gl && this.programs && !this.contextLost) this.rebuildResources();
  }

  setParameter(name: string, value: number | boolean): void {
    const previousSourceMode = this.sourceMode();
    switch (name) {
      case 'sourceMode':
        this.parameters.sourceMode = Math.max(0, Math.min(2, Math.round(finiteOr(value, this.parameters.sourceMode)))) as 0 | 1 | 2;
        break;
      case 'sourceSize':
        this.parameters.sourceSize = Math.max(0.3, Math.min(3, finiteOr(value, this.parameters.sourceSize)));
        break;
      case 'flameHeight':
        this.parameters.flameHeight = Math.max(0.4, Math.min(2.5, finiteOr(value, this.parameters.flameHeight)));
        break;
      case 'emission':
        this.parameters.emission = finiteOr(value, this.parameters.emission);
        break;
      case 'buoyancy':
        this.parameters.buoyancy = finiteOr(value, this.parameters.buoyancy);
        break;
      case 'vorticity':
        this.parameters.vorticity = finiteOr(value, this.parameters.vorticity);
        break;
      case 'cooling':
        this.parameters.cooling = finiteOr(value, this.parameters.cooling);
        break;
      case 'density':
        this.parameters.density = finiteOr(value, this.parameters.density);
        break;
      case 'detail':
        this.parameters.detail = finiteOr(value, this.parameters.detail);
        break;
      case 'exposure':
        this.parameters.exposure = Math.max(0.25, Math.min(3, finiteOr(value, this.parameters.exposure)));
        break;
      case 'cameraYaw':
        this.parameters.cameraYaw = finiteOr(value, this.parameters.cameraYaw);
        break;
      case 'cameraPitch':
        this.parameters.cameraPitch = finiteOr(value, this.parameters.cameraPitch);
        break;
      case 'cameraDistance':
        this.parameters.cameraDistance = finiteOr(value, this.parameters.cameraDistance);
        break;
      case 'viewZoom':
        this.parameters.viewZoom = Math.max(0.5, Math.min(3, finiteOr(value, this.parameters.viewZoom)));
        break;
      default:
        break;
    }
    if (previousSourceMode !== this.sourceMode() && this.gl && this.resources && !this.contextLost) {
      this.simulationTime = 0;
      this.accumulator = 0;
      this.clearResources();
      this.resetParticles();
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.canvas.removeEventListener('webglcontextlost', this.handleContextLost);
    this.canvas.removeEventListener('webglcontextrestored', this.handleContextRestored);
    this.disposeGpuObjects();
    this.gl = null;
    this.offlineReadbackPixels = null;
    this.initialized = false;
    this.disposed = true;
  }

  private setupGpuObjects(): void {
    const gl = this.requireGl();
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float was unavailable after context restoration.');
    }
    if (!gl.getExtension('OES_texture_float_linear')) {
      throw new Error('OES_texture_float_linear was unavailable after context restoration.');
    }

    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.SCISSOR_TEST);
    const vertexArray = gl.createVertexArray();
    if (!vertexArray) throw new Error('Unable to allocate a fullscreen vertex array.');
    this.vertexArray = vertexArray;
    gl.bindVertexArray(vertexArray);

    this.programs = {
      thermo: new GLProgram(gl, FULLSCREEN_VERTEX, THERMO_FRAGMENT),
      sourceVelocity: new GLProgram(gl, FULLSCREEN_VERTEX, SOURCE_VELOCITY_FRAGMENT),
      advectState: new GLProgram(gl, FULLSCREEN_VERTEX, ADVECT_STATE_FRAGMENT),
      advectVelocity: new GLProgram(gl, FULLSCREEN_VERTEX, ADVECT_VELOCITY_FRAGMENT),
      detail: new GLProgram(gl, FULLSCREEN_VERTEX, DETAIL_FRAGMENT),
      curl: new GLProgram(gl, FULLSCREEN_VERTEX, CURL_FRAGMENT),
      force: new GLProgram(gl, FULLSCREEN_VERTEX, FORCE_FRAGMENT),
      divergence: new GLProgram(gl, FULLSCREEN_VERTEX, DIVERGENCE_FRAGMENT),
      jacobi: new GLProgram(gl, FULLSCREEN_VERTEX, JACOBI_FRAGMENT),
      project: new GLProgram(gl, FULLSCREEN_VERTEX, PROJECT_FRAGMENT),
      render: new GLProgram(gl, FULLSCREEN_VERTEX, RENDER_FRAGMENT),
      updateParticles: new GLProgram(
        gl,
        PARTICLE_UPDATE_VERTEX,
        PARTICLE_UPDATE_FRAGMENT,
        ['vPositionAge', 'vVelocitySeed'],
      ),
      projectParticles: new GLProgram(gl, PARTICLE_PROJECT_VERTEX, PARTICLE_PROJECT_FRAGMENT),
      sparks: new GLProgram(gl, SPARK_VERTEX, SPARK_FRAGMENT),
    };
    this.rebuildResources();
  }

  private renderReadyFrame(
    gl: WebGL2RenderingContext,
    resources: SimulationResources,
    deltaSeconds: number,
    advanceSimulation: boolean,
    consumeFullDelta: boolean,
  ): number {
    const start = performance.now();
    if (advanceSimulation) {
      const boundedDelta = consumeFullDelta
        ? Math.max(0, deltaSeconds)
        : Math.max(0, Math.min(deltaSeconds, 1 / 15));
      const maximumAccumulation = resources.fixedTimeStep * resources.maximumSubsteps;
      this.accumulator = consumeFullDelta
        ? this.accumulator + boundedDelta
        : Math.min(this.accumulator + boundedDelta, maximumAccumulation);

      let substeps = 0;
      while (
        this.accumulator + 1e-9 >= resources.fixedTimeStep
        && (consumeFullDelta || substeps < resources.maximumSubsteps)
      ) {
        this.stepSimulation(resources.fixedTimeStep);
        this.accumulator -= resources.fixedTimeStep;
        this.simulationTime += resources.fixedTimeStep;
        substeps += 1;
      }
    }

    this.renderVolume();
    gl.flush();
    return performance.now() - start;
  }

  private rebuildResources(): void {
    const gl = this.requireGl();
    const configuration = this.activeConfiguration();
    const columns = configuration.columns;
    const rows = Math.ceil(configuration.slices / columns);
    const maximumTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    let tileHeight = configuration.tileHeight;
    let tileWidth = this.desiredTileWidth(tileHeight);
    const atlasScale = Math.min(
      1,
      maximumTextureSize / Math.max(tileWidth * columns, 1),
      maximumTextureSize / Math.max(tileHeight * rows, 1),
    );
    if (atlasScale < 1) {
      if (this.offlineRenderTier !== null) {
        const requiredWidth = tileWidth * columns;
        const requiredHeight = tileHeight * rows;
        throw new Error(
          `The exact ${this.offlineRenderTier.toUpperCase()} slice atlas requires ${requiredWidth}×${requiredHeight}, above this device's ${maximumTextureSize}-pixel WebGL2 texture limit. The solver was not silently downscaled.`,
        );
      }
      tileHeight = Math.max(24, roundToMultiple(tileHeight * atlasScale, 4));
      tileWidth = Math.max(24, roundToMultiple(tileWidth * atlasScale, 4));
    }
    const atlasWidth = tileWidth * columns;
    const atlasHeight = tileHeight * rows;
    for (let index = 0; index < 16 && gl.getError() !== gl.NO_ERROR; index += 1) {
      // Drain earlier interactive errors so residency validation below belongs to this allocation.
    }

    const allocated: RenderTarget[] = [];
    let particleFramebuffer: WebGLFramebuffer | null = null;
    let particles: ParticleResources | null = null;
    let candidate: SimulationResources | null = null;
    try {
      for (let index = 0; index < 9; index += 1) {
        allocated.push(createHalfFloatTarget(gl, atlasWidth, atlasHeight));
      }
      particleFramebuffer = gl.createFramebuffer();
      if (!particleFramebuffer) throw new Error('Unable to allocate the particle projection framebuffer.');
      gl.bindFramebuffer(gl.FRAMEBUFFER, particleFramebuffer);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        allocated[7].texture,
        0,
      );
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT1,
        gl.TEXTURE_2D,
        allocated[8].texture,
        0,
      );
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Particle projection MRT framebuffer is incomplete.');
      }
      particles = this.createParticleResources(configuration.particleCount);
      candidate = {
        tileWidth,
        tileHeight,
        columns,
        rows,
        slices: configuration.slices,
        atlasWidth,
        atlasHeight,
        jacobiIterations: configuration.jacobiIterations,
        particleRadius: configuration.particleRadius,
        raySteps: configuration.raySteps,
        fixedTimeStep: configuration.fixedTimeStep,
        maximumSubsteps: configuration.maximumSubsteps,
        detailFrequencyScale: this.offlineRenderTier !== null
          ? tileHeight / HORVATH_QUALITY.cinematic.tileHeight
          : 1,
        state: new PingPongTarget(allocated[0], allocated[1]),
        velocity: new PingPongTarget(allocated[2], allocated[3]),
        pressure: new PingPongTarget(allocated[4], allocated[5]),
        auxiliary: allocated[6],
        particleMotion: allocated[7],
        particleThermo: allocated[8],
        particleFramebuffer,
        particles,
      };
    } catch (error) {
      if (particles) this.deleteParticleResources(particles);
      if (particleFramebuffer) gl.deleteFramebuffer(particleFramebuffer);
      for (const target of allocated) deleteTarget(gl, target);
      throw error;
    }
    if (!candidate) throw new Error('The Horvath solver did not produce a resource set.');

    const previous = this.resources;
    const previousSimulationTime = this.simulationTime;
    const previousAccumulator = this.accumulator;
    this.resources = candidate;
    try {
      this.simulationTime = 0;
      this.accumulator = 0;
      this.clearResources();
      gl.finish();
      const allocationError = gl.getError();
      if (allocationError !== gl.NO_ERROR || gl.isContextLost()) {
        throw new Error(
          `The exact ${this.offlineRenderTier?.toUpperCase() ?? this.quality.toUpperCase()} solver allocation failed GPU residency validation (WebGL error ${allocationError}).`,
        );
      }
    } catch (error) {
      this.resources = previous;
      this.simulationTime = previousSimulationTime;
      this.accumulator = previousAccumulator;
      this.deleteSimulationResources(candidate);
      throw error;
    }
    if (previous) this.deleteSimulationResources(previous);
  }

  private desiredTileWidth(tileHeight: number): number {
    return roundToMultiple(tileHeight * 1.22, 4);
  }

  private activeConfiguration(): QualityConfiguration {
    return this.offlineRenderTier !== null
      ? HORVATH_OFFLINE_QUALITY[this.offlineRenderTier]
      : HORVATH_QUALITY[this.quality];
  }

  private usesCinematicPresentation(): boolean {
    return this.offlineRenderTier !== null || this.quality === 'cinematic';
  }

  private sourceMode(): 0 | 1 | 2 {
    return this.parameters.sourceMode;
  }

  private createParticleResources(count: number): ParticleResources {
    const gl = this.requireGl();
    const initialData = this.createInitialParticleData(count);
    const buffers: WebGLBuffer[] = [];
    const vertexArrays: WebGLVertexArrayObject[] = [];
    const transformFeedbacks: WebGLTransformFeedback[] = [];
    try {
      for (let index = 0; index < 2; index += 1) {
        const buffer = gl.createBuffer();
        const vertexArray = gl.createVertexArray();
        const transformFeedback = gl.createTransformFeedback();
        if (!buffer || !vertexArray || !transformFeedback) {
          if (buffer) gl.deleteBuffer(buffer);
          if (vertexArray) gl.deleteVertexArray(vertexArray);
          if (transformFeedback) gl.deleteTransformFeedback(transformFeedback);
          throw new Error('Unable to allocate transform-feedback particle resources.');
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, initialData, gl.DYNAMIC_COPY);
        gl.bindVertexArray(vertexArray);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 32, 0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 16);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, transformFeedback);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buffer);
        buffers.push(buffer);
        vertexArrays.push(vertexArray);
        transformFeedbacks.push(transformFeedback);
      }
    } catch (error) {
      for (const transformFeedback of transformFeedbacks) gl.deleteTransformFeedback(transformFeedback);
      for (const vertexArray of vertexArrays) gl.deleteVertexArray(vertexArray);
      for (const buffer of buffers) gl.deleteBuffer(buffer);
      throw error;
    } finally {
      gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
      gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }
    return {
      count,
      buffers: buffers as [WebGLBuffer, WebGLBuffer],
      vertexArrays: vertexArrays as [WebGLVertexArrayObject, WebGLVertexArrayObject],
      transformFeedbacks: transformFeedbacks as [WebGLTransformFeedback, WebGLTransformFeedback],
      readIndex: 0,
    };
  }

  private createInitialParticleData(count: number): Float32Array {
    const data = new Float32Array(count * 8);
    for (let index = 0; index < count; index += 1) {
      const offset = index * 8;
      data[offset] = -2;
      data[offset + 1] = -2;
      data[offset + 2] = -2;
      data[offset + 3] = 99;
      data[offset + 4] = 0;
      data[offset + 5] = 0;
      data[offset + 6] = 0;
      data[offset + 7] = (index + 0.5) / count;
    }
    return data;
  }

  private resetParticles(): void {
    const gl = this.gl;
    const particles = this.resources?.particles;
    if (!gl || !particles) return;
    const initialData = this.createInitialParticleData(particles.count);
    for (const buffer of particles.buffers) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, initialData);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    particles.readIndex = 0;
  }

  private updateParticles(deltaTime: number, validateErrors = false): void {
    const gl = this.requireGl();
    const programs = this.requirePrograms();
    const resources = this.requireResources();
    const particles = resources.particles;
    const readIndex = particles.readIndex;
    const writeIndex = (1 - readIndex) as 0 | 1;
    const assertStage = (stage: string): void => {
      if (!validateErrors) return;
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        throw new Error(`Horvath offline particle ${stage} failed with WebGL error ${error}.`);
      }
    };

    // The preceding atlas pass leaves its destination framebuffer bound. After
    // ping-pong swap that texture becomes the next velocity sampler; drawing
    // transform feedback while it is still attached is an illegal feedback
    // loop in WebGL2, even with rasterizer discard enabled.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    programs.updateParticles.use();
    this.setAtlasUniforms(programs.updateParticles);
    bindTextureUnit(gl, resources.velocity.read.texture, 0);
    programs.updateParticles.set1i('uVelocity', 0);
    programs.updateParticles.set1i('uSourceMode', this.sourceMode());
    programs.updateParticles.set1f('uTime', this.simulationTime);
    programs.updateParticles.set1f('uDeltaTime', deltaTime);
    programs.updateParticles.set1f('uEmission', this.parameters.emission);
    programs.updateParticles.set1f('uBuoyancy', this.parameters.buoyancy);
    programs.updateParticles.set1f('uSourceSize', this.parameters.sourceSize);
    programs.updateParticles.set1f('uFlameHeight', this.parameters.flameHeight);
    assertStage('uniform setup');
    gl.bindVertexArray(particles.vertexArrays[readIndex]);
    assertStage('vertex input binding');
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, particles.transformFeedbacks[writeIndex]);
    assertStage('transform feedback binding');
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    assertStage('transform feedback begin');
    gl.drawArrays(gl.POINTS, 0, particles.count);
    assertStage('draw');
    gl.endTransformFeedback();
    assertStage('transform feedback end');
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    particles.readIndex = writeIndex;
  }

  private projectParticles(): void {
    const gl = this.requireGl();
    const programs = this.requirePrograms();
    const resources = this.requireResources();
    gl.bindFramebuffer(gl.FRAMEBUFFER, resources.particleFramebuffer);
    gl.viewport(0, 0, resources.atlasWidth, resources.atlasHeight);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.clearBufferfv(gl.COLOR, 0, this.zeroColor);
    gl.clearBufferfv(gl.COLOR, 1, this.zeroColor);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    programs.projectParticles.use();
    this.setAtlasUniforms(programs.projectParticles);
    const sourceMode = this.sourceMode();
    const wallSourceGain = sourceMode === 2 ? 1.25 : 1.0;
    const wallSourceRadius = sourceMode === 2 ? 1.24 : 1.0;
    // Source size changes the physical support, so scale total injected mass
    // with its measure instead of spreading a fixed weak source over more
    // cells. Fireballs grow volumetrically; burners and walls grow by area.
    const sourceMeasureGain = Math.max(
      0.08,
      Math.min(9, this.parameters.sourceSize ** (sourceMode === 1 ? 3 : 2)),
    );
    // The old three-slab kernel integrated to 1.6. Preserve that source energy
    // while the corrected two-slab projection uses fractional weights summing
    // to one. Compensate smaller normalized kernels at higher field resolution.
    const depthNormalization = (resources.slices / 18) * 1.6;
    const referenceNormalizedRadius = 3.7 / 88;
    const normalizedRadius = resources.particleRadius / resources.tileHeight;
    const kernelMassCorrection = (referenceNormalizedRadius / normalizedRadius) ** 2;
    programs.projectParticles.set1i('uSourceMode', sourceMode);
    programs.projectParticles.set1f(
      'uPointRadius',
      resources.particleRadius * wallSourceRadius,
    );
    programs.projectParticles.set1f(
      'uParticleWeight',
      (4096 / resources.particles.count)
        * sourceMeasureGain
        * wallSourceGain
        * depthNormalization
        * kernelMassCorrection,
    );
    programs.projectParticles.set1f('uFlameHeight', this.parameters.flameHeight);
    gl.bindVertexArray(resources.particles.vertexArrays[resources.particles.readIndex]);
    gl.drawArraysInstanced(gl.POINTS, 0, resources.particles.count, 2);
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private deleteParticleResources(particles: ParticleResources): void {
    const gl = this.requireGl();
    for (const transformFeedback of particles.transformFeedbacks) {
      gl.deleteTransformFeedback(transformFeedback);
    }
    for (const vertexArray of particles.vertexArrays) gl.deleteVertexArray(vertexArray);
    for (const buffer of particles.buffers) gl.deleteBuffer(buffer);
  }

  private stepSimulation(deltaTime: number, validateErrors = false): void {
    const programs = this.requirePrograms();
    const resources = this.requireResources();
    const assertStage = (stage: string): void => {
      if (!validateErrors) return;
      const gl = this.requireGl();
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        throw new Error(`Horvath offline ${stage} failed with WebGL error ${error}.`);
      }
    };

    this.updateParticles(deltaTime, validateErrors);
    assertStage('particle update');
    this.projectParticles();
    assertStage('particle projection');

    this.runAtlasPass(programs.thermo, resources.state.write, [
      resources.state.read.texture,
      resources.particleMotion.texture,
      resources.particleThermo.texture,
    ], (program) => {
      program.set1i('uState', 0);
      program.set1i('uParticleMotion', 1);
      program.set1i('uParticleThermo', 2);
      program.set1f('uDeltaTime', deltaTime);
      this.setSourceUniforms(program);
      program.set1f('uCooling', this.parameters.cooling);
    });
    resources.state.swap();
    assertStage('thermochemical source');

    this.runAtlasPass(
      programs.sourceVelocity,
      resources.velocity.write,
      [resources.velocity.read.texture, resources.particleMotion.texture],
      (program) => {
        program.set1i('uVelocity', 0);
        program.set1i('uParticleMotion', 1);
        program.set1f('uDeltaTime', deltaTime);
        this.setSourceUniforms(program);
      },
    );
    resources.velocity.swap();
    assertStage('source velocity');

    this.runAtlasPass(
      programs.advectState,
      resources.state.write,
      [resources.state.read.texture, resources.velocity.read.texture],
      (program) => {
        program.set1i('uState', 0);
        program.set1i('uVelocity', 1);
        program.set1f('uDeltaTime', deltaTime);
      },
    );
    resources.state.swap();
    assertStage('state advection');

    this.runAtlasPass(
      programs.advectVelocity,
      resources.velocity.write,
      [resources.velocity.read.texture],
      (program) => {
        program.set1i('uVelocity', 0);
        program.set1f('uDeltaTime', deltaTime);
      },
    );
    resources.velocity.swap();
    assertStage('velocity advection');

    this.runAtlasPass(programs.detail, resources.state.write, [resources.state.read.texture], (program) => {
      program.set1i('uState', 0);
      program.set1f('uDeltaTime', deltaTime);
      program.set1f('uTime', this.simulationTime);
      program.set1f('uDetail', this.parameters.detail);
      program.set1f('uDetailFrequencyScale', resources.detailFrequencyScale);
    });
    resources.state.swap();
    assertStage('detail reconstruction');

    this.runAtlasPass(programs.curl, resources.auxiliary, [resources.velocity.read.texture], (program) => {
      program.set1i('uVelocity', 0);
      program.set1f('uTime', this.simulationTime);
      program.set1f('uDetail', this.parameters.detail);
      program.set1f('uVorticity', this.parameters.vorticity);
    });
    assertStage('curl');

    this.runAtlasPass(
      programs.force,
      resources.velocity.write,
      [resources.velocity.read.texture, resources.auxiliary.texture, resources.state.read.texture],
      (program) => {
        program.set1i('uVelocity', 0);
        program.set1i('uCurl', 1);
        program.set1i('uState', 2);
        program.set1f('uDeltaTime', deltaTime);
        program.set1f('uBuoyancy', this.parameters.buoyancy);
        program.set1f('uVorticity', this.parameters.vorticity);
      },
    );
    resources.velocity.swap();
    assertStage('force');

    this.runAtlasPass(
      programs.divergence,
      resources.auxiliary,
      [resources.velocity.read.texture],
      (program) => program.set1i('uVelocity', 0),
    );
    assertStage('divergence');

    for (let iteration = 0; iteration < resources.jacobiIterations; iteration += 1) {
      this.runAtlasPass(
        programs.jacobi,
        resources.pressure.write,
        [resources.pressure.read.texture, resources.auxiliary.texture],
        (program) => {
          program.set1i('uPressure', 0);
          program.set1i('uDivergence', 1);
        },
      );
      resources.pressure.swap();
    }
    assertStage('pressure projection');

    this.runAtlasPass(
      programs.project,
      resources.velocity.write,
      [resources.velocity.read.texture, resources.pressure.read.texture],
      (program) => {
        program.set1i('uVelocity', 0);
        program.set1i('uPressure', 1);
      },
    );
    resources.velocity.swap();
    assertStage('velocity projection');
  }

  private renderVolume(signal?: AbortSignal): void {
    const gl = this.requireGl();
    const programs = this.requirePrograms();
    const resources = this.requireResources();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.BLEND);
    programs.render.use();
    this.setAtlasUniforms(programs.render);
    bindTextureUnit(gl, resources.state.read.texture, 0);
    programs.render.set1i('uState', 0);
    programs.render.set1f('uDensityGain', this.parameters.density);
    programs.render.set1f('uDetail', this.parameters.detail);
    programs.render.set1f('uExposure', this.parameters.exposure);
    programs.render.set1f('uCinematic', this.usesCinematicPresentation() ? 1 : 0);
    programs.render.set1f('uDetailFrequencyScale', resources.detailFrequencyScale);
    programs.render.set1f('uCameraYaw', this.parameters.cameraYaw);
    programs.render.set1f('uCameraPitch', this.parameters.cameraPitch);
    programs.render.set1f('uCameraDistance', this.parameters.cameraDistance);
    programs.render.set1f('uViewZoom', this.parameters.viewZoom);
    programs.render.set1f('uTime', this.simulationTime);
    programs.render.set2f('uViewportSize', this.canvas.width, this.canvas.height);
    programs.render.set1i(
      'uRaySteps',
      Math.min(512, resources.raySteps),
    );
    programs.render.set1i('uSourceMode', this.sourceMode());
    gl.bindVertexArray(this.vertexArray);
    if (this.offlineRenderTier !== null) {
      const stripeHeight = 128;
      gl.enable(gl.SCISSOR_TEST);
      for (let y = 0; y < this.canvas.height; y += stripeHeight) {
        if (signal?.aborted) {
          gl.disable(gl.SCISSOR_TEST);
          throw new DOMException('Offline render canceled.', 'AbortError');
        }
        gl.scissor(0, y, this.canvas.width, Math.min(stripeHeight, this.canvas.height - y));
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.flush();
      }
      gl.disable(gl.SCISSOR_TEST);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    this.renderSparks();
  }

  private renderSparks(): void {
    const gl = this.requireGl();
    const programs = this.requirePrograms();
    const resources = this.requireResources();
    programs.sparks.use();
    programs.sparks.set2i('uTileSize', resources.tileWidth, resources.tileHeight);
    programs.sparks.set1i('uSourceMode', this.sourceMode());
    programs.sparks.set1f('uCameraYaw', this.parameters.cameraYaw);
    programs.sparks.set1f('uCameraPitch', this.parameters.cameraPitch);
    programs.sparks.set1f('uCameraDistance', this.parameters.cameraDistance);
    programs.sparks.set1f('uViewZoom', this.parameters.viewZoom);
    programs.sparks.set2f('uViewportSize', this.canvas.width, this.canvas.height);
    programs.sparks.set1f('uPointSize', Math.max(1.5, Math.min(3.4, this.canvas.height / 420)));
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindVertexArray(resources.particles.vertexArrays[resources.particles.readIndex]);
    gl.drawArrays(gl.POINTS, 0, resources.particles.count);
    gl.disable(gl.BLEND);
  }

  private runAtlasPass(
    program: GLProgram,
    output: RenderTarget,
    textures: readonly WebGLTexture[],
    setUniforms: (program: GLProgram) => void,
  ): void {
    const gl = this.requireGl();
    const resources = this.requireResources();
    gl.bindFramebuffer(gl.FRAMEBUFFER, output.framebuffer);
    gl.viewport(0, 0, resources.atlasWidth, resources.atlasHeight);
    gl.disable(gl.BLEND);
    program.use();
    this.setAtlasUniforms(program);
    textures.forEach((texture, unit) => bindTextureUnit(gl, texture, unit));
    setUniforms(program);
    gl.bindVertexArray(this.vertexArray);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private setAtlasUniforms(program: GLProgram): void {
    const resources = this.requireResources();
    program.set2i('uTileSize', resources.tileWidth, resources.tileHeight);
    program.set2i('uAtlasGrid', resources.columns, resources.rows);
    program.set1i('uSliceCount', resources.slices);
  }

  private setSourceUniforms(program: GLProgram): void {
    program.set1f('uTime', this.simulationTime);
    program.set1f('uEmission', this.parameters.emission);
    program.set1f('uBuoyancy', this.parameters.buoyancy);
    program.set1f('uDensityGain', this.parameters.density);
    program.set1i('uSourceMode', this.sourceMode());
  }

  private clearResources(): void {
    const gl = this.requireGl();
    const resources = this.requireResources();
    const targets = [
      resources.state.read,
      resources.state.write,
      resources.velocity.read,
      resources.velocity.write,
      resources.pressure.read,
      resources.pressure.write,
      resources.auxiliary,
      resources.particleMotion,
      resources.particleThermo,
    ];
    gl.viewport(0, 0, resources.atlasWidth, resources.atlasHeight);
    gl.clearColor(0, 0, 0, 0);
    for (const target of targets) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private disposeResources(): void {
    const resources = this.resources;
    if (!resources) return;
    this.resources = null;
    this.deleteSimulationResources(resources);
  }

  private deleteSimulationResources(resources: SimulationResources): void {
    const gl = this.gl;
    if (!gl) return;
    const targets = [
      resources.state.read,
      resources.state.write,
      resources.velocity.read,
      resources.velocity.write,
      resources.pressure.read,
      resources.pressure.write,
      resources.auxiliary,
      resources.particleMotion,
      resources.particleThermo,
    ];
    gl.deleteFramebuffer(resources.particleFramebuffer);
    this.deleteParticleResources(resources.particles);
    for (const target of targets) deleteTarget(gl, target);
  }

  private disposeGpuObjects(): void {
    const gl = this.gl;
    if (!gl) return;
    this.disposeResources();
    if (this.programs) {
      for (const program of Object.values(this.programs)) program.dispose();
      this.programs = null;
    }
    if (this.vertexArray) gl.deleteVertexArray(this.vertexArray);
    this.vertexArray = null;
  }

  private requireGl(): WebGL2RenderingContext {
    if (!this.gl) throw new Error('The Horvath WebGL2 engine is not initialized.');
    return this.gl;
  }

  private requireOfflineGl(): WebGL2RenderingContext {
    if (this.disposed) throw new Error('The Horvath engine has been disposed and cannot render an export.');
    if (!this.initialized) throw new Error('The Horvath engine must be initialized before rendering an export.');
    const gl = this.requireGl();
    if (this.contextLost || gl.isContextLost()) {
      throw new Error('The Horvath WebGL2 context is lost and cannot render an export.');
    }
    this.requirePrograms();
    this.requireResources();
    return gl;
  }

  private requirePrograms(): ProgramSet {
    if (!this.programs) throw new Error('Horvath shader programs are unavailable.');
    return this.programs;
  }

  private requireResources(): SimulationResources {
    if (!this.resources) throw new Error('Horvath simulation resources are unavailable.');
    return this.resources;
  }
}
