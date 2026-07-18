export type MethodId = 'horvath' | 'firex';
export const QUALITY_LEVELS = ['low', 'balanced', 'high', 'maximum', 'cinematic'] as const;
export type QualityLevel = (typeof QUALITY_LEVELS)[number];
export const OFFLINE_RENDER_TIERS = ['hd', 'qhd', 'uhd'] as const;
export type OfflineRenderTier = (typeof OFFLINE_RENDER_TIERS)[number];

export interface FrameStats {
  fps: number;
  frameMs: number;
  simulationMs: number;
  backend: string;
  detail: string;
}

export type DisplayUnitSystem = 'metric' | 'imperial';

export interface SimulationGridInfo {
  dimensions: readonly [number, number, number];
  domainMeters: readonly [number, number, number];
  cellSizeMeters: readonly [number, number, number];
  cellCount: number;
  estimatedBytes: number;
  custom: boolean;
}

export interface SimulationGridCapabilities {
  maxTextureDimension3D: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  /** Largest four-cell-aligned cube allowed by one vec4 field and the 3D texture axis limit. */
  maximumDenseCubeAxis: number;
}

export interface SimulationEngine {
  readonly backend: string;
  readonly detail: string;
  initialize(): Promise<void>;
  resize(width: number, height: number, pixelRatio: number): void;
  /** Sets the exact intrinsic canvas size used by deterministic offline exports. */
  resizeOutput(widthPx: number, heightPx: number): void | Promise<void>;
  /** Advances and renders a frame; returns synchronous CPU encode/submit time in milliseconds. */
  frame(timeSeconds: number, deltaSeconds: number): number;
  /** Advances on the exact offline timeline and optionally presents, then waits for GPU completion. */
  renderOfflineFrame(deltaSeconds: number, present?: boolean, signal?: AbortSignal): Promise<void>;
  /** Advances, renders, and returns an immutable raw frame before the browser compositor can recycle the canvas. */
  renderOfflineVideoFrame(
    deltaSeconds: number,
    timestampSeconds: number,
    durationSeconds: number,
    signal?: AbortSignal,
  ): Promise<VideoFrame>;
  /** Allocates the exact solver bundle for an HD, QHD, or 4K offline render. */
  setOfflineRenderTier(tier: OfflineRenderTier | null): void | Promise<void>;
  reset(): void;
  setPaused(paused: boolean): void;
  setQuality(quality: QualityLevel): void | Promise<void>;
  setParameter(name: string, value: number | boolean): void;
  setGridMemoryBudget?(bytes: number): void;
  setGridDimensions?(dimensions: readonly [number, number, number] | null): void | Promise<void>;
  getGridInfo?(): SimulationGridInfo;
  getGridCapabilities?(): SimulationGridCapabilities | null;
  dispose(): void;
}

export interface ControlDefinition {
  id: string;
  label: string;
  group: 'source' | 'dynamics' | 'water' | 'domain' | 'appearance';
  min: number;
  max: number;
  step: number;
  value: number;
  unit?: string;
  quantity?: 'length';
  description?: string;
  commitOnly?: boolean;
  resetOnCommit?: boolean;
}
