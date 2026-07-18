import { getOfflineVideoPreset, type OfflineVideoPresetId } from '../export/offlineVideo';
import { FIREX_MAX_GRID_MEMORY_BUDGET_BYTES, type FireXGridDimensions } from '../sim/firex/gridConfiguration';
import {
  getFireXFieldView,
  getFireXScene,
  type FireXFieldViewId,
  type FireXSceneId,
} from '../sim/firex/scenes';
import { OFFLINE_RENDER_TIERS, type OfflineRenderTier } from '../sim/types';

export const FIRE_CLI_CAPTURE_SCHEMA_VERSION = 1;
const FIRE_CLI_FRAME_RATE = 60;
const FIRE_CLI_MAX_DURATION_SECONDS = 60 * 60;
const FRAME_ALIGNMENT_EPSILON = 1e-7;

export interface FireCliCaptureConfig {
  readonly schemaVersion: typeof FIRE_CLI_CAPTURE_SCHEMA_VERSION;
  readonly jobId: string;
  readonly method: 'firex';
  readonly scene: FireXSceneId;
  readonly fieldView: FireXFieldViewId;
  readonly outputPreset: OfflineVideoPresetId;
  readonly solverTier: OfflineRenderTier;
  readonly durationSeconds: number;
  readonly warmupSeconds: number;
  readonly gridDimensions: FireXGridDimensions | null;
  readonly gridMemoryBudgetBytes: number;
  readonly opticalDetailTarget: 0 | 256 | 512 | 1024;
  readonly camera: Readonly<{
    yaw: number;
    pitch: number;
    distance: number;
  }>;
}

export function parseFireCliCaptureConfig(input: unknown): FireCliCaptureConfig {
  const value = requireObject(input, 'capture configuration');
  if (value.schemaVersion !== FIRE_CLI_CAPTURE_SCHEMA_VERSION) {
    throw new Error(`Unsupported capture schema version: ${String(value.schemaVersion)}.`);
  }
  if (value.method !== 'firex') throw new Error('CLI capture currently supports method "firex" only.');

  const jobId = requireString(value.jobId, 'jobId');
  if (!/^[a-z0-9][a-z0-9._-]{0,95}$/i.test(jobId)) {
    throw new Error('jobId must contain only letters, numbers, dots, underscores, or hyphens.');
  }

  const scene = getFireXScene(requireString(value.scene, 'scene')).id;
  const fieldView = getFireXFieldView(requireString(value.fieldView, 'fieldView')).id;
  const outputPreset = getOfflineVideoPreset(requireString(value.outputPreset, 'outputPreset')).id;
  const solverTierValue = requireString(value.solverTier, 'solverTier');
  if (!(OFFLINE_RENDER_TIERS as readonly string[]).includes(solverTierValue)) {
    throw new Error(`Unknown solver tier: ${solverTierValue}.`);
  }
  const solverTier = solverTierValue as OfflineRenderTier;
  const durationSeconds = requireFiniteNumber(value.durationSeconds, 'durationSeconds');
  if (durationSeconds < 1 / 60 || durationSeconds > FIRE_CLI_MAX_DURATION_SECONDS) {
    throw new Error('durationSeconds must be between one frame and one hour.');
  }
  requireFrameAlignment(durationSeconds, 'durationSeconds');
  const warmupSeconds = requireFiniteNumber(value.warmupSeconds, 'warmupSeconds');
  if (warmupSeconds < 0 || warmupSeconds > 60) {
    throw new Error('warmupSeconds must be between 0 and 60.');
  }
  requireFrameAlignment(warmupSeconds, 'warmupSeconds');

  const gridDimensions = parseGridDimensions(value.gridDimensions);
  const gridMemoryBudgetBytes = requireFiniteNumber(value.gridMemoryBudgetBytes, 'gridMemoryBudgetBytes');
  if (!Number.isSafeInteger(gridMemoryBudgetBytes)
    || gridMemoryBudgetBytes < 256 * 1024 * 1024
    || gridMemoryBudgetBytes > FIREX_MAX_GRID_MEMORY_BUDGET_BYTES) {
    throw new Error('gridMemoryBudgetBytes must be a whole number from 256 MiB through 32 GiB.');
  }

  const opticalDetailTarget = requireFiniteNumber(value.opticalDetailTarget, 'opticalDetailTarget');
  if (![0, 256, 512, 1024].includes(opticalDetailTarget)) {
    throw new Error('opticalDetailTarget must be 0, 256, 512, or 1024.');
  }
  const cameraValue = requireObject(value.camera, 'camera');
  const camera = Object.freeze({
    yaw: requireFiniteNumber(cameraValue.yaw, 'camera.yaw'),
    pitch: requireFiniteNumber(cameraValue.pitch, 'camera.pitch'),
    distance: requireFiniteNumber(cameraValue.distance, 'camera.distance'),
  });
  if (camera.pitch < -0.7 || camera.pitch > 0.7) {
    throw new Error('camera.pitch must be between -0.7 and 0.7 radians.');
  }
  if (camera.distance < 2.5 || camera.distance > 8) {
    throw new Error('camera.distance must be between 2.5 and 8.');
  }

  return Object.freeze({
    schemaVersion: FIRE_CLI_CAPTURE_SCHEMA_VERSION,
    jobId,
    method: 'firex',
    scene,
    fieldView,
    outputPreset,
    solverTier,
    durationSeconds,
    warmupSeconds,
    gridDimensions,
    gridMemoryBudgetBytes,
    opticalDetailTarget: opticalDetailTarget as 0 | 256 | 512 | 1024,
    camera,
  });
}

function requireFrameAlignment(seconds: number, name: string): void {
  const frames = seconds * FIRE_CLI_FRAME_RATE;
  if (Math.abs(frames - Math.round(frames)) > FRAME_ALIGNMENT_EPSILON) {
    throw new Error(`${name} must end on an exact 60 fps frame boundary.`);
  }
}

function parseGridDimensions(value: unknown): FireXGridDimensions | null {
  if (value === null) return null;
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error('gridDimensions must be null or an exact [x, y, z] tuple.');
  }
  const dimensions = value.map((axis) => requireFiniteNumber(axis, 'gridDimensions axis'));
  if (dimensions.some((axis) => !Number.isSafeInteger(axis) || axis < 16 || axis > 1024 || axis % 4 !== 0)) {
    throw new Error('Every grid axis must be a whole multiple of 4 from 16 through 1024; values are never rounded.');
  }
  return Object.freeze([dimensions[0]!, dimensions[1]!, dimensions[2]!] as const);
}

function requireObject(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new TypeError(`${name} must be a non-empty string.`);
  return value;
}

function requireFiniteNumber(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new TypeError(`${name} must be a finite number.`);
  return value;
}
