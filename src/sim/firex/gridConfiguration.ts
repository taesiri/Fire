export type FireXGridDimensions = readonly [number, number, number];
export type FireXDomainExtent = readonly [number, number, number];
export type FireXGridPresetId =
  | 'auto'
  | 'maximum'
  | 'ultra'
  | 'experimental'
  | 'extreme'
  | 'expert'
  | 'workstation'
  | 'studio'
  | 'reference'
  | 'custom';
export type FireXUnitSystem = 'metric' | 'imperial';

export interface FireXGridPreset {
  readonly id: FireXGridPresetId;
  readonly label: string;
  /** `null` means that the caller supplies either its quality-profile or custom dimensions. */
  readonly dimensions: FireXGridDimensions | null;
  readonly experimental?: boolean;
  /** Large real allocations require a deliberate second action after selection. */
  readonly requiresConfirmation?: boolean;
  readonly minimumGuardBytes?: number;
}

export interface FireXGridMemoryGuard {
  readonly bytes: number;
  readonly label: string;
}

export interface FireXGridNormalizationOptions {
  readonly minimumAxis?: number;
  readonly maximumAxis?: number;
  readonly fallback?: FireXGridDimensions;
}

export interface FireXGridMemoryEstimate {
  readonly cellCount: number;
  readonly gridBytes: number;
  readonly particleBytes: number;
  readonly fixedBytes: number;
  readonly totalBytes: number;
  readonly largestStorageBufferBytes: number;
}

/** Structural subset of `GPUSupportedLimits`, kept usable in unit tests and capability probes. */
export interface FireXGridDeviceLimits {
  readonly maxTextureDimension3D: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
}

export type FireXGridPreflightIssueCode =
  | 'invalid-dimension'
  | 'unaligned-dimension'
  | 'texture-dimension-limit'
  | 'storage-binding-limit'
  | 'buffer-size-limit'
  | 'memory-budget';

export interface FireXGridPreflightIssue {
  readonly code: FireXGridPreflightIssueCode;
  readonly message: string;
}

export interface FireXGridPreflightOptions {
  readonly particleCount?: number;
  /** Set to `null` to disable the conservative aggregate-allocation guard. */
  readonly memoryBudgetBytes?: number | null;
}

export interface FireXGridPreflightResult {
  readonly ok: boolean;
  readonly dimensions: FireXGridDimensions;
  readonly estimate: FireXGridMemoryEstimate;
  readonly issues: readonly FireXGridPreflightIssue[];
}

export interface FireXGridInfo extends FireXGridMemoryEstimate {
  readonly dimensions: FireXGridDimensions;
  /** Canonical scene extent. The UI may describe this as metres or as a metre-scaled display unit. */
  readonly domainMeters: FireXDomainExtent;
  readonly voxelMeters: FireXDomainExtent;
  readonly volumeCubicMeters: number;
}

export interface FormattedFireXGridInfo {
  readonly resolution: string;
  readonly cellCount: string;
  readonly domain: string;
  readonly voxelSize: string;
  readonly volume: string;
  readonly memory: string;
  readonly summary: string;
}

export const FIREX_GRID_AXIS_QUANTUM = 4;
export const FIREX_GRID_MINIMUM_AXIS = 16;
export const FIREX_GRID_MAXIMUM_AXIS = 1024;
// Smaller grids retain four f32 pressure fields: two for the primary solve and
// two for residual correction. Heavy grids keep the same 8 bytes/cell pressure
// residency with four f16 fields when shader-f16 is available, or two f32 fields
// reused sequentially otherwise. Transport and the remaining fields stay f32.
export const FIREX_HEAVY_GRID_CELL_THRESHOLD = 64 * 1024 * 1024;
export const FIREX_GRID_BYTES_PER_CELL = 116;
export const FIREX_HEAVY_GRID_BYTES_PER_CELL = 108;
export const FIREX_PARTICLE_BYTES = 80;
export const FIREX_LARGEST_STORAGE_BYTES_PER_CELL = 16;
export const FIREX_DETAIL_TEXTURE_BYTES = 64 * 64 * 64 * 4;
export const FIREX_DEFAULT_GRID_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024;
export const FIREX_MAX_NEGOTIATED_STORAGE_BYTES = 512 ** 3 * FIREX_LARGEST_STORAGE_BYTES_PER_CELL;

const GIBIBYTE = 1024 ** 3;

export const FIREX_GRID_MEMORY_GUARDS: readonly FireXGridMemoryGuard[] = [
  { bytes: 512 * 1024 * 1024, label: 'Safe · 512 MiB' },
  { bytes: GIBIBYTE, label: 'Heavy · 1 GiB' },
  { bytes: 2 * GIBIBYTE, label: 'Expert · 2 GiB' },
  { bytes: 4 * GIBIBYTE, label: 'Workstation · 4 GiB' },
  { bytes: 8 * GIBIBYTE, label: 'Studio · 8 GiB' },
  { bytes: 16 * GIBIBYTE, label: 'Reference · 16 GiB' },
  { bytes: 24 * GIBIBYTE, label: 'Reference swap · 24 GiB' },
  { bytes: 32 * GIBIBYTE, label: 'Maximum opt-in · 32 GiB' },
] as const;

export const FIREX_MAX_GRID_MEMORY_BUDGET_BYTES = FIREX_GRID_MEMORY_GUARDS.at(-1)?.bytes
  ?? FIREX_DEFAULT_GRID_MEMORY_BUDGET_BYTES;

const DEFAULT_GRID_FALLBACK: FireXGridDimensions = [64, 96, 64];
const METERS_TO_FEET = 3.280839895013123;
const METERS_TO_INCHES = 39.37007874015748;
const CUBIC_METERS_TO_CUBIC_FEET = 35.31466672148859;

export const FIREX_GRID_PRESETS: readonly FireXGridPreset[] = [
  { id: 'auto', label: 'Auto · follows compute profile', dimensions: null },
  { id: 'maximum', label: 'Maximum · 80×120×80', dimensions: [80, 120, 80] },
  { id: 'ultra', label: 'Ultra · 96×144×96', dimensions: [96, 144, 96] },
  {
    id: 'experimental',
    label: 'Experimental · 128×192×128',
    dimensions: [128, 192, 128],
    experimental: true,
  },
  {
    id: 'extreme',
    label: 'Extreme dense · 160×160×160',
    dimensions: [160, 160, 160],
    experimental: true,
  },
  {
    id: 'expert',
    label: 'Expert dense · 256³ · 2 GiB guard',
    dimensions: [256, 256, 256],
    experimental: true,
    requiresConfirmation: true,
    minimumGuardBytes: 2 * GIBIBYTE,
  },
  {
    id: 'workstation',
    label: 'Workstation dense · 320³ · 4 GiB guard',
    dimensions: [320, 320, 320],
    experimental: true,
    requiresConfirmation: true,
    minimumGuardBytes: 4 * GIBIBYTE,
  },
  {
    id: 'studio',
    label: 'Studio dense · 384³ · 8 GiB guard',
    dimensions: [384, 384, 384],
    experimental: true,
    requiresConfirmation: true,
    minimumGuardBytes: 8 * GIBIBYTE,
  },
  {
    id: 'reference',
    label: 'Reference dense · 508³ · 16+ GiB guard',
    dimensions: [508, 508, 508],
    experimental: true,
    requiresConfirmation: true,
    minimumGuardBytes: 16 * GIBIBYTE,
  },
  { id: 'custom', label: 'Custom', dimensions: null },
] as const;

/**
 * Requests only limits that the adapter reports and only as high as the exact
 * 512³ custom boundary. Raising a limit validates larger resources; it does not
 * reserve memory or imply that the adapter has enough free VRAM.
 */
export function selectFireXRequiredDeviceLimits(
  limits: Pick<FireXGridDeviceLimits, 'maxStorageBufferBindingSize' | 'maxBufferSize'>,
): Pick<FireXGridDeviceLimits, 'maxStorageBufferBindingSize' | 'maxBufferSize'> {
  const maxBufferSize = Math.min(
    Math.max(0, Math.floor(limits.maxBufferSize)),
    FIREX_MAX_NEGOTIATED_STORAGE_BYTES,
  );
  const maxStorageBufferBindingSize = Math.min(
    Math.max(0, Math.floor(limits.maxStorageBufferBindingSize)),
    maxBufferSize,
    FIREX_MAX_NEGOTIATED_STORAGE_BYTES,
  );
  return { maxStorageBufferBindingSize, maxBufferSize };
}

/** Device-limit ceiling for one cubic vec4 field. Aggregate memory is separate. */
export function maximumFireXDeviceDenseCubeAxis(limits: FireXGridDeviceLimits): number {
  const candidateBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  const largestFieldBytes = Number.isFinite(candidateBytes) ? Math.max(0, candidateBytes) : 0;
  // Correct cbrt rounding at exact power-of-two boundaries (some browsers
  // produce 511.99999999999994 for the exact 512³ field requirement).
  let bufferAxis = Math.floor(Math.cbrt(largestFieldBytes / FIREX_LARGEST_STORAGE_BYTES_PER_CELL));
  while ((bufferAxis + 1) ** 3 * FIREX_LARGEST_STORAGE_BYTES_PER_CELL <= largestFieldBytes) {
    bufferAxis += 1;
  }
  while (bufferAxis ** 3 * FIREX_LARGEST_STORAGE_BYTES_PER_CELL > largestFieldBytes) {
    bufferAxis -= 1;
  }
  const maximumAxis = Math.min(
    FIREX_GRID_MAXIMUM_AXIS,
    Math.max(0, Math.floor(limits.maxTextureDimension3D)),
    bufferAxis,
  );
  const aligned = Math.floor(maximumAxis / FIREX_GRID_AXIS_QUANTUM) * FIREX_GRID_AXIS_QUANTUM;
  return aligned >= FIREX_GRID_MINIMUM_AXIS ? aligned : 0;
}

function finiteNonnegativeInteger(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
}

function alignedBounds(minimumAxis: number, maximumAxis: number): readonly [number, number] {
  const minimum = Math.ceil(minimumAxis / FIREX_GRID_AXIS_QUANTUM) * FIREX_GRID_AXIS_QUANTUM;
  const maximum = Math.floor(maximumAxis / FIREX_GRID_AXIS_QUANTUM) * FIREX_GRID_AXIS_QUANTUM;
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum <= 0 || maximum < minimum) {
    throw new RangeError('Fire-X grid bounds must contain at least one positive multiple of four.');
  }
  return [minimum, maximum];
}

function normalizeAxis(value: number, fallback: number, minimum: number, maximum: number): number {
  const finiteValue = Number.isFinite(value) ? value : fallback;
  const snapped = Math.round(finiteValue / FIREX_GRID_AXIS_QUANTUM) * FIREX_GRID_AXIS_QUANTUM;
  return Math.max(minimum, Math.min(maximum, snapped));
}

/**
 * Snaps a user-entered grid to efficient 4×4×4 workgroup boundaries and clamps
 * it to the UI's supported range. Device-specific rejection remains a preflight concern.
 */
export function normalizeFireXGridDimensions(
  dimensions: readonly [number, number, number],
  options: FireXGridNormalizationOptions = {},
): FireXGridDimensions {
  const [minimum, maximum] = alignedBounds(
    options.minimumAxis ?? FIREX_GRID_MINIMUM_AXIS,
    options.maximumAxis ?? FIREX_GRID_MAXIMUM_AXIS,
  );
  const fallback = options.fallback ?? DEFAULT_GRID_FALLBACK;
  return [
    normalizeAxis(dimensions[0], fallback[0], minimum, maximum),
    normalizeAxis(dimensions[1], fallback[1], minimum, maximum),
    normalizeAxis(dimensions[2], fallback[2], minimum, maximum),
  ];
}

export function fireXGridCellCount(dimensions: FireXGridDimensions): number {
  return dimensions[0] * dimensions[1] * dimensions[2];
}

/** Permanent Fire-X grid allocations; swap-chain and cinematic presentation targets are excluded. */
export function estimateFireXGridMemory(
  dimensions: FireXGridDimensions,
  particleCount = 0,
): FireXGridMemoryEstimate {
  const cellCount = fireXGridCellCount(dimensions);
  const bytesPerCell = cellCount >= FIREX_HEAVY_GRID_CELL_THRESHOLD
    ? FIREX_HEAVY_GRID_BYTES_PER_CELL
    : FIREX_GRID_BYTES_PER_CELL;
  const gridBytes = cellCount * bytesPerCell;
  const particleBytes = finiteNonnegativeInteger(particleCount) * FIREX_PARTICLE_BYTES;
  const fixedBytes = FIREX_DETAIL_TEXTURE_BYTES;
  return {
    cellCount,
    gridBytes,
    particleBytes,
    fixedBytes,
    totalBytes: gridBytes + particleBytes + fixedBytes,
    largestStorageBufferBytes: cellCount * FIREX_LARGEST_STORAGE_BYTES_PER_CELL,
  };
}

/**
 * Checks limits before the engine destroys its current resource set. It does not
 * silently normalize dimensions, so malformed programmatic input remains visible.
 */
export function preflightFireXGrid(
  dimensions: FireXGridDimensions,
  limits: FireXGridDeviceLimits,
  options: FireXGridPreflightOptions = {},
): FireXGridPreflightResult {
  const estimate = estimateFireXGridMemory(dimensions, options.particleCount);
  const issues: FireXGridPreflightIssue[] = [];
  const dimensionLabels = ['X', 'Y', 'Z'] as const;

  dimensions.forEach((dimension, index) => {
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      issues.push({
        code: 'invalid-dimension',
        message: `${dimensionLabels[index]} must be a positive integer.`,
      });
    } else if (dimension % FIREX_GRID_AXIS_QUANTUM !== 0) {
      issues.push({
        code: 'unaligned-dimension',
        message: `${dimensionLabels[index]} must be a multiple of ${FIREX_GRID_AXIS_QUANTUM}.`,
      });
    }
    if (dimension > limits.maxTextureDimension3D) {
      issues.push({
        code: 'texture-dimension-limit',
        message: `${dimensionLabels[index]}=${dimension} exceeds this device's 3D-texture limit of ${limits.maxTextureDimension3D}.`,
      });
    }
  });

  if (estimate.largestStorageBufferBytes > limits.maxStorageBufferBindingSize) {
    issues.push({
      code: 'storage-binding-limit',
      message: `A Fire-X vector field requires ${formatFireXBytes(estimate.largestStorageBufferBytes)}, above this device's ${formatFireXBytes(limits.maxStorageBufferBindingSize)} storage-binding limit.`,
    });
  }
  if (estimate.largestStorageBufferBytes > limits.maxBufferSize) {
    issues.push({
      code: 'buffer-size-limit',
      message: `A Fire-X vector field requires ${formatFireXBytes(estimate.largestStorageBufferBytes)}, above this device's ${formatFireXBytes(limits.maxBufferSize)} buffer limit.`,
    });
  }

  const memoryBudgetBytes = options.memoryBudgetBytes === undefined
    ? FIREX_DEFAULT_GRID_MEMORY_BUDGET_BYTES
    : options.memoryBudgetBytes;
  if (memoryBudgetBytes !== null && estimate.totalBytes > memoryBudgetBytes) {
    issues.push({
      code: 'memory-budget',
      message: `Estimated persistent Fire-X resources require ${formatFireXBytes(estimate.totalBytes)}, above the configured ${formatFireXBytes(memoryBudgetBytes)} budget.`,
    });
  }

  return { ok: issues.length === 0, dimensions, estimate, issues };
}

export function deriveFireXGridInfo(
  dimensions: FireXGridDimensions,
  domainMeters: FireXDomainExtent,
  particleCount = 0,
): FireXGridInfo {
  const memory = estimateFireXGridMemory(dimensions, particleCount);
  const voxelMeters: FireXDomainExtent = [
    domainMeters[0] / dimensions[0],
    domainMeters[1] / dimensions[1],
    domainMeters[2] / dimensions[2],
  ];
  return {
    ...memory,
    dimensions,
    domainMeters,
    voxelMeters,
    volumeCubicMeters: domainMeters[0] * domainMeters[1] * domainMeters[2],
  };
}

function formatDecimal(value: number, fractionDigits: number): string {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(fractionDigits).replace(/\.0+$|(?<=\.[0-9]*?)0+$/u, '');
}

function formatInteger(value: number): string {
  const digits = String(Math.max(0, Math.floor(value)));
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatTuple(values: FireXDomainExtent, scale: number, fractionDigits: number): string {
  return values.map((value) => formatDecimal(value * scale, fractionDigits)).join(' × ');
}

export function formatFireXBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  const mebibytes = bytes / (1024 * 1024);
  if (mebibytes < 1) return `${formatDecimal(bytes / 1024, 1)} KiB`;
  const gibibytes = mebibytes / 1024;
  if (gibibytes >= 1024) {
    const tebibytes = gibibytes / 1024;
    return `${formatDecimal(tebibytes, tebibytes >= 10 ? 1 : 2)} TiB`;
  }
  if (gibibytes >= 1) {
    return `${formatDecimal(gibibytes, gibibytes >= 10 ? 1 : 2)} GiB`;
  }
  return `${formatDecimal(mebibytes, mebibytes >= 10 ? 1 : 2)} MiB`;
}

export function formatFireXGridInfo(
  info: FireXGridInfo,
  unitSystem: FireXUnitSystem,
): FormattedFireXGridInfo {
  const resolution = info.dimensions.join('×');
  const cellCount = `${formatInteger(info.cellCount)} cells`;
  const metric = unitSystem === 'metric';
  const domain = metric
    ? `${formatTuple(info.domainMeters, 1, 2)} m`
    : `${formatTuple(info.domainMeters, METERS_TO_FEET, 2)} ft`;
  const voxelSize = metric
    ? `${formatTuple(info.voxelMeters, 100, 2)} cm`
    : `${formatTuple(info.voxelMeters, METERS_TO_INCHES, 2)} in`;
  const volume = metric
    ? `${formatDecimal(info.volumeCubicMeters, 2)} m³`
    : `${formatDecimal(info.volumeCubicMeters * CUBIC_METERS_TO_CUBIC_FEET, 2)} ft³`;
  const memory = formatFireXBytes(info.totalBytes);
  return {
    resolution,
    cellCount,
    domain,
    voxelSize,
    volume,
    memory,
    summary: `${resolution} · ${domain} · Δ ${voxelSize} · ${volume} · ${memory}`,
  };
}
