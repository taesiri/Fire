import { describe, expect, it } from 'vitest';
import {
  FIREX_DEFAULT_GRID_MEMORY_BUDGET_BYTES,
  FIREX_DETAIL_TEXTURE_BYTES,
  FIREX_GRID_BYTES_PER_CELL,
  FIREX_GRID_MEMORY_GUARDS,
  FIREX_GRID_PRESETS,
  FIREX_HEAVY_GRID_BYTES_PER_CELL,
  FIREX_HEAVY_GRID_CELL_THRESHOLD,
  FIREX_MAX_GRID_MEMORY_BUDGET_BYTES,
  FIREX_MAX_NEGOTIATED_STORAGE_BYTES,
  deriveFireXGridInfo,
  estimateFireXGridMemory,
  formatFireXBytes,
  formatFireXGridInfo,
  maximumFireXDeviceDenseCubeAxis,
  normalizeFireXGridDimensions,
  preflightFireXGrid,
  selectFireXRequiredDeviceLimits,
} from './gridConfiguration';

const generousDeviceLimits = {
  maxTextureDimension3D: 2048,
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxBufferSize: 256 * 1024 * 1024,
} as const;

describe('Fire-X grid presets', () => {
  it('exposes auto, fixed high-resolution, experimental, and custom choices', () => {
    expect(FIREX_GRID_PRESETS).toEqual([
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
        minimumGuardBytes: 2 * 1024 ** 3,
      },
      {
        id: 'workstation',
        label: 'Workstation dense · 320³ · 4 GiB guard',
        dimensions: [320, 320, 320],
        experimental: true,
        requiresConfirmation: true,
        minimumGuardBytes: 4 * 1024 ** 3,
      },
      {
        id: 'studio',
        label: 'Studio dense · 384³ · 8 GiB guard',
        dimensions: [384, 384, 384],
        experimental: true,
        requiresConfirmation: true,
        minimumGuardBytes: 8 * 1024 ** 3,
      },
      {
        id: 'reference',
        label: 'Reference dense · 508³ · 16+ GiB guard',
        dimensions: [508, 508, 508],
        experimental: true,
        requiresConfirmation: true,
        minimumGuardBytes: 16 * 1024 ** 3,
      },
      { id: 'custom', label: 'Custom', dimensions: null },
    ]);
  });

  it('exposes explicit opt-in guards through 32 GiB', () => {
    expect(FIREX_GRID_MEMORY_GUARDS.map((guard) => guard.bytes)).toEqual([
      512 * 1024 ** 2,
      1024 ** 3,
      2 * 1024 ** 3,
      4 * 1024 ** 3,
      8 * 1024 ** 3,
      16 * 1024 ** 3,
      24 * 1024 ** 3,
      32 * 1024 ** 3,
    ]);
    expect(FIREX_MAX_GRID_MEMORY_BUDGET_BYTES).toBe(32 * 1024 ** 3);
  });
});

describe('Fire-X grid normalization', () => {
  it('snaps to multiples of four and clamps to the supported custom range', () => {
    expect(normalizeFireXGridDimensions([81, 142, 999])).toEqual([80, 144, 1000]);
    expect(normalizeFireXGridDimensions([-3, 17, 18])).toEqual([16, 16, 20]);
  });

  it('uses an explicit fallback for non-finite input', () => {
    expect(normalizeFireXGridDimensions(
      [Number.NaN, Number.POSITIVE_INFINITY, 66],
      { fallback: [80, 120, 80] },
    )).toEqual([80, 120, 68]);
  });

  it('aligns caller-provided bounds to the workgroup quantum', () => {
    expect(normalizeFireXGridDimensions([1, 200, 77], {
      minimumAxis: 18,
      maximumAxis: 126,
    })).toEqual([20, 124, 76]);
  });
});

describe('Fire-X memory and device preflight', () => {
  it('switches to compact pressure residency exactly at the heavy-grid threshold', () => {
    const belowThreshold = estimateFireXGridMemory([1020, 256, 256]);
    const atThreshold = estimateFireXGridMemory([1024, 256, 256]);

    expect(belowThreshold.cellCount).toBeLessThan(FIREX_HEAVY_GRID_CELL_THRESHOLD);
    expect(belowThreshold.gridBytes).toBe(belowThreshold.cellCount * FIREX_GRID_BYTES_PER_CELL);
    expect(atThreshold.cellCount).toBe(FIREX_HEAVY_GRID_CELL_THRESHOLD);
    expect(atThreshold.gridBytes).toBe(atThreshold.cellCount * FIREX_HEAVY_GRID_BYTES_PER_CELL);
  });

  it('accounts for all persistent grid, particle, and fixed detail allocations', () => {
    const estimate = estimateFireXGridMemory([80, 120, 80], 1536);
    expect(estimate).toEqual({
      cellCount: 768_000,
      gridBytes: 768_000 * 116,
      particleBytes: 1536 * 80,
      fixedBytes: FIREX_DETAIL_TEXTURE_BYTES,
      totalBytes: 768_000 * 116 + 1536 * 80 + FIREX_DETAIL_TEXTURE_BYTES,
      largestStorageBufferBytes: 768_000 * 16,
    });
  });

  it('accepts the 128×192×128 experimental preset under conservative WebGPU limits', () => {
    const result = preflightFireXGrid(
      [128, 192, 128],
      generousDeviceLimits,
      { particleCount: 3072 },
    );
    expect(result.ok).toBe(true);
    expect(result.issues).toEqual([]);
    expect(result.estimate.totalBytes).toBeLessThan(FIREX_DEFAULT_GRID_MEMORY_BUDGET_BYTES);
  });

  it('accepts a 192³ dense grid only with the expert aggregate budget', () => {
    const result = preflightFireXGrid(
      [192, 192, 192],
      generousDeviceLimits,
      { particleCount: 1536, memoryBudgetBytes: 1024 * 1024 * 1024 },
    );
    expect(result.ok).toBe(true);
    expect(result.estimate.totalBytes).toBeGreaterThan(FIREX_DEFAULT_GRID_MEMORY_BUDGET_BYTES);
  });

  it.each([
    [256, 16_777_216, 1_947_328_512, 268_435_456, 2 * 1024 ** 3, '1.81 GiB', '256 MiB'],
    [320, 32_768_000, 3_802_259_456, 524_288_000, 4 * 1024 ** 3, '3.54 GiB', '500 MiB'],
    [384, 56_623_104, 6_569_451_520, 905_969_664, 8 * 1024 ** 3, '6.12 GiB', '864 MiB'],
    [512, 134_217_728, 14_496_686_080, 2_147_483_648, 16 * 1024 ** 3, '13.5 GiB', '2 GiB'],
  ])(
    'accepts a real %i³ dense grid when its exact field limit and guard are available',
    (axis, cellCount, totalBytes, largestStorageBufferBytes, memoryBudgetBytes, totalLabel, fieldLabel) => {
      const dimensions = [axis, axis, axis] as const;
      const estimate = estimateFireXGridMemory(dimensions, 1536);
      expect(estimate).toMatchObject({ cellCount, totalBytes, largestStorageBufferBytes });
      expect(formatFireXBytes(estimate.totalBytes)).toBe(totalLabel);
      expect(formatFireXBytes(estimate.largestStorageBufferBytes)).toBe(fieldLabel);

      const result = preflightFireXGrid(
        dimensions,
        {
          maxTextureDimension3D: 2048,
          maxStorageBufferBindingSize: largestStorageBufferBytes,
          maxBufferSize: largestStorageBufferBytes,
        },
        { particleCount: 1536, memoryBudgetBytes },
      );
      expect(result.ok).toBe(true);
      expect(result.issues).toEqual([]);
    },
  );

  it('reports alignment, texture, binding, buffer, and aggregate-budget failures', () => {
    const result = preflightFireXGrid(
      [162, 240, 160],
      {
        maxTextureDimension3D: 192,
        maxStorageBufferBindingSize: 32 * 1024 * 1024,
        maxBufferSize: 48 * 1024 * 1024,
      },
      { particleCount: 4096, memoryBudgetBytes: 256 * 1024 * 1024 },
    );
    expect(result.ok).toBe(false);
    expect(result.issues.map((issue) => issue.code)).toEqual(expect.arrayContaining([
      'unaligned-dimension',
      'texture-dimension-limit',
      'storage-binding-limit',
      'buffer-size-limit',
      'memory-budget',
    ]));
  });

  it('allows the caller to disable only the aggregate memory guard', () => {
    const result = preflightFireXGrid(
      [160, 240, 160],
      {
        maxTextureDimension3D: 2048,
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxBufferSize: 256 * 1024 * 1024,
      },
      { memoryBudgetBytes: null },
    );
    expect(result.ok).toBe(true);
  });

  it('preflights an exact 1024×1024×1024 request without normalizing it', () => {
    const particleCount = 1536;
    const result = preflightFireXGrid(
      [1024, 1024, 1024],
      generousDeviceLimits,
      { particleCount },
    );

    expect(result.dimensions).toEqual([1024, 1024, 1024]);
    expect(result.estimate).toEqual({
      cellCount: 1_073_741_824,
      gridBytes: 1_073_741_824 * 108,
      particleBytes: particleCount * 80,
      fixedBytes: FIREX_DETAIL_TEXTURE_BYTES,
      totalBytes: 1_073_741_824 * 108
        + particleCount * 80
        + FIREX_DETAIL_TEXTURE_BYTES,
      largestStorageBufferBytes: 16 * 1024 * 1024 * 1024,
    });
    expect(formatFireXBytes(result.estimate.gridBytes)).toBe('108 GiB');
    expect(formatFireXBytes(result.estimate.largestStorageBufferBytes)).toBe('16 GiB');

    const issueCodes = result.issues.map((issue) => issue.code);
    expect(issueCodes).toEqual(expect.arrayContaining([
      'storage-binding-limit',
      'buffer-size-limit',
      'memory-budget',
    ]));
    expect(issueCodes).not.toContain('texture-dimension-limit');
  });

  it('reports an unaligned 1023-cell axis instead of normalizing it', () => {
    const result = preflightFireXGrid([1023, 1024, 1024], generousDeviceLimits);

    expect(result.dimensions).toEqual([1023, 1024, 1024]);
    expect(result.issues.map((issue) => issue.code)).toContain('unaligned-dimension');
  });
});

describe('Fire-X WebGPU limit negotiation', () => {
  it('keeps default adapters at their supported limits', () => {
    expect(selectFireXRequiredDeviceLimits(generousDeviceLimits)).toEqual({
      maxStorageBufferBindingSize: 128 * 1024 ** 2,
      maxBufferSize: 256 * 1024 ** 2,
    });
  });

  it('requests stronger adapter limits only through the 512³ field requirement', () => {
    expect(selectFireXRequiredDeviceLimits({
      maxStorageBufferBindingSize: 8 * 1024 ** 3,
      maxBufferSize: 16 * 1024 ** 3,
    })).toEqual({
      maxStorageBufferBindingSize: FIREX_MAX_NEGOTIATED_STORAGE_BYTES,
      maxBufferSize: FIREX_MAX_NEGOTIATED_STORAGE_BYTES,
    });
  });

  it.each([
    [128 * 1024 ** 2, 256 * 1024 ** 2, 200],
    [256 * 1024 ** 2, 256 * 1024 ** 2, 256],
    [500 * 1024 ** 2, 500 * 1024 ** 2, 320],
    [864 * 1024 ** 2, 864 * 1024 ** 2, 384],
    [2 * 1024 ** 3, 2 * 1024 ** 3, 512],
  ])('reports a %i-byte binding ceiling as a %i-byte buffer and %i³ cube', (binding, buffer, axis) => {
    expect(maximumFireXDeviceDenseCubeAxis({
      maxTextureDimension3D: 2048,
      maxStorageBufferBindingSize: binding,
      maxBufferSize: buffer,
    })).toBe(axis);
  });

  it('does not round a one-byte-short 512³ field up to the 512³ tier', () => {
    expect(maximumFireXDeviceDenseCubeAxis({
      maxTextureDimension3D: 2048,
      maxStorageBufferBindingSize: 2 * 1024 ** 3 - 1,
      maxBufferSize: 2 * 1024 ** 3,
    })).toBe(508);
  });
});

describe('Fire-X byte formatting', () => {
  it('preserves MiB output and promotes large values to GiB and TiB', () => {
    expect(formatFireXBytes(86.125 * 1024 * 1024)).toBe('86.1 MiB');
    expect(formatFireXBytes(1024 * 1024 * 1024)).toBe('1 GiB');
    expect(formatFireXBytes(1024 * 1024 * 1024 * 1024)).toBe('1 TiB');
  });
});

describe('Fire-X derived grid information', () => {
  const info = deriveFireXGridInfo([80, 120, 80], [1.15, 1.35, 1.15], 1536);

  it('derives per-axis voxel size and physical volume', () => {
    expect(info.voxelMeters[0]).toBeCloseTo(0.014375);
    expect(info.voxelMeters[1]).toBeCloseTo(0.01125);
    expect(info.voxelMeters[2]).toBeCloseTo(0.014375);
    expect(info.volumeCubicMeters).toBeCloseTo(1.785375);
  });

  it('formats metric grid and physical-domain information', () => {
    expect(formatFireXGridInfo(info, 'metric')).toEqual({
      resolution: '80×120×80',
      cellCount: '768,000 cells',
      domain: '1.15 × 1.35 × 1.15 m',
      voxelSize: '1.44 × 1.13 × 1.44 cm',
      volume: '1.79 m³',
      memory: '86.1 MiB',
      summary: '80×120×80 · 1.15 × 1.35 × 1.15 m · Δ 1.44 × 1.13 × 1.44 cm · 1.79 m³ · 86.1 MiB',
    });
  });

  it('formats the same canonical domain in imperial units', () => {
    expect(formatFireXGridInfo(info, 'imperial')).toEqual({
      resolution: '80×120×80',
      cellCount: '768,000 cells',
      domain: '3.77 × 4.43 × 3.77 ft',
      voxelSize: '0.57 × 0.44 × 0.57 in',
      volume: '63.05 ft³',
      memory: '86.1 MiB',
      summary: '80×120×80 · 3.77 × 4.43 × 3.77 ft · Δ 0.57 × 0.44 × 0.57 in · 63.05 ft³ · 86.1 MiB',
    });
  });
});
