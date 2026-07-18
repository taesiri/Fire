import { describe, expect, it } from 'vitest';
import { fitCanvasSize } from './canvasSizing';
import { estimateFireXGridMemory } from './firex/gridConfiguration';
import {
  FIREX_OFFLINE_QUALITY,
  FIREX_PIXEL_RATIO_CAP,
  FIREX_QUALITY,
  resolveFireXOfflineDimensions,
} from './firex/FireXEngine';
import {
  estimateHorvathMemory,
  HORVATH_OFFLINE_QUALITY,
  HORVATH_QUALITY,
} from './horvath/HorvathEngine';
import { OFFLINE_RENDER_TIERS, QUALITY_LEVELS } from './types';

describe('quality profile budgets', () => {
  it('defines every shared quality level in both engines', () => {
    expect(Object.keys(FIREX_QUALITY)).toEqual([...QUALITY_LEVELS]);
    expect(Object.keys(HORVATH_QUALITY)).toEqual([...QUALITY_LEVELS]);
    expect(Object.keys(FIREX_PIXEL_RATIO_CAP)).toEqual([...QUALITY_LEVELS]);
  });

  it('keeps Fire-X Maximum inside shader ceilings and above High', () => {
    const high = FIREX_QUALITY.high;
    const maximum = FIREX_QUALITY.maximum;
    const cells = (profile: typeof high) => profile.dimensions.reduce((total, value) => total * value, 1);

    expect(maximum).toMatchObject({
      dimensions: [80, 120, 80],
      pressureIterations: 48,
      correctionIterations: 48,
      raySteps: 192,
      renderScale: 1.25,
      particleCount: 1536,
    });
    expect(cells(maximum)).toBeGreaterThan(cells(high));
    expect(maximum.pressureIterations).toBeGreaterThan(high.pressureIterations);
    expect(maximum.correctionIterations).toBeGreaterThan(high.correctionIterations);
    expect(maximum.particleCount).toBeGreaterThan(high.particleCount);
    expect(maximum.raySteps).toBeGreaterThan(high.raySteps);
    expect(maximum.raySteps).toBeLessThanOrEqual(192);
    expect(FIREX_PIXEL_RATIO_CAP.maximum).toBe(2);
  });

  it('uses the bounded Fire-X simulation budget for Cinematic presentation', () => {
    const cinematic = FIREX_QUALITY.cinematic;

    expect(cinematic).toMatchObject({
      dimensions: [80, 120, 80],
      pressureIterations: 48,
      correctionIterations: 48,
      raySteps: 192,
      renderScale: 1,
      particleCount: 1536,
    });
    expect(cinematic.raySteps).toBeLessThanOrEqual(192);
  });

  it('keeps Horvath Maximum inside its ray loop and preserves kernel scale', () => {
    const high = HORVATH_QUALITY.high;
    const maximum = HORVATH_QUALITY.maximum;

    expect(maximum).toMatchObject({
      tileHeight: 144,
      slices: 48,
      columns: 8,
      jacobiIterations: 18,
      particleCount: 32768,
      particleRadius: 4.7,
      raySteps: 96,
      renderScale: 1.25,
    });
    expect(maximum.particleCount).toBeGreaterThan(high.particleCount);
    expect(maximum.raySteps).toBeGreaterThan(high.raySteps);
    expect(maximum.raySteps).toBeLessThanOrEqual(128);
    expect(maximum.particleRadius / maximum.tileHeight)
      .toBeLessThan(high.particleRadius / high.tileHeight);
    expect(Math.ceil(maximum.slices / maximum.columns) * maximum.columns)
      .toBeGreaterThanOrEqual(maximum.slices);
  });

  it('spends the Horvath Cinematic budget on real-ray presentation', () => {
    const cinematic = HORVATH_QUALITY.cinematic;
    const tileWidth = Math.round((cinematic.tileHeight * 1.22) / 4) * 4;

    expect(cinematic).toMatchObject({
      tileHeight: 176,
      slices: 56,
      columns: 8,
      jacobiIterations: 24,
      particleCount: 65536,
      particleRadius: 3,
      raySteps: 128,
      renderScale: 1.15,
    });
    expect([tileWidth, cinematic.tileHeight, cinematic.slices]).toEqual([216, 176, 56]);
    expect(cinematic.raySteps).toBeLessThanOrEqual(128);
    expect(Math.ceil(cinematic.slices / cinematic.columns) * cinematic.columns)
      .toBeGreaterThanOrEqual(cinematic.slices);
  });

  it('allocates dedicated Fire-X solver bundles instead of upscaling the live field', () => {
    expect(Object.keys(FIREX_OFFLINE_QUALITY)).toEqual([...OFFLINE_RENDER_TIERS]);
    expect(FIREX_OFFLINE_QUALITY.hd).toMatchObject({
      dimensions: [128, 192, 128],
      pressureIterations: 64,
      correctionIterations: 64,
      particleCount: 2048,
      raySteps: 256,
      timeStep: 1 / 60,
      opticalDetailTarget: 512,
    });
    expect(FIREX_OFFLINE_QUALITY.qhd).toMatchObject({
      dimensions: [160, 240, 160],
      pressureIterations: 80,
      correctionIterations: 80,
      particleCount: 3072,
      raySteps: 320,
      timeStep: 1 / 60,
      opticalDetailTarget: 768,
    });
    expect(FIREX_OFFLINE_QUALITY.uhd).toMatchObject({
      dimensions: [192, 288, 192],
      pressureIterations: 96,
      correctionIterations: 96,
      particleCount: 4096,
      raySteps: 384,
      timeStep: 1 / 60,
      opticalDetailTarget: 1024,
    });
    for (const tier of OFFLINE_RENDER_TIERS) {
      const profile = FIREX_OFFLINE_QUALITY[tier];
      const liveCells = FIREX_QUALITY.cinematic.dimensions.reduce((total, value) => total * value, 1);
      const offlineCells = profile.dimensions.reduce((total, value) => total * value, 1);
      expect(Object.isFrozen(profile)).toBe(true);
      expect(offlineCells).toBeGreaterThan(liveCells);
      expect(estimateFireXGridMemory(profile.dimensions, profile.particleCount).totalBytes)
        .toBeLessThan(profile.minimumMemoryGuardBytes);
    }
    expect(resolveFireXOfflineDimensions('hd', [160, 128, 192])).toEqual([160, 192, 192]);
    expect(resolveFireXOfflineDimensions('uhd', null)).toEqual([192, 288, 192]);
  });

  it('allocates exact screen-plane-heavy Horvath offline atlases at 120 Hz', () => {
    expect(Object.keys(HORVATH_OFFLINE_QUALITY)).toEqual([...OFFLINE_RENDER_TIERS]);
    expect(HORVATH_OFFLINE_QUALITY.hd).toMatchObject({
      tileHeight: 256,
      slices: 64,
      particleCount: 131072,
      jacobiIterations: 32,
      raySteps: 256,
      fixedTimeStep: 1 / 120,
      minimumTextureSize: 2496,
    });
    expect(HORVATH_OFFLINE_QUALITY.qhd).toMatchObject({
      tileHeight: 352,
      slices: 80,
      particleCount: 262144,
      jacobiIterations: 48,
      raySteps: 352,
      fixedTimeStep: 1 / 120,
      minimumTextureSize: 3520,
    });
    expect(HORVATH_OFFLINE_QUALITY.uhd).toMatchObject({
      tileHeight: 448,
      slices: 96,
      particleCount: 524288,
      jacobiIterations: 64,
      raySteps: 448,
      fixedTimeStep: 1 / 120,
      minimumTextureSize: 5376,
    });
    expect(estimateHorvathMemory(HORVATH_OFFLINE_QUALITY.hd)).toMatchObject({
      atlasWidth: 2496,
      atlasHeight: 2048,
    });
    expect(estimateHorvathMemory(HORVATH_OFFLINE_QUALITY.qhd)).toMatchObject({
      atlasWidth: 3424,
      atlasHeight: 3520,
    });
    expect(estimateHorvathMemory(HORVATH_OFFLINE_QUALITY.uhd)).toMatchObject({
      atlasWidth: 4384,
      atlasHeight: 5376,
    });
    expect(estimateHorvathMemory(HORVATH_OFFLINE_QUALITY.uhd).totalBytes)
      .toBeGreaterThan(estimateHorvathMemory(HORVATH_OFFLINE_QUALITY.qhd).totalBytes);
  });
});

describe('canvas backing-store sizing', () => {
  it('preserves CSS aspect ratio when either output cap binds', () => {
    for (const [width, height, scale, maximumWidth, maximumHeight] of [
      [1633, 1016, 2, 3072, 1728],
      [900, 1600, 2, 3072, 1728],
      [3200, 900, 1.5, 3072, 1728],
      [1633, 1016, 1.1, 3072, 1728],
    ] as const) {
      const fitted = fitCanvasSize(width, height, scale, maximumWidth, maximumHeight);
      expect(Math.abs(fitted.width / fitted.height - width / height)).toBeLessThan(0.002);
      expect(fitted.width).toBeLessThanOrEqual(maximumWidth);
      expect(fitted.height).toBeLessThanOrEqual(maximumHeight);
    }
  });
});
