import { describe, expect, it } from 'vitest';
import {
  ADVECT_STATE_FRAGMENT,
  ADVECT_VELOCITY_FRAGMENT,
  PARTICLE_UPDATE_VERTEX,
  PARTICLE_PROJECT_VERTEX,
  DETAIL_FRAGMENT,
  RENDER_FRAGMENT,
  SPARK_VERTEX,
  SOURCE_VELOCITY_FRAGMENT,
  THERMO_FRAGMENT,
} from './shaders';

function continuousSlicePosition(z: number, sliceCount: number): number {
  return Math.min(sliceCount - 1, Math.max(0, z * sliceCount - 0.5));
}

describe('Horvath slab coordinates', () => {
  it('maps every slab center back to the exact same layer', () => {
    for (const sliceCount of [12, 18, 28, 48]) {
      for (let layer = 0; layer < sliceCount; layer += 1) {
        const z = (layer + 0.5) / sliceCount;
        expect(continuousSlicePosition(z, sliceCount)).toBeCloseTo(layer, 12);
      }
    }
  });

  it('uses the cell-center convention in both volume sampling and particle projection', () => {
    expect(RENDER_FRAGMENT).toContain('position.z * float(uSliceCount) - 0.5');
    expect(PARTICLE_PROJECT_VERTEX).toContain('position.z * float(uSliceCount) - 0.5');
    expect(PARTICLE_PROJECT_VERTEX).toContain('1.0 - fraction');
    expect(PARTICLE_PROJECT_VERTEX).toContain('vTileBounds');
  });
});

describe('Horvath characteristic tracing', () => {
  it('uses an RK2 midpoint backtrace without a fake reverse correction', () => {
    for (const shader of [ADVECT_STATE_FRAGMENT, ADVECT_VELOCITY_FRAGMENT]) {
      expect(shader).toContain('midpoint = position - 0.5 * uDeltaTime * velocity');
      expect(shader).toContain('position - uDeltaTime * midpointVelocity');
      expect(shader).not.toContain('reverseValue');
      expect(shader).not.toContain('corrected');
    }
  });
});

describe('Horvath source scaling', () => {
  it('changes emitter footprint and particle reach independently', () => {
    expect(PARTICLE_UPDATE_VERTEX).toContain('uniform float uSourceSize;');
    expect(PARTICLE_UPDATE_VERTEX).toContain('uniform float uFlameHeight;');
    expect(PARTICLE_UPDATE_VERTEX).toContain('baseLife * clamp(uFlameHeight, 0.4, 2.5)');
    expect(PARTICLE_PROJECT_VERTEX).toContain('life *= clamp(uFlameHeight, 0.4, 2.5)');
    expect(PARTICLE_UPDATE_VERTEX).toContain('* uSourceSize');
  });

  it('keeps view zoom anchored at the burner', () => {
    const burnerTarget = 'vec3 target = vec3(0.5, 0.468 / uViewZoom, 0.5);';
    expect(RENDER_FRAGMENT).toContain(burnerTarget);
    expect(SPARK_VERTEX).toContain(burnerTarget);
  });

  it('treats source intensity as injection strength rather than shorter lifetime', () => {
    expect(PARTICLE_UPDATE_VERTEX).toContain('age += uDeltaTime;');
    expect(PARTICLE_UPDATE_VERTEX).not.toContain('age += uDeltaTime * mix');
    for (const shader of [THERMO_FRAGMENT, SOURCE_VELOCITY_FRAGMENT]) {
      expect(shader).toContain('float intensity = max(uEmission / 0.86, 0.10);');
      expect(shader).toContain('* 0.18 * intensity');
    }
  });
});

describe('Horvath Cinematic rendering', () => {
  it('marches a real view ray and gates the showcase transfer function explicitly', () => {
    expect(RENDER_FRAGMENT).toContain('uniform float uCinematic;');
    expect(RENDER_FRAGMENT).toMatch(/vec3\s+rayOrigin\s*=/);
    expect(RENDER_FRAGMENT).toMatch(/vec3\s+rayDirection\s*=/);
    expect(RENDER_FRAGMENT).toContain('rayOrigin + rayDirection *');
    expect(RENDER_FRAGMENT).not.toContain('parallaxDirection');
  });

  it('applies the ACES curve per channel instead of scaling every channel by one peak', () => {
    expect(RENDER_FRAGMENT).toMatch(
      /\(color \* \(2\.51 \* color \+ 0\.03\)\)\s*\/ \(color \* \(2\.43 \* color \+ 0\.59\) \+ 0\.14\)/,
    );
    expect(RENDER_FRAGMENT).not.toContain('float peak = max(color.r');
    expect(RENDER_FRAGMENT).toContain('uCinematic');
  });

  it('reserves a 512-sample ray loop and scales resolved detail with offline field resolution', () => {
    expect(RENDER_FRAGMENT).toContain('index < 512');
    expect(RENDER_FRAGMENT).toContain('uniform float uDetailFrequencyScale;');
    expect(RENDER_FRAGMENT).toContain('vec3 finePosition = position * uDetailFrequencyScale;');
    expect(DETAIL_FRAGMENT).toContain('uniform float uDetailFrequencyScale;');
    expect(DETAIL_FRAGMENT).toContain('* uDetailFrequencyScale');
  });
});
