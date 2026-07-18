import { describe, expect, it } from 'vitest';
import { FIREX_F16_PRESSURE_WGSL } from './pressureShaders';
import { FIREX_PACK_WGSL, FIREX_POST_WGSL, FIREX_QUALITY_RENDER_WGSL } from './qualityShaders';
import { FIREX_COMPUTE_WGSL, FIREX_PARTICLE_WGSL } from './shaders';

describe('Fire-X shader contracts', () => {
  it('limits half precision to heavy-grid pressure storage boundaries', () => {
    expect(FIREX_F16_PRESSURE_WGSL).toContain('enable f16;');
    expect(FIREX_F16_PRESSURE_WGSL).toContain('pressureIn: array<f16>');
    expect(FIREX_F16_PRESSURE_WGSL).toContain('pressureOut: array<f16>');
    expect(FIREX_F16_PRESSURE_WGSL).toContain('pressureTile: array<f32, 600>');
    expect(FIREX_F16_PRESSURE_WGSL).toContain('return f32(pressureIn[cellIndex(cell)]);');
    expect(FIREX_F16_PRESSURE_WGSL).toContain('pressureOut[id] = f16(pressure);');
  });

  it('preserves signed projected divergence for the diagnostic view', () => {
    expect(FIREX_QUALITY_RENDER_WGSL).toContain(
      'let media = vec4f(max(rawMedia.xyz, vec3f(0.0)), rawMedia.w);',
    );
    expect(FIREX_QUALITY_RENDER_WGSL).not.toContain(
      'max(sampleMedia(position), vec4f(0.0))',
    );
    expect(FIREX_PACK_WGSL).toContain(
      'if (u32(clamp(round(params.controls.x), 0.0, 7.0)) == 7u)',
    );
    expect(FIREX_PACK_WGSL).toContain('displayDivergence = 0.0;');
    expect(FIREX_PACK_WGSL).toContain('if (!boundaryShell)');
    expect(FIREX_PACK_WGSL).toContain('displayDivergence = divergenceField[id];');
  });

  it('shares exact Beauty packing between the standalone and corrected-flow copy paths', () => {
    expect(FIREX_PACK_WGSL).toContain('fn storePackedFields(invocation: vec3u, flow: vec4f)');
    expect(FIREX_PACK_WGSL).toContain('fn copyAndPackBeautyFields(');
    expect(FIREX_PACK_WGSL).toContain('copiedFlowField[id] = flow;');
    expect(FIREX_PACK_WGSL).toContain('storePackedFields(invocation, flow);');
  });

  it('uses consistent solid-floor and open-top projection boundaries', () => {
    expect(FIREX_COMPUTE_WGSL).toContain('if (cell.y >= gridDimensions().y)');
    expect(FIREX_COMPUTE_WGSL).toContain('pressureOut[cellIndex(cell)] = 0.0;');
    expect(FIREX_COMPUTE_WGSL).toContain(
      'flow = vec4f(belowFlow.xyz - pressureGradient(below, dimensions), flow.w);',
    );
    expect(FIREX_COMPUTE_WGSL).toContain('flow.y = 0.0;');
    expect(FIREX_COMPUTE_WGSL).toContain('flow.y = max(0.0, flow.y);');
    expect(FIREX_COMPUTE_WGSL).toContain('fn copyFlow(');
  });

  it('pairs backward divergence with a forward pressure gradient', () => {
    expect(FIREX_COMPUTE_WGSL).toContain('(center.x - left) * dimensions.x');
    expect(FIREX_COMPUTE_WGSL).toContain('(center.y - bottom) * dimensions.y');
    expect(FIREX_COMPUTE_WGSL).toContain('(center.z - back) * dimensions.z');
    expect(FIREX_COMPUTE_WGSL).toContain(
      '(loadPressure(cell + vec3i(1, 0, 0)) - center) * dimensions.x',
    );
    expect(FIREX_COMPUTE_WGSL).not.toContain('divergenceField[cellIndex(cell)] = 0.5 *');
  });

  it('conservatively deposits along the swept droplet segment', () => {
    expect(FIREX_PARTICLE_WGSL).toContain('const SWEEP_SAMPLES: u32 = 3u;');
    expect(FIREX_PARTICLE_WGSL).toContain('previous.positionAge.xyz + displacement * fraction');
    expect(FIREX_PARTICLE_WGSL).toContain('mass / f32(SWEEP_SAMPLES)');
    expect(FIREX_PARTICLE_WGSL).toContain('particle.positionAge.w >= previous.positionAge.w');
  });

  it('contains no analytic water-cone fallback', () => {
    expect(FIREX_COMPUTE_WGSL).not.toContain('waterSpray');
    expect(FIREX_PARTICLE_WGSL).not.toContain('waterSpray');
  });

  it('propagates independent source dimensions and physical domain extent through the solver', () => {
    expect(FIREX_COMPUTE_WGSL).toContain('let radiusX = min(baseRadius * clamp(params.tuning.x, 0.35, 3.2)');
    expect(FIREX_COMPUTE_WGSL).toContain('let radiusZ = min(baseRadius * clamp(params.tuning.y, 0.35, 3.2)');
    expect(FIREX_COMPUTE_WGSL).toContain('let sourceThickness = clamp(params.tuning.z, 0.35, 2.5)');
    expect(FIREX_COMPUTE_WGSL).toContain('source * params.chemistry.x * firePower');
    expect(FIREX_COMPUTE_WGSL).toContain('originalFlow.xyz * dt * metric');
    expect(FIREX_COMPUTE_WGSL).toContain('return params.grid.xyz / domainExtent();');
    expect(FIREX_PARTICLE_WGSL).toContain('velocity * dt / domainExtent()');
    expect(FIREX_PARTICLE_WGSL).toContain('* domainExtent());');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('position / domainExtent()');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('domainExtent() - origin');
  });

  it('exposes plume energy, air mixing, persistence and framing as separate controls', () => {
    expect(FIREX_COMPUTE_WGSL).toContain('clamp(params.fire.x, 0.0, 4.0)');
    expect(FIREX_COMPUTE_WGSL).toContain('clamp(params.fire.y, 0.0, 4.0)');
    expect(FIREX_COMPUTE_WGSL).toContain('let persistence = clamp(params.fire.z, 0.4, 3.0);');
    expect(FIREX_COMPUTE_WGSL).toContain('let entrainment = clamp(params.mixing.x, 0.0, 3.0);');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('return clamp(params.fire.w, 0.5, 3.0);');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let lens = 0.72 / viewZoom();');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('clamp(params.mixing.y, 0.5, 2.5)');
  });

  it('keeps extreme fires bounded and renders emission through optical depth', () => {
    expect(FIREX_COMPUTE_WGSL).toContain('let thermalPower = min(powerRoot, 1.0 + 0.22 * (powerRoot - 1.0));');
    expect(FIREX_COMPUTE_WGSL).toContain('let interfaceWeight = oxygenDeficit * mix(1.0, 0.18, fuelCore);');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('smoothstep(0.004, 0.28, reactionRate)');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let sourceRadiance = (emissive + blueEmission + scattered)');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('radiance += transmittance * alpha * sourceRadiance;');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let rawMedia = sampleMedia(position);');
    expect(FIREX_QUALITY_RENDER_WGSL).not.toContain('reconstructionOffset');
    expect(FIREX_QUALITY_RENDER_WGSL).not.toContain('stepLength * 7.2');
  });

  it('keeps Cinematic detail field-driven and tone-maps the linear HDR composite once', () => {
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('var detailTexture: texture_3d<f32>;');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let noise = sampleDetail(detailUv);');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain(
      'let outputColor = select(aces(color), color, cinematic >= 0.5);',
    );
    expect(FIREX_POST_WGSL).toContain('fn brightExtract(color: vec3f) -> vec3f');
    expect(FIREX_POST_WGSL).toContain('fn bloomFragment(');
    expect(FIREX_POST_WGSL).toContain('fn compositeFragment(');
    expect(FIREX_POST_WGSL).toContain('output.uv = vec2f(corner.x, 1.0 - corner.y);');
    expect(FIREX_POST_WGSL).not.toContain('output.uv = corner;');
    expect(FIREX_POST_WGSL).toContain('filmic(scene + bloom * 0.10)');
    expect(FIREX_POST_WGSL).toContain('(dither(input.position.xy) - 0.5) / 255.0');
  });

  it('reserves a higher ray ceiling for offline quality sampling', () => {
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('min(u32(params.render.z), 384u)');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('step < 384u');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('clamp(params.render.z, 1.0, 384.0)');
  });

  it('keeps virtual optical detail separate from the dense solver grid', () => {
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let requestedDetail = params.mixing.w;');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let tileScale = max(detailCells / 64.0');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let rotated = rotateDetailCoordinate');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('let breakup = clamp(1.0 + fbm * 0.44, 0.84, 1.16);');
    expect(FIREX_QUALITY_RENDER_WGSL).toContain('if (cinematic > 0.5 && reactionFront > 0.002)');
  });
});
