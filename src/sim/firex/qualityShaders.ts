// Render-facing Fire-X shaders are kept separate from the solver shader so the
// simulation can remain storage-buffer based while presentation uses hardware
// filtered 3D textures.  This avoids nearest-neighbour voxel blocks without
// forcing a risky solver rewrite.

export const FIREX_PACK_WGSL = /* wgsl */ `
struct Parameters {
  grid: vec4f,
  time: vec4f,
  chemistry: vec4f,
  water: vec4f,
  camera: vec4f,
  render: vec4f,
  controls: vec4f,
  tuning: vec4f,
  domain: vec4f,
  fire: vec4f,
  mixing: vec4f,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> flowField: array<vec4f>;
@group(0) @binding(2) var<storage, read> speciesField: array<vec4f>;
@group(0) @binding(3) var<storage, read> reactionField: array<f32>;
@group(0) @binding(4) var<storage, read> liquidField: array<u32>;
@group(0) @binding(5) var<storage, read> divergenceField: array<f32>;
@group(0) @binding(6) var flowTexture: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var chemistryTexture: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var mediaTexture: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var<storage, read_write> copiedFlowField: array<vec4f>;

fn cellIndex(cell: vec3u) -> u32 {
  let dimensions = vec3u(params.grid.xyz);
  return cell.x + dimensions.x * (cell.y + dimensions.y * cell.z);
}

fn storePackedFields(invocation: vec3u, flow: vec4f) {
  let dimensions = vec3u(params.grid.xyz);
  let id = cellIndex(invocation);
  let species = max(speciesField[id], vec4f(0.0));
  let reaction = clamp(reactionField[id], 0.0, 4.0);
  let richness = clamp(species.x / max(species.x + species.y * 0.25, 0.0001), 0.0, 1.0);
  let soot = species.z * smoothstep(0.25, 0.82, richness);
  let liquid = f32(liquidField[id]) * (3.4 / 4096.0);
  // The outer cell centers implement the mixed wall/open boundary stencil and
  // are not interior fluid samples. Suppress only that one-cell visualization
  // shell; every interior signed post-projection residual remains unchanged.
  let boundaryShell = any(invocation == vec3u(0)) || any(invocation + vec3u(1) >= dimensions);
  var displayDivergence = max(
    max(reaction, soot),
    max(liquid, max(species.w, smoothstep(420.0, 700.0, flow.w)))
  );
  if (u32(clamp(round(params.controls.x), 0.0, 7.0)) == 7u) {
    displayDivergence = 0.0;
    if (!boundaryShell) {
      displayDivergence = divergenceField[id];
    }
  }
  let coordinate = vec3i(invocation);
  textureStore(flowTexture, coordinate, vec4f(flow.xyz, clamp((flow.w - 300.0) / 2600.0, 0.0, 1.0)));
  textureStore(chemistryTexture, coordinate, species);
  textureStore(mediaTexture, coordinate, vec4f(reaction, soot, liquid, displayDivergence));
}

@compute @workgroup_size(4, 4, 4)
fn packFields(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensions = vec3u(params.grid.xyz);
  if (any(invocation >= dimensions)) {
    return;
  }
  storePackedFields(invocation, flowField[cellIndex(invocation)]);
}

// The heavy mixed-precision offline Beauty path already has its corrected
// velocity in flowField. Copy it into the canonical solver buffer while it is
// hot and populate the three filterable render volumes in the same traversal.
@compute @workgroup_size(4, 4, 4)
fn copyAndPackBeautyFields(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensions = vec3u(params.grid.xyz);
  if (any(invocation >= dimensions)) {
    return;
  }
  let id = cellIndex(invocation);
  let flow = flowField[id];
  copiedFlowField[id] = flow;
  storePackedFields(invocation, flow);
}
`;

export const FIREX_QUALITY_RENDER_WGSL = /* wgsl */ `
struct Parameters {
  grid: vec4f,
  time: vec4f,
  chemistry: vec4f,
  water: vec4f,
  camera: vec4f,
  render: vec4f,
  controls: vec4f,
  tuning: vec4f,
  domain: vec4f,
  fire: vec4f,
  mixing: vec4f,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var volumeSampler: sampler;
@group(0) @binding(2) var flowTexture: texture_3d<f32>;
@group(0) @binding(3) var chemistryTexture: texture_3d<f32>;
@group(0) @binding(4) var mediaTexture: texture_3d<f32>;
@group(0) @binding(6) var detailTexture: texture_3d<f32>;
@group(0) @binding(7) var detailSampler: sampler;

struct Particle {
  positionAge: vec4f,
  velocityMass: vec4f,
};

@group(0) @binding(5) var<storage, read> particles: array<Particle>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

fn domainExtent() -> vec3f {
  return max(params.domain.xyz, vec3f(0.001));
}

fn viewZoom() -> f32 {
  return clamp(params.fire.w, 0.5, 3.0);
}

fn cameraTarget() -> vec3f {
  let extent = domainExtent();
  return vec3f(extent.x * 0.5, extent.y * 0.40, extent.z * 0.5);
}

fn cameraEye() -> vec3f {
  let yaw = params.camera.x;
  let pitch = params.camera.y;
  let orbit = vec3f(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch));
  let extent = domainExtent();
  let frameScale = max(extent.x, max(extent.y, extent.z));
  return cameraTarget() + orbit * (0.90 + params.camera.z * 0.18) * frameScale;
}

@vertex
fn fullscreenVertex(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let corner = vec2f(f32((vertexIndex << 1u) & 2u), f32(vertexIndex & 2u));
  var output: VertexOutput;
  output.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  output.uv = corner;
  return output;
}

struct DropletOutput {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) massSpeed: vec2f,
};

fn projectWorld(point: vec3f) -> vec3f {
  let lookAtPoint = cameraTarget();
  let eye = cameraEye();
  let forward = normalize(lookAtPoint - eye);
  let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, forward));
  let relative = point - eye;
  let depth = max(dot(relative, forward), 0.0001);
  let lens = 0.72 / viewZoom();
  return vec3f(
    dot(relative, right) / (depth * params.camera.w * lens),
    dot(relative, up) / (depth * lens),
    depth
  );
}

@vertex
fn dropletVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> DropletOutput {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  let corner = corners[vertexIndex];
  let particle = particles[instanceIndex];
  let particleVisible = particle.velocityMass.w > 0.0001
    && all(particle.positionAge.xyz >= vec3f(0.0))
    && all(particle.positionAge.xyz <= vec3f(1.0));
  var output: DropletOutput;
  if (!particleVisible) {
    output.position = vec4f(2.0, 2.0, 0.0, 1.0);
    output.local = corner;
    output.massSpeed = vec2f(0.0);
    return output;
  }
  let worldPosition = particle.positionAge.xyz * domainExtent();
  let projected = projectWorld(worldPosition);
  let projectedTail = projectWorld(worldPosition - particle.velocityMass.xyz * 0.018);
  let streak = projectedTail.xy - projected.xy;
  let streakLength = min(length(streak), 0.032);
  let tangent = select(vec2f(1.0, 0.0), normalize(streak), streakLength > 0.00001);
  let normal = vec2f(-tangent.y, tangent.x);
  let pixelRadius = select(1.65, 1.15, params.controls.z >= 0.5);
  let radius = pixelRadius * 2.0 / max(params.render.y, 1.0);
  let offset = normal * corner.x * radius + tangent * corner.y * (radius + streakLength * 0.5);
  output.position = vec4f(projected.xy + offset, 0.0, 1.0);
  output.local = corner;
  output.massSpeed = vec2f(particle.velocityMass.w, length(particle.velocityMass.xyz));
  return output;
}

@fragment
fn dropletFragment(input: DropletOutput) -> @location(0) vec4f {
  let viewMode = u32(clamp(round(params.controls.x), 0.0, 7.0));
  if ((viewMode != 0u && viewMode != 6u) || input.massSpeed.x <= 0.0001) {
    discard;
  }
  let capsuleDistance = length(vec2f(input.local.x, max(abs(input.local.y) - 0.48, 0.0)));
  let coverage = 1.0 - smoothstep(0.64, 1.0, capsuleDistance);
  if (coverage <= 0.001) {
    discard;
  }
  let highlight = pow(clamp(1.0 - abs(input.local.x + 0.30), 0.0, 1.0), 5.0);
  let baseColor = mix(vec3f(0.27, 0.34, 0.38), vec3f(0.76, 0.86, 0.91), highlight * 0.62);
  let diagnosticBoost = select(1.0, 1.38, viewMode == 6u);
  let alpha = coverage * clamp(0.13 + input.massSpeed.x * 0.15, 0.12, 0.42) * diagnosticBoost;
  let linearColor = baseColor * 0.72;
  let outputColor = select(aces(linearColor), linearColor, params.mixing.z >= 0.5);
  return vec4f(outputColor, alpha);
}

fn sampleFlow(position: vec3f) -> vec4f {
  let halfTexel = 0.5 / params.grid.xyz;
  let uvw = clamp(position / domainExtent(), halfTexel, vec3f(1.0) - halfTexel);
  return textureSampleLevel(flowTexture, volumeSampler, uvw, 0.0);
}

fn sampleChemistry(position: vec3f) -> vec4f {
  let halfTexel = 0.5 / params.grid.xyz;
  let uvw = clamp(position / domainExtent(), halfTexel, vec3f(1.0) - halfTexel);
  return textureSampleLevel(chemistryTexture, volumeSampler, uvw, 0.0);
}

fn sampleMedia(position: vec3f) -> vec4f {
  let halfTexel = 0.5 / params.grid.xyz;
  let uvw = clamp(position / domainExtent(), halfTexel, vec3f(1.0) - halfTexel);
  return textureSampleLevel(mediaTexture, volumeSampler, uvw, 0.0);
}

fn sampleDetail(position: vec3f) -> vec4f {
  return textureSampleLevel(detailTexture, detailSampler, position, 0.0);
}

fn hash21(value: vec2f) -> f32 {
  let p = fract(value * vec2f(123.34, 456.21));
  return fract((p.x + p.y) * (p.x + p.y + 45.32));
}

fn rayBox(origin: vec3f, direction: vec3f) -> vec2f {
  let safeDirection = select(vec3f(0.000001), direction, abs(direction) > vec3f(0.000001));
  let inverseDirection = 1.0 / safeDirection;
  let lower = -origin * inverseDirection;
  let upper = (domainExtent() - origin) * inverseDirection;
  let nearPlane = min(lower, upper);
  let farPlane = max(lower, upper);
  return vec2f(max(max(nearPlane.x, nearPlane.y), nearPlane.z), min(min(farPlane.x, farPlane.y), farPlane.z));
}

fn blackbody(temperature: f32) -> vec3f {
  let t = clamp((temperature - 720.0) / 1900.0, 0.0, 1.0);
  let red = vec3f(1.0, 0.035, 0.002);
  let orange = vec3f(1.0, 0.24, 0.008);
  let gold = vec3f(1.0, 0.68, 0.12);
  let white = vec3f(1.0, 0.94, 0.72);
  return mix(mix(red, orange, smoothstep(0.0, 0.32, t)),
    mix(gold, white, smoothstep(0.58, 1.0, t)), smoothstep(0.26, 0.72, t));
}

fn thermalMap(value: f32) -> vec3f {
  let t = clamp(value, 0.0, 1.0);
  let cold = vec3f(0.015, 0.025, 0.12);
  let blue = vec3f(0.02, 0.36, 1.0);
  let amber = vec3f(1.0, 0.34, 0.01);
  let white = vec3f(1.0, 0.96, 0.76);
  return mix(mix(cold, blue, smoothstep(0.0, 0.30, t)),
    mix(amber, white, smoothstep(0.62, 1.0, t)), smoothstep(0.24, 0.72, t));
}

fn acesCurve(value: f32) -> f32 {
  return clamp(
    (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14),
    0.0,
    1.0
  );
}

fn aces(color: vec3f) -> vec3f {
  let x = max(color * max(params.controls.w, 0.25), vec3f(0.0));
  let peak = max(x.x, max(x.y, x.z));
  let mappedPeak = acesCurve(peak);
  let mapped = clamp(x * (mappedPeak / max(peak, 0.000001)), vec3f(0.0), vec3f(1.0));
  return pow(mapped, vec3f(1.0 / 2.2));
}

fn diagnosticSample(mode: u32, flow: vec4f, chemistry: vec4f, media: vec4f) -> vec4f {
  if (mode == 1u) {
    let strength = smoothstep(0.015, 0.72, flow.w);
    return vec4f(thermalMap(flow.w), strength);
  }
  if (mode == 2u) {
    let strength = smoothstep(0.003, 0.38, media.x);
    return vec4f(mix(vec3f(0.45, 0.01, 0.0), vec3f(1.0, 0.95, 0.52), strength), strength);
  }
  if (mode == 3u) {
    let fuel = clamp(chemistry.x, 0.0, 1.0);
    let ambientOxygen = max(params.chemistry.y, 0.001);
    let oxygenDeficit = clamp((ambientOxygen - chemistry.y) / ambientOxygen, 0.0, 1.0);
    let overlap = min(fuel, clamp(chemistry.y / ambientOxygen, 0.0, 1.0));
    let strength = max(fuel, max(oxygenDeficit * 0.72, overlap * 0.32));
    return vec4f(vec3f(fuel, overlap * 0.36, oxygenDeficit), strength);
  }
  if (mode == 4u) {
    let strength = smoothstep(0.005, 0.62, chemistry.z);
    return vec4f(mix(vec3f(0.06, 0.02, 0.12), vec3f(0.74, 0.26, 1.0), strength), strength);
  }
  if (mode == 5u) {
    let vapor = smoothstep(0.004, 0.72, chemistry.w);
    let soot = smoothstep(0.003, 0.55, media.y);
    return vec4f(mix(vec3f(0.56, 0.72, 0.82), vec3f(0.17, 0.12, 0.10), soot), max(vapor, soot));
  }
  if (mode == 6u) {
    let liquid = smoothstep(0.002, 0.32, media.z);
    let speed = smoothstep(0.035, 1.25, length(flow.xyz));
    let direction = 0.5 + 0.5 * normalize(flow.xyz + vec3f(0.0001));
    let color = mix(direction * vec3f(0.25, 0.31, 0.36), vec3f(0.78, 0.91, 1.0), liquid);
    return vec4f(color, max(liquid, speed * 0.038));
  }
  let divergence = media.w;
  let magnitude = smoothstep(0.01, 1.2, abs(divergence));
  let signedColor = select(vec3f(0.12, 0.42, 1.0), vec3f(1.0, 0.18, 0.045), divergence >= 0.0);
  return vec4f(signedColor, magnitude);
}

fn studioBackground(uv: vec2f, eye: vec3f, rayDirection: vec3f) -> vec3f {
  var color = mix(vec3f(0.020, 0.024, 0.030), vec3f(0.0025, 0.004, 0.007), clamp(uv.y, 0.0, 1.0));
  let floorDistance = -eye.y / min(rayDirection.y, -0.00001);
  if (rayDirection.y < -0.00001 && floorDistance > 0.0) {
    let floorPosition = eye + rayDirection * floorDistance;
    let rings = smoothstep(0.012, 0.0, abs(length(floorPosition.xz - domainExtent().xz * 0.5) - 0.10));
    let grid = pow(max(abs(sin(floorPosition.x * 31.4159)), abs(sin(floorPosition.z * 31.4159))), 28.0);
    color = mix(vec3f(0.018, 0.020, 0.022), vec3f(0.045, 0.048, 0.050), grid * 0.22) + rings * vec3f(0.055, 0.020, 0.006);
  }
  return color;
}

fn rotateDetailCoordinate(position: vec3f) -> vec3f {
  // A fixed, orthonormal two-axis rotation keeps the second octave from
  // exposing the periodic detail volume's axis-aligned repetition.
  let xz = vec2f(
    position.x * 0.771 + position.z * 0.637,
    -position.x * 0.637 + position.z * 0.771
  );
  return vec3f(
    xz.x,
    position.y * 0.909 - xz.y * 0.417,
    position.y * 0.417 + xz.y * 0.909
  );
}

@fragment
fn volumeFragment(input: VertexOutput) -> @location(0) vec4f {
  let screen = input.uv * 2.0 - 1.0;
  let lookAtPoint = cameraTarget();
  let eye = cameraEye();
  let forward = normalize(lookAtPoint - eye);
  let right = normalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
  let up = normalize(cross(right, forward));
  let lens = 0.72 / viewZoom();
  let rayDirection = normalize(forward + right * screen.x * params.camera.w * lens + up * screen.y * lens);
  let interval = rayBox(eye, rayDirection);
  let background = studioBackground(input.uv, eye, rayDirection);
  let cinematic = clamp(params.mixing.z, 0.0, 1.0);
  if (interval.x > interval.y || interval.y <= 0.0) {
    let outputBackground = select(aces(background), background, cinematic >= 0.5);
    return vec4f(outputBackground, 1.0);
  }

  let startDistance = max(interval.x, 0.0);
  let stepCount = min(u32(params.render.z), 384u);
  let stepLength = (interval.y - startDistance) / max(f32(stepCount), 1.0);
  let jitter = hash21(floor(input.position.xy));
  var distance = startDistance + jitter * stepLength;
  var transmittance = 1.0;
  var radiance = vec3f(0.0);
  let viewMode = u32(clamp(round(params.controls.x), 0.0, 7.0));

  for (var step = 0u; step < 384u; step += 1u) {
    if (step >= stepCount || distance >= interval.y || transmittance < 0.006) {
      break;
    }
    let position = eye + rayDirection * distance;
    let rawMedia = sampleMedia(position);
    if (viewMode == 0u && cinematic >= 0.5) {
      let occupancy = max(max(rawMedia.x, rawMedia.y), max(rawMedia.z, rawMedia.w));
      if (occupancy < 0.0005) {
        let nextDistance = distance + stepLength;
        if (nextDistance < interval.y) {
          let nextPosition = eye + rayDirection * nextDistance;
          let nextMedia = sampleMedia(nextPosition);
          let nextOccupancy = max(max(nextMedia.x, nextMedia.y), max(nextMedia.z, nextMedia.w));
          distance += select(stepLength, stepLength * 2.0, nextOccupancy < 0.0005);
        } else {
          distance = interval.y;
        }
        continue;
      }
    }
    let flow = sampleFlow(position);
    let chemistry = max(sampleChemistry(position), vec4f(0.0));
    let media = vec4f(max(rawMedia.xyz, vec3f(0.0)), rawMedia.w);

    if (viewMode == 0u) {
      let temperature = 300.0 + flow.w * 2600.0;
      let reactionRate = max(media.x, 0.0);
      let reactionFront = mix(
        smoothstep(0.006, 0.32, reactionRate),
        smoothstep(0.004, 0.28, reactionRate),
        cinematic
      );
      let thermal = smoothstep(620.0, 1780.0, temperature);
      // Keep render-only breakup bounded so it reveals an advected reaction
      // front without replacing that front with thresholded procedural bands.
      let detailPosition = position - flow.xyz * 0.08;
      let detailA = sin(dot(detailPosition, vec3f(27.0, 35.0, 13.0)) + params.time.y * 3.4);
      let detailB = sin(dot(detailPosition, vec3f(-21.0, 29.0, 17.0)) - params.time.y * 4.7);
      var opticalDetail = 0.90 + 0.20 * detailA * detailB;
      if (cinematic > 0.5 && reactionFront > 0.002) {
        let normalized = detailPosition / domainExtent();
        // mixing.w is a render-only optical-detail target. Zero follows the
        // actual solver dimensions; explicit targets tile the resident 64^3
        // periodic texture to 256^3, 512^3 or 1024^3 sample frequency without
        // allocating a correspondingly dense simulation field.
        let requestedDetail = params.mixing.w;
        let explicitDetail = clamp(requestedDetail, 64.0, 1024.0);
        let detailCells = select(params.grid.xyz, vec3f(explicitDetail), requestedDetail >= 1.0);
        let tileScale = max(detailCells / 64.0, vec3f(0.015625));
        let timeOffset = params.time.y * vec3f(0.013, -0.021, 0.017);
        let detailUv = normalized * tileScale + timeOffset;
        let noise = sampleDetail(detailUv);
        let primaryNoise = noise.x - 0.5;
        let rotated = rotateDetailCoordinate(normalized - vec3f(0.5));
        let secondaryNoise = sampleDetail(
          rotated * tileScale * 0.47
            + vec3f(0.371, 0.113, 0.733)
            - timeOffset.yzx * 0.61
        ).y - 0.5;

        // The full-frequency octave is attenuated as it outruns the bounded
        // ray budget; the lower rotated octave retains coherent breakup. This
        // keeps the 1024 target visible without turning high-sample marching into
        // unstable high-frequency flicker.
        let maximumDetailCells = max(detailCells.x, max(detailCells.y, detailCells.z));
        let rayBudget = clamp(params.render.z, 1.0, 384.0);
        let primaryBand = clamp((rayBudget * 2.0) / maximumDetailCells, 0.25, 1.0);
        let secondaryBand = clamp(
          (rayBudget * 2.0) / max(maximumDetailCells * 0.47, 1.0),
          0.35,
          1.0
        );
        let fbm = primaryNoise * (0.55 * primaryBand)
          + secondaryNoise * (0.45 * secondaryBand);
        let edgeMask = smoothstep(0.025, 0.24, reactionFront)
          * (1.0 - smoothstep(0.72, 0.96, reactionFront));
        let breakup = clamp(1.0 + fbm * 0.44, 0.84, 1.16);
        opticalDetail = mix(1.0, breakup, edgeMask * 0.85);
      }
      // A power transfer rejects the broad, low-energy interpolation tail while
      // retaining the resolved reaction front. This is field reconstruction,
      // not an image-space sharpen pass.
      let frontExponent = mix(
        0.85 + clamp(params.mixing.y, 0.5, 2.5) * 0.35,
        0.92 + clamp(params.mixing.y, 0.5, 2.5) * 0.16,
        cinematic
      );
      let cinematicFront = reactionFront * (0.42 + 0.58 * thermal);
      let standardFront = reactionFront * thermal;
      let flameDensity = pow(
        clamp(mix(standardFront, cinematicFront, cinematic), 0.0, 1.0),
        frontExponent
      ) * opticalDetail;
      let soot = clamp(
        1.0 - exp(-max(media.y, 0.0) * mix(1.15, 0.90, cinematic)),
        0.0,
        mix(1.0, 0.96, cinematic)
      );
      // Grid liquid is a coupling/verification field, not a second droplet
      // renderer. Showing it in Beauty produced voxel-sized white capsules on
      // top of the instanced particle streaks. Keep liquid in view mode 6 and
      // render only softly scattering vapor here.
      let vapor = 1.0 - exp(-max(chemistry.w, 0.0) * mix(0.34, 0.30, cinematic));
      let richness = clamp(chemistry.x / max(chemistry.x + chemistry.y * 0.25, 0.0001), 0.0, 1.0);
      let sootReveal = mix(1.0, 1.0 - smoothstep(0.045, 0.30, reactionFront) * 0.78, cinematic);
      let flameSigma = flameDensity * mix(3.20, 0.88, cinematic);
      let sootSigma = soot * sootReveal * mix(1.6, 1.62, cinematic);
      let vaporSigma = vapor * mix(0.20, 0.32, cinematic);
      let extinction = min(flameSigma + sootSigma + vaporSigma, mix(100.0, 5.5, cinematic));
      let opticalDepth = min(
        extinction * stepLength * mix(5.2, 3.15, cinematic),
        mix(100.0, 0.28, cinematic)
      );
      let alpha = 1.0 - exp(-opticalDepth);
      let hotSoot = soot * thermal;
      let emissionTemperature = mix(
        temperature + reactionFront * 650.0,
        820.0 + 2150.0 * pow(clamp(reactionFront * 0.74 + thermal * 0.38, 0.0, 1.0), 0.72),
        cinematic
      );
      let edgeWarmth = 1.0 - smoothstep(0.18, 0.72, reactionFront * thermal);
      let standardFlameColor = mix(
        blackbody(emissionTemperature),
        vec3f(1.0, 0.075, 0.002),
        edgeWarmth * 0.68
      );
      let cinematicAmber = mix(
        vec3f(1.0, 0.055, 0.002),
        vec3f(1.0, 0.43, 0.018),
        smoothstep(0.025, 0.30, flameDensity)
      );
      let cinematicFlameColor = mix(
        cinematicAmber,
        vec3f(1.0, 0.91, 0.60),
        smoothstep(0.52, 0.92, flameDensity) * smoothstep(980.0, 2050.0, temperature)
      );
      let flameColor = mix(standardFlameColor, cinematicFlameColor, cinematic);
      let cinematicEmission = flameDensity * (1.35 + 3.65 * reactionFront)
        * (0.92 + 0.32 * params.chemistry.z)
        + hotSoot * 0.08;
      let standardEmission = flameDensity * (1.20 + 1.80 * reactionFront)
        * (0.90 + 0.30 * params.chemistry.z)
        + hotSoot * 0.25;
      let emissive = flameColor
        * mix(standardEmission, cinematicEmission, cinematic);
      let baseZone = 1.0 - smoothstep(0.16, 0.48, position.y / domainExtent().y);
      let blueEmission = vec3f(0.018, 0.12, 0.74)
        * flameDensity * (1.0 - richness) * mix(0.48, baseZone * 0.34, cinematic);
      let sourcePoint = vec3f(domainExtent().x * 0.5, 0.08, domainExtent().z * 0.5);
      let toFire = sourcePoint - position;
      let fireFalloff = 1.0 / (1.0 + dot(toFire, toFire) * 8.0);
      let ambientLight = vec3f(0.018, 0.022, 0.028);
      let fireLight = vec3f(1.0, 0.24, 0.035) * fireFalloff;
      let standardScatter = sootSigma * vec3f(0.012, 0.009, 0.007)
        + vaporSigma * vec3f(0.12, 0.15, 0.17);
      let cinematicScatter = sootSigma * (ambientLight * 0.12 + fireLight * 0.055)
        + vaporSigma * (ambientLight * 0.55 + fireLight * 0.05);
      let scattered = mix(standardScatter, cinematicScatter, cinematic);
      let sourceRadiance = (emissive + blueEmission + scattered) / max(extinction, 0.0001);
      radiance += transmittance * alpha * sourceRadiance;
      transmittance *= 1.0 - alpha;
    } else {
      let fieldValue = diagnosticSample(viewMode, flow, chemistry, media);
      let alpha = 1.0 - exp(-fieldValue.a * stepLength * 8.5);
      radiance += transmittance * alpha * fieldValue.rgb * 1.35;
      transmittance *= 1.0 - alpha;
    }
    distance += stepLength;
  }

  let color = radiance + transmittance * background;
  let outputColor = select(aces(color), color, cinematic >= 0.5);
  return vec4f(outputColor, 1.0);
}
`;

export const FIREX_POST_WGSL = /* wgsl */ `
struct Parameters {
  grid: vec4f,
  time: vec4f,
  chemistry: vec4f,
  water: vec4f,
  camera: vec4f,
  render: vec4f,
  controls: vec4f,
  tuning: vec4f,
  domain: vec4f,
  fire: vec4f,
  mixing: vec4f,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var postSampler: sampler;
@group(0) @binding(2) var sceneTexture: texture_2d<f32>;
@group(0) @binding(3) var bloomTexture: texture_2d<f32>;

struct PostOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn postVertex(@builtin(vertex_index) vertexIndex: u32) -> PostOutput {
  let corner = vec2f(f32((vertexIndex << 1u) & 2u), f32(vertexIndex & 2u));
  var output: PostOutput;
  output.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  // WebGPU render targets and sampled textures use opposite Y conventions at
  // this fullscreen boundary. Flip exactly once while reading the HDR scene.
  output.uv = vec2f(corner.x, 1.0 - corner.y);
  return output;
}

fn brightExtract(color: vec3f) -> vec3f {
  let luminance = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  let threshold = 1.15;
  let knee = 0.35;
  let soft = clamp((luminance - threshold + knee) / (2.0 * knee), 0.0, 1.0);
  let contribution = max(luminance - threshold, 0.0) + soft * soft * knee;
  return color * contribution / max(luminance, 0.0001);
}

@fragment
fn bloomFragment(input: PostOutput) -> @location(0) vec4f {
  let dimensions = vec2f(textureDimensions(sceneTexture));
  let texel = 1.0 / max(dimensions, vec2f(1.0));
  let radius = texel * 5.5;
  var color = textureSampleLevel(sceneTexture, postSampler, input.uv, 0.0).rgb * 0.20;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv + vec2f(radius.x, 0.0), 0.0).rgb * 0.10;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv - vec2f(radius.x, 0.0), 0.0).rgb * 0.10;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv + vec2f(0.0, radius.y), 0.0).rgb * 0.10;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv - vec2f(0.0, radius.y), 0.0).rgb * 0.10;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv + radius, 0.0).rgb * 0.10;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv - radius, 0.0).rgb * 0.10;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv + vec2f(radius.x, -radius.y), 0.0).rgb * 0.10;
  color += textureSampleLevel(sceneTexture, postSampler, input.uv + vec2f(-radius.x, radius.y), 0.0).rgb * 0.10;
  return vec4f(brightExtract(color), 1.0);
}

fn acesCurve(value: f32) -> f32 {
  return clamp(
    (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14),
    0.0,
    1.0
  );
}

fn filmic(color: vec3f) -> vec3f {
  let exposed = max(color * max(params.controls.w, 0.25), vec3f(0.0));
  let mapped = clamp(
    (exposed * (2.51 * exposed + 0.03))
      / (exposed * (2.43 * exposed + 0.59) + 0.14),
    vec3f(0.0),
    vec3f(1.0)
  );
  return pow(mapped, vec3f(1.0 / 2.2));
}

fn dither(position: vec2f) -> f32 {
  let p = fract(position * vec2f(0.06711056, 0.00583715));
  return fract(52.9829189 * (p.x + p.y));
}

@fragment
fn compositeFragment(input: PostOutput) -> @location(0) vec4f {
  let scene = textureSampleLevel(sceneTexture, postSampler, input.uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTexture, postSampler, input.uv, 0.0).rgb;
  var color = filmic(scene + bloom * 0.10);
  color += vec3f((dither(input.position.xy) - 0.5) / 255.0);
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;
