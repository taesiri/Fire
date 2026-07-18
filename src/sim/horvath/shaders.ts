export const FULLSCREEN_VERTEX = `#version 300 es
precision highp float;
out vec2 vUv;

void main() {
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = corner;
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
`;

const ATLAS_COMMON = `
precision highp float;
precision highp int;
layout(location = 0) out vec4 outColor;

uniform ivec2 uTileSize;
uniform ivec2 uAtlasGrid;
uniform int uSliceCount;

int atlasLayer() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  ivec2 tile = pixel / uTileSize;
  return tile.y * uAtlasGrid.x + tile.x;
}

vec2 atlasLocalUv() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  ivec2 localPixel = pixel - (pixel / uTileSize) * uTileSize;
  return (vec2(localPixel) + 0.5) / vec2(uTileSize);
}

vec2 atlasUv(vec2 localUv, int layer) {
  int safeLayer = clamp(layer, 0, uSliceCount - 1);
  ivec2 tile = ivec2(safeLayer % uAtlasGrid.x, safeLayer / uAtlasGrid.x);
  vec2 halfTexel = 0.5 / vec2(uTileSize);
  vec2 safeUv = clamp(localUv, halfTexel, 1.0 - halfTexel);
  vec2 atlasPixel = vec2(tile * uTileSize) + safeUv * vec2(uTileSize);
  return atlasPixel / vec2(uTileSize * uAtlasGrid);
}

vec4 sampleLayer(sampler2D field, vec2 localUv, int layer) {
  return texture(field, atlasUv(localUv, layer));
}

vec4 sampleVolume(sampler2D field, vec3 position) {
  // Atlas layers represent cell centers at (layer + 0.5) / N. Converting
  // normalized depth back to continuous layer space must therefore subtract
  // half a cell; using z * (N - 1) would blur even an exact layer-center read.
  float slicePosition = clamp(
    position.z * float(uSliceCount) - 0.5,
    0.0,
    float(uSliceCount - 1)
  );
  int lowerLayer = int(floor(slicePosition));
  int upperLayer = min(lowerLayer + 1, uSliceCount - 1);
  float blend = fract(slicePosition);
  return mix(
    sampleLayer(field, position.xy, lowerLayer),
    sampleLayer(field, position.xy, upperLayer),
    blend
  );
}

bool insideVolume(vec3 position) {
  return all(greaterThanEqual(position, vec3(0.0)))
    && all(lessThanEqual(position, vec3(1.0)));
}

float hash4(vec4 value) {
  return fract(sin(dot(value, vec4(127.1, 311.7, 74.7, 269.5))) * 43758.5453123);
}

float noise4(vec4 value) {
  vec4 cell = floor(value);
  vec4 blend = fract(value);
  blend = blend * blend * (3.0 - 2.0 * blend);

  float n0000 = hash4(cell + vec4(0.0, 0.0, 0.0, 0.0));
  float n1000 = hash4(cell + vec4(1.0, 0.0, 0.0, 0.0));
  float n0100 = hash4(cell + vec4(0.0, 1.0, 0.0, 0.0));
  float n1100 = hash4(cell + vec4(1.0, 1.0, 0.0, 0.0));
  float n0010 = hash4(cell + vec4(0.0, 0.0, 1.0, 0.0));
  float n1010 = hash4(cell + vec4(1.0, 0.0, 1.0, 0.0));
  float n0110 = hash4(cell + vec4(0.0, 1.0, 1.0, 0.0));
  float n1110 = hash4(cell + vec4(1.0, 1.0, 1.0, 0.0));
  float n0001 = hash4(cell + vec4(0.0, 0.0, 0.0, 1.0));
  float n1001 = hash4(cell + vec4(1.0, 0.0, 0.0, 1.0));
  float n0101 = hash4(cell + vec4(0.0, 1.0, 0.0, 1.0));
  float n1101 = hash4(cell + vec4(1.0, 1.0, 0.0, 1.0));
  float n0011 = hash4(cell + vec4(0.0, 0.0, 1.0, 1.0));
  float n1011 = hash4(cell + vec4(1.0, 0.0, 1.0, 1.0));
  float n0111 = hash4(cell + vec4(0.0, 1.0, 1.0, 1.0));
  float n1111 = hash4(cell + vec4(1.0, 1.0, 1.0, 1.0));

  float z000 = mix(mix(n0000, n1000, blend.x), mix(n0100, n1100, blend.x), blend.y);
  float z100 = mix(mix(n0010, n1010, blend.x), mix(n0110, n1110, blend.x), blend.y);
  float z001 = mix(mix(n0001, n1001, blend.x), mix(n0101, n1101, blend.x), blend.y);
  float z101 = mix(mix(n0011, n1011, blend.x), mix(n0111, n1111, blend.x), blend.y);
  return mix(mix(z000, z100, blend.z), mix(z001, z101, blend.z), blend.w);
}

float fbm4(vec4 value) {
  float result = 0.0;
  float amplitude = 0.5714286;
  for (int octave = 0; octave < 3; ++octave) {
    result += amplitude * noise4(value);
    value = value * 2.03 + vec4(9.2, 3.7, 5.1, 1.9);
    amplitude *= 0.5;
  }
  return result;
}
`;

export const PARTICLE_UPDATE_VERTEX = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec4 aPositionAge;
layout(location = 1) in vec4 aVelocitySeed;

out vec4 vPositionAge;
out vec4 vVelocitySeed;

uniform sampler2D uVelocity;
uniform ivec2 uTileSize;
uniform ivec2 uAtlasGrid;
uniform int uSliceCount;
uniform int uSourceMode;
uniform float uTime;
uniform float uDeltaTime;
uniform float uEmission;
uniform float uBuoyancy;
uniform float uSourceSize;
uniform float uFlameHeight;

float hash11(float value) {
  return fract(sin(value * 127.1 + 17.17) * 43758.5453123);
}

vec3 hash31(float value) {
  return fract(sin(vec3(value, value + 19.19, value + 47.47)
    * vec3(127.1, 311.7, 74.7)) * 43758.5453123);
}

vec2 atlasUvFor(vec2 localUv, int layer) {
  int safeLayer = clamp(layer, 0, uSliceCount - 1);
  ivec2 tile = ivec2(safeLayer % uAtlasGrid.x, safeLayer / uAtlasGrid.x);
  vec2 halfTexel = 0.5 / vec2(uTileSize);
  vec2 safeUv = clamp(localUv, halfTexel, 1.0 - halfTexel);
  vec2 atlasPixel = vec2(tile * uTileSize) + safeUv * vec2(uTileSize);
  return atlasPixel / vec2(uTileSize * uAtlasGrid);
}

vec3 sampleVelocity(vec3 position) {
  float slicePosition = clamp(
    position.z * float(uSliceCount) - 0.5,
    0.0,
    float(uSliceCount - 1)
  );
  int lowerLayer = int(floor(slicePosition));
  int upperLayer = min(lowerLayer + 1, uSliceCount - 1);
  float blend = fract(slicePosition);
  return mix(
    texture(uVelocity, atlasUvFor(position.xy, lowerLayer)).xyz,
    texture(uVelocity, atlasUvFor(position.xy, upperLayer)).xyz,
    blend
  );
}

float particleLife(float seed) {
  float baseLife;
  if (uSourceMode == 1) baseLife = mix(0.62, 1.18, hash11(seed + 3.1));
  else if (uSourceMode == 2) baseLife = mix(0.68, 1.15, hash11(seed + 7.3));
  else baseLife = mix(0.82, 1.62, hash11(seed + 5.7));
  return baseLife * clamp(uFlameHeight, 0.4, 2.5);
}

void respawn(inout vec3 position, inout vec3 velocity, inout float age, inout float seed, bool warmStart) {
  seed = fract(seed + 0.61803398875);
  vec3 randomValue = hash31(seed * 4093.0 + floor(uTime * 0.17));
  float angle = 6.2831853 * randomValue.x;

  if (uSourceMode == 1) {
    vec3 direction = normalize(randomValue * 2.0 - 1.0 + vec3(0.34, 0.18, 0.0));
    position = vec3(0.36, 0.27, 0.5)
      + (randomValue - 0.5) * vec3(0.026, 0.022, 0.034) * uSourceSize;
    velocity = direction * mix(0.38, 0.76, randomValue.y) + vec3(0.10, 0.08, 0.0);
  } else if (uSourceMode == 2) {
    float sheet = floor(randomValue.x * 5.0);
    float sheetCenter = 0.18 + sheet * 0.16;
    float sheetJitter = (hash11(seed * 733.0 + 2.9) - 0.5) * 0.065;
    vec3 basePosition = vec3(
      sheetCenter + sheetJitter,
      mix(0.025, 0.052, randomValue.y),
      mix(0.18, 0.82, randomValue.z)
    );
    float wallScale = clamp(0.7 + 0.3 * uSourceSize, 0.79, 1.25);
    position = vec3(
      0.5 + (basePosition.x - 0.5) * wallScale,
      basePosition.y,
      0.5 + (basePosition.z - 0.5) * wallScale
    );
    velocity = vec3(
      mix(-0.065, 0.065, randomValue.z) + 0.018 * sin(sheet * 2.3),
      mix(0.42, 0.70, randomValue.y),
      0.060 * cos(angle * 1.7)
    );
  } else {
    // Deposit a continuous elliptical burner. Discrete jet columns read as
    // pipes once the resident field is large enough to resolve them.
    float localAngle = angle * 1.7 + hash11(seed * 521.0 + 9.7) * 2.4;
    float radius = sqrt(randomValue.y);
    vec2 burnerOffset = vec2(cos(localAngle), sin(localAngle))
      * radius * vec2(0.145, 0.095) * uSourceSize;
    position = vec3(
      0.5 + burnerOffset.x,
      mix(0.022, 0.043, randomValue.z),
      0.5 + burnerOffset.y
    );
    velocity = vec3(
      -burnerOffset.x * 0.18 + mix(-0.12, 0.12, randomValue.z),
      mix(0.40, 0.72, randomValue.y),
      -burnerOffset.y * 0.16 + mix(-0.075, 0.075, randomValue.z)
    );
  }

  age = warmStart ? hash11(seed * 177.0 + 1.7) * particleLife(seed) * 0.88 : 0.0;
  if (warmStart) {
    float warmDistance = age * (uSourceMode == 1 ? 0.72 : 0.56);
    position += velocity * warmDistance;
    if (uSourceMode != 1) {
      position.x += 0.11 * sin(seed * 91.0 + age * 5.0) * age;
      position.z += 0.050 * cos(seed * 73.0 - age * 4.0) * age;
    }
  }
}

void main() {
  vec3 position = aPositionAge.xyz;
  float age = aPositionAge.w;
  vec3 velocity = aVelocitySeed.xyz;
  float seed = aVelocitySeed.w;
  float life = particleLife(seed);
  bool invalid = age > 50.0 || any(lessThan(position, vec3(-0.08)))
    || any(greaterThan(position, vec3(1.08)));

  if (invalid || age >= life) {
    respawn(position, velocity, age, seed, invalid);
  } else {
    vec3 flow = sampleVelocity(clamp(position, vec3(0.0), vec3(1.0)));
    float phase = seed * 63.0 + uTime;
    vec3 fineMotion = vec3(
      sin(phase * 1.37 + position.y * 13.0),
      0.35 * sin(phase * 0.73 + position.x * 9.0),
      cos(phase * 1.11 - position.y * 11.0)
    );

    if (uSourceMode == 1) {
      vec3 radial = normalize(position - vec3(0.36, 0.27, 0.5) + vec3(0.0001));
      velocity += uDeltaTime * (0.22 * radial + vec3(0.0, 0.10 * uBuoyancy, 0.0));
      velocity *= exp(-0.42 * uDeltaTime);
    } else {
      velocity += uDeltaTime * vec3(0.0, 0.18 + 0.16 * uBuoyancy, 0.0);
      velocity += uDeltaTime * fineMotion * vec3(0.34, 0.045, 0.18);
      velocity = mix(velocity, flow + vec3(0.0, 0.16, 0.0), clamp(uDeltaTime * 0.75, 0.0, 0.08));
      velocity *= exp(-0.12 * uDeltaTime);
    }

    float maximumSpeed = uSourceMode == 1 ? 0.94 : 0.82;
    float speed = length(velocity);
    if (speed > maximumSpeed) velocity *= maximumSpeed / speed;
    position += velocity * uDeltaTime;
    age += uDeltaTime;
  }

  vPositionAge = vec4(position, age);
  vVelocitySeed = vec4(velocity, seed);
  gl_Position = vec4(0.0);
  gl_PointSize = 1.0;
}
`;

export const PARTICLE_UPDATE_FRAGMENT = `#version 300 es
precision highp float;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(0.0); }
`;

export const PARTICLE_PROJECT_VERTEX = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec4 aPositionAge;
layout(location = 1) in vec4 aVelocitySeed;

out vec3 vVelocity;
out float vHeat;
out float vDepthWeight;
flat out vec4 vTileBounds;

uniform ivec2 uTileSize;
uniform ivec2 uAtlasGrid;
uniform int uSliceCount;
uniform int uSourceMode;
uniform float uPointRadius;
uniform float uParticleWeight;
uniform float uFlameHeight;

float hash11(float value) {
  return fract(sin(value * 127.1 + 17.17) * 43758.5453123);
}

void main() {
  vec3 position = aPositionAge.xyz;
  float age = aPositionAge.w;
  float seed = aVelocitySeed.w;
  bool visible = age >= 0.0 && age < 50.0
    && all(greaterThanEqual(position, vec3(0.0)))
    && all(lessThanEqual(position, vec3(1.0)));
  if (!visible) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    vVelocity = vec3(0.0);
    vHeat = 0.0;
    vDepthWeight = 0.0;
    vTileBounds = vec4(0.0);
    return;
  }

  // Deposit into the same two cell-centered slabs that sampleVolume blends.
  // Fractional weights conserve each particle's projected mass in depth.
  float slicePosition = clamp(
    position.z * float(uSliceCount) - 0.5,
    0.0,
    float(uSliceCount - 1)
  );
  int lowerLayer = int(floor(slicePosition));
  int upperLayer = min(lowerLayer + 1, uSliceCount - 1);
  float fraction = fract(slicePosition);
  int layer = gl_InstanceID == 0 ? lowerLayer : upperLayer;
  float depthWeight = gl_InstanceID == 0 ? 1.0 - fraction : fraction;
  ivec2 tile = ivec2(layer % uAtlasGrid.x, layer / uAtlasGrid.x);
  vec2 atlasUv = (vec2(tile) + clamp(position.xy, vec2(0.001), vec2(0.999))) / vec2(uAtlasGrid);
  gl_Position = vec4(atlasUv * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uPointRadius;
  vTileBounds = vec4(vec2(tile * uTileSize), vec2((tile + ivec2(1)) * uTileSize));

  float life = uSourceMode == 1
    ? mix(0.62, 1.18, hash11(seed + 3.1))
    : (uSourceMode == 2
      ? mix(0.68, 1.15, hash11(seed + 7.3))
      : mix(0.82, 1.62, hash11(seed + 5.7)));
  life *= clamp(uFlameHeight, 0.4, 2.5);
  float normalizedAge = clamp(age / max(life, 0.001), 0.0, 1.0);
  float heatDecay = uSourceMode == 1 ? 1.35 : (uSourceMode == 2 ? 1.85 : 2.05);
  vHeat = exp(-normalizedAge * heatDecay);
  vVelocity = aVelocitySeed.xyz;
  float driverTail = uSourceMode == 1
    ? mix(0.14, 1.0, exp(-normalizedAge * 3.2))
    : (uSourceMode == 2
      ? mix(0.12, 1.0, exp(-normalizedAge * 3.6))
      : mix(0.10, 1.0, exp(-normalizedAge * 4.2)));
  vDepthWeight = uParticleWeight * driverTail * depthWeight;
}
`;

export const PARTICLE_PROJECT_FRAGMENT = `#version 300 es
precision highp float;

in vec3 vVelocity;
in float vHeat;
in float vDepthWeight;
flat in vec4 vTileBounds;

layout(location = 0) out vec4 outMotion;
layout(location = 1) out vec4 outThermo;

void main() {
  if (any(lessThan(gl_FragCoord.xy, vTileBounds.xy))
      || any(greaterThanEqual(gl_FragCoord.xy, vTileBounds.zw))) discard;
  vec2 point = gl_PointCoord * 2.0 - 1.0;
  float radiusSquared = dot(point, point);
  if (radiusSquared >= 1.0) discard;
  float kernel = (1.0 - radiusSquared);
  kernel = kernel * kernel * vDepthWeight;
  outMotion = vec4(vVelocity * kernel, kernel);
  outThermo = vec4(vHeat * kernel, 0.0, 0.0, kernel);
}
`;

export const THERMO_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uState;
uniform sampler2D uParticleMotion;
uniform sampler2D uParticleThermo;
uniform float uDeltaTime;
uniform float uCooling;
uniform float uEmission;
uniform int uSourceMode;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }

  vec2 uv = atlasLocalUv();
  vec4 state = sampleLayer(uState, uv, layer);
  vec4 particleMotion = sampleLayer(uParticleMotion, uv, layer);
  vec4 particleThermo = sampleLayer(uParticleThermo, uv, layer);

  float projectedMass = max(particleMotion.a, 0.0);
  float intensity = max(uEmission / 0.86, 0.10);
  float source = 1.0 - exp(-projectedMass * 0.18 * intensity);
  float sourceHeat = particleThermo.r / max(projectedMass, 0.0001);
  sourceHeat = clamp(sourceHeat, 0.0, 1.0);

  float coolingRate = mix(0.12, 0.68, clamp(uCooling / 1.4, 0.0, 1.0));
  state.g = max(0.0, state.g - uDeltaTime * coolingRate * pow(max(state.g, 0.0), 3.35));
  state.r *= exp(-uDeltaTime * (0.18 + 0.18 * uCooling));
  float reactionDecay = mix(2.40, 3.40, clamp(uCooling, 0.0, 1.4) / 1.4);
  reactionDecay *= uSourceMode == 2 ? 0.92 : 1.0;
  state.a *= exp(-uDeltaTime * reactionDecay);

  float profileDensity = uSourceMode == 2 ? 0.47 : (uSourceMode == 1 ? 0.42 : 0.38);
  float profileHeat = uSourceMode == 1 ? 1.08 : (uSourceMode == 2 ? 1.02 : 1.00);
  float reactionGain = uSourceMode == 2 ? 1.08 : 1.00;
  state.r = max(state.r, source * profileDensity);
  state.g = max(state.g, source * mix(0.54, profileHeat, sourceHeat));
  state.a = max(state.a, source * sourceHeat * 0.92 * reactionGain);

  float openBoundary = smoothstep(0.0, 0.024, uv.x)
    * smoothstep(0.0, 0.024, 1.0 - uv.x)
    * smoothstep(0.0, 0.035, uv.y)
    * smoothstep(0.0, 0.052, 1.0 - uv.y);
  state.r *= exp(-uDeltaTime * 7.0 * (1.0 - openBoundary));
  state.g *= exp(-uDeltaTime * 4.0 * (1.0 - openBoundary));
  state.a *= openBoundary;
  outColor = clamp(state, vec4(0.0, 0.0, -1.0, 0.0), vec4(3.0, 1.25, 1.0, 1.4));
}
`;

export const SOURCE_VELOCITY_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uVelocity;
uniform sampler2D uParticleMotion;
uniform float uDeltaTime;
uniform float uEmission;
uniform float uBuoyancy;
uniform int uSourceMode;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec2 uv = atlasLocalUv();
  vec3 velocity = sampleLayer(uVelocity, uv, layer).xyz;
  vec4 projected = sampleLayer(uParticleMotion, uv, layer);
  float intensity = max(uEmission / 0.86, 0.10);
  float source = 1.0 - exp(-max(projected.a, 0.0) * 0.18 * intensity);
  vec3 projectedVelocity = projected.xyz / max(projected.a, 0.0001);
  float coupling = clamp(source * uDeltaTime * 26.0, 0.0, 0.44);
  velocity = mix(velocity, projectedVelocity, coupling);
  if (uSourceMode != 1) velocity.y += uDeltaTime * source * (0.18 + 0.12 * uBuoyancy);
  float speed = length(velocity);
  if (speed > 1.2) velocity *= 1.2 / speed;
  outColor = vec4(velocity, 1.0);
}
`;

export const ADVECT_STATE_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uState;
uniform sampler2D uVelocity;
uniform float uDeltaTime;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec3 position = vec3(atlasLocalUv(), (float(layer) + 0.5) / float(uSliceCount));
  vec3 velocity = sampleVolume(uVelocity, position).xyz;
  vec3 midpoint = position - 0.5 * uDeltaTime * velocity;
  vec3 midpointVelocity = sampleVolume(uVelocity, clamp(midpoint, vec3(0.0), vec3(1.0))).xyz;
  vec3 backPosition = position - uDeltaTime * midpointVelocity;
  if (!insideVolume(backPosition)) {
    outColor = vec4(0.0);
    return;
  }
  // Second-order midpoint characteristic tracing. Reconstruction remains
  // monotone because this is a single filtered semi-Lagrangian field sample.
  outColor = sampleVolume(uState, backPosition);
}
`;

export const ADVECT_VELOCITY_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uVelocity;
uniform float uDeltaTime;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec3 position = vec3(atlasLocalUv(), (float(layer) + 0.5) / float(uSliceCount));
  vec3 velocity = sampleVolume(uVelocity, position).xyz;
  vec3 midpoint = position - 0.5 * uDeltaTime * velocity;
  vec3 midpointVelocity = sampleVolume(uVelocity, clamp(midpoint, vec3(0.0), vec3(1.0))).xyz;
  vec3 backPosition = position - uDeltaTime * midpointVelocity;
  if (!insideVolume(backPosition)) {
    outColor = vec4(0.0);
    return;
  }
  vec3 advected = sampleVolume(uVelocity, backPosition).xyz;
  float speed = length(advected);
  if (speed > 1.2) advected *= 1.2 / speed;
  outColor = vec4(advected, 1.0);
}
`;

export const DETAIL_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uState;
uniform float uDeltaTime;
uniform float uTime;
uniform float uDetail;
uniform float uDetailFrequencyScale;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec2 uv = atlasLocalUv();
  float z = (float(layer) + 0.5) / float(uSliceCount);
  vec4 state = sampleLayer(uState, uv, layer);

  float broad = noise4(vec4(uv * vec2(5.2, 7.4), z * 4.1, uTime * 0.48));
  float medium = 0.5 + 0.5 * sin(
    uv.x * 71.0 - uv.y * 47.0 + z * 39.0 + uTime * 3.1 + broad * 4.6
  );
  float fine = 0.5 + 0.5 * sin(
    (uv.x * 173.0 + uv.y * 131.0 - z * 97.0) * uDetailFrequencyScale
      - uTime * 6.4 + medium * 2.8
  );
  float signedDetail = (broad * 0.84 + medium * 0.46 + fine * 0.22) / 1.52 * 2.0 - 1.0;
  float response = 1.0 - exp(-uDeltaTime * 8.0);
  state.b = mix(state.b * exp(-uDeltaTime * 0.62), signedDetail, response * uDetail * 0.72);

  // Keep extinction and emission openings spatially aligned: negative detail
  // marks a void, so density/reaction are eroded in the same cells that the
  // beauty pass closes instead of in the bright cells.
  float breakup = max(0.0, -state.b) * uDetail;
  state.r *= exp(-uDeltaTime * breakup * 2.5);
  state.a *= exp(-uDeltaTime * breakup * 8.0);
  state.g = clamp(state.g * (1.0 + state.b * uDetail * uDeltaTime * 0.065), 0.0, 1.25);
  outColor = state;
}
`;

export const CURL_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uVelocity;
uniform float uTime;
uniform float uDetail;
uniform float uVorticity;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec3 position = vec3(atlasLocalUv(), (float(layer) + 0.5) / float(uSliceCount));
  vec3 cell = vec3(1.0 / vec2(uTileSize), 1.0 / float(uSliceCount));
  vec3 leftVelocity = sampleVolume(uVelocity, position - vec3(cell.x, 0.0, 0.0)).xyz;
  vec3 rightVelocity = sampleVolume(uVelocity, position + vec3(cell.x, 0.0, 0.0)).xyz;
  vec3 bottomVelocity = sampleVolume(uVelocity, position - vec3(0.0, cell.y, 0.0)).xyz;
  vec3 topVelocity = sampleVolume(uVelocity, position + vec3(0.0, cell.y, 0.0)).xyz;
  vec3 backVelocity = sampleVolume(uVelocity, position - vec3(0.0, 0.0, cell.z)).xyz;
  vec3 frontVelocity = sampleVolume(uVelocity, position + vec3(0.0, 0.0, cell.z)).xyz;

  vec3 curl = vec3(
    0.5 * ((topVelocity.z - bottomVelocity.z) / cell.y - (frontVelocity.y - backVelocity.y) / cell.z),
    0.5 * ((frontVelocity.x - backVelocity.x) / cell.z - (rightVelocity.z - leftVelocity.z) / cell.x),
    0.5 * ((rightVelocity.y - leftVelocity.y) / cell.x - (topVelocity.x - bottomVelocity.x) / cell.y)
  );
  vec3 turbulence = vec3(
    sin(dot(position, vec3(23.0, 31.0, 17.0)) + uTime * 2.1),
    sin(dot(position, vec3(-19.0, 27.0, 37.0)) - uTime * 1.7),
    sin(dot(position, vec3(41.0, -13.0, 29.0)) + uTime * 1.3)
  );
  curl += turbulence * uDetail * 0.085 * min(uVorticity, 2.5);
  outColor = vec4(curl, length(curl));
}
`;

export const FORCE_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform sampler2D uState;
uniform float uDeltaTime;
uniform float uBuoyancy;
uniform float uVorticity;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec3 position = vec3(atlasLocalUv(), (float(layer) + 0.5) / float(uSliceCount));
  vec3 cell = vec3(1.0 / vec2(uTileSize), 1.0 / float(uSliceCount));
  float leftMagnitude = sampleVolume(uCurl, position - vec3(cell.x, 0.0, 0.0)).a;
  float rightMagnitude = sampleVolume(uCurl, position + vec3(cell.x, 0.0, 0.0)).a;
  float bottomMagnitude = sampleVolume(uCurl, position - vec3(0.0, cell.y, 0.0)).a;
  float topMagnitude = sampleVolume(uCurl, position + vec3(0.0, cell.y, 0.0)).a;
  float backMagnitude = sampleVolume(uCurl, position - vec3(0.0, 0.0, cell.z)).a;
  float frontMagnitude = sampleVolume(uCurl, position + vec3(0.0, 0.0, cell.z)).a;
  vec3 magnitudeGradient = 0.5 * vec3(
    (rightMagnitude - leftMagnitude) / cell.x,
    (topMagnitude - bottomMagnitude) / cell.y,
    (frontMagnitude - backMagnitude) / cell.z
  );
  vec3 gradientDirection = normalize(magnitudeGradient + vec3(0.00001));
  vec3 curl = sampleVolume(uCurl, position).xyz;
  vec3 confinement = cross(gradientDirection, curl) * (0.032 * uVorticity);

  vec3 velocity = sampleVolume(uVelocity, position).xyz;
  vec4 state = sampleVolume(uState, position);
  float buoyantDensity = max(0.0, state.g - 0.075 * state.r);
  velocity *= exp(-0.085 * uDeltaTime);
  velocity += uDeltaTime * (confinement + vec3(0.0, 0.72 * uBuoyancy * buoyantDensity, 0.0));
  float speed = length(velocity);
  if (speed > 1.2) velocity *= 1.2 / speed;
  outColor = vec4(velocity, 1.0);
}
`;

export const DIVERGENCE_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uVelocity;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec3 position = vec3(atlasLocalUv(), (float(layer) + 0.5) / float(uSliceCount));
  vec3 cell = vec3(1.0 / vec2(uTileSize), 1.0 / float(uSliceCount));
  vec3 leftVelocity = sampleVolume(uVelocity, position - vec3(cell.x, 0.0, 0.0)).xyz;
  vec3 rightVelocity = sampleVolume(uVelocity, position + vec3(cell.x, 0.0, 0.0)).xyz;
  vec3 bottomVelocity = sampleVolume(uVelocity, position - vec3(0.0, cell.y, 0.0)).xyz;
  vec3 topVelocity = sampleVolume(uVelocity, position + vec3(0.0, cell.y, 0.0)).xyz;
  vec3 backVelocity = sampleVolume(uVelocity, position - vec3(0.0, 0.0, cell.z)).xyz;
  vec3 frontVelocity = sampleVolume(uVelocity, position + vec3(0.0, 0.0, cell.z)).xyz;
  float divergence = 0.5 * (
    (rightVelocity.x - leftVelocity.x) / cell.x
    + (topVelocity.y - bottomVelocity.y) / cell.y
    + (frontVelocity.z - backVelocity.z) / cell.z
  );
  outColor = vec4(divergence, 0.0, 0.0, 1.0);
}
`;

export const JACOBI_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uPressure;
uniform sampler2D uDivergence;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec3 position = vec3(atlasLocalUv(), (float(layer) + 0.5) / float(uSliceCount));
  vec3 inverseCellSquared = vec3(
    float(uTileSize.x * uTileSize.x),
    float(uTileSize.y * uTileSize.y),
    float(uSliceCount * uSliceCount)
  );
  vec3 cell = 1.0 / sqrt(inverseCellSquared);
  float leftPressure = sampleVolume(uPressure, position - vec3(cell.x, 0.0, 0.0)).r;
  float rightPressure = sampleVolume(uPressure, position + vec3(cell.x, 0.0, 0.0)).r;
  float bottomPressure = sampleVolume(uPressure, position - vec3(0.0, cell.y, 0.0)).r;
  float topPressure = sampleVolume(uPressure, position + vec3(0.0, cell.y, 0.0)).r;
  float backPressure = sampleVolume(uPressure, position - vec3(0.0, 0.0, cell.z)).r;
  float frontPressure = sampleVolume(uPressure, position + vec3(0.0, 0.0, cell.z)).r;
  float divergence = sampleVolume(uDivergence, position).r;
  float pressure = (
    inverseCellSquared.x * (leftPressure + rightPressure)
    + inverseCellSquared.y * (bottomPressure + topPressure)
    + inverseCellSquared.z * (backPressure + frontPressure)
    - divergence
  ) / (2.0 * (inverseCellSquared.x + inverseCellSquared.y + inverseCellSquared.z));
  outColor = vec4(pressure, 0.0, 0.0, 1.0);
}
`;

export const PROJECT_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
uniform sampler2D uVelocity;
uniform sampler2D uPressure;

void main() {
  int layer = atlasLayer();
  if (layer >= uSliceCount) {
    outColor = vec4(0.0);
    return;
  }
  vec3 position = vec3(atlasLocalUv(), (float(layer) + 0.5) / float(uSliceCount));
  vec3 cell = vec3(1.0 / vec2(uTileSize), 1.0 / float(uSliceCount));
  float leftPressure = sampleVolume(uPressure, position - vec3(cell.x, 0.0, 0.0)).r;
  float rightPressure = sampleVolume(uPressure, position + vec3(cell.x, 0.0, 0.0)).r;
  float bottomPressure = sampleVolume(uPressure, position - vec3(0.0, cell.y, 0.0)).r;
  float topPressure = sampleVolume(uPressure, position + vec3(0.0, cell.y, 0.0)).r;
  float backPressure = sampleVolume(uPressure, position - vec3(0.0, 0.0, cell.z)).r;
  float frontPressure = sampleVolume(uPressure, position + vec3(0.0, 0.0, cell.z)).r;
  vec3 gradient = 0.5 * vec3(
    (rightPressure - leftPressure) / cell.x,
    (topPressure - bottomPressure) / cell.y,
    (frontPressure - backPressure) / cell.z
  );
  vec3 velocity = sampleVolume(uVelocity, position).xyz - gradient;

  if (position.x <= cell.x || position.x >= 1.0 - cell.x) velocity.x = 0.0;
  if (position.y <= cell.y) velocity.y = max(0.0, velocity.y);
  if (position.z <= cell.z || position.z >= 1.0 - cell.z) velocity.z = 0.0;
  float speed = length(velocity);
  if (speed > 1.2) velocity *= 1.2 / speed;
  outColor = vec4(velocity, 1.0);
}
`;

export const RENDER_FRAGMENT = `#version 300 es
${ATLAS_COMMON}
in vec2 vUv;
uniform sampler2D uState;
uniform float uDensityGain;
uniform float uDetail;
uniform float uExposure;
uniform float uCinematic;
uniform float uCameraYaw;
uniform float uCameraPitch;
uniform float uCameraDistance;
uniform float uViewZoom;
uniform float uTime;
uniform vec2 uViewportSize;
uniform int uRaySteps;
uniform int uSourceMode;
uniform float uDetailFrequencyScale;

vec3 blackbody(float kelvin) {
  float temperature = clamp(kelvin, 800.0, 12000.0) / 100.0;
  float red;
  float green;
  float blue;
  if (temperature <= 66.0) {
    red = 1.0;
    green = clamp(0.39008158 * log(max(temperature, 1.0)) - 0.63184144, 0.0, 1.0);
    blue = temperature <= 19.0
      ? 0.0
      : clamp(0.54320679 * log(max(temperature - 10.0, 1.0)) - 1.19625409, 0.0, 1.0);
  } else {
    red = clamp(1.29293619 * pow(temperature - 60.0, -0.13320476), 0.0, 1.0);
    green = clamp(1.12989086 * pow(temperature - 60.0, -0.07551485), 0.0, 1.0);
    blue = 1.0;
  }
  return vec3(red, green, blue);
}

vec3 acesToneMap(vec3 color) {
  return clamp(
    (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14),
    0.0,
    1.0
  );
}

vec3 linearToSrgb(vec3 color) {
  vec3 low = color * 12.92;
  vec3 high = 1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
  return mix(low, high, step(vec3(0.0031308), color));
}

vec2 intersectUnitBox(vec3 origin, vec3 direction) {
  vec3 inverseDirection = mix(
    vec3(-1.0),
    vec3(1.0),
    step(vec3(0.0), direction)
  ) / max(abs(direction), vec3(0.0001));
  vec3 nearValues = (vec3(0.0) - origin) * inverseDirection;
  vec3 farValues = (vec3(1.0) - origin) * inverseDirection;
  vec3 minimumValues = min(nearValues, farValues);
  vec3 maximumValues = max(nearValues, farValues);
  return vec2(
    max(max(minimumValues.x, minimumValues.y), minimumValues.z),
    min(min(maximumValues.x, maximumValues.y), maximumValues.z)
  );
}

// Cinematic-quality detail is evaluated in volume space, so it remains attached
// to the plume instead of swimming with pixels as the camera moves.
float cinematicMicrostructure(vec3 position, float time) {
  vec3 finePosition = position * uDetailFrequencyScale;
  float verticalWarp = sin(position.y * 17.0 - time * 2.9 + position.z * 11.0)
    + 0.55 * sin(position.y * 31.0 + time * 3.8 - position.x * 13.0);
  float ribbons = sin((finePosition.x + verticalWarp * 0.034) * 39.0 + finePosition.z * 9.0)
    * sin((finePosition.z - verticalWarp * 0.028) * 31.0 - finePosition.x * 7.0);
  float billow = sin(
    position.y * 27.0 - time * 5.1
      + sin(position.x * 17.0) * 1.6
      + cos(position.z * 15.0) * 1.4
  );
  return clamp(ribbons * 0.62 + billow * 0.38, -1.0, 1.0);
}

void main() {
  vec2 screen = vUv;
  vec2 centered = screen - 0.5;
  float viewportAspect = uViewportSize.x / max(uViewportSize.y, 1.0);
  float domainAspect = float(uTileSize.x) / float(uTileSize.y);
  float dolly = uCameraDistance / 4.9;
  vec2 domainScale = vec2(viewportAspect / domainAspect, 1.0) * dolly * 1.04 / uViewZoom;
  float cosinePitch = cos(uCameraPitch);
  vec3 rayDirection = normalize(vec3(
    sin(uCameraYaw) * cosinePitch,
    -sin(uCameraPitch),
    cos(uCameraYaw) * cosinePitch
  ));
  vec3 cameraRight = normalize(cross(vec3(0.0, 1.0, 0.0), rayDirection));
  vec3 cameraUp = normalize(cross(rayDirection, cameraRight));
  // Move the target down while zooming so the emitter remains in frame.
  vec3 target = vec3(0.5, 0.468 / uViewZoom, 0.5);
  vec3 rayOrigin = target
    + cameraRight * centered.x * domainScale.x
    + cameraUp * centered.y * domainScale.y
    - rayDirection * 2.0;
  vec2 hit = intersectUnitBox(rayOrigin, rayDirection);
  float rayStart = max(hit.x, 0.0);
  float rayLength = max(hit.y - rayStart, 0.0);
  float transmittance = 1.0;
  vec3 accumulated = vec3(0.0);
  float heatIntegral = 0.0;
  float previousReaction = 0.0;
  // Stable stratification avoids 24 Hz shimmer when no temporal accumulation
  // buffer is present.
  float jitter = hash4(vec4(gl_FragCoord.xy, 3.7, 0.0));

  for (int index = 0; index < 512; ++index) {
    if (index >= uRaySteps || transmittance < 0.006 || rayLength <= 0.0) break;
    float sampleDistance = rayStart
      + (float(index) + jitter) / float(uRaySteps) * rayLength;
    vec3 samplePosition = rayOrigin + rayDirection * sampleDistance;
    vec4 state = sampleVolume(uState, samplePosition);
    float microstructure = 0.0;
    if (uCinematic > 0.5) {
      microstructure = cinematicMicrostructure(samplePosition, uTime);
    }
    float edgeBand = smoothstep(0.012, 0.080, state.a)
      * (1.0 - smoothstep(1.05, 1.35, state.a));
    float cinematicDetail = mix(state.b, microstructure, edgeBand * 0.78);
    float detailSignal = clamp(
      mix(state.b, cinematicDetail, uCinematic),
      -1.0,
      1.0
    );
    float detailAmount = clamp(uDetail * 0.78, 0.0, 0.88);
    float density = max(0.0, state.r * (1.0 + detailSignal * detailAmount * 0.26));
    float temperature = clamp(
      state.g + detailSignal * detailAmount * 0.03,
      0.0,
      1.2
    );
    float reaction = max(
      0.0,
      state.a * (1.0 + detailSignal * detailAmount * 0.42)
    );
    float reactionSurface = smoothstep(
      0.006,
      0.075,
      abs(reaction - previousReaction)
    );
    if (uCinematic > 0.5) {
      vec3 cell = vec3(
        1.0 / float(uTileSize.x),
        1.0 / float(uTileSize.y),
        1.0 / float(uSliceCount)
      );
      vec3 reactionForward = vec3(
        sampleVolume(uState, samplePosition + vec3(cell.x, 0.0, 0.0)).a,
        sampleVolume(uState, samplePosition + vec3(0.0, cell.y, 0.0)).a,
        sampleVolume(uState, samplePosition + vec3(0.0, 0.0, cell.z)).a
      );
      float fieldSurface = smoothstep(
        0.010,
        0.095,
        length(vec3(reaction) - reactionForward)
      );
      reactionSurface = max(reactionSurface, fieldSurface);
    }
    previousReaction = reaction;

    float hotGas = smoothstep(0.16, 0.54, temperature);
    float cellularOpening = mix(
      1.0,
      smoothstep(-0.38, 0.16, detailSignal),
      detailAmount * mix(1.0, 0.30, uCinematic)
    );
    float reactionEdge = smoothstep(0.065, 0.34, reaction);
    float frontStrength = max(reaction - 0.035, 0.0) * (0.48 + 2.15 * reactionEdge);
    float flameFront = pow(
      clamp(frontStrength * reactionEdge * hotGas * cellularOpening, 0.0, 1.0),
      1.62
    );
    flameFront *= mix(0.045, 1.62, reactionSurface);
    float coolSmoke = density
      * (1.0 - smoothstep(0.12, 0.42, temperature))
      * (1.0 - smoothstep(0.035, 0.20, reaction));
    float opticalDensity = coolSmoke * 0.58 + flameFront * mix(0.48, 0.11, uCinematic);
    float stepLength = rayLength / float(uRaySteps);
    float alpha = 1.0 - exp(-opticalDensity * uDensityGain * 4.8 * stepLength);

    float heat = clamp(temperature / 1.1, 0.0, 1.0);
    float kelvin = 980.0 + 2100.0 * pow(heat, 0.86);
    vec3 physicalColor = blackbody(kelvin);
    float paleCore = smoothstep(0.50, 0.94, temperature)
      * smoothstep(0.16, 0.58, reaction)
      * mix(0.28, 1.0, reactionSurface);
    float orangeRim = reactionEdge * (1.0 - paleCore);
    vec3 flameColor = mix(
      vec3(1.0, 0.030, 0.001),
      vec3(1.0, 0.205, 0.006),
      smoothstep(0.035, 0.34, flameFront)
    );
    flameColor = mix(flameColor, physicalColor, smoothstep(0.38, 0.82, flameFront) * 0.26);
    flameColor = mix(flameColor, vec3(1.0, 0.86, 0.46), paleCore * 0.42);
    flameColor += vec3(0.12, 0.016, 0.001) * orangeRim;
    vec3 emission = flameColor * flameFront * pow(heat, 1.18)
      * (0.72 + 1.42 * heat);
    vec3 smokeColor = mix(
      vec3(0.012, 0.014, 0.018),
      vec3(0.080, 0.030, 0.008),
      heatIntegral
    );
    vec3 smokeScattering = smokeColor * coolSmoke
      * (0.10 + 0.55 * heatIntegral);
    accumulated += transmittance * (
      emission * stepLength * mix(3.6, 3.8, uCinematic)
      + alpha * smokeScattering
    );
    transmittance *= 1.0 - alpha;
    heatIntegral = min(1.0, heatIntegral + flameFront * stepLength * 3.0);
  }

  float horizon = smoothstep(0.0, 1.0, screen.y);
  vec3 background = mix(vec3(0.012, 0.011, 0.010), vec3(0.0014, 0.0026, 0.0052), horizon);
  float groundGlow = exp(-abs(screen.y - 0.075) * 17.0) * min(1.0, dot(accumulated, vec3(0.2126, 0.7152, 0.0722)));
  background += vec3(0.13, 0.023, 0.004) * groundGlow * 0.20;
  float vignette = 1.0 - 0.30 * dot(centered, centered);
  vec3 color = (accumulated + transmittance * background) * vignette * uExposure;
  color = acesToneMap(color);
  float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
  color = clamp(mix(vec3(luminance), color, 1.04), 0.0, 1.0);
  color = linearToSrgb(color);
  outColor = vec4(color, 1.0);
}
`;

export const SPARK_VERTEX = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec4 aPositionAge;
layout(location = 1) in vec4 aVelocitySeed;

out float vSparkIntensity;

uniform ivec2 uTileSize;
uniform int uSourceMode;
uniform float uCameraYaw;
uniform float uCameraPitch;
uniform float uCameraDistance;
uniform float uViewZoom;
uniform vec2 uViewportSize;
uniform float uPointSize;

float hash11(float value) {
  return fract(sin(value * 127.1 + 17.17) * 43758.5453123);
}

void main() {
  vec3 position = aPositionAge.xyz;
  float age = aPositionAge.w;
  float selector = hash11(aVelocitySeed.w * 911.0 + 4.3);
  float threshold = uSourceMode == 1 ? 0.986 : (uSourceMode == 2 ? 0.998 : 0.994);
  bool visible = selector > threshold && age > 0.10 && age < 50.0
    && all(greaterThanEqual(position, vec3(0.0)))
    && all(lessThanEqual(position, vec3(1.0)));
  if (!visible) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    vSparkIntensity = 0.0;
    return;
  }

  float viewportAspect = uViewportSize.x / max(uViewportSize.y, 1.0);
  float domainAspect = float(uTileSize.x) / float(uTileSize.y);
  float dolly = uCameraDistance / 4.9;
  vec2 domainScale = vec2(viewportAspect / domainAspect, 1.0)
    * dolly * 1.04 / uViewZoom;
  float cosinePitch = cos(uCameraPitch);
  vec3 rayDirection = normalize(vec3(
    sin(uCameraYaw) * cosinePitch,
    -sin(uCameraPitch),
    cos(uCameraYaw) * cosinePitch
  ));
  vec3 cameraRight = normalize(cross(vec3(0.0, 1.0, 0.0), rayDirection));
  vec3 cameraUp = normalize(cross(rayDirection, cameraRight));
  vec3 target = vec3(0.5, 0.468 / uViewZoom, 0.5);
  vec3 relative = position - target;
  vec2 domain = vec2(
    dot(relative, cameraRight),
    dot(relative, cameraUp)
  );
  vec2 screenCentered = domain / domainScale;
  gl_Position = vec4(screenCentered * 2.0, 0.0, 1.0);
  gl_PointSize = uPointSize * mix(0.72, 1.35, selector);
  vSparkIntensity = (selector - threshold) / max(1.0 - threshold, 0.0001)
    * smoothstep(0.08, 0.26, position.y);
}
`;

export const SPARK_FRAGMENT = `#version 300 es
precision highp float;
in float vSparkIntensity;
layout(location = 0) out vec4 outColor;

void main() {
  vec2 point = gl_PointCoord * 2.0 - 1.0;
  float radiusSquared = dot(point, point);
  if (radiusSquared >= 1.0) discard;
  float glow = exp(-radiusSquared * 3.8) * vSparkIntensity;
  outColor = vec4(vec3(1.0, 0.24, 0.025) * glow * 1.8, glow);
}
`;
