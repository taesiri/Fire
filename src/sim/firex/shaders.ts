export const FIREX_COMPUTE_WGSL = /* wgsl */ `
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
@group(0) @binding(1) var<storage, read> flowIn: array<vec4f>;
@group(0) @binding(2) var<storage, read> speciesIn: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> flowOut: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> speciesOut: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> divergenceField: array<f32>;
@group(0) @binding(6) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(8) var<storage, read> liquidField: array<u32>;

const PRESSURE_GROUP_SIZE = vec3u(8u, 8u, 4u);
const PRESSURE_TILE_SIZE = vec3u(10u, 10u, 6u);
const PRESSURE_TILE_CELL_COUNT = 600u;
var<workgroup> pressureTile: array<f32, 600>;

fn gridDimensions() -> vec3i {
  return vec3i(params.grid.xyz);
}

fn domainExtent() -> vec3f {
  return max(params.domain.xyz, vec3f(0.001));
}

fn cellMetric() -> vec3f {
  return params.grid.xyz / domainExtent();
}

fn clampCell(cell: vec3i) -> vec3i {
  return clamp(cell, vec3i(0), gridDimensions() - vec3i(1));
}

fn cellIndex(cell: vec3i) -> u32 {
  let bounded = clampCell(cell);
  let dimensions = gridDimensions();
  return u32(bounded.x + dimensions.x * (bounded.y + dimensions.y * bounded.z));
}

fn loadFlow(cell: vec3i) -> vec4f {
  return flowIn[cellIndex(cell)];
}

fn loadSpecies(cell: vec3i) -> vec4f {
  return speciesIn[cellIndex(cell)];
}

fn trilinearFlow(position: vec3f) -> vec4f {
  let dimensions = vec3f(params.grid.xyz);
  let samplePosition = clamp(position - vec3f(0.5), vec3f(0.0), dimensions - vec3f(1.001));
  let base = vec3i(floor(samplePosition));
  let fraction = fract(samplePosition);
  let c000 = loadFlow(base);
  let c100 = loadFlow(base + vec3i(1, 0, 0));
  let c010 = loadFlow(base + vec3i(0, 1, 0));
  let c110 = loadFlow(base + vec3i(1, 1, 0));
  let c001 = loadFlow(base + vec3i(0, 0, 1));
  let c101 = loadFlow(base + vec3i(1, 0, 1));
  let c011 = loadFlow(base + vec3i(0, 1, 1));
  let c111 = loadFlow(base + vec3i(1, 1, 1));
  let z0 = mix(mix(c000, c100, fraction.x), mix(c010, c110, fraction.x), fraction.y);
  let z1 = mix(mix(c001, c101, fraction.x), mix(c011, c111, fraction.x), fraction.y);
  return mix(z0, z1, fraction.z);
}

fn trilinearSpecies(position: vec3f) -> vec4f {
  let dimensions = vec3f(params.grid.xyz);
  let samplePosition = clamp(position - vec3f(0.5), vec3f(0.0), dimensions - vec3f(1.001));
  let base = vec3i(floor(samplePosition));
  let fraction = fract(samplePosition);
  let c000 = loadSpecies(base);
  let c100 = loadSpecies(base + vec3i(1, 0, 0));
  let c010 = loadSpecies(base + vec3i(0, 1, 0));
  let c110 = loadSpecies(base + vec3i(1, 1, 0));
  let c001 = loadSpecies(base + vec3i(0, 0, 1));
  let c101 = loadSpecies(base + vec3i(1, 0, 1));
  let c011 = loadSpecies(base + vec3i(0, 1, 1));
  let c111 = loadSpecies(base + vec3i(1, 1, 1));
  let z0 = mix(mix(c000, c100, fraction.x), mix(c010, c110, fraction.x), fraction.y);
  let z1 = mix(mix(c001, c101, fraction.x), mix(c011, c111, fraction.x), fraction.y);
  return mix(z0, z1, fraction.z);
}

fn depositedLiquid(cell: vec3i) -> f32 {
  return f32(liquidField[cellIndex(cell)]) * (1.0 / 4096.0);
}

fn emitter(normalizedPosition: vec3f) -> f32 {
  let time = params.time.y;
  let extent = domainExtent();
  let position = normalizedPosition * extent;
  let center = vec3f(
    extent.x * 0.5 + sin(time * 1.73) * 0.013,
    0.075,
    extent.z * 0.5 + cos(time * 1.37) * 0.011
  );
  let offset = position - center;
  let baseRadius = sqrt(0.02024);
  let radiusX = min(baseRadius * clamp(params.tuning.x, 0.35, 3.2), extent.x * 0.42);
  let radiusZ = min(baseRadius * clamp(params.tuning.y, 0.35, 3.2), extent.z * 0.42);
  let sourceThickness = clamp(params.tuning.z, 0.35, 2.5);
  let radial = offset.x * offset.x / (radiusX * radiusX)
    + offset.z * offset.z / (radiusZ * radiusZ);
  let vertical = offset.y * offset.y / (0.0022 * sourceThickness * sourceThickness);
  let cellular = 0.76 + 0.24 * sin(position.x * 49.0 + time * 5.2)
    * sin(position.z * 43.0 - time * 4.4);
  let cinematicLobes = mix(
    0.48,
    1.18,
    smoothstep(
      -0.42,
      0.48,
      sin(offset.x / max(radiusX, 0.001) * 6.2 + time * 1.7)
        * sin(offset.z / max(radiusZ, 0.001) * 5.4 - time * 1.3)
    )
  );
  let sourceTexture = mix(cellular, cellular * cinematicLobes, clamp(params.mixing.z, 0.0, 1.0));
  let pulse = 0.84 + 0.16 * sin(time * 3.7 + position.x * 27.0 + position.z * 21.0);
  return exp(-(radial + vertical)) * sourceTexture * pulse;
}

@compute @workgroup_size(256)
fn initializeFields(
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(num_workgroups) workgroupCount: vec3u,
) {
  let groupIndex = workgroup.x
    + workgroupCount.x * (workgroup.y + workgroupCount.y * workgroup.z);
  let id = groupIndex * 256u + localIndex;
  if (id >= arrayLength(&flowOut)) {
    return;
  }
  flowOut[id] = vec4f(0.0, 0.0, 0.0, params.water.w);
  speciesOut[id] = vec4f(0.0, max(params.chemistry.y, 0.0), 0.0, 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn simulate(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensionsU = vec3u(params.grid.xyz);
  if (any(invocation >= dimensionsU)) {
    return;
  }
  let cell = vec3i(invocation);
  let id = cellIndex(cell);
  let dimensions = vec3f(params.grid.xyz);
  let positionInCells = vec3f(invocation) + vec3f(0.5);
  let normalizedPosition = positionInCells / dimensions;
  let dt = params.time.x;

  let originalFlow = loadFlow(cell);
  let metric = cellMetric();
  let backtrace = positionInCells - originalFlow.xyz * dt * metric;
  var flow = trilinearFlow(backtrace);
  var species = max(trilinearSpecies(backtrace), vec4f(0.0));

  let left = loadFlow(cell + vec3i(-1, 0, 0));
  let right = loadFlow(cell + vec3i(1, 0, 0));
  let bottom = loadFlow(cell + vec3i(0, -1, 0));
  let top = loadFlow(cell + vec3i(0, 1, 0));
  let back = loadFlow(cell + vec3i(0, 0, -1));
  let front = loadFlow(cell + vec3i(0, 0, 1));

  let velocityAverage = (left.xyz + right.xyz + bottom.xyz + top.xyz + back.xyz + front.xyz) / 6.0;
  flow = vec4f(mix(flow.xyz, velocityAverage, clamp(dt * 0.32, 0.0, 0.12)), flow.w);
  let temperatureAverage = (left.w + right.w + bottom.w + top.w + back.w + front.w) / 6.0;
  flow.w = mix(flow.w, temperatureAverage, clamp(dt * 0.18, 0.0, 0.08));

  let speciesAverage = (
    loadSpecies(cell + vec3i(-1, 0, 0)) + loadSpecies(cell + vec3i(1, 0, 0))
    + loadSpecies(cell + vec3i(0, -1, 0)) + loadSpecies(cell + vec3i(0, 1, 0))
    + loadSpecies(cell + vec3i(0, 0, -1)) + loadSpecies(cell + vec3i(0, 0, 1))
  ) / 6.0;
  species = mix(species, speciesAverage, clamp(dt * 0.12, 0.0, 0.06));

  let source = emitter(normalizedPosition);
  let firePower = clamp(params.tuning.w, 0.1, 4.0);
  let powerRoot = sqrt(firePower);
  let thermalPower = min(powerRoot, 1.0 + 0.22 * (powerRoot - 1.0));
  species.x = max(species.x, source * params.chemistry.x * firePower);
  flow.w = max(flow.w, params.water.w + source * 1180.0 * thermalPower);
  let sourceSwirl = vec3f(
    sin(params.time.y * 2.3 + normalizedPosition.z * 22.0),
    0.0,
    cos(params.time.y * 1.9 + normalizedPosition.x * 25.0)
  );
  flow = vec4f(flow.xyz + sourceSwirl * (source * dt * 0.11 * params.chemistry.w * powerRoot), flow.w);
  flow.y += source * dt * 0.52 * clamp(params.fire.x, 0.0, 4.0) * powerRoot;

  let ambientBoundary = cell.x == 0 || cell.x == gridDimensions().x - 1
    || cell.z == 0 || cell.z == gridDimensions().z - 1 || cell.y == gridDimensions().y - 1;
  if (ambientBoundary) {
    species.y = max(species.y, params.chemistry.y);
    flow.w = mix(flow.w, params.water.w, 0.035);
  }

  // Transfer liquid cooling before chemistry so source cells can actually be
  // extinguished instead of burning first and only cooling afterward.
  let water = clamp(depositedLiquid(cell) * 3.4, 0.0, 5.0);
  let evaporation = water * smoothstep(335.0, 1040.0, flow.w) * dt * 13.5;
  flow.w = max(params.water.w, flow.w - evaporation * 2080.0 - water * dt * 260.0);
  species.w = clamp(species.w + evaporation * 2.15, 0.0, 3.0);

  // Large burners become oxygen-limited unless ambient air is entrained. Keep
  // that mixing concentrated at oxygen-deficient fuel interfaces so a broad
  // burner retains a flame sheet instead of reacting as one solid volume.
  let entrainment = clamp(params.mixing.x, 0.0, 3.0);
  let ambientOxygen = max(params.chemistry.y, 0.001);
  let oxygenDeficit = clamp((ambientOxygen - species.y) / ambientOxygen, 0.0, 1.0);
  let fuelCore = smoothstep(0.30, 0.78, species.x);
  let interfaceWeight = oxygenDeficit * mix(1.0, 0.18, fuelCore);
  let entrainmentMix = clamp(
    dt * entrainment * (
      0.12 + 0.28 * length(flow.xyz) + 0.22 * smoothstep(400.0, 1500.0, flow.w)
    ) * interfaceWeight,
    0.0,
    0.08
  );
  species.y = mix(species.y, ambientOxygen, entrainmentMix);

  let fuelBefore = max(species.x, 0.0);
  let oxygenBefore = max(species.y, 0.0);
  let availableFuel = min(fuelBefore, oxygenBefore * 0.25);
  let safeFuel = max(fuelBefore, 0.00001);
  let safeOxygen = max(oxygenBefore, 0.00001);
  let activation = exp(-15000.0 / max(flow.w, params.water.w));
  let arrhenius = clamp(
    2.5e6 * activation * pow(safeFuel, -0.30) * pow(safeOxygen, 1.30),
    0.0,
    80.0
  );
  let ignition = smoothstep(690.0, 930.0, flow.w);
  let requestedBurn = availableFuel * (1.0 - exp(-arrhenius * ignition * dt)) * exp(-water * 18.0);
  let burn = clamp(requestedBurn, 0.0, availableFuel);
  let richness = clamp(1.0 - oxygenBefore / max(fuelBefore * 4.0, 0.00001), 0.0, 1.0);
  species.x = max(0.0, fuelBefore - burn);
  species.y = max(0.0, oxygenBefore - burn * 4.0);
  species.z = clamp(species.z + burn * (1.0 + richness * 1.8), 0.0, 3.0);
  species.w = clamp(species.w + burn * 0.62, 0.0, 3.0);
  flow.w += burn * (2150.0 + 1450.0 * params.chemistry.z);
  // This is an instantaneous, localized heat-release diagnostic.  It is kept
  // separate from temperature so hot products and steam never become flames.
  pressureOut[id] = clamp(burn / max(dt, 0.0001) * 0.22, 0.0, 4.0);

  let curl = 0.5 * vec3f(
    (top.z - bottom.z) * metric.y - (front.y - back.y) * metric.z,
    (front.x - back.x) * metric.z - (right.z - left.z) * metric.x,
    (right.y - left.y) * metric.x - (top.x - bottom.x) * metric.y
  );
  let temperatureGradient = 0.5 * vec3f(
    (right.w - left.w) * metric.x,
    (top.w - bottom.w) * metric.y,
    (front.w - back.w) * metric.z
  );
  let gradientDirection = normalize(temperatureGradient + vec3f(0.00001));
  flow = vec4f(
    flow.xyz + cross(gradientDirection, curl) * (0.00135 * params.chemistry.w * dt),
    flow.w
  );

  let hotPlume = smoothstep(520.0, 1650.0, flow.w);
  let heightEnvelope = smoothstep(0.045, 0.15, normalizedPosition.y)
    * (1.0 - smoothstep(0.80, 1.0, normalizedPosition.y));
  let lateralTurbulence = vec3f(
    sin(normalizedPosition.y * 31.0 + normalizedPosition.z * 19.0 + params.time.y * 3.1)
      + 0.45 * sin(normalizedPosition.y * 67.0 - params.time.y * 5.3),
    0.12 * sin(normalizedPosition.x * 29.0 + normalizedPosition.z * 23.0 - params.time.y * 4.1),
    cos(normalizedPosition.y * 27.0 - normalizedPosition.x * 21.0 + params.time.y * 2.7)
      + 0.45 * cos(normalizedPosition.y * 59.0 + params.time.y * 4.7)
  );
  let turbulenceStrength = (0.075 + 0.16 * params.chemistry.w) * hotPlume * heightEnvelope;
  flow = vec4f(flow.xyz + lateralTurbulence * (turbulenceStrength * dt), flow.w);

  let buoyancy = clamp((flow.w - params.water.w) / 1500.0, -0.1, 1.2);
  flow.y += buoyancy * dt * 0.72 * clamp(params.fire.y, 0.0, 4.0);

  // Liquid exists only where particles deposited it.  Avoiding an analytic
  // cone makes the coupling and the rendered diagnostic independently testable.
  species.z *= exp(-water * dt * 4.8);
  let extent = domainExtent();
  let nozzleOrigin = vec3f(0.035, 0.50, 0.5) * extent;
  let sprayTarget = vec3f(0.52, params.controls.y, 0.5) * extent;
  let sprayMomentum = normalize(sprayTarget - nozzleOrigin);
  flow = vec4f(flow.xyz + sprayMomentum * water * dt * 0.82, flow.w);

  let persistence = clamp(params.fire.z, 0.4, 3.0);
  let radiation = 5.67e-8 * (pow(params.water.w, 4.0) - pow(max(flow.w, 0.0), 4.0))
    / (1005.0 * persistence);
  flow.w = max(params.water.w, flow.w + dt * radiation);
  species.z *= exp(-dt * 0.075 / persistence);
  species.w *= exp(-dt * 0.14 / persistence);

  // Open/outflow top: remove transported scalars and relax temperature instead
  // of repeatedly sampling a clamped cap cell.
  let outflow = smoothstep(0.84, 0.995, normalizedPosition.y);
  species.x *= 1.0 - outflow;
  let outflowDecay = exp(-outflow * dt * 15.0 / persistence);
  species.z *= outflowDecay;
  species.w *= outflowDecay;
  flow.w = mix(flow.w, params.water.w, outflow * 0.16 / persistence);
  pressureOut[id] *= 1.0 - outflow;
  if (cell.y == gridDimensions().y - 1) {
    species.x = 0.0;
    species.z = 0.0;
    species.w = 0.0;
    flow.w = params.water.w;
    pressureOut[id] = 0.0;
  }

  if (cell.x == 0 || cell.x == gridDimensions().x - 1) {
    flow.x = 0.0;
  }
  if (cell.z == 0 || cell.z == gridDimensions().z - 1) {
    flow.z = 0.0;
  }
  if (cell.y == 0) {
    // Solid floor: no normal flux. The source lives above this boundary cell.
    flow.y = 0.0;
  }
  if (cell.y == gridDimensions().y - 1) {
    // Open top permits outflow but rejects numerical inflow.
    flow.y = max(0.0, flow.y);
  }
  flow = vec4f(clamp(flow.xyz, vec3f(-1.4), vec3f(1.4)), flow.w);
  flow.w = clamp(flow.w, params.water.w, 2900.0);
  species = clamp(species, vec4f(0.0), vec4f(3.0));
  flowOut[id] = flow;
  speciesOut[id] = species;
}

@compute @workgroup_size(4, 4, 4)
fn computeDivergence(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensionsU = vec3u(params.grid.xyz);
  if (any(invocation >= dimensionsU)) {
    return;
  }
  let cell = vec3i(invocation);
  let dimensions = cellMetric();
  let center = loadFlow(cell).xyz;
  let left = loadFlow(cell + vec3i(-1, 0, 0)).x;
  let bottom = loadFlow(cell + vec3i(0, -1, 0)).y;
  let back = loadFlow(cell + vec3i(0, 0, -1)).z;
  // Backward divergence is paired with the forward pressure gradient below.
  // Their composition is the same nearest-neighbor Laplacian solved by Jacobi;
  // centered D/G operators would instead produce an uncancellable stride-2 mode.
  divergenceField[cellIndex(cell)] = (
    (center.x - left) * dimensions.x
    + (center.y - bottom) * dimensions.y
    + (center.z - back) * dimensions.z
  );
}

fn loadPressure(cell: vec3i) -> f32 {
  // The side walls and floor use clamped (Neumann) pressure. The open top uses
  // ambient Dirichlet pressure so upward flow can leave the domain cleanly.
  if (cell.y >= gridDimensions().y) {
    return 0.0;
  }
  return pressureIn[cellIndex(cell)];
}

@compute @workgroup_size(4, 4, 4)
fn solvePressureGlobal(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensionsU = vec3u(params.grid.xyz);
  if (any(invocation >= dimensionsU)) {
    return;
  }
  let cell = vec3i(invocation);
  if (cell.y == gridDimensions().y - 1) {
    pressureOut[cellIndex(cell)] = 0.0;
    return;
  }
  let dimensions = cellMetric();
  let wx = dimensions.x * dimensions.x;
  let wy = dimensions.y * dimensions.y;
  let wz = dimensions.z * dimensions.z;
  let neighbors = wx * (
      loadPressure(cell + vec3i(-1, 0, 0))
      + loadPressure(cell + vec3i(1, 0, 0))
    )
    + wy * (
      loadPressure(cell + vec3i(0, -1, 0))
      + loadPressure(cell + vec3i(0, 1, 0))
    )
    + wz * (
      loadPressure(cell + vec3i(0, 0, -1))
      + loadPressure(cell + vec3i(0, 0, 1))
    );
  pressureOut[cellIndex(cell)] = (neighbors - divergenceField[cellIndex(cell)]) / (2.0 * (wx + wy + wz));
}

fn pressureTileIndex(cell: vec3u) -> u32 {
  return cell.x + PRESSURE_TILE_SIZE.x * (cell.y + PRESSURE_TILE_SIZE.y * cell.z);
}

@compute @workgroup_size(8, 8, 4)
fn solvePressureTiled(
  @builtin(global_invocation_id) invocation: vec3u,
  @builtin(local_invocation_id) localInvocation: vec3u,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(workgroup_id) workgroup: vec3u,
) {
  var tileIndex = localIndex;
  while (tileIndex < PRESSURE_TILE_CELL_COUNT) {
    let tileX = tileIndex % PRESSURE_TILE_SIZE.x;
    let tileYZ = tileIndex / PRESSURE_TILE_SIZE.x;
    let tileY = tileYZ % PRESSURE_TILE_SIZE.y;
    let tileZ = tileYZ / PRESSURE_TILE_SIZE.y;
    let globalCell = vec3i(workgroup * PRESSURE_GROUP_SIZE)
      + vec3i(vec3u(tileX, tileY, tileZ))
      - vec3i(1);
    pressureTile[tileIndex] = loadPressure(globalCell);
    tileIndex += 256u;
  }
  workgroupBarrier();

  let dimensionsU = vec3u(params.grid.xyz);
  if (any(invocation >= dimensionsU)) {
    return;
  }
  let cell = vec3i(invocation);
  if (cell.y == gridDimensions().y - 1) {
    // The top cell is the ambient-pressure boundary, not a pressure unknown.
    pressureOut[cellIndex(cell)] = 0.0;
    return;
  }
  let dimensions = cellMetric();
  let wx = dimensions.x * dimensions.x;
  let wy = dimensions.y * dimensions.y;
  let wz = dimensions.z * dimensions.z;
  let tileCell = localInvocation + vec3u(1);
  let neighbors = wx * (
      pressureTile[pressureTileIndex(tileCell - vec3u(1, 0, 0))]
      + pressureTile[pressureTileIndex(tileCell + vec3u(1, 0, 0))]
    )
    + wy * (
      pressureTile[pressureTileIndex(tileCell - vec3u(0, 1, 0))]
      + pressureTile[pressureTileIndex(tileCell + vec3u(0, 1, 0))]
    )
    + wz * (
      pressureTile[pressureTileIndex(tileCell - vec3u(0, 0, 1))]
      + pressureTile[pressureTileIndex(tileCell + vec3u(0, 0, 1))]
    );
  let pressure = (neighbors - divergenceField[cellIndex(cell)]) / (2.0 * (wx + wy + wz));
  pressureOut[cellIndex(cell)] = pressure;
}

fn pressureGradient(cell: vec3i, dimensions: vec3f) -> vec3f {
  let center = loadPressure(cell);
  return vec3f(
    (loadPressure(cell + vec3i(1, 0, 0)) - center) * dimensions.x,
    (loadPressure(cell + vec3i(0, 1, 0)) - center) * dimensions.y,
    (loadPressure(cell + vec3i(0, 0, 1)) - center) * dimensions.z
  );
}

@compute @workgroup_size(4, 4, 4)
fn projectVelocity(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensionsU = vec3u(params.grid.xyz);
  if (any(invocation >= dimensionsU)) {
    return;
  }
  let cell = vec3i(invocation);
  let dimensions = cellMetric();
  let gradient = pressureGradient(cell, dimensions);
  var flow = loadFlow(cell);
  flow = vec4f(flow.xyz - gradient, flow.w);
  if (cell.y == gridDimensions().y - 1) {
    // Zero-gradient velocity outflow: duplicate the projected interior value,
    // rather than leaving an artificial jump in the adjacent divergence cell.
    let below = cell - vec3i(0, 1, 0);
    let belowFlow = loadFlow(below);
    flow = vec4f(belowFlow.xyz - pressureGradient(below, dimensions), flow.w);
  }
  if (cell.x == 0 || cell.x == gridDimensions().x - 1) {
    flow.x = 0.0;
  }
  if (cell.z == 0 || cell.z == gridDimensions().z - 1) {
    flow.z = 0.0;
  }
  if (cell.y == 0) {
    flow.y = 0.0;
  }
  if (cell.y == gridDimensions().y - 1) {
    flow.y = max(0.0, flow.y);
  }
  flowOut[cellIndex(cell)] = flow;
}

@compute @workgroup_size(4, 4, 4)
fn copyFlow(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensionsU = vec3u(params.grid.xyz);
  if (any(invocation >= dimensionsU)) {
    return;
  }
  let cell = vec3i(invocation);
  flowOut[cellIndex(cell)] = loadFlow(cell);
}
`;

export const FIREX_PARTICLE_WGSL = /* wgsl */ `
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

struct Particle {
  positionAge: vec4f,
  velocityMass: vec4f,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> flowField: array<vec4f>;
@group(0) @binding(2) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(3) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(4) var<storage, read_write> densityData: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> liquidField: array<atomic<u32>>;

const PARTICLE_RADIUS: f32 = 0.060;
const LIQUID_FIXED_SCALE: f32 = 4096.0;

fn domainExtent() -> vec3f {
  return max(params.domain.xyz, vec3f(0.001));
}

fn particleCount() -> u32 {
  return u32(params.render.w);
}

fn hash11(value: f32) -> f32 {
  return fract(sin(value * 12.9898 + 1.317) * 43758.5453);
}

fn gridDimensions() -> vec3i {
  return vec3i(params.grid.xyz);
}

fn cellIndex(cell: vec3i) -> u32 {
  let dimensions = gridDimensions();
  let bounded = clamp(cell, vec3i(0), dimensions - vec3i(1));
  return u32(bounded.x + dimensions.x * (bounded.y + dimensions.y * bounded.z));
}

fn sampleFlow(position: vec3f) -> vec4f {
  let cell = vec3i(clamp(position, vec3f(0.0), vec3f(0.99999)) * params.grid.xyz);
  return flowField[cellIndex(cell)];
}

fn nozzleOrigin() -> vec3f {
  return vec3f(0.035, 0.50, 0.5);
}

fn nozzleDirection() -> vec3f {
  let nozzleTarget = vec3f(0.52, clamp(params.controls.y, 0.08, 0.82), 0.5);
  return normalize((nozzleTarget - nozzleOrigin()) * domainExtent());
}

fn spawnParticle(id: u32, cycle: f32) -> Particle {
  let seed = f32(id) * 3.771 + cycle * 19.173;
  let radialRandom = sqrt(hash11(seed + 0.17));
  let angularRandom = hash11(seed + 4.91) * 6.2831853;
  let speedRandom = hash11(seed + 9.37);
  let direction = nozzleDirection();
  let tangent = normalize(cross(direction, vec3f(0.0, 0.0, 1.0)));
  let bitangent = normalize(cross(direction, tangent));
  let diskDirection = tangent * cos(angularRandom) + bitangent * sin(angularRandom);
  let nozzleSpread = select(0.10, 0.72, params.controls.z >= 0.5);
  let cone = tan(clamp(params.water.y, 0.005, 0.78)) * radialRandom * nozzleSpread;
  let launchDirection = normalize(direction + diskDirection * cone);
  let originRadius = select(0.0028, 0.006, params.controls.z >= 0.5);
  let origin = nozzleOrigin() + diskDirection * radialRandom * originRadius / domainExtent();
  let speedVariation = select(0.05, 0.24, params.controls.z >= 0.5);
  let speed = (0.82 + params.water.x * 0.82) * mix(1.0 - speedVariation * 0.5, 1.0 + speedVariation * 0.5, speedRandom);
  var particle: Particle;
  particle.positionAge = vec4f(origin, 0.0);
  particle.velocityMass = vec4f(launchDirection * speed, 0.52 + params.water.x * 0.88);
  return particle;
}

@compute @workgroup_size(64)
fn computeParticleDensity(@builtin(global_invocation_id) invocation: vec3u) {
  let id = invocation.x;
  let count = particleCount();
  if (id >= count) {
    return;
  }
  let particle = particlesIn[id];
  if (particle.velocityMass.w <= 0.0001) {
    densityData[id] = vec4f(0.0);
    return;
  }

  var density = 0.0;
  var nearDensity = 0.0;
  let interactionRadius = select(0.038, PARTICLE_RADIUS, params.controls.z >= 0.5);
  for (var neighborId = 0u; neighborId < count; neighborId += 1u) {
    if (neighborId == id) {
      continue;
    }
    let neighbor = particlesIn[neighborId];
    if (neighbor.velocityMass.w <= 0.0001) {
      continue;
    }
    let distance = length((particle.positionAge.xyz - neighbor.positionAge.xyz) * domainExtent());
    if (distance > 0.00001 && distance < interactionRadius) {
      let q = 1.0 - distance / interactionRadius;
      density += q * q;
      nearDensity += q * q * q;
    }
  }

  let restDensity = 2.7 + params.water.x * 0.9;
  let pressure = max(density - restDensity, 0.0) * 0.055;
  let nearPressure = nearDensity * 0.065;
  densityData[id] = vec4f(density, nearDensity, pressure, nearPressure);
}

@compute @workgroup_size(64)
fn integrateParticles(@builtin(global_invocation_id) invocation: vec3u) {
  let id = invocation.x;
  let count = particleCount();
  if (id >= count) {
    return;
  }

  let dt = params.time.x;
  var particle = particlesIn[id];
  if (dt <= 0.0) {
    particlesOut[id] = particle;
    return;
  }

  if (particle.velocityMass.w <= 0.0001) {
    let wait = particle.positionAge.w + dt;
    if (params.water.z >= 0.5 && params.water.x > 0.001 && wait >= 0.0) {
      particlesOut[id] = spawnParticle(id, wait + f32(params.time.z));
    } else {
      particle.positionAge = vec4f(nozzleOrigin(), select(wait, -0.72, wait > 0.0));
      particle.velocityMass = vec4f(0.0);
      particlesOut[id] = particle;
    }
    return;
  }

  let lifetime = 0.88 + hash11(f32(id) * 2.31 + 7.0) * 0.52;
  let nextAge = particle.positionAge.w + dt;
  let outside = any(particle.positionAge.xyz < vec3f(-0.03)) || any(particle.positionAge.xyz > vec3f(1.03));
  if (nextAge >= lifetime || outside) {
    if (params.water.z >= 0.5 && params.water.x > 0.001) {
      particlesOut[id] = spawnParticle(id, nextAge + f32(params.time.z));
    } else {
      particle.positionAge = vec4f(nozzleOrigin(), -hash11(f32(id) * 8.17) * 0.72);
      particle.velocityMass = vec4f(0.0);
      particlesOut[id] = particle;
    }
    return;
  }

  let position = particle.positionAge.xyz;
  var velocity = particle.velocityMass.xyz;
  let particleDensity = densityData[id];
  var pressureAcceleration = vec3f(0.0);
  var viscosityAcceleration = vec3f(0.0);
  let interactionRadius = select(0.038, PARTICLE_RADIUS, params.controls.z >= 0.5);
  for (var neighborId = 0u; neighborId < count; neighborId += 1u) {
    if (neighborId == id) {
      continue;
    }
    let neighbor = particlesIn[neighborId];
    if (neighbor.velocityMass.w <= 0.0001) {
      continue;
    }
    let separation = (position - neighbor.positionAge.xyz) * domainExtent();
    let distance = length(separation);
    if (distance > 0.00001 && distance < interactionRadius) {
      let q = 1.0 - distance / interactionRadius;
      let direction = separation / distance;
      let neighborDensity = densityData[neighborId];
      let pressure = (particleDensity.z + neighborDensity.z) * q;
      let nearPressure = (particleDensity.w + neighborDensity.w) * q * q;
      pressureAcceleration += direction * (pressure + nearPressure);
      viscosityAcceleration += (neighbor.velocityMass.xyz - velocity) * (q * 0.018);
    }
  }

  let gas = sampleFlow(position);
  let heat = smoothstep(380.0, 1500.0, gas.w);
  var acceleration = vec3f(0.0, -0.58, 0.0);
  acceleration += (gas.xyz - velocity) * (0.30 + heat * 0.18);
  let pressureDispersion = select(0.055, 1.0, params.controls.z >= 0.5);
  let viscousCohesion = select(2.2, 1.0, params.controls.z >= 0.5);
  acceleration += pressureAcceleration * pressureDispersion + viscosityAcceleration * viscousCohesion;
  velocity += acceleration * dt;
  var nextPosition = position + velocity * dt / domainExtent();

  if (nextPosition.y < 0.012) {
    nextPosition.y = 0.012;
    velocity.y = abs(velocity.y) * 0.18;
    velocity = vec3f(velocity.x * 0.72, velocity.y, velocity.z * 0.72);
  }
  if (nextPosition.x < 0.008 || nextPosition.x > 0.992) {
    nextPosition.x = clamp(nextPosition.x, 0.008, 0.992);
    velocity.x *= -0.24;
  }
  if (nextPosition.z < 0.008 || nextPosition.z > 0.992) {
    nextPosition.z = clamp(nextPosition.z, 0.008, 0.992);
    velocity.z *= -0.24;
  }

  let remainingMass = particle.velocityMass.w * exp(-heat * dt * 0.34);
  particle.positionAge = vec4f(nextPosition, nextAge);
  particle.velocityMass = vec4f(clamp(velocity, vec3f(-2.0), vec3f(2.0)), remainingMass);
  particlesOut[id] = particle;
}

fn depositCorner(cell: vec3i, weight: f32, mass: f32) {
  let fixedMass = u32(clamp(weight * mass * LIQUID_FIXED_SCALE, 0.0, 65535.0));
  atomicAdd(&liquidField[cellIndex(cell)], fixedMass);
}

fn depositAtPosition(position: vec3f, mass: f32) {
  // Compact normalized 3x3x3 kernel.  It remains strictly particle-driven but
  // represents each droplet's finite SPH support instead of a single sparse
  // trilinear point, improving conservative contact with thin flame fronts.
  let voxelPosition = position * params.grid.xyz - vec3f(0.5);
  if (params.controls.z < 0.5) {
    // A laminar jet is a narrow coherent stream: use the compact trilinear
    // transfer footprint and a lower entrained-mist fraction.  The spray path
    // below represents atomized droplets with broad SPH support.
    let base = vec3i(floor(voxelPosition));
    let fraction = fract(voxelPosition);
    let inverse = vec3f(1.0) - fraction;
    // Only the atomized/entrained fraction exchanges heat with the gas grid;
    // the coherent liquid core remains represented by the rendered particles.
    let laminarMass = mass * 0.20;
    depositCorner(base, inverse.x * inverse.y * inverse.z, laminarMass);
    depositCorner(base + vec3i(1, 0, 0), fraction.x * inverse.y * inverse.z, laminarMass);
    depositCorner(base + vec3i(0, 1, 0), inverse.x * fraction.y * inverse.z, laminarMass);
    depositCorner(base + vec3i(1, 1, 0), fraction.x * fraction.y * inverse.z, laminarMass);
    depositCorner(base + vec3i(0, 0, 1), inverse.x * inverse.y * fraction.z, laminarMass);
    depositCorner(base + vec3i(1, 0, 1), fraction.x * inverse.y * fraction.z, laminarMass);
    depositCorner(base + vec3i(0, 1, 1), inverse.x * fraction.y * fraction.z, laminarMass);
    depositCorner(base + vec3i(1, 1, 1), fraction.x * fraction.y * fraction.z, laminarMass);
    return;
  }
  let center = vec3i(round(voxelPosition));
  var totalWeight = 0.0;
  for (var offsetZ = -1; offsetZ <= 1; offsetZ += 1) {
    for (var offsetY = -1; offsetY <= 1; offsetY += 1) {
      for (var offsetX = -1; offsetX <= 1; offsetX += 1) {
        let sampleCell = center + vec3i(offsetX, offsetY, offsetZ);
        let distance = length(voxelPosition - vec3f(sampleCell));
        let kernel = max(0.0, 1.0 - distance * 0.52);
        totalWeight += kernel * kernel;
      }
    }
  }
  for (var offsetZ = -1; offsetZ <= 1; offsetZ += 1) {
    for (var offsetY = -1; offsetY <= 1; offsetY += 1) {
      for (var offsetX = -1; offsetX <= 1; offsetX += 1) {
        let sampleCell = center + vec3i(offsetX, offsetY, offsetZ);
        let distance = length(voxelPosition - vec3f(sampleCell));
        let kernel = max(0.0, 1.0 - distance * 0.52);
        depositCorner(sampleCell, kernel * kernel / max(totalWeight, 0.0001), mass);
      }
    }
  }
}

@compute @workgroup_size(64)
fn depositParticles(@builtin(global_invocation_id) invocation: vec3u) {
  let id = invocation.x;
  if (id >= particleCount()) {
    return;
  }
  let particle = particlesOut[id];
  let position = particle.positionAge.xyz;
  let mass = particle.velocityMass.w;
  if (mass <= 0.0001 || any(position < vec3f(0.0)) || any(position >= vec3f(1.0))) {
    return;
  }

  let previous = particlesIn[id];
  let displacement = position - previous.positionAge.xyz;
  let canSweep = previous.velocityMass.w > 0.0001
    && particle.positionAge.w >= previous.positionAge.w
    && length(displacement * domainExtent()) < 0.20;
  if (!canSweep) {
    // New and recycled particles must not draw a transfer segment across the
    // domain from their prior lifetime.
    depositAtPosition(position, mass);
    return;
  }

  // Three conservative samples keep a 30 Hz droplet from tunnelling across a
  // thin flame front. Dividing the mass preserves total per-step transfer.
  const SWEEP_SAMPLES: u32 = 3u;
  for (var sampleIndex = 0u; sampleIndex < SWEEP_SAMPLES; sampleIndex += 1u) {
    let fraction = f32(sampleIndex) / f32(SWEEP_SAMPLES - 1u);
    depositAtPosition(previous.positionAge.xyz + displacement * fraction, mass / f32(SWEEP_SAMPLES));
  }
}
`;
