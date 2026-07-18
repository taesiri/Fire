export const FIREX_F16_PRESSURE_WGSL = /* wgsl */ `
enable f16;

// Dedicated 32-byte pressure uniform. Both members are 16-byte aligned, so
// this has the same layout in WGSL's uniform address space and a Float32Array.
// grid.xyz contains cell dimensions; domain.xyz contains physical dimensions.
struct Parameters {
  grid: vec4f,
  domain: vec4f,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> divergenceField: array<f32>;
@group(0) @binding(2) var<storage, read> pressureIn: array<f16>;
@group(0) @binding(3) var<storage, read_write> pressureOut: array<f16>;
@group(0) @binding(4) var<storage, read> flowIn: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> flowOut: array<vec4f>;

const PRESSURE_GROUP_SIZE = vec3u(8u, 8u, 4u);
const PRESSURE_TILE_SIZE = vec3u(10u, 10u, 6u);
const PRESSURE_TILE_CELL_COUNT = 600u;

// Pressure is stored as f16 to halve global-memory traffic and residency. All
// stencil arithmetic remains f32; only the load/store boundaries convert.
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

fn loadPressure(cell: vec3i) -> f32 {
  // Side walls and the floor clamp to the nearest pressure cell (Neumann).
  // A sample beyond the open top sees ambient zero pressure (Dirichlet).
  if (cell.y >= gridDimensions().y) {
    return 0.0;
  }
  return f32(pressureIn[cellIndex(cell)]);
}

fn pressureTileIndex(cell: vec3u) -> u32 {
  return cell.x + PRESSURE_TILE_SIZE.x * (cell.y + PRESSURE_TILE_SIZE.y * cell.z);
}

@compute @workgroup_size(8, 8, 4)
fn solvePressureF16Tiled(
  @builtin(global_invocation_id) invocation: vec3u,
  @builtin(local_invocation_id) localInvocation: vec3u,
  @builtin(local_invocation_index) localIndex: u32,
  @builtin(workgroup_id) workgroup: vec3u,
) {
  // Cooperatively fetch the 8x8x4 interior plus a one-cell halo. Every lane
  // reaches the barrier, including lanes outside a partial edge workgroup.
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
  let id = cellIndex(cell);
  if (cell.y == gridDimensions().y - 1) {
    // The top cell itself is the fixed ambient-pressure boundary.
    pressureOut[id] = f16(0.0);
    return;
  }

  let metric = cellMetric();
  let wx = metric.x * metric.x;
  let wy = metric.y * metric.y;
  let wz = metric.z * metric.z;
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
  let pressure = (neighbors - divergenceField[id]) / (2.0 * (wx + wy + wz));
  pressureOut[id] = f16(pressure);
}

fn pressureGradient(cell: vec3i, metric: vec3f) -> vec3f {
  let center = loadPressure(cell);
  return vec3f(
    (loadPressure(cell + vec3i(1, 0, 0)) - center) * metric.x,
    (loadPressure(cell + vec3i(0, 1, 0)) - center) * metric.y,
    (loadPressure(cell + vec3i(0, 0, 1)) - center) * metric.z
  );
}

@compute @workgroup_size(4, 4, 4)
fn projectVelocityF16(@builtin(global_invocation_id) invocation: vec3u) {
  let dimensionsU = vec3u(params.grid.xyz);
  if (any(invocation >= dimensionsU)) {
    return;
  }

  let cell = vec3i(invocation);
  let metric = cellMetric();
  var flow = flowIn[cellIndex(cell)];
  flow = vec4f(flow.xyz - pressureGradient(cell, metric), flow.w);

  if (cell.y == gridDimensions().y - 1) {
    // Zero-gradient velocity outflow duplicates the projected interior value.
    // Preserve the top cell's scalar component exactly as the f32 path does.
    let below = cell - vec3i(0, 1, 0);
    let belowFlow = flowIn[cellIndex(below)];
    flow = vec4f(belowFlow.xyz - pressureGradient(below, metric), flow.w);
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
`;
