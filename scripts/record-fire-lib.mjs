import path from 'node:path';

export const FRAME_RATE = 60;
export const GIB = 1024 ** 3;
export const MIB = 1024 ** 2;
export const MAX_GRID_MEMORY_BYTES = 32 * GIB;
export const MAX_STORAGE_BINDING_BYTES = 2 * GIB;
export const MAX_CLI_DURATION_SECONDS = 60 * 60;
const FRAME_ALIGNMENT_EPSILON = 1e-7;

export const FIELD_IDS = Object.freeze([
  'beauty',
  'temperature',
  'reaction',
  'fuel-oxygen',
  'products',
  'vapor-soot',
  'liquid-velocity',
  'divergence',
]);

export const SCENE_IDS = Object.freeze([
  'methane',
  'large-fire',
  'inferno',
  'rich',
  'top-jet',
  'base-spray',
]);

export const OUTPUT_PRESETS = Object.freeze({
  hd: Object.freeze({ width: 1920, height: 1080, bitrate: 24_000_000, codecString: 'avc1.64002a', level: 42 }),
  qhd: Object.freeze({ width: 2560, height: 1440, bitrate: 40_000_000, codecString: 'avc1.640033', level: 51 }),
  uhd: Object.freeze({ width: 3840, height: 2160, bitrate: 65_000_000, codecString: 'avc1.640034', level: 52 }),
});

export const SOLVER_TIERS = Object.freeze({
  hd: Object.freeze({ dimensions: [128, 192, 128], particleCount: 2048, minimumGuardBytes: 512 * MIB }),
  qhd: Object.freeze({ dimensions: [160, 240, 160], particleCount: 3072, minimumGuardBytes: GIB }),
  uhd: Object.freeze({ dimensions: [192, 288, 192], particleCount: 4096, minimumGuardBytes: 2 * GIB }),
});

export const GRID_PRESETS = Object.freeze({
  auto: null,
  maximum: Object.freeze([80, 120, 80]),
  ultra: Object.freeze([96, 144, 96]),
  experimental: Object.freeze([128, 192, 128]),
  extreme: Object.freeze([160, 160, 160]),
  expert: Object.freeze([256, 256, 256]),
  workstation: Object.freeze([320, 320, 320]),
  studio: Object.freeze([384, 384, 384]),
  reference: Object.freeze([508, 508, 508]),
});

const GRID_MINIMUM_GUARDS = Object.freeze({
  expert: 2 * GIB,
  workstation: 4 * GIB,
  studio: 8 * GIB,
  reference: 16 * GIB,
});

const FIELD_ALIASES = Object.freeze({
  'fuel-o2': 'fuel-oxygen',
  'fuel/oxygen': 'fuel-oxygen',
  'co2': 'products',
  'liquid': 'liquid-velocity',
  'vapor': 'vapor-soot',
});

const OUTPUT_ALIASES = Object.freeze({
  hd: 'hd',
  '1080': 'hd',
  '1080p': 'hd',
  qhd: 'qhd',
  '1440': 'qhd',
  '1440p': 'qhd',
  uhd: 'uhd',
  '2160': 'uhd',
  '2160p': 'uhd',
  '4k': 'uhd',
});

export function parseDuration(value) {
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value <= 0) throw new Error('Duration must be positive.');
    if (value < 1 / FRAME_RATE || value > MAX_CLI_DURATION_SECONDS) {
      throw new Error('Duration must be between one 60 fps frame and one hour.');
    }
    return requireFrameAlignedDuration(value);
  }
  if (typeof value !== 'string' || value.trim() === '') throw new Error('Duration is required.');
  const match = /^([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)?$/i.exec(value.trim());
  if (!match) throw new Error(`Invalid duration "${value}". Use values such as 30s, 5m, or 1h.`);
  const amount = Number(match[1]);
  const unit = (match[2] ?? 's').toLowerCase();
  const seconds = amount * (unit === 'ms' ? 0.001 : unit === 'm' ? 60 : unit === 'h' ? 3600 : 1);
  if (!Number.isFinite(seconds) || seconds < 1 / FRAME_RATE || seconds > MAX_CLI_DURATION_SECONDS) {
    throw new Error('Duration must be between one 60 fps frame and one hour.');
  }
  return requireFrameAlignedDuration(seconds);
}

function requireFrameAlignedDuration(seconds) {
  const frames = seconds * FRAME_RATE;
  if (Math.abs(frames - Math.round(frames)) > FRAME_ALIGNMENT_EPSILON) {
    throw new Error(`Duration ${seconds}s is not an exact 60 fps frame boundary.`);
  }
  return Math.round(frames) / FRAME_RATE;
}

export function parseByteSize(value) {
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error('Memory budget must be a positive whole byte count.');
    return value;
  }
  const match = /^([0-9]+(?:\.[0-9]+)?)\s*(b|kib|mib|gib|kb|mb|gb)?$/i.exec(String(value).trim());
  if (!match) throw new Error(`Invalid byte size "${value}". Use values such as 8GiB or 32GiB.`);
  const amount = Number(match[1]);
  const unit = (match[2] ?? 'b').toLowerCase();
  const multiplier = unit === 'kib' ? 1024
    : unit === 'mib' ? MIB
      : unit === 'gib' ? GIB
        : unit === 'kb' ? 1000
          : unit === 'mb' ? 1000 ** 2
            : unit === 'gb' ? 1000 ** 3
              : 1;
  const bytes = Math.floor(amount * multiplier);
  if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error('Memory budget is outside the supported numeric range.');
  return bytes;
}

export function parseGrid(value) {
  const normalized = String(value).trim().toLowerCase();
  if (Object.hasOwn(GRID_PRESETS, normalized)) {
    return { id: normalized, dimensions: GRID_PRESETS[normalized] };
  }
  const match = /^(\d+)\s*[x×,]\s*(\d+)\s*[x×,]\s*(\d+)$/.exec(normalized);
  if (!match) {
    throw new Error(`Unknown grid "${value}". Use auto, maximum, ultra, experimental, extreme, expert, workstation, studio, reference, or XxYxZ.`);
  }
  const dimensions = match.slice(1).map(Number);
  validateGridDimensions(dimensions);
  const estimate = estimateGridMemory(dimensions, 4096);
  if (estimate.largestStorageBufferBytes > MAX_STORAGE_BINDING_BYTES) {
    throw new Error(
      `Dense ${dimensions.join('x')} needs a ${formatBytes(estimate.largestStorageBufferBytes)} vector field, above this engine's 2 GiB negotiated WebGPU ceiling.`,
    );
  }
  if (estimate.totalBytes > MAX_GRID_MEMORY_BYTES) {
    throw new Error(
      `Dense ${dimensions.join('x')} needs about ${formatBytes(estimate.totalBytes)}, above the 32 GiB allocation guard.`,
    );
  }
  return { id: 'custom', dimensions: Object.freeze(dimensions) };
}

export function validateGridDimensions(dimensions) {
  if (!Array.isArray(dimensions) || dimensions.length !== 3) throw new Error('Grid must contain exactly three axes.');
  if (dimensions.some((axis) => !Number.isSafeInteger(axis) || axis < 16 || axis > 1024 || axis % 4 !== 0)) {
    throw new Error('Every grid axis must be a whole multiple of four from 16 through 1024; no axis is rounded.');
  }
}

export function estimateGridMemory(dimensions, particleCount = 0) {
  validateGridDimensions(dimensions);
  const cellCount = dimensions[0] * dimensions[1] * dimensions[2];
  const gridBytes = cellCount * (cellCount >= 64 * 1024 * 1024 ? 108 : 116);
  const particleBytes = particleCount * 80;
  const fixedBytes = 64 * 64 * 64 * 4;
  return Object.freeze({
    cellCount,
    gridBytes,
    particleBytes,
    fixedBytes,
    totalBytes: gridBytes + particleBytes + fixedBytes,
    largestStorageBufferBytes: cellCount * 16,
  });
}

export function resolveGridDimensions(solverTier, requestedDimensions) {
  const baseline = SOLVER_TIERS[solverTier]?.dimensions;
  if (!baseline) throw new Error(`Unknown solver tier: ${solverTier}.`);
  if (!requestedDimensions) return Object.freeze([...baseline]);
  return Object.freeze(baseline.map((axis, index) => Math.max(axis, requestedDimensions[index])));
}

export function estimateOutputBytes(outputPreset, durationSeconds) {
  const preset = OUTPUT_PRESETS[outputPreset];
  if (!preset) throw new Error(`Unknown output preset: ${outputPreset}.`);
  return Math.ceil((preset.bitrate * durationSeconds) / 8);
}

export function parseArguments(argv) {
  const options = {
    scene: 'inferno',
    fields: ['beauty'],
    durationSeconds: 30,
    outputPreset: 'hd',
    solverTier: null,
    grid: parseGrid('auto'),
    gridMemoryBudgetBytes: null,
    opticalDetailTarget: 1024,
    warmupSeconds: 1,
    camera: { yaw: -0.34, pitch: 0.03, distance: 4.9 },
    output: null,
    outputDir: 'renders',
    browser: null,
    headless: false,
    background: false,
    dryRun: false,
    confirmHeavy: false,
    overwrite: false,
    verify: 'auto',
    help: false,
    listFields: false,
  };
  let fieldWasSet = false;
  let solverWasSet = false;

  for (let index = 0; index < argv.length; index += 1) {
    const raw = argv[index];
    if (!raw.startsWith('--') && raw !== '-h') {
      throw new Error(
        `Unexpected positional argument: ${raw}. On Windows PowerShell, use .\\scripts\\record-fire.cmd so the npm shim cannot consume recorder options.`,
      );
    }
    const equals = raw.indexOf('=');
    const flag = equals >= 0 ? raw.slice(0, equals) : raw;
    const inline = equals >= 0 ? raw.slice(equals + 1) : null;
    const takeValue = () => {
      if (inline !== null) return inline;
      const next = argv[index + 1];
      if (!next || next.startsWith('--')) throw new Error(`${flag} requires a value.`);
      index += 1;
      return next;
    };

    switch (flag) {
      case '--help':
      case '-h':
        options.help = true;
        break;
      case '--list-fields':
        options.listFields = true;
        break;
      case '--scene': {
        const scene = takeValue().toLowerCase();
        if (!SCENE_IDS.includes(scene)) throw new Error(`Unknown scene "${scene}".`);
        options.scene = scene;
        break;
      }
      case '--field':
      case '--fields': {
        const requested = takeValue().toLowerCase().split(',').map((item) => item.trim()).filter(Boolean);
        const resolved = requested.includes('all')
          ? [...FIELD_IDS]
          : requested.map((field) => FIELD_ALIASES[field] ?? field);
        for (const field of resolved) {
          if (!FIELD_IDS.includes(field)) throw new Error(`Unknown field view "${field}".`);
        }
        if (!fieldWasSet) options.fields = [];
        options.fields.push(...resolved);
        options.fields = [...new Set(options.fields)];
        fieldWasSet = true;
        break;
      }
      case '--duration':
        options.durationSeconds = parseDuration(takeValue());
        break;
      case '--hour':
        options.durationSeconds = 3600;
        break;
      case '--resolution':
      case '--preset': {
        const requested = takeValue().toLowerCase();
        const outputPreset = OUTPUT_ALIASES[requested];
        if (!outputPreset) throw new Error(`Unknown output resolution "${requested}".`);
        options.outputPreset = outputPreset;
        break;
      }
      case '--solver-tier':
      case '--solver': {
        const solverTier = takeValue().toLowerCase();
        if (!Object.hasOwn(SOLVER_TIERS, solverTier)) throw new Error(`Unknown solver tier "${solverTier}".`);
        options.solverTier = solverTier;
        solverWasSet = true;
        break;
      }
      case '--grid':
        options.grid = parseGrid(takeValue());
        break;
      case '--memory-budget':
        options.gridMemoryBudgetBytes = parseByteSize(takeValue());
        break;
      case '--optical-detail': {
        const target = Number(takeValue());
        if (![0, 256, 512, 1024].includes(target)) throw new Error('--optical-detail must be 0, 256, 512, or 1024.');
        options.opticalDetailTarget = target;
        break;
      }
      case '--warmup': {
        const warmup = takeValue();
        options.warmupSeconds = warmup === '0' ? 0 : parseDuration(warmup);
        if (options.warmupSeconds > 60) throw new Error('--warmup cannot exceed 60 seconds.');
        break;
      }
      case '--camera-yaw':
        options.camera.yaw = parseFinite(takeValue(), '--camera-yaw');
        break;
      case '--camera-pitch':
        options.camera.pitch = parseFinite(takeValue(), '--camera-pitch');
        break;
      case '--camera-distance':
        options.camera.distance = parseFinite(takeValue(), '--camera-distance');
        break;
      case '--output':
        options.output = takeValue();
        break;
      case '--output-dir':
        options.outputDir = takeValue();
        break;
      case '--browser':
        options.browser = takeValue();
        break;
      case '--headless':
        options.headless = true;
        break;
      case '--background':
        options.background = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--confirm-heavy':
        options.confirmHeavy = true;
        break;
      case '--overwrite':
        options.overwrite = true;
        break;
      case '--verify': {
        const verify = takeValue().toLowerCase();
        if (!['auto', 'metadata', 'sample', 'full'].includes(verify)) {
          throw new Error('--verify must be auto, metadata, sample, or full.');
        }
        options.verify = verify;
        break;
      }
      case '--quality': {
        const quality = takeValue().toLowerCase();
        if (quality !== 'highest') throw new Error('The CLI quality shortcut currently supports "highest" only.');
        options.outputPreset = 'uhd';
        options.solverTier = 'uhd';
        options.grid = parseGrid('reference');
        options.gridMemoryBudgetBytes = 32 * GIB;
        options.opticalDetailTarget = 1024;
        solverWasSet = true;
        break;
      }
      case '--test-views':
        options.fields = ['beauty', 'temperature', 'reaction', 'vapor-soot'];
        options.durationSeconds = 30;
        options.outputPreset = 'hd';
        options.solverTier = 'hd';
        options.grid = parseGrid('auto');
        options.gridMemoryBudgetBytes = 2 * GIB;
        fieldWasSet = true;
        solverWasSet = true;
        break;
      default:
        throw new Error(`Unknown option: ${flag}. Run .\\scripts\\record-fire.cmd --help.`);
    }
  }

  if (!solverWasSet) options.solverTier = options.outputPreset;
  if (options.camera.pitch < -0.7 || options.camera.pitch > 0.7) throw new Error('Camera pitch must be between -0.7 and 0.7.');
  if (options.camera.distance < 2.5 || options.camera.distance > 8) throw new Error('Camera distance must be between 2.5 and 8.');
  const effectiveDimensions = resolveGridDimensions(options.solverTier, options.grid.dimensions);
  const memoryEstimate = estimateGridMemory(effectiveDimensions, SOLVER_TIERS[options.solverTier].particleCount);
  if (memoryEstimate.largestStorageBufferBytes > MAX_STORAGE_BINDING_BYTES) {
    throw new Error(`Effective dense grid requires a ${formatBytes(memoryEstimate.largestStorageBufferBytes)} vector field, above 2 GiB.`);
  }
  if (memoryEstimate.totalBytes > MAX_GRID_MEMORY_BYTES) {
    throw new Error(`Effective dense grid requires ${formatBytes(memoryEstimate.totalBytes)}, above the 32 GiB guard.`);
  }
  if (options.gridMemoryBudgetBytes === null) {
    const gridGuard = GRID_MINIMUM_GUARDS[options.grid.id] ?? 0;
    options.gridMemoryBudgetBytes = Math.max(
      SOLVER_TIERS[options.solverTier].minimumGuardBytes,
      gridGuard,
      chooseMemoryGuard(memoryEstimate.totalBytes + outputPresentationBytes(options.outputPreset)),
    );
  }
  if (options.gridMemoryBudgetBytes < memoryEstimate.totalBytes + outputPresentationBytes(options.outputPreset)) {
    throw new Error(
      `The exact solver and output need about ${formatBytes(memoryEstimate.totalBytes + outputPresentationBytes(options.outputPreset))}, above --memory-budget ${formatBytes(options.gridMemoryBudgetBytes)}.`,
    );
  }
  if (options.gridMemoryBudgetBytes > MAX_GRID_MEMORY_BYTES) throw new Error('Memory budget cannot exceed 32 GiB.');
  return Object.freeze({
    ...options,
    fields: Object.freeze([...options.fields]),
    grid: Object.freeze({ ...options.grid }),
    camera: Object.freeze({ ...options.camera }),
    effectiveDimensions,
    memoryEstimate,
  });
}

export function buildCaptureConfig(options, field, jobId) {
  if (!FIELD_IDS.includes(field)) throw new Error(`Unknown field view "${field}".`);
  return Object.freeze({
    schemaVersion: 1,
    jobId,
    method: 'firex',
    scene: options.scene,
    fieldView: field,
    outputPreset: options.outputPreset,
    solverTier: options.solverTier,
    durationSeconds: options.durationSeconds,
    warmupSeconds: options.warmupSeconds,
    gridDimensions: options.grid.dimensions,
    gridMemoryBudgetBytes: options.gridMemoryBudgetBytes,
    opticalDetailTarget: options.opticalDetailTarget,
    camera: options.camera,
  });
}

export function resolveOutputPath(options, field, cwd = process.cwd()) {
  if (options.output) {
    const requested = path.resolve(cwd, options.output);
    if (options.fields.length === 1) return requested;
    const extension = path.extname(requested) || '.mp4';
    const base = requested.slice(0, requested.length - path.extname(requested).length);
    return `${base}-${field}${extension}`;
  }
  const durationLabel = formatDurationSlug(options.durationSeconds);
  const fileName = `fire-replica-firex-${options.scene}-${field}-${options.outputPreset}-60fps-${durationLabel}.mp4`;
  return path.resolve(cwd, options.outputDir, fileName);
}

export function isHeavyJob(options) {
  const largestAxis = Math.max(...options.effectiveDimensions);
  return options.durationSeconds >= 10 * 60
    || largestAxis >= 384
    || (options.outputPreset === 'uhd' && options.durationSeconds >= 60);
}

export function createDryRunPlan(options, cwd = process.cwd()) {
  return options.fields.map((field) => ({
    field,
    output: resolveOutputPath(options, field, cwd),
    frames: Math.round(options.durationSeconds * FRAME_RATE),
    durationSeconds: options.durationSeconds,
    resolution: `${OUTPUT_PRESETS[options.outputPreset].width}x${OUTPUT_PRESETS[options.outputPreset].height}`,
    solverTier: options.solverTier,
    grid: options.effectiveDimensions.join('x'),
    gridCells: options.memoryEstimate.cellCount,
    estimatedGpuBytes: options.memoryEstimate.totalBytes,
    estimatedVideoBytes: estimateOutputBytes(options.outputPreset, options.durationSeconds),
  }));
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return String(bytes);
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = Math.abs(bytes);
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  const signed = bytes < 0 ? -value : value;
  return `${signed.toFixed(index === 0 ? 0 : signed >= 10 ? 1 : 2)} ${units[index]}`;
}

export function formatDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(seconds % 1 === 0 ? 0 : 2)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(seconds % 60 === 0 ? 0 : 2)}m`;
  return `${(seconds / 3600).toFixed(seconds % 3600 === 0 ? 0 : 2)}h`;
}

function formatDurationSlug(seconds) {
  if (seconds % 3600 === 0) return `${seconds / 3600}h`;
  if (seconds % 60 === 0) return `${seconds / 60}m`;
  return `${String(seconds).replace('.', 'p')}s`;
}

function parseFinite(value, flag) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${flag} requires a finite number.`);
  return number;
}

function outputPresentationBytes(outputPreset) {
  const preset = OUTPUT_PRESETS[outputPreset];
  const bloomWidth = Math.ceil(preset.width / 4);
  const bloomHeight = Math.ceil(preset.height / 4);
  return preset.width * preset.height * 8
    + bloomWidth * bloomHeight * 8
    + preset.width * preset.height * 4;
}

function chooseMemoryGuard(requiredBytes) {
  for (const guard of [512 * MIB, GIB, 2 * GIB, 4 * GIB, 8 * GIB, 16 * GIB, 24 * GIB, 32 * GIB]) {
    if (requiredBytes <= guard) return guard;
  }
  return 32 * GIB;
}
