import './style.css';
import {
  OFFLINE_VIDEO_PRESETS,
  estimateOfflineVideoBytes,
  getOfflineVideoFrameCount,
  getOfflineVideoPreset,
  isOfflineVideoSupported,
  renderOfflineVideo,
  type OfflineVideoPreset,
  type OfflineVideoPresetId,
  type OfflineVideoProgress,
} from './export/offlineVideo';
import {
  FireXEngine,
  FIREX_OFFLINE_QUALITY,
  FIREX_QUALITY,
  resolveFireXOfflineDimensions,
} from './sim/firex/FireXEngine';
import {
  FIREX_GRID_MEMORY_GUARDS,
  FIREX_GRID_PRESETS,
  deriveFireXGridInfo,
  estimateFireXGridMemory,
  formatFireXBytes,
  formatFireXGridInfo,
  preflightFireXGrid,
  type FireXGridDimensions,
  type FireXGridPresetId,
  type FireXUnitSystem,
} from './sim/firex/gridConfiguration';
import { FIREX_FIELD_VIEWS, FIREX_SCENE_PRESETS } from './sim/firex/scenes';
import {
  estimateHorvathMemory,
  HORVATH_OFFLINE_QUALITY,
  HorvathEngine,
} from './sim/horvath/HorvathEngine';
import { QUALITY_LEVELS } from './sim/types';
import type { ControlDefinition, MethodId, QualityLevel, SimulationEngine } from './sim/types';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('Application root is missing.');

const methodCopy = {
  horvath: {
    year: 'SIGGRAPH 2009 · HORVATH + GEIGER',
    title: 'View-aligned refinement preview',
    subtitle: 'Realtime research preview of independent camera-facing fluid slices',
    paper: 'Directable, High-Resolution Simulation of Fire on the GPU',
    detail:
      'WebGL2 research preview of the paper’s refinement stage. It uses compact resident slabs and projected source fields, not the production-scale coarse PIC/FLIP driver.',
  },
  firex: {
    year: 'SIGGRAPH ASIA 2025 · WREDE ET AL.',
    title: 'Stoichiometric suppression preview',
    subtitle: 'Realtime research preview of coupled droplets and thermochemical fields',
    paper: 'Fire-X: Extinguishing Fire with Stoichiometric Heat Release',
    detail:
      'WebGPU research preview with bounded species transport, combustion, pressure projection and droplet coupling. Use the diagnostic views to inspect fields directly.',
  },
} as const;

interface ScenarioPreset {
  readonly label: string;
  readonly values: Readonly<Record<string, number | boolean>>;
  readonly expected: string;
}

const firexFieldViews = FIREX_FIELD_VIEWS;

const controls: Record<MethodId, ControlDefinition[]> = {
  horvath: [
    { id: 'sourceSize', label: 'Emitter size', group: 'source', min: 0.3, max: 3, step: 0.01, value: 2.1, unit: '×', description: 'Scales the actual particle source footprint.', resetOnCommit: true },
    { id: 'emission', label: 'Source intensity', group: 'source', min: 0.1, max: 3, step: 0.01, value: 1.2 },
    { id: 'flameHeight', label: 'Flame reach', group: 'source', min: 0.4, max: 2.5, step: 0.01, value: 1.55, unit: '×', description: 'Changes particle lifetime and plume reach.', resetOnCommit: true },
    { id: 'buoyancy', label: 'Thermal buoyancy', group: 'dynamics', min: 0, max: 4, step: 0.01, value: 1.2 },
    { id: 'vorticity', label: 'Vorticity strength', group: 'dynamics', min: 0, max: 4, step: 0.01, value: 2.4 },
    { id: 'cooling', label: 'Radiative cooling', group: 'dynamics', min: 0.05, max: 1.4, step: 0.01, value: 0.5 },
    { id: 'viewZoom', label: 'Camera framing zoom', group: 'appearance', min: 0.5, max: 3, step: 0.01, value: 1.1, unit: '×', description: 'Display framing only; the simulation box stays normalized.' },
    { id: 'density', label: 'Smoke density', group: 'appearance', min: 0.2, max: 3, step: 0.01, value: 1.2 },
    { id: 'detail', label: '4D detail breakup', group: 'appearance', min: 0, max: 1.5, step: 0.01, value: 0.82 },
    { id: 'exposure', label: 'Render exposure', group: 'appearance', min: 0.25, max: 3, step: 0.01, value: 1.08, unit: '×', description: 'Linear-light exposure before the final filmic curve.' },
  ],
  firex: [
    { id: 'burnerSize', label: 'Burner width', group: 'source', min: 0.35, max: 3.2, step: 0.01, value: 3.1, unit: '×', description: 'Scales the fuel source along X.' },
    { id: 'burnerDepth', label: 'Burner depth', group: 'source', min: 0.35, max: 3.2, step: 0.01, value: 2.7, unit: '×', description: 'Scales the fuel source along Z.' },
    { id: 'sourceThickness', label: 'Fuel-layer thickness', group: 'source', min: 0.35, max: 2.5, step: 0.01, value: 1.35, unit: '×', description: 'Controls the vertical thickness of injected fuel.' },
    { id: 'fuelRate', label: 'Injected fuel fraction', group: 'source', min: 0.05, max: 1, step: 0.01, value: 0.3, description: 'Sets the normalized fuel fraction injected into the source region.' },
    { id: 'firePower', label: 'Fuel-feed scale', group: 'source', min: 0.1, max: 4, step: 0.01, value: 2.5, unit: '×', description: 'Controls injected fuel mass independently of concentration.' },
    { id: 'sourceLift', label: 'Burner jet momentum', group: 'source', min: 0, max: 4, step: 0.01, value: 1.6, unit: '×', description: 'Controls upward momentum at the burner.' },
    { id: 'oxygenRate', label: 'Ambient O₂ fraction', group: 'dynamics', min: 0.05, max: 1, step: 0.01, value: 1 },
    { id: 'airEntrainment', label: 'Oxygen entrainment', group: 'dynamics', min: 0, max: 3, step: 0.01, value: 1.65, description: 'Mixes additional oxygen into large, fuel-rich plumes.' },
    { id: 'heatEfficiency', label: 'Heat-release efficiency', group: 'dynamics', min: 0, max: 1, step: 0.01, value: 0.86 },
    { id: 'flamePersistence', label: 'Thermal persistence', group: 'dynamics', min: 0.4, max: 3, step: 0.01, value: 1.45, unit: '×', description: 'Higher values retain heat and combustion products for a longer plume.' },
    { id: 'buoyancyScale', label: 'Plume buoyancy', group: 'dynamics', min: 0, max: 4, step: 0.01, value: 2.15, unit: '×', description: 'Directly controls the rise of the hot plume.' },
    { id: 'vorticity', label: 'Vorticity strength', group: 'dynamics', min: 0, max: 4, step: 0.01, value: 3.6 },
    { id: 'waterFlow', label: 'Droplet-flow scale', group: 'water', min: 0, max: 1, step: 0.01, value: 0 },
    { id: 'sprayAngle', label: 'Spray cone angle', group: 'water', min: 4, max: 52, step: 1, value: 10, unit: '°' },
    { id: 'aimHeight', label: 'Target Y (domain height)', group: 'water', min: 0.08, max: 0.82, step: 0.01, value: 0.42, description: 'Normalized vertical target within the simulation domain.' },
    { id: 'domainWidth', label: 'Domain width (X)', group: 'domain', min: 0.5, max: 4, step: 0.01, value: 1.15, quantity: 'length', description: 'Nominal physical width. Applied when the drag is released.', commitOnly: true, resetOnCommit: true },
    { id: 'domainHeight', label: 'Domain height (Y)', group: 'domain', min: 0.5, max: 6, step: 0.01, value: 1.35, quantity: 'length', description: 'Nominal physical height. Applied when the drag is released.', commitOnly: true, resetOnCommit: true },
    { id: 'domainDepth', label: 'Domain depth (Z)', group: 'domain', min: 0.5, max: 4, step: 0.01, value: 1.15, quantity: 'length', description: 'Nominal physical depth. Applied when the drag is released.', commitOnly: true, resetOnCommit: true },
    { id: 'viewZoom', label: 'Camera framing zoom', group: 'appearance', min: 0.5, max: 5, step: 0.01, value: 2.1, unit: '×', description: 'Changes camera framing without changing the simulated domain.' },
    { id: 'frontDefinition', label: 'Reaction-edge contrast', group: 'appearance', min: 0.5, max: 2.5, step: 0.01, value: 1.65, unit: '×', description: 'Render-only contrast for resolved reaction fronts; it does not change combustion.' },
    { id: 'exposure', label: 'Render exposure', group: 'appearance', min: 0.25, max: 3, step: 0.01, value: 1.25, unit: '×' },
  ],
};

const controlGroupCopy: Record<MethodId, Record<ControlDefinition['group'], { label: string; hint: string }>> = {
  horvath: {
    source: { label: 'Emitter & emission', hint: 'Source geometry, feed strength and particle lifetime.' },
    dynamics: { label: 'Plume dynamics', hint: 'Buoyant rise, rotational transport and radiative cooling.' },
    water: { label: 'Water suppression', hint: 'Not available in the slice-refinement preview.' },
    domain: { label: 'Physical domain', hint: 'The slice preview uses a normalized simulation box.' },
    appearance: { label: 'Slice appearance & camera', hint: 'Camera framing, exposure, smoke density and reconstructed detail.' },
  },
  firex: {
    source: { label: 'Fuel source & burner', hint: 'Burner geometry, fuel mixture, feed and source momentum.' },
    dynamics: { label: 'Combustion & plume', hint: 'Reactant mixing and heat release first; thermal transport and vorticity follow.' },
    water: { label: 'Water suppression', hint: 'Nozzle pattern, droplet flow, cone angle and target height.' },
    domain: { label: 'Physical domain', hint: 'Nominal X/Y/Z extents combine with the dense grid to determine voxel spacing.' },
    appearance: { label: 'Flame rendering & camera', hint: 'Display-only framing, reaction contrast and final exposure.' },
  },
};

const domainControlGroupOrder: ControlDefinition['group'][] = ['domain'];
const simulationControlGroupOrder: ControlDefinition['group'][] = ['source', 'dynamics', 'water'];
const renderingControlGroupOrder: ControlDefinition['group'][] = ['appearance'];

const beautyLegends: Record<MethodId, { label: string; color: string }[]> = {
  horvath: [
    { label: 'Hot core', color: '#fff6c9' },
    { label: 'Flame body', color: '#ff7a24' },
    { label: 'Smoke', color: '#656b6c' },
    { label: 'Refined detail', color: '#ffd070' },
  ],
  firex: [
    { label: 'Hot reaction', color: '#fff0af' },
    { label: 'Blackbody', color: '#ff6828' },
    { label: 'Droplets / steam', color: '#b9d9df' },
    { label: 'Residual / soot', color: '#42484b' },
  ],
};

const presets: Record<MethodId, readonly ScenarioPreset[]> = {
  horvath: [
    { label: 'Campfire', values: { sourceMode: 0, sourceSize: 1, flameHeight: 1, emission: 0.86, buoyancy: 0.92, vorticity: 1.45, cooling: 0.58, density: 1.36, detail: 0.64, viewZoom: 1, exposure: 1.04 }, expected: 'A narrow source should rise, curl and cool while remaining attached to the emitter.' },
    { label: 'Cinematic bonfire', values: { sourceMode: 0, sourceSize: 1.55, flameHeight: 1.65, emission: 1.2, buoyancy: 1.12, vorticity: 1.65, cooling: 0.43, density: 0.76, detail: 0.55, viewZoom: 1.32, exposure: 1.02 }, expected: 'A coherent broad plume with dark gaps, fine curling tongues, pale hot cores, orange edges and visible smoke.' },
    { label: 'Fireball', values: { sourceMode: 1, sourceSize: 1, flameHeight: 1, emission: 1.4, buoyancy: 0.22, vorticity: 0.62, cooling: 0.32, density: 1.1, detail: 0.88, viewZoom: 1, exposure: 1.02 }, expected: 'A broad, slower-rising hot body should retain more heat and injected detail.' },
    { label: 'Fire wall', values: { sourceMode: 2, sourceSize: 1, flameHeight: 1, emission: 1.15, buoyancy: 0.56, vorticity: 0.18, cooling: 0.72, density: 2.15, detail: 0.42, viewZoom: 1, exposure: 0.96 }, expected: 'A dense, comparatively coherent sheet should rise with limited lateral curl.' },
  ],
  firex: FIREX_SCENE_PRESETS,
};

app.innerHTML = `
  <div class="lab-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true"></div>
        <div class="brand-copy">
          <div class="brand-title">Fire Replica Lab</div>
          <div class="brand-subtitle">Realtime research previews · local GPU compute</div>
        </div>
      </div>
      <nav class="method-tabs" aria-label="Simulation method">
        <button class="method-tab" data-method="horvath">2009 · Slice refinement</button>
        <button class="method-tab" data-method="firex">2025 · Fire-X</button>
      </nav>
      <div class="top-actions">
        <button class="action-button" id="water-toggle" type="button" aria-pressed="false">Water spray</button>
        <button class="icon-button" id="pause-button" type="button" aria-label="Pause simulation">Ⅱ</button>
        <button class="icon-button" id="reset-button" type="button" aria-label="Reset simulation">↻</button>
        <button class="icon-button" id="panel-button" type="button" aria-label="Toggle controls" aria-controls="control-panel" aria-expanded="false">☷</button>
      </div>
    </header>
    <main class="workspace">
      <section class="viewport" id="viewport" aria-label="Realtime fire simulation viewport">
        <canvas class="simulation-canvas" id="gl-canvas"></canvas>
        <canvas class="simulation-canvas" id="gpu-canvas" hidden></canvas>
        <div class="view-heading">
          <div class="eyebrow" id="view-eyebrow"></div>
          <h1 id="view-title"></h1>
          <p id="view-subtitle"></p>
        </div>
        <div class="status-cluster">
          <div class="status-chip"><span class="status-dot"></span><span id="backend-chip">Detecting GPU</span></div>
          <div class="status-chip" id="quality-chip">HIGH</div>
        </div>
        <div class="view-footer">
          <div class="metric"><div class="metric-label">Frame rate</div><div class="metric-value" id="fps-value">—</div></div>
          <div class="metric"><div class="metric-label">Frame time</div><div class="metric-value" id="frame-value">—</div></div>
          <div class="metric"><div class="metric-label">CPU encode / submit</div><div class="metric-value" id="sim-value">—</div></div>
          <div class="metric"><div class="metric-label">Grid / domain</div><div class="metric-value" id="detail-value">—</div></div>
          <div class="interaction-hint">DRAG TO ORBIT · WHEEL TO DOLLY</div>
        </div>
        <div class="loading-card" id="loading-card">
          <div class="loading-flame" aria-hidden="true"></div>
          <strong id="loading-title">Preparing simulation</strong>
          <span id="loading-detail">Compiling GPU pipelines and allocating fields…</span>
        </div>
      </section>
      <aside class="control-panel" id="control-panel" aria-label="Simulation controls">
        <section class="control-section">
          <div class="section-heading"><h2>Scenario</h2><span class="mono-label">Presets</span></div>
          <div class="preset-row" id="preset-row"></div>
        </section>
        <section class="control-section">
          <div class="section-heading"><h2>Compute &amp; resolution</h2><span class="mono-label">GPU</span></div>
          <label class="select-field" for="quality-select">
            <span>Compute profile</span>
            <span class="select-wrap">
              <select id="quality-select" aria-label="Compute quality">
                <option value="low">Fast · reduced field</option>
                <option value="balanced">Balanced · interactive</option>
                <option value="high" selected>High · best available</option>
                <option value="maximum">Maximum · inspection budget</option>
                <option value="cinematic">Cinematic · showcase GPU</option>
              </select>
            </span>
          </label>
          <p class="control-status" id="quality-status" role="status" aria-live="polite" aria-atomic="true"></p>
          <div class="firex-resolution-controls" id="firex-resolution-controls">
            <label class="select-field" for="grid-preset-select">
              <span class="field-label-row"><span>Dense solver grid</span><strong id="grid-selection-state">Applied</strong></span>
              <span class="select-wrap">
                <select id="grid-preset-select" aria-label="Fire-X dense solver grid">
                  <option value="auto">Loading dense-grid tiers…</option>
                </select>
              </span>
            </label>
            <label class="select-field" id="grid-budget-control" for="grid-budget-select">
              <span>Peak GPU allocation guard</span>
              <span class="select-wrap">
                <select id="grid-budget-select" aria-label="Fire-X peak GPU allocation guard">
                  <option value="536870912" selected>Loading memory guards…</option>
                </select>
              </span>
            </label>
            <p class="dependency-note" id="resolution-help"></p>
            <div class="grid-custom-editor" id="grid-custom-editor" hidden>
              <label class="grid-axis-field"><span>X voxels</span><input id="grid-x-input" type="number" min="16" max="1024" step="4" value="64" inputmode="numeric" /></label>
              <label class="grid-axis-field"><span>Y voxels</span><input id="grid-y-input" type="number" min="16" max="1024" step="4" value="96" inputmode="numeric" /></label>
              <label class="grid-axis-field"><span>Z voxels</span><input id="grid-z-input" type="number" min="16" max="1024" step="4" value="64" inputmode="numeric" /></label>
              <button class="grid-apply-button" id="grid-apply-button" type="button">Check &amp; apply dense grid</button>
            </div>
            <div class="grid-summary allocation-summary">
              <div><span>Applied solver</span><strong id="grid-resolution-value">—</strong></div>
              <div><span>Applied cells / estimated GPU memory</span><strong id="grid-memory-value">—</strong></div>
              <div><span>Requested dense grid</span><strong id="grid-request-value">—</strong></div>
              <div><span>Requested cells / estimated GPU memory</span><strong id="grid-request-cost">—</strong></div>
              <div><span>Transactional rebuild peak</span><strong id="grid-request-peak">—</strong></div>
              <div><span>Device field ceiling</span><strong id="grid-device-value">—</strong></div>
              <p id="grid-guidance" role="status" aria-live="polite" aria-atomic="true">Resolution reallocates the solver on commit.</p>
            </div>
          </div>
        </section>
        <section class="control-section firex-only" id="firex-grid-section">
          <div class="section-heading"><h2>Domain &amp; units</h2><span class="mono-label">Fire-X</span></div>
          <div class="inspection-controls">
            <label class="select-field" for="unit-system-select">
              <span>Display units</span>
              <span class="select-wrap">
                <select id="unit-system-select" aria-label="Physical domain display units">
                  <option value="metric">Metric · m / cm</option>
                  <option value="imperial">Imperial · ft / in</option>
                </select>
              </span>
            </label>
          </div>
          <div class="controls-list domain-controls-list" id="domain-controls-list"></div>
          <div class="grid-summary domain-summary">
            <div><span>Domain</span><strong id="grid-domain-value">—</strong></div>
            <div><span>Applied spacing</span><strong id="grid-cell-value">—</strong></div>
            <p>Display units change labels only. Domain size changes physical scale and voxel spacing, never the selected grid tuple.</p>
          </div>
        </section>
        <section class="control-section">
          <div class="section-heading"><h2>Simulation controls</h2><span class="mono-label">Live</span></div>
          <div class="controls-list" id="controls-list"></div>
        </section>
        <section class="control-section">
          <div class="section-heading"><h2>Rendering &amp; camera</h2><span class="mono-label">Display</span></div>
          <div class="controls-list" id="render-controls-list"></div>
          <label class="select-field rendering-detail-control" id="optical-detail-control" for="optical-detail-select">
            <span class="field-label-row"><span>Cinematic detail frequency</span><strong id="optical-selection-state">Stored</strong></span>
            <span class="select-wrap">
              <select id="optical-detail-select" aria-label="Cinematic render-only detail frequency">
                <option value="0">Native solver frequency · render only</option>
                <option value="256">Virtual 256³ frequency · render only</option>
                <option value="512">Virtual 512³ frequency · render only</option>
                <option value="1024" selected>Virtual 1024³ frequency · render only</option>
              </select>
            </span>
          </label>
          <p class="dependency-note rendering-detail-help" id="optical-detail-help"></p>
          <div class="field-legend" id="field-legend-section">
            <div class="section-heading legend-heading"><h3 id="legend-title">Beauty legend</h3><span class="mono-label" id="legend-context">Display</span></div>
            <div class="legend" id="legend-items"></div>
          </div>
        </section>
        <section class="control-section" id="offline-export-section" aria-labelledby="offline-export-heading">
          <div class="section-heading"><h2 id="offline-export-heading">Offline video render</h2><span class="mono-label">60 fps</span></div>
          <div class="offline-export-controls">
            <label class="select-field" for="offline-preset-select">
              <span>Output preset</span>
              <span class="select-wrap">
                <select id="offline-preset-select" aria-label="Offline video output preset"></select>
              </span>
            </label>
            <label class="range-control offline-duration-control" for="offline-duration-input">
              <span class="range-label">Duration</span>
              <output class="range-value" id="offline-duration-value" for="offline-duration-input">5 seconds</output>
              <input id="offline-duration-input" type="range" min="2" max="10" step="1" value="5" />
              <span class="range-endpoints" aria-hidden="true"><span>2 s</span><span id="offline-duration-max">10 s</span></span>
            </label>
            <div class="offline-export-summary">
              <div><span>Scene snapshot</span><strong id="offline-source-value">Engine pending</strong></div>
              <div><span>Render solver</span><strong id="offline-solver-value">Engine pending</strong></div>
              <div><span>Video</span><strong id="offline-video-value">1920×1080 · 60 fps</strong></div>
              <div><span>Estimated file</span><strong id="offline-size-value">—</strong></div>
              <div><span>Verified file</span><strong id="offline-verified-value">Not rendered yet</strong></div>
            </div>
            <p class="dependency-note" id="offline-export-note"></p>
            <details class="cli-render-help">
              <summary>Long-form / one-hour CLI recorder</summary>
              <p>The CLI streams deterministic 60 fps fMP4 directly to disk, then verifies frame count, cadence and decoding outside the browser. It does not keep the full movie in RAM.</p>
              <code>.\scripts\record-fire.cmd --test-views --background</code>
              <code>.\scripts\record-fire.cmd --quality highest --hour --field beauty --confirm-heavy --background --output renders/firex-4k60-1h.mp4</code>
              <small>Highest means real Reference 508³ + UHD solver + 4K60. It needs roughly 13.19 GiB of solver resources and produces about 29.25 GB per hour; unsupported hardware fails instead of downscaling.</small>
            </details>
            <button class="offline-render-button" id="offline-render-button" type="button">Render 60 fps MP4</button>
            <div class="offline-progress" id="offline-progress" hidden>
              <progress id="offline-progress-bar" max="1" value="0">0%</progress>
              <div class="offline-progress-copy"><span id="offline-progress-value">Preparing</span><span id="offline-eta-value">ETA —</span></div>
              <button class="offline-cancel-button" id="offline-cancel-button" type="button">Cancel render</button>
            </div>
            <p class="control-status offline-phase-status" id="offline-phase-status" role="status" aria-live="polite" aria-atomic="true"></p>
            <p class="control-status is-error" id="offline-error" role="alert" hidden></p>
            <div class="offline-result-actions">
              <a class="offline-save-link" id="offline-save-link" href="" download hidden>Save 60 fps master</a>
              <a class="offline-compatibility-link" id="offline-compatibility-link" href="" download hidden>Save 720p60 playback copy</a>
              <button class="offline-test-playback-button" id="offline-test-playback-button" type="button" hidden>Test real 60 fps playback</button>
            </div>
            <div class="offline-preview" id="offline-preview" hidden>
              <video
                id="offline-preview-video"
                controls
                muted
                playsinline
                preload="metadata"
                aria-label="Rendered MP4 preview"
                aria-describedby="offline-playback-status"
              ></video>
              <p class="control-status offline-playback-status" id="offline-playback-status" role="status" aria-live="polite" aria-atomic="true">Play the 720p60 copy to measure real 60 fps presentation on this device.</p>
            </div>
          </div>
        </section>
        <section class="control-section firex-only" id="firex-inspection-section">
          <div class="section-heading"><h2>Inspect Fire-X</h2><span class="mono-label">Verification</span></div>
          <div class="inspection-controls">
            <label class="select-field" for="field-view-select">
              <span>Field view</span>
              <span class="select-wrap">
                <select id="field-view-select" aria-label="Fire-X diagnostic field"></select>
              </span>
            </label>
          </div>
          <div class="verification-card" aria-live="polite">
            <div class="verification-row">
              <span>Current field</span>
              <strong id="verification-view">Beauty</strong>
            </div>
            <p id="verification-help"></p>
            <div class="verification-callout">
              <span>Expected result</span>
              <p id="verification-expected"></p>
            </div>
            <div class="verification-profile">
              <span>Active profile</span>
              <code id="verification-profile">Engine pending</code>
            </div>
          </div>
        </section>
        <section class="control-section" id="about-section">
          <div class="section-heading"><h2>About this method</h2><span class="mono-label">Paper</span></div>
          <div class="paper-card">
            <div class="paper-year" id="paper-year"></div>
            <div class="paper-title" id="paper-title"></div>
            <div class="paper-detail" id="paper-detail"></div>
          </div>
          <div class="capability-note" id="capability-note"><strong>Research previews, not safety models.</strong> Each path reproduces selected paper mechanisms at a browser-scale budget. Diagnostic fields expose what is actually simulated.</div>
        </section>
        <div id="method-control-pool" hidden>
          <label class="suppression-control" id="suppression-enabled-control" for="suppression-enabled-input">
            <span>Suppression enabled</span>
            <input id="suppression-enabled-input" type="checkbox" />
          </label>
          <label class="select-field" id="nozzle-type-control" for="nozzle-type-select">
            <span>Nozzle type</span>
            <span class="select-wrap">
              <select id="nozzle-type-select" aria-label="Water nozzle type">
                <option value="0">Laminar jet</option>
                <option value="1">Spray</option>
              </select>
            </span>
          </label>
        </div>
      </aside>
    </main>
  </div>
`;

function required<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`Missing UI element: ${selector}`);
  return node;
}

const viewport = required<HTMLElement>('#viewport');
const glCanvas = required<HTMLCanvasElement>('#gl-canvas');
const gpuCanvas = required<HTMLCanvasElement>('#gpu-canvas');
const loadingCard = required<HTMLElement>('#loading-card');
const loadingTitle = required<HTMLElement>('#loading-title');
const loadingDetail = required<HTMLElement>('#loading-detail');
const controlPanel = required<HTMLElement>('#control-panel');
const panelButton = required<HTMLButtonElement>('#panel-button');
const pauseButton = required<HTMLButtonElement>('#pause-button');
const waterButton = required<HTMLButtonElement>('#water-toggle');
const qualitySelect = required<HTMLSelectElement>('#quality-select');
const qualityStatus = required<HTMLElement>('#quality-status');
const firexResolutionControls = required<HTMLElement>('#firex-resolution-controls');
const gridSelectionState = required<HTMLElement>('#grid-selection-state');
const opticalDetailControl = required<HTMLElement>('#optical-detail-control');
const opticalSelectionState = required<HTMLElement>('#optical-selection-state');
const opticalDetailHelp = required<HTMLElement>('#optical-detail-help');
const resolutionHelp = required<HTMLElement>('#resolution-help');
const domainControlsList = required<HTMLElement>('#domain-controls-list');
const simulationControlsList = required<HTMLElement>('#controls-list');
const renderingControlsList = required<HTMLElement>('#render-controls-list');
const methodControlPool = required<HTMLElement>('#method-control-pool');
const suppressionEnabledControl = required<HTMLElement>('#suppression-enabled-control');
const suppressionEnabledInput = required<HTMLInputElement>('#suppression-enabled-input');
const nozzleTypeControl = required<HTMLElement>('#nozzle-type-control');
const fieldLegendSection = required<HTMLElement>('#field-legend-section');
const legendTitle = required<HTMLElement>('#legend-title');
const legendContext = required<HTMLElement>('#legend-context');
const legendItems = required<HTMLElement>('#legend-items');
const fieldViewSelect = required<HTMLSelectElement>('#field-view-select');
const nozzleTypeSelect = required<HTMLSelectElement>('#nozzle-type-select');
const firexInspectionSection = required<HTMLElement>('#firex-inspection-section');
const firexGridSection = required<HTMLElement>('#firex-grid-section');
const gridPresetSelect = required<HTMLSelectElement>('#grid-preset-select');
const unitSystemSelect = required<HTMLSelectElement>('#unit-system-select');
const opticalDetailSelect = required<HTMLSelectElement>('#optical-detail-select');
const gridBudgetSelect = required<HTMLSelectElement>('#grid-budget-select');
const gridCustomEditor = required<HTMLElement>('#grid-custom-editor');
const gridXInput = required<HTMLInputElement>('#grid-x-input');
const gridYInput = required<HTMLInputElement>('#grid-y-input');
const gridZInput = required<HTMLInputElement>('#grid-z-input');
const gridApplyButton = required<HTMLButtonElement>('#grid-apply-button');
const gridAxisFields = Array.from(gridCustomEditor.querySelectorAll<HTMLElement>('.grid-axis-field'));
const offlineExportSection = required<HTMLElement>('#offline-export-section');
const offlinePresetSelect = required<HTMLSelectElement>('#offline-preset-select');
const offlineDurationInput = required<HTMLInputElement>('#offline-duration-input');
const offlineDurationValue = required<HTMLOutputElement>('#offline-duration-value');
const offlineDurationMax = required<HTMLElement>('#offline-duration-max');
const offlineSourceValue = required<HTMLElement>('#offline-source-value');
const offlineSolverValue = required<HTMLElement>('#offline-solver-value');
const offlineVideoValue = required<HTMLElement>('#offline-video-value');
const offlineSizeValue = required<HTMLElement>('#offline-size-value');
const offlineVerifiedValue = required<HTMLElement>('#offline-verified-value');
const offlineExportNote = required<HTMLElement>('#offline-export-note');
const offlineRenderButton = required<HTMLButtonElement>('#offline-render-button');
const offlineCancelButton = required<HTMLButtonElement>('#offline-cancel-button');
const offlineProgress = required<HTMLElement>('#offline-progress');
const offlineProgressBar = required<HTMLProgressElement>('#offline-progress-bar');
const offlineProgressValue = required<HTMLElement>('#offline-progress-value');
const offlineEtaValue = required<HTMLElement>('#offline-eta-value');
const offlinePhaseStatus = required<HTMLElement>('#offline-phase-status');
const offlineError = required<HTMLElement>('#offline-error');
const offlineSaveLink = required<HTMLAnchorElement>('#offline-save-link');
const offlineCompatibilityLink = required<HTMLAnchorElement>('#offline-compatibility-link');
const offlineTestPlaybackButton = required<HTMLButtonElement>('#offline-test-playback-button');
const offlinePreview = required<HTMLElement>('#offline-preview');
const offlinePreviewVideo = required<HTMLVideoElement>('#offline-preview-video');
const offlinePlaybackStatus = required<HTMLElement>('#offline-playback-status');

fieldViewSelect.innerHTML = firexFieldViews
  .map((view) => `<option value="${view.value}">${view.label}</option>`)
  .join('');

const linkedGridPresets = FIREX_GRID_PRESETS.filter((preset) => preset.id === 'auto');
const independentGridPresets = FIREX_GRID_PRESETS.filter((preset) => preset.id !== 'auto');
gridPresetSelect.innerHTML = `
  <optgroup label="Linked to compute profile">
    ${linkedGridPresets.map((preset) => `<option value="${preset.id}">${preset.label}</option>`).join('')}
  </optgroup>
  <optgroup label="Independent fixed or custom grid">
    ${independentGridPresets.map((preset) => `<option value="${preset.id}">${preset.label}</option>`).join('')}
  </optgroup>
`;

gridBudgetSelect.innerHTML = FIREX_GRID_MEMORY_GUARDS
  .map((guard) => `<option value="${guard.bytes}">${guard.label}</option>`)
  .join('');

offlinePresetSelect.innerHTML = OFFLINE_VIDEO_PRESETS
  .map((preset) => `<option value="${preset.id}">${preset.label}</option>`)
  .join('');
offlinePresetSelect.value = 'hd';

let activeMethod: MethodId = navigator.gpu ? 'firex' : 'horvath';
let activeEngine: SimulationEngine | null = null;
let paused = false;
let quality: QualityLevel = 'high';
let animationFrame = 0;
let previousTime = performance.now();
let lastStatsUpdate = 0;
let smoothedFrameMs = 16.67;
let smoothedSimMs = 0;
let cameraYaw = -0.34;
let cameraPitch = 0.03;
let cameraDistance = 4.9;
let firexViewMode = 0;
let firexNozzleType = 0;
let firexUnitSystem: FireXUnitSystem = 'metric';
let firexOpticalDetailTarget = 1024;
let firexGridMemoryBudgetBytes = 512 * 1024 * 1024;
let firexAppliedGridPreset: FireXGridPresetId = 'auto';
let firexGridPickerValue: FireXGridPresetId = 'auto';
let firexCustomGrid: FireXGridDimensions = [64, 96, 64];
let firexGridStatusMessage: string | null = null;
let firexRejectedGridRequest: {
  presetId: FireXGridPresetId;
  dimensions: FireXGridDimensions | null;
} | null = null;
let horvathSourceMode = 0;
const engines: Partial<Record<MethodId, SimulationEngine>> = {};
const enginePromises: Partial<Record<MethodId, Promise<SimulationEngine>>> = {};
// Start Fire-X on the deliberately oversized Inferno profile so source-scale
// controls are obvious on first load. Methane remains the neutral comparison
// and Base spray remains a single-click deterministic suppression case.
const activePreset: Record<MethodId, number> = { horvath: 1, firex: 2 };
const openControlGroups: Record<MethodId, Set<ControlDefinition['group']>> = {
  horvath: new Set(['source', 'appearance']),
  firex: new Set(['domain', 'source', 'appearance']),
};
let requestedMethod = activeMethod;
let switchGeneration = 0;
let resizeQueued = false;
let resourceControlsLocked = false;
let resourceControlsLockDepth = 0;
let lastResize: {
  engine: SimulationEngine;
  width: number;
  height: number;
  pixelRatio: number;
} | null = null;
let offlinePresetId: OfflineVideoPresetId = 'hd';
let offlineExportRunning = false;
let offlineExportAbortController: AbortController | null = null;
let offlineDownloadUrl: string | null = null;
let offlinePlaybackPreviewUrl: string | null = null;
let offlineCompletedSourceSummary: string | null = null;
let offlineRenderStartedAt = 0;
let offlineLastAnnouncedPhase: OfflineVideoProgress['phase'] | null = null;
const offlineControlStates = new Map<HTMLButtonElement | HTMLInputElement | HTMLSelectElement, boolean>();

interface OfflinePlaybackQuality {
  totalVideoFrames: number;
  droppedVideoFrames: number;
  corruptedVideoFrames: number;
}

interface OfflineVideoFrameMetadata {
  mediaTime: number;
  presentedFrames: number;
}

type MeasurableVideoElement = HTMLVideoElement & {
  getVideoPlaybackQuality?: () => OfflinePlaybackQuality;
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: OfflineVideoFrameMetadata) => void,
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

interface OfflinePlaybackMeasurement {
  baselineDropped: number;
  baselineTotal: number;
  baselineCorrupted: number;
  callbacks: number;
  estimatedDropped: number;
  lastMediaTime: number | null;
  lastPresentedFrames: number | null;
  lastCallbackNow: number | null;
  wallIntervalsMs: number[];
  stallCount: number;
  stallDurationMs: number;
  stallStartedAt: number | null;
}

let offlinePlaybackMeasurement: OfflinePlaybackMeasurement | null = null;
let offlinePlaybackFrameCallback: number | null = null;
let offlinePlaybackPoll: number | null = null;
let offlinePlaybackExpectedFrameRate = 60;
let offlinePlaybackSuspendedEngine: SimulationEngine | null = null;

function formatControlValue(control: ControlDefinition, value: number): string {
  if (control.quantity === 'length') {
    const converted = firexUnitSystem === 'metric' ? value : value * 3.280839895013123;
    return `${converted.toFixed(2)} ${firexUnitSystem === 'metric' ? 'm' : 'ft'}`;
  }
  if (control.id === 'aimHeight') return `${Math.round(value * 100)}%`;
  const precision = control.step >= 1 ? 0 : control.step >= 0.1 ? 1 : 2;
  return `${value.toFixed(precision)}${control.unit ?? ''}`;
}

function formatOfflineBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'] as const;
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const precision = value >= 100 || unitIndex === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}

function formatOfflineBitrate(bitsPerSecond: number): string {
  if (!Number.isFinite(bitsPerSecond) || bitsPerSecond <= 0) return '—';
  return `${(bitsPerSecond / 1_000_000).toFixed(1)} Mb/s`;
}

function selectedOfflinePresetId(): OfflineVideoPresetId {
  const value = offlinePresetSelect.value;
  return OFFLINE_VIDEO_PRESETS.some((preset) => preset.id === value)
    ? value as OfflineVideoPresetId
    : 'hd';
}

function offlineSolverBudgetSummary(preset: OfflineVideoPreset): string {
  if (activeMethod === 'firex') {
    const configuration = FIREX_OFFLINE_QUALITY[preset.offlineTier];
    const currentDimensions = selectedFireXGridDimensions();
    const dimensions = resolveFireXOfflineDimensions(preset.offlineTier, currentDimensions);
    const memory = estimateFireXGridMemory(dimensions, configuration.particleCount);
    return `${dimensions.join('×')} real grid · ${configuration.pressureIterations}+${configuration.correctionIterations} pressure · ${configuration.particleCount.toLocaleString('en-US')} particles · ${configuration.raySteps} rays · ${configuration.opticalDetailTarget}³ optical shading · ${formatOfflineBytes(memory.totalBytes)} solver`;
  }
  const configuration = HORVATH_OFFLINE_QUALITY[preset.offlineTier];
  const memory = estimateHorvathMemory(configuration);
  const tileWidth = Math.round((configuration.tileHeight * 1.22) / 4) * 4;
  return `${tileWidth}×${configuration.tileHeight}×${configuration.slices} real atlas field · ${configuration.jacobiIterations} pressure · ${configuration.particleCount.toLocaleString('en-US')} particles · ${configuration.raySteps} rays · ${formatOfflineBytes(memory.totalBytes)} solver`;
}

function currentOfflineSourceSummary(): string {
  const method = activeMethod === 'firex' ? 'Fire-X 2025' : 'Slice 2009';
  const scenario = presets[activeMethod][activePreset[activeMethod]]?.label ?? 'Custom scene';
  const view = activeMethod === 'firex'
    ? firexFieldViews.find((candidate) => candidate.value === firexViewMode)?.label ?? 'Beauty'
    : 'Beauty';
  const grid = activeMethod === 'firex'
    ? activeEngine?.getGridInfo?.().dimensions.join('×') ?? 'grid pending'
    : 'slice atlas';
  return `${method} · ${scenario} · ${quality.toUpperCase()} · ${grid} · ${view} · current camera`;
}

function updateOfflineExportPanel(): void {
  if (offlineExportRunning) return;
  offlinePresetId = selectedOfflinePresetId();
  offlinePresetSelect.value = offlinePresetId;
  const preset = getOfflineVideoPreset(offlinePresetId);
  offlineDurationInput.max = String(preset.maxDurationSeconds);
  offlineDurationMax.textContent = `${preset.maxDurationSeconds} s`;
  const durationSeconds = Math.max(
    2,
    Math.min(preset.maxDurationSeconds, Math.round(Number(offlineDurationInput.value) || 5)),
  );
  offlineDurationInput.value = String(durationSeconds);
  offlineDurationValue.value = `${durationSeconds} second${durationSeconds === 1 ? '' : 's'}`;
  setRangeFill(offlineDurationInput);

  const frameCount = getOfflineVideoFrameCount(durationSeconds, preset.frameRate);
  offlineSourceValue.textContent = offlineCompletedSourceSummary
    ?? (activeEngine ? currentOfflineSourceSummary() : 'Engine pending');
  offlineSolverValue.textContent = activeEngine
    ? offlineSolverBudgetSummary(preset)
    : 'Engine pending';
  offlineVideoValue.textContent = `${preset.width}×${preset.height} · ${preset.frameRate} fps · ${frameCount} frames · ${preset.codecLabel} MP4`;
  if (offlineSaveLink.hidden) {
    offlineSizeValue.textContent = `~${formatOfflineBytes(estimateOfflineVideoBytes(preset, durationSeconds))} target file size`;
  }

  const supported = isOfflineVideoSupported();
  if (!supported) {
    offlineExportNote.textContent = 'Offline export needs WebCodecs in a secure browser context. Localhost and the published HTTPS site qualify when the browser exposes VideoEncoder.';
  } else if (!activeEngine) {
    offlineExportNote.textContent = 'Waiting for the active GPU engine before rendering can start.';
  } else {
    offlineExportNote.textContent = 'This is a dedicated offline solver allocation for 60 fps output, not an upscale of the live field. Every frame is copied directly from the GPU before the browser compositor can recycle the canvas, then decoded frame by frame to verify cadence and visible motion. A lighter 720p60 copy retains all frames for honest playback testing; no 30 fps result is labeled as proof of 60 fps. Unsupported exact tiers fail explicitly—resolution and fps are never reduced.';
  }
  offlineRenderButton.disabled = resourceControlsLocked || !activeEngine || !supported;
  offlineCancelButton.disabled = true;
}

function setOfflineMutationControlsLocked(locked: boolean): void {
  if (locked) {
    offlineControlStates.clear();
    document.querySelectorAll<HTMLButtonElement | HTMLInputElement | HTMLSelectElement>('button, input, select')
      .forEach((control) => {
        if (control === panelButton || control === offlineCancelButton) return;
        offlineControlStates.set(control, control.disabled);
        control.disabled = true;
      });
    offlineCancelButton.disabled = false;
    return;
  }
  offlineControlStates.forEach((wasDisabled, control) => {
    control.disabled = wasDisabled;
  });
  offlineControlStates.clear();
  offlineCancelButton.disabled = true;
}

function offlinePlaybackQuality(): OfflinePlaybackQuality | null {
  const video = offlinePreviewVideo as MeasurableVideoElement;
  return typeof video.getVideoPlaybackQuality === 'function'
    ? video.getVideoPlaybackQuality()
    : null;
}

function suspendInteractiveRenderingForPlayback(): void {
  if (offlinePlaybackSuspendedEngine || !activeEngine || offlineExportRunning) return;
  offlinePlaybackSuspendedEngine = activeEngine;
  offlinePlaybackSuspendedEngine.setPaused(true);
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = 0;
}

function resumeInteractiveRenderingAfterPlayback(): void {
  const suspendedEngine = offlinePlaybackSuspendedEngine;
  if (!suspendedEngine) return;
  offlinePlaybackSuspendedEngine = null;
  suspendedEngine.setPaused(paused);
  if (!offlineExportRunning && activeEngine) {
    previousTime = performance.now();
    if (!animationFrame) animationFrame = requestAnimationFrame(animate);
  }
}

function stopOfflinePlaybackMeasurement(): void {
  const video = offlinePreviewVideo as MeasurableVideoElement;
  if (offlinePlaybackFrameCallback !== null && typeof video.cancelVideoFrameCallback === 'function') {
    video.cancelVideoFrameCallback(offlinePlaybackFrameCallback);
  }
  offlinePlaybackFrameCallback = null;
  if (offlinePlaybackPoll !== null) window.clearInterval(offlinePlaybackPoll);
  offlinePlaybackPoll = null;
}

function resetOfflinePlaybackHealth(): void {
  stopOfflinePlaybackMeasurement();
  offlinePlaybackMeasurement = null;
}

function renderOfflinePlaybackHealth(): void {
  const measurement = offlinePlaybackMeasurement;
  if (!measurement) return;
  const quality = offlinePlaybackQuality();
  const total = quality
    ? Math.max(0, quality.totalVideoFrames - measurement.baselineTotal)
    : measurement.callbacks;
  const dropped = quality
    ? Math.max(0, quality.droppedVideoFrames - measurement.baselineDropped)
    : measurement.estimatedDropped;
  const corrupted = quality
    ? Math.max(0, quality.corruptedVideoFrames - measurement.baselineCorrupted)
    : 0;
  if (total < 30) {
    offlinePlaybackStatus.textContent = `Measuring local playback\u2026 ${total.toLocaleString('en-US')} frames checked.`;
    return;
  }
  const droppedPercent = total > 0 ? (dropped / total) * 100 : 0;
  const corruptionCopy = corrupted > 0 ? ` \u00b7 ${corrupted} corrupted` : '';
  const sortedIntervals = [...measurement.wallIntervalsMs].sort((a, b) => a - b);
  const percentile = (fraction: number): number => {
    if (sortedIntervals.length === 0) return 0;
    return sortedIntervals[Math.min(
      sortedIntervals.length - 1,
      Math.floor((sortedIntervals.length - 1) * fraction),
    )]!;
  };
  const medianInterval = percentile(0.5);
  const p95Interval = percentile(0.95);
  const maxInterval = sortedIntervals.at(-1) ?? 0;
  const measuredPresentationFps = medianInterval > 0 ? 1000 / medianInterval : 0;
  const timingCopy = sortedIntervals.length >= 10
    ? ` \u00b7 presented ~${measuredPresentationFps.toFixed(0)} fps \u00b7 p95 ${p95Interval.toFixed(1)} ms \u00b7 max ${maxInterval.toFixed(1)} ms`
    : '';
  const stallCopy = measurement.stallCount > 0
    ? ` \u00b7 ${measurement.stallCount} stall${measurement.stallCount === 1 ? '' : 's'} (${(measurement.stallDurationMs / 1000).toFixed(2)} s)`
    : '';
  const hasPresentationJitter = p95Interval > (1000 / offlinePlaybackExpectedFrameRate) * 1.5;
  if (dropped === 0 && corrupted === 0 && !hasPresentationJitter && measurement.stallCount === 0) {
    offlinePlaybackStatus.textContent = `Smooth ${offlinePlaybackExpectedFrameRate} fps presentation on this device \u00b7 ${total.toLocaleString('en-US')} frames checked \u00b7 0 dropped${timingCopy}`;
  } else if (droppedPercent <= 1 && corrupted === 0) {
    offlinePlaybackStatus.textContent = `Presentation jitter detected \u00b7 ${dropped} / ${total.toLocaleString('en-US')} dropped (${droppedPercent.toFixed(1)}%)${timingCopy}${stallCopy}`;
  } else {
    const displayLimitCopy = measuredPresentationFps > 0
      && measuredPresentationFps < offlinePlaybackExpectedFrameRate * 0.8
      ? ` This browser window is presenting about ${measuredPresentationFps.toFixed(0)} fps, so it cannot visibly prove 60 fps; save the verified file and play it in a 60 Hz system player.`
      : ' Save the verified file and test it in a hardware-accelerated system player.';
    offlinePlaybackStatus.textContent = `60 fps playback is dropping frames in this browser \u00b7 ${dropped} / ${total.toLocaleString('en-US')} (${droppedPercent.toFixed(1)}%)${corruptionCopy}${timingCopy}${stallCopy}.${displayLimitCopy}`;
  }
}

function scheduleOfflinePlaybackFrameMeasurement(): void {
  const video = offlinePreviewVideo as MeasurableVideoElement;
  if (!offlinePlaybackMeasurement || typeof video.requestVideoFrameCallback !== 'function') return;
  offlinePlaybackFrameCallback = video.requestVideoFrameCallback((now, metadata) => {
    offlinePlaybackFrameCallback = null;
    const measurement = offlinePlaybackMeasurement;
    if (!measurement) return;
    measurement.callbacks += 1;
    if (measurement.lastCallbackNow !== null) {
      measurement.wallIntervalsMs.push(now - measurement.lastCallbackNow);
      if (measurement.wallIntervalsMs.length > 1200) measurement.wallIntervalsMs.shift();
    }
    measurement.lastCallbackNow = now;
    if (measurement.lastMediaTime !== null && measurement.lastPresentedFrames !== null) {
      const mediaIntervals = Math.max(
        1,
        Math.round(
          (metadata.mediaTime - measurement.lastMediaTime) * offlinePlaybackExpectedFrameRate,
        ),
      );
      const compositorIntervals = Math.max(
        1,
        metadata.presentedFrames - measurement.lastPresentedFrames,
      );
      measurement.estimatedDropped += Math.max(0, mediaIntervals - compositorIntervals);
    }
    measurement.lastMediaTime = metadata.mediaTime;
    measurement.lastPresentedFrames = metadata.presentedFrames;
    if (!offlinePreviewVideo.paused && !offlinePreviewVideo.ended) {
      scheduleOfflinePlaybackFrameMeasurement();
    }
  });
}

function startOfflinePlaybackMeasurement(): void {
  stopOfflinePlaybackMeasurement();
  const video = offlinePreviewVideo as MeasurableVideoElement;
  const quality = offlinePlaybackQuality();
  const hasFrameCallbacks = typeof video.requestVideoFrameCallback === 'function';
  if (!quality && !hasFrameCallbacks) {
    offlinePlaybackMeasurement = null;
    offlinePlaybackStatus.textContent = 'Preview is playing, but this browser does not expose dropped-frame metrics.';
    return;
  }
  offlinePlaybackMeasurement = {
    baselineDropped: quality?.droppedVideoFrames ?? 0,
    baselineTotal: quality?.totalVideoFrames ?? 0,
    baselineCorrupted: quality?.corruptedVideoFrames ?? 0,
    callbacks: 0,
    estimatedDropped: 0,
    lastMediaTime: null,
    lastPresentedFrames: null,
    lastCallbackNow: null,
    wallIntervalsMs: [],
    stallCount: 0,
    stallDurationMs: 0,
    stallStartedAt: null,
  };
  scheduleOfflinePlaybackFrameMeasurement();
  offlinePlaybackPoll = window.setInterval(renderOfflinePlaybackHealth, 500);
  renderOfflinePlaybackHealth();
}

function resumeOfflinePlaybackMeasurement(): void {
  const measurement = offlinePlaybackMeasurement;
  if (!measurement) {
    startOfflinePlaybackMeasurement();
    return;
  }
  if (measurement.stallStartedAt !== null) {
    measurement.stallDurationMs += performance.now() - measurement.stallStartedAt;
    measurement.stallStartedAt = null;
  }
  stopOfflinePlaybackMeasurement();
  scheduleOfflinePlaybackFrameMeasurement();
  offlinePlaybackPoll = window.setInterval(renderOfflinePlaybackHealth, 500);
  renderOfflinePlaybackHealth();
}

function finishOfflinePlaybackStall(): void {
  const measurement = offlinePlaybackMeasurement;
  if (!measurement || measurement.stallStartedAt === null) return;
  measurement.stallDurationMs += performance.now() - measurement.stallStartedAt;
  measurement.stallStartedAt = null;
}

function revokeOfflineDownload(): void {
  resetOfflinePlaybackHealth();
  offlinePreviewVideo.pause();
  resumeInteractiveRenderingAfterPlayback();
  offlinePreviewVideo.removeAttribute('src');
  offlinePreviewVideo.load();
  offlinePreview.hidden = true;
  offlineTestPlaybackButton.hidden = true;
  offlineCompatibilityLink.hidden = true;
  offlineCompatibilityLink.removeAttribute('href');
  offlinePlaybackStatus.textContent = 'Play the 720p60 copy to measure real 60 fps presentation on this device.';
  if (offlineDownloadUrl) URL.revokeObjectURL(offlineDownloadUrl);
  if (offlinePlaybackPreviewUrl) URL.revokeObjectURL(offlinePlaybackPreviewUrl);
  offlineDownloadUrl = null;
  offlinePlaybackPreviewUrl = null;
  offlineCompletedSourceSummary = null;
  offlineSaveLink.hidden = true;
  offlineSaveLink.removeAttribute('href');
  offlineVerifiedValue.textContent = 'Not rendered yet';
}

function offlinePhaseCopy(phase: OfflineVideoProgress['phase']): string {
  switch (phase) {
    case 'probing': return 'Preflighting the exact H.264 profile, resolution, bitrate, and 60 fps cadence…';
    case 'allocating': return 'Allocating the dedicated high-resolution solver bundle…';
    case 'warmup': return 'Warming the solver at full field resolution without wasting work on presentation…';
    case 'capture': return 'Advancing, ray-marching, and encoding every exact output frame…';
    case 'finalizing': return 'Finalizing the fast-start H.264 MP4 container…';
    case 'verifying': return 'Demuxing and decoding the finished MP4 to verify timestamps, keyframes, and visible motion…';
    case 'proxying': return 'Building and decoding a lighter 720p60 playback proof while retaining every verified master frame…';
  }
}

function updateOfflineProgress(progress: OfflineVideoProgress): void {
  const fraction = Math.max(0, Math.min(1, progress.fraction));
  offlineProgressBar.max = 1;
  offlineProgressBar.value = fraction;
  const percent = Math.round(fraction * 100);
  const phaseLabel: Record<OfflineVideoProgress['phase'], string> = {
    probing: 'Encoder check',
    allocating: 'Solver allocation',
    warmup: 'Warm-up',
    capture: 'Render',
    finalizing: 'Finalizing',
    verifying: 'Verifying',
    proxying: 'Playback proof',
  };
  offlineProgressValue.textContent = `${phaseLabel[progress.phase]} · ${progress.completedFrames.toLocaleString('en-US')} / ${progress.totalFrames.toLocaleString('en-US')} · ${percent}%`;

  const elapsedSeconds = Math.max((performance.now() - offlineRenderStartedAt) / 1000, 0);
  const remainingSeconds = fraction > 0.01 && fraction < 1
    ? Math.max(0, elapsedSeconds * (1 - fraction) / fraction)
    : 0;
  offlineEtaValue.textContent = fraction >= 1
    ? 'Complete'
    : fraction > 0.01
      ? `ETA ${remainingSeconds < 60 ? `${Math.ceil(remainingSeconds)} s` : `${Math.ceil(remainingSeconds / 60)} min`}`
      : 'ETA measuring…';

  if (offlineLastAnnouncedPhase !== progress.phase) {
    offlineLastAnnouncedPhase = progress.phase;
    offlinePhaseStatus.textContent = offlinePhaseCopy(progress.phase);
  }
}

function isOfflineAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

function offlineFileName(presetId: OfflineVideoPresetId, durationSeconds: number): string {
  const scenario = (presets[activeMethod][activePreset[activeMethod]]?.label ?? 'custom')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
  return `fire-replica-${activeMethod}-${scenario || 'custom'}-${presetId}-60fps-${durationSeconds}s.mp4`;
}

async function startOfflineExport(): Promise<void> {
  if (offlineExportRunning || resourceControlsLocked || !activeEngine) return;
  if (!isOfflineVideoSupported()) {
    offlineError.hidden = false;
    offlineError.textContent = 'This browser does not expose WebCodecs, so it cannot create a deterministic H.264 MP4.';
    return;
  }

  offlinePresetId = selectedOfflinePresetId();
  const preset = getOfflineVideoPreset(offlinePresetId);
  const durationSeconds = Math.max(
    2,
    Math.min(preset.maxDurationSeconds, Math.round(Number(offlineDurationInput.value) || 5)),
  );
  const engine = activeEngine;
  const method = activeMethod;
  const canvas = method === 'firex' ? gpuCanvas : glCanvas;
  const pausedBefore = paused;
  const fileName = offlineFileName(offlinePresetId, durationSeconds);
  const sceneSnapshot = currentOfflineSourceSummary();

  revokeOfflineDownload();
  offlineError.hidden = true;
  offlineError.textContent = '';
  offlineProgress.hidden = false;
  offlineProgressBar.value = 0;
  offlineProgressValue.textContent = 'Preparing exact output';
  offlineEtaValue.textContent = 'ETA measuring…';
  offlinePhaseStatus.textContent = `Preparing ${preset.width}×${preset.height} · ${preset.frameRate} fps cinematic render…`;
  offlineLastAnnouncedPhase = null;
  offlineRenderStartedAt = performance.now();
  offlineExportAbortController = new AbortController();
  offlineExportRunning = true;
  offlineExportSection.setAttribute('aria-busy', 'true');

  cancelAnimationFrame(animationFrame);
  animationFrame = 0;
  setOfflineMutationControlsLocked(true);

  try {
    const result = await renderOfflineVideo({
      canvas,
      preset,
      durationSeconds,
      signal: offlineExportAbortController.signal,
      prepare: async () => {
        offlinePhaseStatus.textContent = `Allocating the exact ${preset.label} solver profile…`;
        await engine.setOfflineRenderTier(preset.offlineTier);
        await engine.resizeOutput(preset.width, preset.height);
        offlineSolverValue.textContent = offlineSolverBudgetSummary(preset);
      },
      renderFrame: async (phase, _index, timing) => {
        const signal = offlineExportAbortController?.signal;
        if (phase === 'warmup') {
          await engine.renderOfflineFrame(1 / preset.frameRate, false, signal);
          return;
        }
        if (!timing) throw new Error('Offline capture timing was not provided.');
        return engine.renderOfflineVideoFrame(
          1 / preset.frameRate,
          timing.timestampSeconds,
          timing.durationSeconds,
          signal,
        );
      },
      onProgress: updateOfflineProgress,
    });

    offlineDownloadUrl = URL.createObjectURL(result.blob);
    offlinePlaybackPreviewUrl = URL.createObjectURL(result.playbackPreviewBlob);
    offlineSaveLink.href = offlineDownloadUrl;
    offlineSaveLink.download = fileName;
    offlineSaveLink.hidden = false;
    offlineCompatibilityLink.href = offlinePlaybackPreviewUrl;
    offlineCompatibilityLink.download = fileName.replace('-60fps-', '-playback-720p60-');
    offlineCompatibilityLink.hidden = false;
    offlineTestPlaybackButton.hidden = false;
    offlinePlaybackExpectedFrameRate = result.playbackPreviewVerification.frameRate;
    resetOfflinePlaybackHealth();
    offlinePreviewVideo.src = offlinePlaybackPreviewUrl;
    offlinePreviewVideo.load();
    offlinePreview.hidden = false;
    offlineCompletedSourceSummary = sceneSnapshot;
    const verified = result.verification;
    const previewVerified = result.playbackPreviewVerification;
    const playback = result.playbackCapability;
    const playbackCopy = playback === null
      ? 'This browser did not report a playback capability result.'
      : playback.supported && playback.smooth
        ? `This device predicts smooth${playback.powerEfficient ? ', power-efficient' : ''} playback for the finished tier.`
        : playback.supported
          ? 'This device predicts that local playback may still stutter at this tier.'
          : 'This device does not report support for local playback at this tier.';
    const cadenceErrorMs = Math.max(
      verified.maximumTimestampErrorSeconds,
      verified.maximumDurationErrorSeconds,
    ) * 1000;
    offlineSizeValue.textContent = `${formatOfflineBytes(result.blob.size)} · ${formatOfflineBitrate(verified.averageBitrate)} · H.264 MP4`;
    offlineVerifiedValue.textContent = `${verified.codedWidth}×${verified.codedHeight} · ${verified.frameRate.toFixed(2)} fps · ${verified.decodedFrameCount.toLocaleString('en-US')} decoded · temporal continuity passed · cadence error ≤ ${cadenceErrorMs.toFixed(3)} ms`;
    offlinePhaseStatus.textContent = `Master verification passed: ${verified.decodedFrameCount.toLocaleString('en-US')} decoded frames, exact 60 fps cadence, timeline and continuity audit passed, no decode-order reordering, and ${verified.keyFrameCount} verified keyframes. A separate ${previewVerified.codedWidth}×${previewVerified.codedHeight} · ${previewVerified.frameRate.toFixed(0)} fps playback proof also passed decoded verification. ${playbackCopy}`;
    offlinePlaybackStatus.textContent = `${previewVerified.codedWidth}×${previewVerified.codedHeight} · ${previewVerified.frameRate.toFixed(0)} fps playback proof ready. The saved master remains ${verified.codedWidth}×${verified.codedHeight} · ${verified.frameRate.toFixed(0)} fps.`;
  } catch (error) {
    if (isOfflineAbort(error)) {
      offlinePhaseStatus.textContent = 'Offline render canceled. Interactive playback is restored.';
    } else {
      const message = error instanceof Error ? error.message : String(error);
      console.error('Offline video render failed.', error);
      offlinePhaseStatus.textContent = 'Offline render stopped before a file was created.';
      offlineError.hidden = false;
      offlineError.textContent = message;
    }
  } finally {
    let restoredInteractiveSolver = false;
    try {
      await engine.setOfflineRenderTier(null);
      restoredInteractiveSolver = true;
    } catch (error) {
      console.error('Unable to restore the interactive solver after offline export.', error);
      offlineError.hidden = false;
      offlineError.textContent = 'The video was handled, but the interactive solver could not be restored. Reload the page before continuing.';
    }
    offlineExportAbortController = null;
    offlineExportRunning = false;
    offlineExportSection.removeAttribute('aria-busy');
    offlineProgress.hidden = true;
    if (restoredInteractiveSolver) {
      engine.setPaused(pausedBefore);
      setOfflineMutationControlsLocked(false);
      lastResize = null;
      resizeActiveEngine(true);
      previousTime = performance.now();
      updateOfflineExportPanel();
      if (!animationFrame) animationFrame = requestAnimationFrame(animate);
    } else {
      offlineCancelButton.disabled = true;
    }
  }
}

function isFireXGridPresetId(value: string): value is FireXGridPresetId {
  return FIREX_GRID_PRESETS.some((preset) => preset.id === value);
}

function currentFireXDomain(): readonly [number, number, number] {
  const value = (id: 'domainWidth' | 'domainHeight' | 'domainDepth', fallback: number): number =>
    controls.firex.find((control) => control.id === id)?.value ?? fallback;
  return [
    value('domainWidth', 1.15),
    value('domainHeight', 1.35),
    value('domainDepth', 1.15),
  ];
}

function selectedFireXGridDimensions(): FireXGridDimensions {
  const live = activeMethod === 'firex' ? activeEngine?.getGridInfo?.() : undefined;
  if (live) return live.dimensions;
  if (firexAppliedGridPreset === 'auto') return FIREX_QUALITY[quality].dimensions;
  if (firexAppliedGridPreset === 'custom') return firexCustomGrid;
  return FIREX_GRID_PRESETS.find((preset) => preset.id === firexAppliedGridPreset)?.dimensions
    ?? FIREX_QUALITY[quality].dimensions;
}

interface FireXGridDraft {
  dimensions: FireXGridDimensions | null;
  label: string;
  error: string | null;
}

function gridLabel(dimensions: readonly [number, number, number]): string {
  return dimensions.join('×');
}

function sameGrid(
  left: readonly [number, number, number],
  right: readonly [number, number, number],
): boolean {
  return left.every((value, index) => value === right[index]);
}

function customGridFromInputs(): FireXGridDraft {
  const raw = [gridXInput.value.trim(), gridYInput.value.trim(), gridZInput.value.trim()] as const;
  const values = raw.map((value) => Number(value)) as [number, number, number];
  const label = raw.map((value) => value || '—').join('×');
  if (raw.some((value) => value.length === 0) || values.some((value) => !Number.isSafeInteger(value))) {
    return { dimensions: null, label, error: 'Enter three whole-number voxel counts.' };
  }
  if (values.some((value) => value < 16)) {
    return { dimensions: null, label, error: 'Each dense-grid axis must be at least 16 voxels.' };
  }
  if (values.some((value) => value > 1024)) {
    return { dimensions: null, label, error: 'The request inspector accepts at most 1024 voxels per axis.' };
  }
  if (values.some((value) => value % 4 !== 0)) {
    return { dimensions: null, label, error: 'Each axis must be a multiple of 4; values are never silently rounded.' };
  }
  return { dimensions: [values[0], values[1], values[2]], label, error: null };
}

function requestedFireXGrid(draft: FireXGridDraft): FireXGridDimensions | null {
  if (firexGridPickerValue === 'custom') return draft.dimensions;
  if (firexGridPickerValue === 'auto') return FIREX_QUALITY[quality].dimensions;
  return FIREX_GRID_PRESETS.find((preset) => preset.id === firexGridPickerValue)?.dimensions ?? null;
}

function updateGridPanel(): void {
  gridPresetSelect.value = firexGridPickerValue;
  unitSystemSelect.value = firexUnitSystem;
  opticalDetailSelect.value = String(firexOpticalDetailTarget);
  gridBudgetSelect.value = String(firexGridMemoryBudgetBytes);

  const appliedDimensions = selectedFireXGridDimensions();
  const appliedInfo = deriveFireXGridInfo(
    appliedDimensions,
    currentFireXDomain(),
    FIREX_QUALITY[quality].particleCount,
  );
  const formatted = formatFireXGridInfo(appliedInfo, firexUnitSystem);
  const draft = customGridFromInputs();
  const requestedDimensions = requestedFireXGrid(draft);
  const requestEstimate = requestedDimensions
    ? estimateFireXGridMemory(requestedDimensions, FIREX_QUALITY[quality].particleCount)
    : null;
  const requestNeedsRebuild = requestedDimensions
    ? !sameGrid(appliedDimensions, requestedDimensions)
    : false;
  const rebuildPeakBytes = requestEstimate
    ? requestNeedsRebuild
      ? appliedInfo.totalBytes + requestEstimate.totalBytes
      : appliedInfo.totalBytes
    : null;
  const selectedPreset = FIREX_GRID_PRESETS.find((preset) => preset.id === firexGridPickerValue);
  const confirmationRequired = Boolean(selectedPreset?.requiresConfirmation && requestNeedsRebuild);
  const capabilities = activeMethod === 'firex' ? activeEngine?.getGridCapabilities?.() : null;
  const devicePreflight = requestedDimensions && capabilities
    ? preflightFireXGrid(requestedDimensions, capabilities, {
      particleCount: FIREX_QUALITY[quality].particleCount,
      memoryBudgetBytes: null,
    })
    : null;
  const deviceIssues = devicePreflight?.issues ?? [];
  const deviceBlocked = deviceIssues.length > 0;
  const guardBlocked = requestNeedsRebuild
    && rebuildPeakBytes !== null
    && rebuildPeakBytes > firexGridMemoryBudgetBytes;
  const rejectedRequestMatches = Boolean(
    firexRejectedGridRequest
    && firexRejectedGridRequest.presetId === firexGridPickerValue
    && (
      firexRejectedGridRequest.dimensions === null
      || (requestedDimensions && sameGrid(firexRejectedGridRequest.dimensions, requestedDimensions))
    ),
  );

  let gridState = 'Applied';
  let gridStateKind = 'applied';
  if (firexGridPickerValue === 'custom' && draft.error) {
    gridState = 'Invalid';
    gridStateKind = 'blocked';
  } else if (deviceBlocked) {
    gridState = 'GPU limit';
    gridStateKind = 'blocked';
  } else if (guardBlocked) {
    gridState = 'Raise guard';
    gridStateKind = 'blocked';
  } else if (rejectedRequestMatches) {
    gridState = 'Allocation failed';
    gridStateKind = 'blocked';
  } else if (requestNeedsRebuild) {
    gridState = 'Pending apply';
    gridStateKind = 'pending';
  }
  gridSelectionState.textContent = gridState;
  gridSelectionState.dataset.state = gridStateKind;

  const opticalActive = quality === 'cinematic' && firexOpticalDetailTarget > 0;
  opticalSelectionState.textContent = firexOpticalDetailTarget === 0
    ? 'Native'
    : opticalActive
      ? 'Active'
      : 'Stored · Cinematic only';
  opticalSelectionState.dataset.state = firexOpticalDetailTarget === 0 || opticalActive ? 'applied' : 'pending';

  const solverDependency = firexGridPickerValue === 'auto'
    ? 'Auto resolves its grid from Compute profile; a profile change may change the applied grid.'
    : 'A fixed grid keeps its voxel tuple across Compute profiles; the profile still changes particle, pressure and ray budgets and may require a rebuild.';
  const opticalDependency = firexOpticalDetailTarget === 0
    ? 'Render detail uses the native solver frequency.'
    : quality === 'cinematic'
      ? 'Cinematic detail is active and affects rendering only.'
      : 'Cinematic detail is stored but inactive until Compute profile is Cinematic.';
  resolutionHelp.textContent = `${solverDependency} The allocation guard only accepts or blocks rebuilds; it never downscales or resets a selection.`;
  opticalDetailHelp.textContent = `${opticalDependency} It never changes the dense solver grid.`;

  gridCustomEditor.hidden = firexGridPickerValue !== 'custom' && !requestNeedsRebuild;
  gridAxisFields.forEach((field) => {
    field.hidden = firexGridPickerValue !== 'custom';
  });

  required<HTMLElement>('#grid-resolution-value').textContent = `${formatted.resolution} voxels`;
  required<HTMLElement>('#grid-request-value').textContent = requestedDimensions
    ? `${gridLabel(requestedDimensions)} voxels`
    : draft.label;
  required<HTMLElement>('#grid-request-cost').textContent = requestEstimate
    ? `${requestEstimate.cellCount.toLocaleString('en-US')} cells · ~${formatFireXBytes(requestEstimate.totalBytes)}`
    : 'Invalid request';
  required<HTMLElement>('#grid-request-peak').textContent = rebuildPeakBytes === null
    ? '—'
    : requestNeedsRebuild
      ? `~${formatFireXBytes(rebuildPeakBytes)} · current + replacement`
      : `No rebuild · ${formatFireXBytes(rebuildPeakBytes)} current`;
  required<HTMLElement>('#grid-device-value').textContent = capabilities
    ? `≤${capabilities.maximumDenseCubeAxis}³ single vec4 field · ${formatFireXBytes(capabilities.maxStorageBufferBindingSize)} binding · ${formatFireXBytes(capabilities.maxBufferSize)} buffer · total guard separate`
    : 'Waiting for Fire-X WebGPU device';
  required<HTMLElement>('#grid-domain-value').textContent = `${formatted.domain} · ${formatted.volume}`;
  required<HTMLElement>('#grid-cell-value').textContent = `Δ ${formatted.voxelSize} / cell`;
  required<HTMLElement>('#grid-memory-value').textContent = `${formatted.cellCount} · ~${formatted.memory}`;

  let guidance = 'Resolution reallocates the solver on commit. Units are a nominal scene scale, not an engineering safety calibration.';
  let isError = false;
  if (firexGridStatusMessage) {
    guidance = firexGridStatusMessage;
    isError = guidance.includes('not applied') || guidance.includes('Invalid');
  } else if (firexGridPickerValue === 'custom' && draft.error) {
    guidance = `Invalid dense request — ${draft.error}`;
    isError = true;
  } else if (requestedDimensions && requestEstimate && deviceBlocked) {
    guidance = `Dense ${gridLabel(requestedDimensions)} is structurally unavailable on this GPU: ${deviceIssues.map((issue) => issue.message).join(' ')} The exact request remains visible and will not be downscaled.`;
    isError = true;
  } else if (requestedDimensions && requestEstimate && guardBlocked && rebuildPeakBytes !== null) {
    guidance = `Dense ${gridLabel(requestedDimensions)} needs ~${formatFireXBytes(requestEstimate.totalBytes)} persistent GPU memory and a ${formatFireXBytes(requestEstimate.largestStorageBufferBytes)} vector field; keeping the current grid alive during the safe swap raises peak allocation to ~${formatFireXBytes(rebuildPeakBytes)}, above the selected ${formatFireXBytes(firexGridMemoryBudgetBytes)} guard. Apply Auto first to reduce the live allocation or raise the guard; the request will never be downscaled.`;
    isError = true;
  } else if (requestedDimensions && requestEstimate && !requestNeedsRebuild) {
    guidance = `${gridLabel(requestedDimensions)} is the active real dense solver: ${requestEstimate.cellCount.toLocaleString('en-US')} simulated cells and ~${formatFireXBytes(requestEstimate.totalBytes)} persistent resources. No hidden rounding or virtual-grid substitution is active.`;
  } else if (requestedDimensions && requestEstimate && confirmationRequired) {
    guidance = `Real dense ${gridLabel(requestedDimensions)} is limit-compatible: ${requestEstimate.cellCount.toLocaleString('en-US')} simulated cells, ~${formatFireXBytes(requestEstimate.totalBytes)} persistent resources, a ${formatFireXBytes(requestEstimate.largestStorageBufferBytes)} vector field and ~${formatFireXBytes(rebuildPeakBytes ?? requestEstimate.totalBytes)} transactional peak. This workstation/reference tier may run far below realtime or exhaust available VRAM even when structural limits pass. Apply is an explicit opt-in.`;
  } else if (selectedPreset?.experimental && requestEstimate) {
    guidance = `${selectedPreset.label}: ${requestEstimate.cellCount.toLocaleString('en-US')} real simulated cells and ~${formatFireXBytes(requestEstimate.totalBytes)} persistent resources. Expect a severe frame-rate cost.`;
  } else if (requestedDimensions && requestEstimate && firexGridPickerValue === 'custom') {
    guidance = 'Exact request ready. Apply checks this tuple against the negotiated GPU buffer, texture and transactional-memory limits; it is never silently reduced.';
  }
  const guidanceElement = required<HTMLElement>('#grid-guidance');
  if (guidanceElement.textContent !== guidance) guidanceElement.textContent = guidance;
  guidanceElement.classList.toggle('is-error', isError);
  gridApplyButton.textContent = deviceBlocked
    ? 'Unavailable on this GPU'
    : guardBlocked
      ? 'Raise peak guard to apply'
      : rejectedRequestMatches
        ? 'Retry allocation'
      : !requestNeedsRebuild
        ? 'Dense grid already applied'
        : requestedDimensions
          ? `Apply real ${gridLabel(requestedDimensions)} solver`
          : 'Check & apply dense grid';
  const requestBlocked = !requestedDimensions || !requestNeedsRebuild || deviceBlocked || guardBlocked;
  gridApplyButton.dataset.requestBlocked = String(requestBlocked);
  gridApplyButton.disabled = resourceControlsLocked || requestBlocked;
}

function setResourceControlsLocked(locked: boolean): void {
  resourceControlsLockDepth = Math.max(0, resourceControlsLockDepth + (locked ? 1 : -1));
  resourceControlsLocked = resourceControlsLockDepth > 0;
  qualitySelect.disabled = resourceControlsLocked;
  gridPresetSelect.disabled = resourceControlsLocked;
  opticalDetailSelect.disabled = resourceControlsLocked;
  gridBudgetSelect.disabled = resourceControlsLocked;
  gridApplyButton.disabled = resourceControlsLocked || gridApplyButton.dataset.requestBlocked === 'true';
  gridXInput.disabled = resourceControlsLocked;
  gridYInput.disabled = resourceControlsLocked;
  gridZInput.disabled = resourceControlsLocked;
  document.querySelectorAll<HTMLButtonElement>('[data-method]').forEach((button) => {
    button.disabled = resourceControlsLocked;
  });
  if (!offlineExportRunning) updateOfflineExportPanel();
}

async function applyFireXGrid(presetId: FireXGridPresetId, dimensions: FireXGridDimensions | null): Promise<void> {
  if (resourceControlsLocked) return;
  const previousPreset = firexAppliedGridPreset;
  firexGridStatusMessage = null;
  firexRejectedGridRequest = null;
  setResourceControlsLocked(true);
  try {
    await activeEngine?.setGridDimensions?.(dimensions);
    const liveDimensions = activeEngine?.getGridInfo?.().dimensions;
    if (dimensions && liveDimensions && !sameGrid(dimensions, liveDimensions)) {
      throw new Error(`The engine reported ${gridLabel(liveDimensions)} instead of the exact ${gridLabel(dimensions)} request.`);
    }
    firexAppliedGridPreset = presetId;
    firexGridPickerValue = presetId;
    if (presetId === 'custom' && dimensions) {
      firexCustomGrid = dimensions;
      firexGridStatusMessage = `Applied exact dense ${gridLabel(dimensions)}. No axis was rounded or clamped.`;
    } else if (dimensions && FIREX_GRID_PRESETS.find((preset) => preset.id === presetId)?.requiresConfirmation) {
      const estimate = estimateFireXGridMemory(dimensions, FIREX_QUALITY[quality].particleCount);
      firexGridStatusMessage = `Applied real dense ${gridLabel(dimensions)} — ${estimate.cellCount.toLocaleString('en-US')} simulated cells and ~${formatFireXBytes(estimate.totalBytes)} persistent resources. This workstation/reference tier is active and may run far below realtime.`;
    }
    if (activeEngine) required<HTMLElement>('#detail-value').textContent = activeEngine.detail;
    required<HTMLElement>('#capability-note').innerHTML = '<strong>Research previews, not safety models.</strong> Each path reproduces selected paper mechanisms at a browser-scale budget. Diagnostic fields expose what is actually simulated.';
  } catch (error) {
    firexAppliedGridPreset = previousPreset;
    firexGridPickerValue = presetId;
    firexRejectedGridRequest = { presetId, dimensions };
    const message = error instanceof Error ? error.message : String(error);
    const currentDimensions = activeEngine?.getGridInfo?.().dimensions ?? selectedFireXGridDimensions();
    if (dimensions) {
      const estimate = estimateFireXGridMemory(dimensions, FIREX_QUALITY[quality].particleCount);
      firexGridStatusMessage = `Dense ${gridLabel(dimensions)} not applied — ${estimate.cellCount.toLocaleString('en-US')} cells need ~${formatFireXBytes(estimate.totalBytes)} persistent GPU memory and a ${formatFireXBytes(estimate.largestStorageBufferBytes)} vector field. ${message} Current solver remains ${gridLabel(currentDimensions)}.`;
    } else {
      firexGridStatusMessage = `Selected grid not applied — ${message}`;
    }
    required<HTMLElement>('#capability-note').textContent = 'The dense-grid request was rejected before replacing the working solver. See Compute & resolution for the exact requirement and current applied grid.';
  } finally {
    setResourceControlsLocked(false);
  }
  updateGridPanel();
  updateVerificationPanel();
}

function isQualityLevel(value: string): value is QualityLevel {
  return (QUALITY_LEVELS as readonly string[]).includes(value);
}

function setRangeFill(input: HTMLInputElement): void {
  const min = Number(input.min);
  const max = Number(input.max);
  const percent = ((Number(input.value) - min) / Math.max(max - min, 1e-6)) * 100;
  input.style.setProperty('--fill', `${percent}%`);
}

function setWaterEnabled(enabled: boolean): void {
  waterButton.setAttribute('aria-pressed', String(enabled));
  waterButton.classList.toggle('is-active', enabled);
  suppressionEnabledInput.checked = enabled;
}

function markScenarioCustom(): void {
  activePreset[activeMethod] = -1;
  document.querySelectorAll('.preset-button').forEach((button) => button.classList.remove('is-active'));
  updateVerificationPanel();
}

function updateLegend(): void {
  const diagnosticActive = activeMethod === 'firex' && firexViewMode !== 0;
  fieldLegendSection.hidden = diagnosticActive;
  if (diagnosticActive) return;

  legendTitle.textContent = activeMethod === 'firex' ? 'Fire-X beauty legend' : 'Slice beauty legend';
  legendContext.textContent = activeMethod === 'firex' ? 'Composite' : 'Display';
  legendItems.innerHTML = beautyLegends[activeMethod]
    .map((item) => `<div class="legend-item"><span class="legend-swatch" style="--color:${item.color}"></span>${item.label}</div>`)
    .join('');
}

function updateVerificationPanel(): void {
  const view = firexFieldViews.find((candidate) => candidate.value === firexViewMode) ?? firexFieldViews[0];
  required<HTMLElement>('#verification-view').textContent = view.label;
  required<HTMLElement>('#verification-help').textContent = view.help;
  const preset = presets.firex[activePreset.firex];
  required<HTMLElement>('#verification-expected').textContent = preset?.expected
    ?? 'Custom scenario: compare Beauty with Temperature, Reaction and Liquid / velocity before drawing a conclusion.';
  required<HTMLElement>('#verification-profile').textContent = activeMethod === 'firex' && activeEngine
    ? `${quality.toUpperCase()} · ${activeEngine.detail}`
    : `${quality.toUpperCase()} · Fire-X engine inactive`;
  updateLegend();
  updateOfflineExportPanel();
}

function supplementalControlCount(method: MethodId, group: ControlDefinition['group']): number {
  if (method !== 'firex') return 0;
  if (group === 'water') return 2;
  return 0;
}

function renderControlGroupList(
  list: HTMLElement,
  groupOrder: ControlDefinition['group'][],
): void {
  const renderedMethod = activeMethod;
  list.innerHTML = groupOrder
    .filter((group) => controls[renderedMethod].some((control) => control.group === group))
    .map((group) => {
      const definitions = controls[renderedMethod].filter((control) => control.group === group);
      const groupCopy = controlGroupCopy[renderedMethod][group];
      const expanded = openControlGroups[renderedMethod].has(group);
      const count = definitions.length + supplementalControlCount(renderedMethod, group);
      return `
        <details class="control-group" data-control-group="${group}"${expanded ? ' open' : ''}>
          <summary>
            <span>${groupCopy.label}</span>
            <span class="control-count">${count}</span>
          </summary>
          <p class="control-group-hint">${groupCopy.hint}</p>
          <div class="control-group-list">
            ${definitions.map((control) => `
              <label class="range-control" for="control-${control.id}">
                <span class="range-label">${control.label}</span>
                <output class="range-value" id="value-${control.id}">${formatControlValue(control, control.value)}</output>
                <input id="control-${control.id}" data-parameter="${control.id}" type="range" min="${control.min}" max="${control.max}" step="${control.step}" value="${control.value}" aria-description="${control.description ?? ''}" />
                <span class="range-endpoints" aria-hidden="true"><span>${formatControlValue(control, control.min)}</span><span>${formatControlValue(control, control.max)}</span></span>
              </label>`).join('')}
          </div>
        </details>`;
    })
    .join('');

  list.querySelectorAll<HTMLDetailsElement>('[data-control-group]').forEach((details) => {
    details.addEventListener('toggle', () => {
      const group = details.dataset.controlGroup as ControlDefinition['group'] | undefined;
      if (!group) return;
      if (details.open) openControlGroups[renderedMethod].add(group);
      else openControlGroups[renderedMethod].delete(group);
    });
  });
}

function bindRangeControls(): void {
  controlPanel.querySelectorAll<HTMLInputElement>('input[type="range"][data-parameter]').forEach((input) => {
    setRangeFill(input);
    input.addEventListener('input', () => {
      const definition = controls[activeMethod].find((item) => item.id === input.dataset.parameter);
      if (!definition) return;
      definition.value = Number(input.value);
      required<HTMLOutputElement>(`#value-${definition.id}`).value = formatControlValue(definition, definition.value);
      setRangeFill(input);
      if (!definition.commitOnly) activeEngine?.setParameter(definition.id, definition.value);
      if (definition.quantity === 'length') updateGridPanel();
      markScenarioCustom();
    });
    input.addEventListener('change', () => {
      const definition = controls[activeMethod].find((item) => item.id === input.dataset.parameter);
      if (definition?.commitOnly) activeEngine?.setParameter(definition.id, definition.value);
      if (definition?.resetOnCommit) activeEngine?.reset();
      if (definition?.quantity === 'length') updateGridPanel();
    });
  });
}

function updateLengthControlReadouts(): void {
  controls.firex
    .filter((control) => control.quantity === 'length')
    .forEach((control) => {
      const label = document.querySelector<HTMLLabelElement>(`label[for="control-${control.id}"]`);
      const output = document.querySelector<HTMLOutputElement>(`#value-${control.id}`);
      if (output) output.value = formatControlValue(control, control.value);
      const endpoints = label?.querySelectorAll<HTMLElement>('.range-endpoints span');
      if (endpoints?.[0]) endpoints[0].textContent = formatControlValue(control, control.min);
      if (endpoints?.[1]) endpoints[1].textContent = formatControlValue(control, control.max);
    });
}

function renderControls(): void {
  const copy = methodCopy[activeMethod];
  qualityStatus.textContent = '';
  qualityStatus.classList.remove('is-error');
  required<HTMLElement>('#view-eyebrow').textContent = copy.year;
  required<HTMLElement>('#view-title').textContent = copy.title;
  required<HTMLElement>('#view-subtitle').textContent = copy.subtitle;
  required<HTMLElement>('#paper-year').textContent = copy.year;
  required<HTMLElement>('#paper-title').textContent = copy.paper;
  required<HTMLElement>('#paper-detail').textContent = copy.detail;

  document.querySelectorAll<HTMLButtonElement>('[data-method]').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.method === activeMethod);
  });

  const presetRow = required<HTMLElement>('#preset-row');
  presetRow.innerHTML = presets[activeMethod]
    .map((preset, index) => `<button class="preset-button${index === activePreset[activeMethod] ? ' is-active' : ''}" type="button" data-preset="${index}">${preset.label}</button>`)
    .join('');
  presetRow.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach((button) => {
    button.addEventListener('click', () => applyPreset(Number(button.dataset.preset)));
  });

  methodControlPool.append(suppressionEnabledControl, nozzleTypeControl);
  renderControlGroupList(domainControlsList, domainControlGroupOrder);
  renderControlGroupList(simulationControlsList, simulationControlGroupOrder);
  renderControlGroupList(renderingControlsList, renderingControlGroupOrder);

  if (activeMethod === 'firex') {
    simulationControlsList
      .querySelector<HTMLElement>('[data-control-group="water"] .control-group-list')
      ?.prepend(suppressionEnabledControl, nozzleTypeControl);
  }
  bindRangeControls();

  waterButton.hidden = activeMethod !== 'firex';
  firexResolutionControls.hidden = activeMethod !== 'firex';
  opticalDetailControl.hidden = activeMethod !== 'firex';
  opticalDetailHelp.hidden = activeMethod !== 'firex';
  firexGridSection.hidden = activeMethod !== 'firex';
  firexInspectionSection.hidden = activeMethod !== 'firex';
  qualitySelect.value = quality;
  fieldViewSelect.value = String(firexViewMode);
  nozzleTypeSelect.value = String(firexNozzleType);
  updateGridPanel();
  updateVerificationPanel();
}

function applyPreset(index: number): void {
  const preset = presets[activeMethod][index];
  if (!preset) return;
  for (const definition of controls[activeMethod]) {
    const value = preset.values[definition.id];
    if (typeof value !== 'number') continue;
    definition.value = value;
  }
  if (activeMethod === 'firex') {
    const nozzleType = preset.values.nozzleType;
    const waterEnabled = preset.values.waterEnabled;
    if (typeof nozzleType === 'number') firexNozzleType = nozzleType;
    if (typeof waterEnabled === 'boolean') {
      setWaterEnabled(waterEnabled);
      if (waterEnabled) openControlGroups.firex.add('water');
    }
  } else {
    const sourceMode = preset.values.sourceMode;
    if (typeof sourceMode === 'number') horvathSourceMode = Math.max(0, Math.min(2, Math.round(sourceMode)));
  }
  for (const [name, value] of Object.entries(preset.values)) {
    activeEngine?.setParameter(name, value);
  }
  activePreset[activeMethod] = index;
  renderControls();
  activeEngine?.reset();
}

async function pushAllParameters(): Promise<void> {
  for (const parameter of controls[activeMethod]) {
    activeEngine?.setParameter(parameter.id, parameter.value);
  }
  if (activeMethod === 'firex') {
    activeEngine?.setParameter('waterEnabled', waterButton.getAttribute('aria-pressed') === 'true');
    activeEngine?.setParameter('viewMode', firexViewMode);
    activeEngine?.setParameter('nozzleType', firexNozzleType);
    activeEngine?.setParameter('displayImperial', firexUnitSystem === 'imperial');
    activeEngine?.setParameter('opticalDetailTarget', firexOpticalDetailTarget);
    activeEngine?.setGridMemoryBudget?.(firexGridMemoryBudgetBytes);
    const fixedPreset = FIREX_GRID_PRESETS.find((preset) => preset.id === firexAppliedGridPreset);
    const dimensions = firexAppliedGridPreset === 'auto'
      ? null
      : firexAppliedGridPreset === 'custom'
        ? firexCustomGrid
        : fixedPreset?.dimensions ?? null;
    await activeEngine?.setGridDimensions?.(dimensions);
  } else {
    activeEngine?.setParameter('sourceMode', horvathSourceMode);
  }
  activeEngine?.setParameter('cameraYaw', cameraYaw);
  activeEngine?.setParameter('cameraPitch', cameraPitch);
  activeEngine?.setParameter('cameraDistance', cameraDistance);
}

async function createEngine(method: MethodId): Promise<SimulationEngine> {
  if (engines[method]) return engines[method];
  if (enginePromises[method]) return enginePromises[method];
  const promise = (async (): Promise<SimulationEngine> => {
    const engine: SimulationEngine = method === 'horvath' ? new HorvathEngine(glCanvas) : new FireXEngine(gpuCanvas);
    try {
      await engine.initialize();
      engines[method] = engine;
      return engine;
    } catch (error) {
      engine.dispose();
      throw error;
    }
  })();
  enginePromises[method] = promise;
  try {
    return await promise;
  } catch (error) {
    delete enginePromises[method];
    delete engines[method];
    throw error;
  }
}

function disposeEngine(method: MethodId): void {
  const engine = engines[method];
  if (engine) engine.dispose();
  delete engines[method];
  delete enginePromises[method];
}

async function switchMethod(method: MethodId): Promise<void> {
  if (offlineExportRunning) return;
  if (method === activeMethod && activeEngine && requestedMethod === method) return;
  revokeOfflineDownload();
  setResourceControlsLocked(true);
  requestedMethod = method;
  const generation = ++switchGeneration;
  document.querySelectorAll<HTMLButtonElement>('[data-method]').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.method === method);
  });
  loadingCard.hidden = false;
  loadingTitle.textContent = method === 'firex' ? 'Compiling Fire-X compute graph' : 'Compiling slice refinement solver';
  loadingDetail.textContent = method === 'firex' ? 'Allocating 3D species fields and SPH neighborhoods…' : 'Allocating layered half-float slabs and projection buffers…';

  try {
    const nextEngine = await createEngine(method);
    if (generation !== switchGeneration || requestedMethod !== method) {
      if (requestedMethod !== method && engines[method] === nextEngine) disposeEngine(method);
      return;
    }
    const previousMethod = activeMethod;
    const previousEngine = activeEngine;
    activeMethod = method;
    activeEngine = nextEngine;
    if (method !== 'firex') firexViewMode = 0;
    renderControls();
    await activeEngine.setQuality(quality);
    activeEngine.setPaused(paused);
    await pushAllParameters();
    // Engines allocate deterministic initial fields during initialization and
    // quality changes. Re-seed once after the selected scenario parameters are
    // applied so startup state, controls, and the highlighted preset agree.
    activeEngine.reset();
    glCanvas.hidden = method !== 'horvath';
    gpuCanvas.hidden = method !== 'firex';
    required<HTMLElement>('#backend-chip').textContent = activeEngine.backend;
    required<HTMLElement>('#detail-value').textContent = activeEngine.detail;
    updateGridPanel();
    loadingCard.hidden = true;
    scheduleResize(true);
    if (previousEngine && previousEngine !== nextEngine) disposeEngine(previousMethod);
  } catch (error) {
    if (generation !== switchGeneration || requestedMethod !== method) return;
    const message = error instanceof Error ? error.message : String(error);
    loadingTitle.textContent = method === 'firex' ? 'WebGPU path unavailable' : 'WebGL2 path unavailable';
    loadingDetail.textContent = message;
    if (method === 'firex') {
      window.setTimeout(() => {
        if (generation === switchGeneration && requestedMethod === 'firex') void switchMethod('horvath');
      }, 1800);
    }
  } finally {
    setResourceControlsLocked(false);
  }
}

function resizeActiveEngine(force = false): void {
  if (!activeEngine || offlineExportRunning) return;
  const rect = viewport.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio, 2);
  if (
    !force &&
    lastResize?.engine === activeEngine &&
    lastResize.width === rect.width &&
    lastResize.height === rect.height &&
    lastResize.pixelRatio === pixelRatio
  ) return;
  activeEngine.resize(rect.width, rect.height, pixelRatio);
  lastResize = { engine: activeEngine, width: rect.width, height: rect.height, pixelRatio };
}

function scheduleResize(force = false): void {
  if (offlineExportRunning) return;
  if (resizeQueued && !force) return;
  resizeQueued = true;
  requestAnimationFrame(() => {
    resizeQueued = false;
    resizeActiveEngine(force);
  });
}

function animate(now: number): void {
  if (offlineExportRunning) {
    previousTime = now;
    animationFrame = requestAnimationFrame(animate);
    return;
  }
  const renderDeltaMs = Math.max(now - previousTime, 0.01);
  const delta = Math.min(renderDeltaMs / 1000, 1 / 20);
  previousTime = now;
  const simulationMs = activeEngine?.frame(now / 1000, delta) ?? 0;
  smoothedFrameMs += (renderDeltaMs - smoothedFrameMs) * 0.08;
  smoothedSimMs += (simulationMs - smoothedSimMs) * 0.08;

  if (now - lastStatsUpdate > 250) {
    required<HTMLElement>('#fps-value').textContent = `${Math.round(1000 / Math.max(smoothedFrameMs, 1))} FPS`;
    required<HTMLElement>('#frame-value').textContent = `${smoothedFrameMs.toFixed(1)} ms`;
    required<HTMLElement>('#sim-value').textContent = `${smoothedSimMs.toFixed(1)} ms`;
    if (activeEngine) {
      required<HTMLElement>('#detail-value').textContent = activeEngine.detail;
      if (activeMethod === 'firex') {
        required<HTMLElement>('#verification-profile').textContent = `${quality.toUpperCase()} · ${activeEngine.detail}`;
      }
    }
    lastStatsUpdate = now;
  }
  animationFrame = requestAnimationFrame(animate);
}

const mobilePanelMedia = window.matchMedia('(max-width: 900px)');

function syncControlPanelState(): void {
  const isMobile = mobilePanelMedia.matches;
  if (!isMobile) controlPanel.classList.remove('is-open');
  const isOpen = isMobile && controlPanel.classList.contains('is-open');
  panelButton.setAttribute('aria-expanded', String(isOpen));
  if (isMobile && !isOpen && controlPanel.contains(document.activeElement)) panelButton.focus();
  controlPanel.inert = isMobile && !isOpen;
  if (isMobile && !isOpen) controlPanel.setAttribute('aria-hidden', 'true');
  else controlPanel.removeAttribute('aria-hidden');
}

function closeMobileControlPanel(restoreFocus = false): void {
  if (!mobilePanelMedia.matches || !controlPanel.classList.contains('is-open')) return;
  controlPanel.classList.remove('is-open');
  syncControlPanelState();
  if (restoreFocus) panelButton.focus();
}

offlinePresetSelect.addEventListener('change', () => {
  offlinePresetId = selectedOfflinePresetId();
  revokeOfflineDownload();
  updateOfflineExportPanel();
});

offlineDurationInput.addEventListener('input', () => {
  revokeOfflineDownload();
  updateOfflineExportPanel();
});

offlineRenderButton.addEventListener('click', () => {
  void startOfflineExport();
});

offlineTestPlaybackButton.addEventListener('click', () => {
  void (async () => {
    try {
      resetOfflinePlaybackHealth();
      offlinePreviewVideo.currentTime = 0;
      await offlinePreviewVideo.play();
    } catch {
      offlinePlaybackStatus.textContent = 'The browser blocked automatic preview playback. Use the video Play control to run the test.';
    }
  })();
});

offlineCancelButton.addEventListener('click', () => {
  if (!offlineExportAbortController || offlineExportAbortController.signal.aborted) return;
  offlineCancelButton.disabled = true;
  offlinePhaseStatus.textContent = 'Cancel requested. Finishing the current GPU frame…';
  offlineExportAbortController.abort();
});

offlinePreviewVideo.addEventListener('playing', () => {
  suspendInteractiveRenderingForPlayback();
  resumeOfflinePlaybackMeasurement();
});

offlinePreviewVideo.addEventListener('pause', () => {
  finishOfflinePlaybackStall();
  stopOfflinePlaybackMeasurement();
  renderOfflinePlaybackHealth();
  resumeInteractiveRenderingAfterPlayback();
});

offlinePreviewVideo.addEventListener('ended', () => {
  finishOfflinePlaybackStall();
  stopOfflinePlaybackMeasurement();
  renderOfflinePlaybackHealth();
  resumeInteractiveRenderingAfterPlayback();
});

offlinePreviewVideo.addEventListener('seeking', () => {
  resetOfflinePlaybackHealth();
  offlinePlaybackStatus.textContent = 'Seeking. Playback measurement restarts when the preview resumes.';
});

offlinePreviewVideo.addEventListener('waiting', () => {
  const measurement = offlinePlaybackMeasurement;
  if (measurement && measurement.stallStartedAt === null) {
    measurement.stallCount += 1;
    measurement.stallStartedAt = performance.now();
  }
  offlinePlaybackStatus.textContent = 'Local playback stalled. The stall remains included when playback resumes.';
});

offlinePreviewVideo.addEventListener('error', () => {
  if (!offlinePreviewVideo.hasAttribute('src')) return;
  stopOfflinePlaybackMeasurement();
  resumeInteractiveRenderingAfterPlayback();
  offlinePlaybackStatus.textContent = 'This browser could not preview the verified MP4. The Save link remains available.';
});

offlinePreviewVideo.addEventListener('emptied', () => {
  resetOfflinePlaybackHealth();
  resumeInteractiveRenderingAfterPlayback();
});

document.querySelectorAll<HTMLButtonElement>('[data-method]').forEach((button) => {
  button.addEventListener('click', () => void switchMethod(button.dataset.method as MethodId));
});

pauseButton.addEventListener('click', () => {
  paused = !paused;
  pauseButton.textContent = paused ? '▶' : 'Ⅱ';
  pauseButton.setAttribute('aria-label', paused ? 'Resume simulation' : 'Pause simulation');
  activeEngine?.setPaused(paused);
});

required<HTMLButtonElement>('#reset-button').addEventListener('click', () => activeEngine?.reset());
panelButton.addEventListener('click', () => {
  controlPanel.classList.toggle('is-open');
  syncControlPanelState();
  if (controlPanel.classList.contains('is-open')) {
    window.requestAnimationFrame(() => {
      controlPanel
        .querySelector<HTMLElement>('button:not([disabled]), select:not([disabled]), input:not([disabled]), summary')
        ?.focus();
    });
  }
});
mobilePanelMedia.addEventListener('change', syncControlPanelState);
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeMobileControlPanel(true);
});

waterButton.addEventListener('click', () => {
  const enabled = waterButton.getAttribute('aria-pressed') !== 'true';
  setWaterEnabled(enabled);
  activeEngine?.setParameter('waterEnabled', enabled);
  if (enabled) openControlGroups.firex.add('water');
  markScenarioCustom();
  if (enabled) {
    const waterGroup = simulationControlsList.querySelector<HTMLDetailsElement>('[data-control-group="water"]');
    if (waterGroup) waterGroup.open = true;
  }
});

suppressionEnabledInput.addEventListener('change', () => {
  const enabled = suppressionEnabledInput.checked;
  setWaterEnabled(enabled);
  activeEngine?.setParameter('waterEnabled', enabled);
  markScenarioCustom();
});

fieldViewSelect.addEventListener('change', () => {
  firexViewMode = Number(fieldViewSelect.value);
  if (!firexFieldViews.some((view) => view.value === firexViewMode)) firexViewMode = 0;
  activeEngine?.setParameter('viewMode', firexViewMode);
  updateVerificationPanel();
});

nozzleTypeSelect.addEventListener('change', () => {
  firexNozzleType = Number(nozzleTypeSelect.value) >= 0.5 ? 1 : 0;
  activeEngine?.setParameter('nozzleType', firexNozzleType);
  markScenarioCustom();
});

gridPresetSelect.addEventListener('change', () => {
  const next = gridPresetSelect.value;
  if (!isFireXGridPresetId(next)) {
    gridPresetSelect.value = firexGridPickerValue;
    return;
  }
  if (next === 'custom') {
    firexGridPickerValue = 'custom';
    const dimensions = selectedFireXGridDimensions();
    gridXInput.value = String(dimensions[0]);
    gridYInput.value = String(dimensions[1]);
    gridZInput.value = String(dimensions[2]);
    firexGridStatusMessage = null;
    firexRejectedGridRequest = null;
    updateGridPanel();
    return;
  }
  const dimensions = next === 'auto'
    ? null
    : FIREX_GRID_PRESETS.find((preset) => preset.id === next)?.dimensions ?? null;
  const preset = FIREX_GRID_PRESETS.find((candidate) => candidate.id === next);
  if (preset?.requiresConfirmation) {
    firexGridPickerValue = next;
    firexGridStatusMessage = null;
    firexRejectedGridRequest = null;
    updateGridPanel();
    return;
  }
  void applyFireXGrid(next, dimensions);
});

unitSystemSelect.addEventListener('change', () => {
  firexUnitSystem = unitSystemSelect.value === 'imperial' ? 'imperial' : 'metric';
  activeEngine?.setParameter('displayImperial', firexUnitSystem === 'imperial');
  updateLengthControlReadouts();
  updateGridPanel();
  if (activeEngine) required<HTMLElement>('#detail-value').textContent = activeEngine.detail;
});

opticalDetailSelect.addEventListener('change', () => {
  const target = Number(opticalDetailSelect.value);
  firexOpticalDetailTarget = target >= 768 ? 1024 : target >= 384 ? 512 : target >= 192 ? 256 : 0;
  activeEngine?.setParameter('opticalDetailTarget', firexOpticalDetailTarget);
  updateGridPanel();
});

gridBudgetSelect.addEventListener('change', () => {
  const rejected = firexRejectedGridRequest;
  const selectedGuard = FIREX_GRID_MEMORY_GUARDS.find((guard) => guard.bytes === Number(gridBudgetSelect.value));
  if (!selectedGuard) {
    gridBudgetSelect.value = String(firexGridMemoryBudgetBytes);
    return;
  }
  firexGridMemoryBudgetBytes = selectedGuard.bytes;
  activeEngine?.setGridMemoryBudget?.(firexGridMemoryBudgetBytes);
  firexGridStatusMessage = null;
  firexRejectedGridRequest = null;
  const rejectedPreset = rejected
    ? FIREX_GRID_PRESETS.find((preset) => preset.id === rejected.presetId)
    : null;
  if (rejected && rejected.presetId !== 'custom' && !rejectedPreset?.requiresConfirmation) {
    void applyFireXGrid(rejected.presetId, rejected.dimensions);
    return;
  }
  updateGridPanel();
});

[gridXInput, gridYInput, gridZInput].forEach((input) => {
  input.addEventListener('input', () => {
    firexGridStatusMessage = null;
    firexRejectedGridRequest = null;
    updateGridPanel();
  });
});

gridApplyButton.addEventListener('click', () => {
  const draft = customGridFromInputs();
  const dimensions = requestedFireXGrid(draft);
  if (!dimensions) {
    firexGridStatusMessage = `Invalid dense request — ${draft.error ?? 'enter three valid dimensions.'}`;
    updateGridPanel();
    return;
  }
  void applyFireXGrid(firexGridPickerValue, dimensions);
});

qualitySelect.addEventListener('change', async () => {
  if (resourceControlsLocked) {
    qualitySelect.value = quality;
    qualityStatus.textContent = 'Wait for the current GPU rebuild to finish.';
    qualityStatus.classList.add('is-error');
    return;
  }
  const nextQuality = qualitySelect.value;
  if (!isQualityLevel(nextQuality)) {
    qualitySelect.value = quality;
    qualityStatus.textContent = 'That compute profile is not available.';
    qualityStatus.classList.add('is-error');
    return;
  }
  const previousQuality = quality;
  qualityStatus.textContent = `Applying ${nextQuality} compute profile…`;
  qualityStatus.classList.remove('is-error');
  setResourceControlsLocked(true);
  try {
    await activeEngine?.setQuality(nextQuality);
    quality = nextQuality;
    firexGridStatusMessage = null;
    firexRejectedGridRequest = null;
    qualityStatus.textContent = `${nextQuality[0].toUpperCase()}${nextQuality.slice(1)} compute profile active.`;
  } catch (error) {
    quality = previousQuality;
    qualitySelect.value = previousQuality;
    try {
      await activeEngine?.setQuality(previousQuality);
    } catch {
      // The engine's existing unavailable-state UI remains the final fallback.
    }
    const message = error instanceof Error ? error.message : String(error);
    qualityStatus.textContent = `Compute profile unavailable on this device: ${message}`;
    qualityStatus.classList.add('is-error');
  } finally {
    setResourceControlsLocked(false);
  }
  required<HTMLElement>('#quality-chip').textContent = quality.toUpperCase();
  scheduleResize(true);
  updateGridPanel();
  updateVerificationPanel();
});

let dragPointer = -1;
let dragX = 0;
let dragY = 0;
viewport.addEventListener('pointerdown', (event) => {
  if (offlineExportRunning || !event.isPrimary || event.button !== 0) return;
  dragPointer = event.pointerId;
  dragX = event.clientX;
  dragY = event.clientY;
  viewport.setPointerCapture(event.pointerId);
  viewport.classList.add('is-dragging');
});
viewport.addEventListener('pointermove', (event) => {
  if (offlineExportRunning || event.pointerId !== dragPointer) return;
  cameraYaw += (event.clientX - dragX) * 0.006;
  cameraPitch = Math.max(-0.7, Math.min(0.7, cameraPitch + (event.clientY - dragY) * 0.004));
  dragX = event.clientX;
  dragY = event.clientY;
  activeEngine?.setParameter('cameraYaw', cameraYaw);
  activeEngine?.setParameter('cameraPitch', cameraPitch);
});
viewport.addEventListener('pointerup', (event) => {
  if (event.pointerId === dragPointer) dragPointer = -1;
  viewport.classList.remove('is-dragging');
});
const cancelDrag = (event: PointerEvent): void => {
  if (event.pointerId === dragPointer) dragPointer = -1;
  viewport.classList.remove('is-dragging');
};
viewport.addEventListener('pointercancel', cancelDrag);
viewport.addEventListener('lostpointercapture', cancelDrag);
viewport.addEventListener('wheel', (event) => {
  event.preventDefault();
  if (offlineExportRunning) return;
  cameraDistance = Math.max(2.5, Math.min(8, cameraDistance + event.deltaY * 0.004));
  activeEngine?.setParameter('cameraDistance', cameraDistance);
}, { passive: false });

new ResizeObserver(() => scheduleResize()).observe(viewport);
window.addEventListener('beforeunload', () => {
  offlineExportAbortController?.abort();
  revokeOfflineDownload();
  cancelAnimationFrame(animationFrame);
  Object.values(engines).forEach((engine) => engine?.dispose());
});

renderControls();
syncControlPanelState();
if (!navigator.gpu) {
  required<HTMLElement>('#capability-note').innerHTML = '<strong>WebGPU was not detected.</strong> The WebGL2 slice solver remains available; Fire-X requires a browser/device with WebGPU enabled.';
}
void switchMethod(activeMethod);
animationFrame = requestAnimationFrame(animate);
