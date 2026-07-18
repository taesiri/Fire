# Fire Replica Lab

A local GPU lab containing two **realtime research previews** inspired by fire-simulation papers:




https://github.com/user-attachments/assets/5dc056ed-3a68-4115-8f0d-3560b947ac31




## Run locally

```powershell
npm install
npm run dev
```

Open the URL printed by Vite. Verify the repository with:

```powershell
npm test
npm run build
```

WebGPU requires a secure context. Vite's `127.0.0.1` or `localhost` URL qualifies; opening `index.html` directly with `file://` does not. If WebGPU is unavailable, the lab falls back to the WebGL2 preview.

## Offline rendering

Fire Replica Lab includes a deterministic offline renderer for producing smooth,
frame-complete MP4 video without depending on browser display speed. It advances
the simulation by exactly one fixed step for every output frame, so a job may
render much slower than realtime while the finished movie still contains a
verified 60 fps cadence with no dropped frames.

On Windows, start with a short HD validation render:

```powershell
.\scripts\record-fire.cmd --duration 30s --resolution hd --solver-tier hd --grid auto --scene inferno --field beauty --background --verify full --output renders/firex-inferno-hd60-30s.mp4
```

The practical 4K preset keeps the UHD solver grid while raising the presentation
resolution:

```powershell
.\scripts\record-fire.cmd --duration 30s --resolution 4k --solver-tier uhd --grid auto --scene inferno --field beauty --background --verify full --output renders/firex-inferno-4k60-30s.mp4
```

The maximum-compute Reference render is intentionally expensive and requires an
explicit heavy-job confirmation:

```powershell
.\scripts\record-fire.cmd --quality highest --duration 30s --scene inferno --field beauty --warmup 3s --confirm-heavy --background --verify full --output renders/firex-inferno-reference-4k60-30s.mp4
```

| Offline setup | Output | Dense solver grid | Intended use |
|---|---:|---:|---|
| `--resolution hd --solver-tier hd --grid auto` | 1920×1080, 60 fps | 128×192×128 | Fast validation and field checks |
| `--resolution qhd --solver-tier qhd --grid auto` | 2560×1440, 60 fps | 160×240×160 | Balanced final render |
| `--resolution 4k --solver-tier uhd --grid auto` | 3840×2160, 60 fps | 192×288×192 | Practical 4K render |
| `--quality highest` | 3840×2160, 60 fps | Reference 508³ | Maximum-compute research render |

Reference 508³ contains 131.1 million cells and runs 96 primary plus 96
residual pressure iterations per output frame. On the development system, a
short optimized run sustained about 0.21 output frames per wall-clock second;
a 30-second movie therefore takes hours, and a one-hour movie can take many
days. A mature plume or competing GPU workload can increase that time. The
renderer reports measured solver and presentation timings while running—use
those measurements, rather than the movie's 60 fps playback rate, to estimate a
long job. Run `--dry-run` first and close other GPU-heavy browser tabs before a
Reference capture because WebGPU uses one adapter and cannot pool VRAM across
multiple GPUs.

Every successful job writes the MP4 plus a neighboring `.mp4.json` manifest
containing the exact scene, grid, camera, expected frame count, phase timings,
codec information, and independent cadence verification. Use `--field all` for
all eight views, or run the shorter four-view acceptance gate before a long
Beauty render:

```powershell
.\scripts\record-fire.cmd --test-views --background
```

### Recorder architecture and verification

Long renders use a separate capture path from the short in-browser exporter. The
capture page initializes Fire-X directly with the requested scene, diagnostic
view, offline solver tier and exact dense grid, advances one fixed simulation
step per output frame, and streams fragmented H.264 MP4 data to a token-scoped
loopback file sink with backpressure. Neither the complete movie nor raw 4K
frames are retained in browser memory. The fragmented MP4 muxer does retain
lightweight index metadata (216,000 frame entries for an hour), but never the
roughly 29 GB encoded payload. The CLI then checks the file from disk with
`ffprobe` and `ffmpeg` before replacing an existing destination, and writes a
JSON manifest beside the verified movie. `--overwrite` keeps the previous
verified movie in place until its replacement has passed disk verification.
The CLI is intentionally capped at the requested one-hour maximum, which also
caps retained muxer index metadata at 216,000 entries.

Requirements: Node/npm, an installed Chrome or Edge browser with WebGPU and
WebCodecs, and `ffmpeg` plus `ffprobe` on `PATH`.

On Windows PowerShell, use the included `.cmd` launcher. It bypasses the NVM/npm
PowerShell shim, which can consume space-separated recorder options before Node
sees them. Direct `node scripts/record-fire.mjs ...` invocation is the portable
alternative. Inspect every option and run the tested 30-second field-view gate:

```powershell
.\scripts\record-fire.cmd --help
.\scripts\record-fire.cmd --test-views --background
```

That gate performs four independent Inferno resets for `beauty`, `temperature`,
`reaction`, and `vapor-soot`. Each file is 1920×1080, 30.000 seconds, and exactly
1,800 frames at 60 fps. The default `auto` verification fully decodes recordings
up to two minutes and audits every presentation timestamp and duration. Longer
jobs still receive an exact packet/frame-count and duration audit, plus decoded
start/middle/end windows; use `--verify full` to decode an entire long movie.

Always inspect the one-hour plan before starting the heavy job:

```powershell
.\scripts\record-fire.cmd --quality highest --hour --field beauty --dry-run --output renders/firex-inferno-beauty-4k60-1h.mp4
.\scripts\record-fire.cmd --quality highest --hour --field beauty --confirm-heavy --background --output renders/firex-inferno-beauty-4k60-1h.mp4
```

The equivalent convenience script deliberately stops at the heavy-job guard
until `--confirm-heavy` is supplied:

```powershell
npm.cmd run record:4k-hour -- --confirm-heavy --output renders/firex-inferno-beauty-4k60-1h.mp4
```

`--quality highest` is explicit: 3840×2160 at 60 fps, the UHD offline solver,
real Reference `508³` dense fields, the 32 GiB software allocation guard, and
virtual `1024³` optical detail. It never substitutes the optical frequency for
the real solver grid and never silently lowers a grid, frame rate, duration, or
resolution. One hour contains 216,000 frames and is approximately 29.25 GB at
65 Mb/s (container overhead varies). Reference `508³` alone is about 13.19 GiB
of persistent solver resources; it can exhaust a 16 GiB adapter after browser,
presentation, driver, and other desktop allocations are included. Multiple GPUs
do not combine their VRAM for one WebGPU device. Close other GPU-heavy programs
before this tier, and expect wall render time to be much longer than one hour.

Useful independent controls include `--field all`, `--scene base-spray`,
`--solver-tier hd|qhd|uhd`, `--resolution hd|qhd|4k`, named grids from `auto`
through `reference`, exact aligned tuples such as `512x512x512`, camera values,
and `--memory-budget`. Durations and warmups must land on an exact 1/60-second
frame boundary, so requested timing is never rounded. Structurally impossible
requests such as dense `1024³`
fail before Chrome opens. An interrupted job preserves its partial file for
diagnosis, but GPU solver state is not serialized, so a fresh run is required.

## What is implemented

### 2009 · WebGL2 refinement preview

The Horvath–Geiger path concentrates on the paper's camera-facing refinement stage:

- a deterministic WebGL2 transform-feedback driver with 2K/4K/8K/32K/65K directed particles;
- fractional, mass-normalized MRT projection of particle mass, heat and weighted velocity into neighboring depth slabs;
- a resident half-float slice atlas with cross-slice 3D transport and pressure stencils;
- temperature cooling and density/fuel dissipation;
- RK2 midpoint semi-Lagrangian advection, evolving multiscale detail and buoyancy;
- vorticity-like forcing and Jacobi pressure projection;
- trilinear reaction-localized emission, dark-smoke extinction and GPU sparks;
- an opt-in Cinematic transfer function that raises the resident field and driver
  budget, marches real view rays through the slab volume, reconstructs thin reaction
  surfaces and applies per-channel ACES mapping.

The slabs are fixed to the preview view; orbiting does not rebuild them. The preview does **not** reproduce the production coarse 3D PIC/FLIP grid, multiresolution velocity hierarchy, per-camera 2K slice reconstruction, or offline slice output described by the paper.

### 2025 · WebGPU Fire-X preview

The Fire-X path contains a compact hybrid compute graph:

- collocated 3D velocity/temperature and bounded species fields;
- an advanced grid panel with Auto, 80×120×80, 96×144×96,
  128×192×128, 160³, expert 256³, workstation 320³, studio 384³,
  reference 508³ and exact custom X/Y/Z requests. Fire-X negotiates the adapter's
  supported buffer limits up to the 2 GiB field required by 512³, then checks the
  exact request against device limits and selectable 512 MiB–32 GiB transactional
  peak guards before resources are replaced;
- first-order semi-Lagrangian transport and lightweight diffusion;
- methane fuel/oxygen reaction clamped by the 1:4 reactant mass ratio;
- buoyancy, approximate vorticity forcing, matched discrete projection operators,
  and a measured-residual Jacobi correction pass;
- reference-scale grids use adapter-supported `f16` pressure storage with `f32`
  tile/projection arithmetic when available; adapters without `shader-f16` reuse
  two `f32` pressure buffers between the primary and residual solves. This compact
  heavy-grid layout is 108 bytes/cell, while grids below 64 Mi cells retain four
  independent `f32` pressure fields and use 116 bytes/cell;
- a bounded direct-neighborhood droplet system;
- particle-only atomic mass deposition swept across each 30 Hz path segment, with compact laminar or broad spray kernels;
- pre-reaction cooling, vapor generation and approximate momentum coupling;
- filterable `rgba16float` 3D presentation fields, reaction-driven emission,
  softly scattering vapor and non-emissive droplet streaks (grid liquid remains
  available in its diagnostic instead of being rendered twice in Beauty);
- native-scale ray marching plus eight diagnostic field views, including signed post-projection divergence.
- an opt-in Cinematic path that keeps the 80×120×80 simulation budget when Grid
  resolution is Auto, adds a
  periodic detail volume with separately labeled native/256³/512³/1024³ virtual
  optical-frequency targets, renders a linear-HDR scene, and applies restrained
  bloom, final filmic mapping and dithering at native output scale.

This is not yet the paper's full staggered MAC solver, conservative bidirectional thermochemical SPH transfer, multi-species ideal-gas formulation, spatially hashed particle system, high-resolution scene solver, or Monte Carlo renderer.

## Inspecting Fire-X

The **Inspect Fire-X** panel is the source of truth. Beauty is only a presentation view; use the fields to establish whether the simulation is responding:

| View | What to check |
|---|---|
| Beauty | Final compositing only; do not use it alone as proof of correctness. |
| Temperature | Hot gas should rise and base suppression should shrink the hot region. |
| Reaction | Activity should stay near mixed fuel and oxygen and weaken when water reaches the source. |
| Fuel / O₂ | Reactants should meet and deplete around the reaction zone. |
| CO₂ / products | Products should trail active combustion. |
| Vapor / soot | Base spray should add vapor; rich combustion should leave more residue. |
| Liquid / velocity | The deposited jet should intersect the selected target height. |
| Divergence | The projected field should be substantially darker than a strongly divergent flow. |

The **Compute & resolution** panel keeps every interacting dropdown together: compute profile, dense solver grid, render-only cinematic frequency, and peak GPU allocation guard. Auto grid explicitly follows the compute profile; fixed and custom requests remain selected across profile changes. Applied and requested grids are shown separately, and pending or blocked requests are labeled beside the grid selector. Custom inputs preserve the exact entered tuple and never silently round or clamp it. The panel reports replacement cost, the temporary peak while the previous grid remains alive, and the negotiated single-field device ceiling separately from the total-memory guard. The real 256³–508³ tiers require an explicit second action and are described as workstation/reference computation, not realtime profiles. Passing structural WebGPU limits does not prove that enough VRAM is free, so allocation remains guarded and can still fail cleanly. The separate **Domain & units** panel controls physical extent and display units without changing the selected grid tuple. Width, height and depth are independently adjustable on a nominal meter scale, with metric or imperial readouts and per-axis spacing. These units are scene-scale controls, not validated engineering calibration. Suppression enablement, nozzle type, flow and targeting live together under **Water suppression**. Presets reset the simulation to make comparisons reproducible:

The Fire-X path starts on **Inferno** so source-scale controls and the larger burner are obvious immediately.

- **Methane:** clean combustion without water.
- **Large fire:** broader burner and stronger lift without suppression.
- **Inferno:** near-domain-wide turbulent flame with oxygen entrainment.
- **Rich:** oxygen-limited combustion with more residual products.
- **Top jet:** a narrow jet crossing the upper plume; the source reaction should persist.
- **Base spray:** a broad spray reaching the source; reaction and temperature should contract while vapor increases.

Toggle between Beauty and diagnostic fields without resetting. For an A/B suppression comparison, run Methane, inspect Reaction and Temperature, then run Base spray and inspect the same fields from a comparable elapsed time.

## Paper scale versus this preview

The paper reports all final results on an RTX 4090. Its appendix includes 64–128 pressure iterations, particle capacities from 131k to more than one million, and substantially larger grids.

| Case | Gas grid | Active liquid particles | Pressure iterations | Reported render time |
|---|---:|---:|---:|---:|
| Browser low | 32×48×32 (49k) | 384 | 14 + 14 residual | 80-ray realtime preview |
| Browser balanced | 40×64×40 (102k) | 640 | 24 + 24 residual | 112-ray realtime preview |
| Browser high | 64×96×64 (393k) | 1,024 | 36 + 36 residual | 144-ray realtime preview |
| Browser maximum | 80×120×80 (768k) | 1,536 | 48 + 48 residual | 192-ray supersampled inspection preview |
| Browser cinematic | 80×120×80 (768k) | 1,536 | 48 + 48 residual | 192-ray native-scale HDR showcase |
| Browser ultra override | 96×144×96 (1.33M) | Profile-dependent | Inherits profile | Advanced opt-in grid |
| Browser experimental override | 128×192×128 (3.15M) | Profile-dependent | Inherits profile | High-memory opt-in grid |
| Browser extreme override | 160×160×160 (4.10M) | Profile-dependent | Inherits profile | ~454 MiB dense opt-in grid |
| Browser expert override | 256³ (16.78M) | Profile-dependent | Inherits profile | ~1.81 GiB; 2 GiB base-transition guard |
| Browser workstation override | 320³ (32.77M) | Profile-dependent | Inherits profile | ~3.54 GiB; 4 GiB base-transition guard |
| Browser studio override | 384³ (56.62M) | Profile-dependent | Inherits profile | ~6.12 GiB; 8 GiB base-transition guard |
| Browser reference override | 508³ (131.10M) | Profile-dependent | Inherits profile | ~13.19 GiB; 16+ GiB base-transition guard |
| Paper Fig. 4 | 128×128×256 (4.2M) | — | 64–128 range | 10 ms |
| Paper Fig. 8 | 300×150×300 (13.5M) | 6.7k | 64–128 range | 23–35 ms |
| Paper Fig. 1 | 400×300×200 (24M) | 175k | 64–128 range | 960 ms |

The grid simulation and particle times reported by the paper are additional to those render times. The 256³–508³ overrides are real dense allocations but are expected to run far below realtime under this browser Jacobi graph; Reference may exceed free VRAM or a browser/GPU watchdog even when its structural limits pass. The named 508³ tier deliberately fits common WebGPU limits that expose 2 GiB minus a small binding-alignment margin; an exact 512³ remains available through Custom only when the device truly exposes the full 2 GiB field. A dense 1024³ request contains 1.074 billion cells and needs about 108 GiB of persistent Fire-X resources, including a 16 GiB vector buffer, so this architecture still rejects it explicitly. The Virtual 1024³ control is optical reconstruction only and never claims 1024³ simulated cells. Matching offline or workstation-scale images while remaining realtime on an integrated GPU is not a realistic expectation; the lab instead exposes explicit quality and verification controls. Maximum is an opt-in supersampled inspection budget and may run below wall-clock realtime. With Grid resolution set to Auto, Fire-X Cinematic reuses its bounded Maximum simulation state and spends the added budget on linear HDR structure; an explicit grid override increases the simulation allocation independently. The Horvath Cinematic path instead raises the resident field to 216×176×56, uses 24 Jacobi iterations, 65,536 particles, 128 real-ray samples and a 1.15× output scale.

## Measurements

The footer separates display cadence (`FPS` and `Frame time`) from synchronous JavaScript command encoding/submission (`CPU encode / submit`). CPU submission time is **not GPU execution time**. The active profile line reports the engine's actual domain, rays, pressure iterations, fixed-step rate and particle budget. Fire-X uses a 30 Hz bounded step in every tier; the WebGL2 detail line similarly exposes its step, slab and ray budgets. Cinematic is a showcase profile, not a claim of paper-scale or wall-clock-realtime computation.

## Reproducible acceptance checks

Before treating a change as an improvement:

1. Run `npm test` and `npm run build`.
2. Reset Methane and confirm finite Temperature, Reaction, Fuel/O₂ and Divergence views.
3. Reset Rich and confirm reduced oxygen plus increased product/residual response relative to Methane.
4. Compare Top jet and Base spray at the same quality and similar elapsed time. Base spray should reduce the reaction region more strongly.
5. Toggle water off in Base spray, reset, and confirm the liquid field disappears and suppression weakens.
6. Rebuild all quality profiles and switch methods repeatedly without WebGPU/WebGL validation errors.
7. Apply Auto, Ultra and one valid custom Fire-X grid. Then enter 1024³ and confirm
   the exact request remains visible, the ~108 GiB rejection is explicit, and the
   previously applied solver grid remains active.
8. Select Expert 256³ and confirm the UI shows the exact resident/peak cost, the
   negotiated device ceiling and an explicit Apply action. On hardware below the
   required 256 MiB binding or selected 2 GiB peak guard, confirm it explains the
   blocker without attempting or silently reducing the allocation.

Visual similarity alone is not sufficient. Conservation reductions, GPU timings and divergence norms remain future validation work and are documented in the implementation notes.

## References

1. Christopher Horvath and Willi Geiger, [*Directable, High-Resolution Simulation of Fire on the GPU*](https://doi.org/10.1145/1531326.1531347) (SIGGRAPH 2009).
2. Helge Wrede et al., [*Fire-X: Extinguishing Fire with Stoichiometric Heat Release*](https://doi.org/10.1145/3763338.3763378) (SIGGRAPH Asia 2025).


