# Paper-to-code implementation notes

This document distinguishes paper mechanisms, current browser approximations and the checks required before making fidelity claims.

## Scope labels

- **Paper method:** stated or reported by the publication.
- **Current preview:** implemented in this repository today.
- **Planned fidelity work:** required for a closer reproduction but not currently claimed.

Both implementations are realtime research previews. Neither is validated for fire-safety prediction.

## Horvath and Geiger 2009

Source: *Directable, High-Resolution Simulation of Fire on the GPU*, ACM TOG 28(3), Article 41.

### Paper method

The coarse stage uses directed 3D particles, a small PIC/FLIP grid, a multiresolution velocity pyramid, loose projection and scale-dependent vorticity. Its distinctive second stage independently refines camera-facing 2D slices. Algorithm 2 projects particle quantities, cools and dissipates fields, advects, injects detail, applies vorticity/buoyancy/pressure and composites the slices.

### Current preview

The WebGL2 path keeps a smaller half-float slice atlas resident on the GPU and follows the broad refinement ordering:

1. Advance a deterministic transform-feedback particle driver.
2. Additively project particle mass, heat and weighted velocity into neighboring depth slabs through MRT.
3. Cool temperature, dissipate density/fuel and advect the slab fields with RK2 midpoint backtracing and cross-slice sampling.
4. Add bounded evolving multiscale detail.
5. Apply 3D curl, buoyancy, divergence, Jacobi pressure and projection operations.
6. Trilinearly composite reaction emission and smoke extinction, then draw selected hot particles as sparks.

The opt-in Cinematic profile raises the resident slab volume to `216×176×56` with 24
Jacobi iterations and 65,536 driver particles. Its 128-sample renderer constructs a
real view ray through the normalized slab volume, gates its showcase transfer
function through `uCinematic`, and applies the ACES curve independently per color
channel. Its output scale is `1.15×`; this tier spends presentation work on
resolved depth and highlight color rather than backing-store supersampling.

Important limitations:

- The transform-feedback particles provide directed source attributes, but the original coarse PIC/FLIP grid, wavelet velocity hierarchy and multiscale pressure solve are still absent.
- Production dimensions around `2048×1556×32–128` are replaced by quality-bounded resident atlases.
- The paper's per-slice offline workflow and seconds-per-slice budget are replaced by a realtime transfer function.
- Slabs remain fixed-view; moving the preview camera does not rebuild or re-simulate camera-facing planes.
- Boundary stencils, Jacobi counts, source kernels and exact procedural noise are implementation choices because the paper does not fully specify them.
- Printed Eq. 6 appears to update fuel with a temperature-valued quantity while the prose describes heating; the preview follows the prose.

## Fire-X 2025

Source: *Fire-X: Extinguishing Fire with Stoichiometric Heat Release*, ACM TOG 44(6), Article 268.

### Paper method

Algorithm 1 couples a staggered Eulerian gas grid with Lagrangian SPH liquid particles:

1. Spawn and advance particles using pseudo-density, pressure and viscosity.
2. Conservatively transfer liquid mass, temperature and velocity to the grid.
3. Advect gas velocity, temperature and species.
4. Apply buoyancy, reaction, heat release, radiation and diffusion.
5. Evaporate liquid from a local energy budget.
6. Pressure-project the gas velocity.
7. Transfer remaining mass and temperature back to particles.

The paper tracks thermodynamic species, updates density from temperature and composition, uses 64–128 pressure iterations, and renders scalar volumes with ray marching or Monte Carlo transport depending on the scene.

### Current preview

The WebGPU path currently uses:

- collocated `vec4<f32>` buffers for velocity/temperature and compact species state;
- quality-following or explicit `80×120×80`, `96×144×96`, `128×192×128`,
  `160³`, expert `256³`, workstation `320³`, studio `384³`, reference `508³`,
  and exact custom 4-cell-aligned grid requests. Device creation negotiates
  adapter-supported buffer/binding limits up to a 2 GiB single field; requests
  then pass device-limit and selectable 512 MiB–32 GiB transactional-peak
  preflight before the active resource set is destroyed;
- first-order semi-Lagrangian transport;
- a bounded methane-like reaction with fuel consumption limited by `min(fuel, oxygen / 4)`;
- approximate heat, buoyancy, vorticity and radiative cooling terms;
- 14/24/36/48/48 primary Jacobi iterations plus matching 14/24/36/48/48
  post-projection residual-correction iterations by profile;
- four independent `f32` pressure fields below 64 Mi cells. At and above that
  threshold, adapters with `shader-f16` store four half-precision pressure fields
  while retaining `f32` tile and projection arithmetic; the fallback reuses two
  `f32` fields sequentially across the primary and residual solves. Both heavy
  paths reduce persistent pressure residency from 16 to 8 bytes/cell;
- 384/640/1024/1536/1536 droplets with a bounded `O(N²)` direct-neighborhood approximation;
- conservative three-sample swept fixed-point liquid deposition using compact laminar or normalized `3×3×3` spray footprints;
- approximate pre-reaction cooling, vapor and momentum coupling;
- a localized instantaneous-reaction field and an explicit open-top outflow;
- filterable `rgba16float` 3D presentation textures with 80/112/144/192/192-ray marching, 1.25× output supersampling on Maximum, and native-scale linear-HDR presentation on Cinematic;
- non-emissive instanced droplet streaks and eight diagnostic field views, including post-projection divergence;
- a 30 Hz fixed step with at most one simulation step per rendered frame in every profile.

The pressure projection uses a backward divergence and forward pressure gradient,
an adjoint-compatible pair whose composition matches the nearest-neighbor
Laplacian solved by Jacobi. A separately zeroed correction solve projects the
measured residual before the final divergence diagnostic. The presentation masks
the outer one-cell shell because collocated clamped-wall derivatives there are a
boundary-discretization artifact; every interior signed residual remains visible.
Side walls and the floor enforce zero normal velocity; the top uses an
ambient-pressure, zero-gradient velocity outflow and rejects inflow.

The laminar nozzle uses a compact trilinear footprint and transfers 20% of particle
mass into the gas-coupling field. This is an explicit reduced entrainment/atomization
model, not a loss of particle mass: coherent liquid remains visible as particles,
while spray uses its full normalized footprint. Consequently, the Top jet and Base
spray presets compare both impact height and nozzle regime; they are a qualitative
paper-aligned contrast, not a controlled aim-only ablation.

Beauty renders liquid once, through the instanced particle streaks. Deposited grid
liquid remains visible in the Liquid / velocity diagnostic but is omitted from the
beauty volume, avoiding a second set of voxel-sized white capsules. Vapor remains a
low-albedo scattering volume rather than an emissive water effect.

The presentation volume is sampled once at each centered ray position; it no longer
mixes a shifted second media sample that broadened and misregistered reaction fronts.
Emission is integrated through the same Beer-Lambert optical depth as extinction,
and Maximum uses a 1.25× aspect-preserving supersampled backing store.

Cinematic with Grid resolution set to Auto deliberately reuses Maximum's `80×120×80`, 48+48-pressure and
1,536-droplet simulation budget. A periodic 3D detail texture perturbs only the
resolved reaction edge. Its native/256³/512³/1024³ controls describe virtual optical
sampling frequency, not additional solver cells. The volume and droplets remain linear in an `rgba16float`
scene target; a separate post pass extracts restrained bloom, then performs the
single final filmic/gamma mapping and 8-bit dithering. This is a presentation mode,
not additional simulated turbulence or a paper-scale Monte Carlo renderer.

The preview does not currently justify claims of a conservative Fire-X reproduction:

- It is not a staggered MAC grid.
- It cannot run a dense `1024³` grid: the compact heavy-grid topology uses
  108 bytes/cell and would require about 108 GiB persistent GPU memory plus
  16 GiB single vector buffers.
- It does not yet store the paper's complete species/density state.
- Product generation and mixture normalization are simplified.
- Droplets do not carry the complete temperature, heat-capacity, latent-heat and diameter state used by the paper.
- Particle mass loss is not yet a fully bidirectional, cell-energy-limited transfer.
- Only liquid mass is deposited conservatively; temperature and momentum transfer remain approximate.
- Direct all-pairs neighbors prevent paper-scale particle counts.
- Gas and particle work is limited to one fixed step per rendered frame; if display
  cadence falls below 30 FPS, simulation time falls behind wall time instead of
  entering an unbounded catch-up spiral.
- The pressure budget is well below the paper's 64–128 iterations.
- The beauty renderer is not the paper's refractive/path-traced renderer.
- Metric and imperial domain labels are a nominal scene scale. They expose
  per-axis extent and voxel spacing but do not constitute engineering calibration.

For methane, the physically expected global mass relationship is:

```text
CH4 + 2 O2 -> CO2 + 2 H2O
fuel : oxygen : CO2 : H2O = 1 : 4 : 2.75 : 2.25 by mass
```

The current reactant clamp enforces the first two terms. Full product and mixture conservation remains an explicit acceptance requirement rather than a current claim.

## Scale and performance context

| Case | Voxels | Active particles | Grid simulation | Particle simulation | Render |
|---|---:|---:|---:|---:|---:|
| Browser balanced | 102k | 640 | Device-dependent | Device-dependent | 112-ray filtered preview |
| Browser high | 393k | 1,024 | Device-dependent | Device-dependent | 144-ray filtered preview |
| Browser maximum | 768k | 1,536 | Device-dependent | Device-dependent | 192-ray supersampled preview |
| Browser cinematic | 768k | 1,536 | Device-dependent | Device-dependent | 192-ray native-scale HDR showcase |
| Browser ultra override | 1.33M | Profile-dependent | Device-dependent | Device-dependent | Inherits selected profile |
| Browser experimental override | 3.15M | Profile-dependent | Device-dependent | Device-dependent | Inherits selected profile |
| Browser expert override | 16.78M | Profile-dependent | Device-dependent, far below realtime expected | Device-dependent | Inherits selected profile |
| Browser workstation override | 32.77M | Profile-dependent | Device-dependent, reference use | Device-dependent | Inherits selected profile |
| Browser studio override | 56.62M | Profile-dependent | Device-dependent, reference use | Device-dependent | Inherits selected profile |
| Browser reference override | 131.10M | Profile-dependent | Device-dependent, watchdog/OOM risk | Device-dependent | Inherits selected profile |
| Paper Fig. 4 | 4.2M | — | 29 ms | — | 10 ms |
| Paper Fig. 8 | 13.5M | 6.7k | 36–37 ms | 6–8 ms | 23–35 ms |
| Paper Fig. 1 | 24M | 175k | 65 ms | 3 ms | 960 ms |

Paper measurements were produced on an RTX 4090. The Auto-mode 768k-cell Maximum and Cinematic grids remain about 5.5 times smaller than Fig. 4 and about 17.6 times smaller than Fig. 8. High remains the default realtime-oriented tier. Maximum spends extra output resolution on inspection; Cinematic keeps native output scale and changes the presentation path. Explicit 256³–508³ grid overrides change real voxel allocation independently of those presentation and solver profiles, but cubic work growth makes them workstation/reference experiments rather than realtime claims. The 508³ named ceiling leaves room below implementations that report a 2 GiB-minus-alignment storage-binding maximum; exact 512³ stays inspectable through Custom when a device exposes the full field size. A negotiated WebGPU validation limit is not a VRAM availability report, so allocation scopes and exact applied/requested status remain authoritative.

The footer's `CPU encode / submit` value measures synchronous JavaScript work only. It is not GPU simulation time. Reliable GPU timing requires timestamp queries where supported and asynchronous readback.

## Diagnostic views

Fire-X accepts numeric `viewMode` values through `SimulationEngine.setParameter`:

| Value | View | Verification purpose |
|---:|---|---|
| 0 | Beauty | Presentation composite; never sufficient by itself. |
| 1 | Temperature | Inspect heat transport and suppression. |
| 2 | Reaction | Locate active combustion. |
| 3 | Fuel / O₂ | Inspect reactant mixing and depletion. |
| 4 | CO₂ / products | Inspect product transport and history. |
| 5 | Vapor / soot | Separate suppression vapor from rich-burn residue. |
| 6 | Liquid / velocity | Confirm nozzle intersection and flow response. |
| 7 | Divergence | Inspect projection quality. |

The UI also sends `aimHeight` and numeric `nozzleType` (`0` laminar, `1` spray). These parameters reproduce the qualitative intent of the paper's top-versus-base and laminar-versus-spray comparisons within the preview's fixed scene.

## Reproducible acceptance tests

### Automated repository checks

```powershell
npm test
npm run build
```

Existing unit tests cover bounded reaction helpers, heat/radiation behavior,
evaporation bounds, Horvath cooling/support, fixed-step scheduling, slab-coordinate
conventions, RK2 characteristic tracing, matched Fire-X projection stencils,
boundary handling and swept liquid deposition. They do not validate the full GPU
state.

### Interactive A/B checks

Use one quality profile, do not move the camera between comparisons, and reset through each preset:

1. **Methane:** Reaction stays near mixed reactants; Liquid is empty.
2. **Rich:** Fuel/O₂ shows oxygen limitation; Products or Vapor/soot increases relative to Methane.
3. **Top jet:** Liquid intersects the upper plume; base Reaction remains active.
4. **Base spray:** Liquid intersects the source; Temperature and Reaction contract while Vapor increases.
5. Disable water in Base spray and reset. Liquid must disappear and the result should move toward the unsuppressed case.
6. Inspect Divergence after each quality rebuild and check browser logs for validation errors.

### Quantitative work still required

Before claiming physical validation, add GPU reductions or deterministic readbacks for:

- pre/post-projection divergence RMS and maximum;
- total fuel and oxygen consumption;
- instantaneous and integrated reaction rate;
- liquid particle mass versus deposited liquid and vapor gain;
- maximum/average temperature and hot-voxel count;
- energy removed by evaporation versus available sensible and latent energy;
- simulation-time versus wall-time ratio and dropped fixed steps.

Required invariants include nonnegative species, oxygen consumption of four units per fuel unit for methane, product mass conservation, evaporation bounded by available liquid/energy, and a substantially lower integrated reaction/hot volume for Base spray than Top jet.

## Planned fidelity sequence

1. Add measured GPU reductions for divergence, species, heat release, liquid mass and simulation/wall-time ratio.
2. Upgrade gas transport to RK2 and then a correctly staged bounded MacCormack/BFECC scheme; add multigrid pressure projection.
3. Replace all-pairs liquid neighborhoods with GPU spatial hashing and normalized mass/temperature/momentum transfer.
4. Complete methane products, density/heat-capacity coupling and bidirectional energy-limited evaporation.
5. Move to a staggered MAC representation and add a separately labeled workstation/reference profile.
