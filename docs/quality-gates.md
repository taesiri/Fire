# Fire Replica quality gates

This file is the acceptance contract for the local realtime previews. A visually
plausible beauty image is not enough: diagnostic fields must demonstrate that the
underlying transport, reaction, pressure projection, and liquid coupling are active.

The locked pre-upgrade implementation is available at Git tag `baseline-v0`.

## Fire-X 2025 preview

### Runtime and observability

- The WebGPU engine initializes without shader compilation or validation errors.
- Beauty, temperature, reaction, fuel/O2, products, vapor/soot, liquid/velocity,
  and divergence views are selectable and visibly distinct.
- The post-projection divergence view must not contain a domain-wide floor, wall,
  or open-top plane. Boundary conditions must be solved consistently; display
  masking is limited to the documented one-cell numerical shell.
- The UI reports the actual grid, particle count, pressure iteration count, ray
  count, and render scale for the selected profile.
- Fire-X grid resolution is independently selectable and reports X/Y/Z voxels,
  nominal physical X/Y/Z extent, per-axis cell spacing, cell count and estimated
  persistent grid memory in metric or imperial display units.
- Custom dense-grid requests are never silently rounded or clamped. Requested and
  applied tuples remain visible simultaneously, including after rejection.
- Grid changes are committed atomically, preflight device limits before replacing
  resources, and restore the previous grid after a failed allocation.
- CPU encode/submit time is never presented as GPU simulation time.
- Resetting a preset starts from the same deterministic initial state.

### Combustion and rendering

- Luminous flame is localized to an active reaction or hot-soot signal. A warm,
  non-reacting plume must not turn into a solid orange emissive column.
- Fuel and oxygen overlap at the flame base; products and temperature are
  transported upward; reaction occupies a thinner interface than temperature.
- The methane reaction consumes oxygen at approximately four times the fuel mass.
- Volume fields use filtered reconstruction in the beauty render. At native 1080p,
  grid cells must not appear as nearest-neighbor blocks.
- Cinematic detail must be sampled from the bounded periodic 3D detail texture and
  remain localized to the resolved reaction edge; it must not replace the simulated
  reaction field with stationary screen-space noise.
- Virtual 256³/512³/1024³ optical targets must be labeled render-only and must not
  alter or misreport the applied dense solver dimensions.
- Cinematic renders scene and droplets in linear HDR, extracts only thresholded
  bloom, and performs filmic mapping and dithering in the final composite. Bloom
  must not broaden the full plume or erase internal extinction structure.
- Steam and liquid water are non-emissive. Water may scatter reflected flame light,
  but cannot appear as a self-lit cyan beam when the flame is absent.

### Suppression experiment

- With equal flow, a base-directed spray reduces the visible reaction field and hot
  volume more than a top-directed jet.
- A spray produces more distributed liquid/vapor than a laminar jet.
- Turning water off removes both visible droplets and liquid-grid coupling.
- No-water methane, rich burn, top jet, and base spray presets can be reproduced
  without manually coordinating multiple controls.

### Performance profiles

- Balanced is the reduced interactive tier and should sustain at least 30 FPS in
  the local app on the declared test machine.
- High is the default and prioritizes image quality while remaining interactive.
- Maximum is an opt-in inspection budget. It may fall below 30 FPS and wall-clock
  simulation speed, but must remain responsive and label that tradeoff honestly.
- With Grid resolution set to Auto, Cinematic reuses the bounded 80×120×80,
  48+48-pressure, 1,536-droplet and
  192-ray simulation budget at `1×` output scale. It is an opt-in showcase profile,
  not a larger physical simulation or a paper-scale rendering claim.
- A profile that cannot keep simulation time near wall time must state that clearly;
  it must not silently claim realtime operation.

## Horvath-Geiger 2009 preview

### Runtime and observability

- The WebGL2 engine initializes without shader or framebuffer errors.
- The UI truthfully distinguishes the realtime preview from the paper's offline,
  multi-GPU 2K-per-slice result.
- The active slice resolution, depth count, solver budget, and any particle-driver
  count are visible.
- Campfire, fireball, and fire-wall presets produce materially different source and
  motion patterns, not the same blob with different brightness.

### Refinement and rendering

- Campfire contains multiple separated tongues, holes, and wrinkled edges.
- Fireball has a dominant horizontal trajectory, tail, and detached detail.
- Fire wall spans the source width and forms multiple sheets with darker smoke.
- Extinction does not saturate in the first few slabs; internal flame structure
  remains visible through the composite.
- Detail injection adds advected high-frequency structure without turning the field
  into stationary screen-space noise.
- Cinematic constructs a real 3D view ray through the normalized slab volume rather
  than offsetting a 2D UV by a synthetic parallax vector. `uCinematic` must gate the
  showcase transfer function so the lower tiers retain their established render.
- Cinematic applies the ACES rational curve per RGB channel. A single peak-derived
  scale must not wash the entire flame core toward white.
- Camera controls are described honestly: if slices are view-dependent and are not
  rebuilt for a changed camera, the camera must be locked or the limitation shown.

### Performance profiles

- Balanced should sustain at least 30 FPS and at least 0.9x wall-clock simulation
  speed on the local test machine.
- High may trade speed for slice resolution, but should remain at least 15 FPS.
- Maximum may trade realtime cadence for denser slabs, more particles and more
  ray samples; it must remain an explicitly opt-in inspection profile.
- Cinematic uses a `216×176×56` slab volume, 24 Jacobi iterations, 65,536 particles,
  128 real-ray samples and a `1.15×` output scale. It may trade cadence for showcase
  rendering but must remain responsive and explicitly opt-in.
- Increasing quality must preserve the source's physical footprint and core
  brightness. A larger atlas must not collapse the flame into a smaller or dimmer
  emitter because a particle kernel stayed fixed in texel units.
- GPU timings are reported only when a GPU timer extension is actually available.

## Paper-scale reality check

These are comparison targets, not claims about the realtime tiers:

| Method | Paper-scale example | Locked baseline high tier |
| --- | --- | --- |
| Fire-X | 128x128x256 to 300x150x300 gas grids, 64-128 pressure iterations | 44x68x44, 18 iterations, 512 particles |
| Horvath-Geiger | 2048x1556 per slice, 32-128 slices, 20k-100k driver particles | 196x120 per slice, 32 slices, analytic source |

Passing the realtime gates does not make either path an exact paper reproduction.
That label requires the missing paper-scale algorithms and a quantitative comparison
against aligned reference captures.
