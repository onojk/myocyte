# Tier 2 Plan: Myocyte Engine

*Written 2026-05-18. This document is the working spec for the Tier 2
implementation. It does not substitute for the design doc — it references
and expands it for implementation purposes.*

---

## 1. What We Are Building

Tier 2 replaces the four-sphere cursor demo (v1) with a new engine built
around a richer cell primitive. Where v1 represents each cell as a deformed
triangle mesh driven by 360 surface cursors, Tier 2 represents each cell as
a **19-float splat** — a parameterized volumetric kernel that is composited
into the image the way 3D Gaussian Splatting composites Gaussians. The scene
is a 3D grid of these cells (8³ to 32³), animated by a procedural field
(Gray-Scott reaction-diffusion) rather than per-cell scripts.

The central research artifact is the **cell-count-versus-fidelity curve**: how
few cells does it take to represent a given scene at a given quality level?
Tier 2 generates the procedural scenes and the rendering engine. Tier 3
(differentiable optimization) will close the loop and actually optimize cell
parameters against a target image.

### Relationship to 3D Gaussian Splatting

Kerbl et al. (SIGGRAPH 2023) demonstrated that a scene can be represented as
a collection of 3D Gaussian primitives, composited via back-to-front alpha
blending, and that gradient descent over these Gaussians from multi-view
photographs produces state-of-the-art novel-view synthesis quality. Our
approach shares the compositing framework but differs on three axes:

**Richer primitive.** A 3DGS Gaussian is fixed to an exp(-0.5 r^T Σ^{-1} r)
profile with view-dependent color via spherical harmonics. Our cell has two
density-shape parameters (`falloff` and `sharpness`) that let it range
continuously from a soft exponential blob to a near-hard-edged ellipsoid, and
replaces SH with a simpler view-independent radial color gradient
(`color_inner`/`color_outer`). The primitive is expressive with fewer floats
than high-order SH.

**Different research question.** 3DGS asks "what image quality can we reach?"
We ask "what is the minimum cell count to reach quality Q?" These require
different evaluation methodologies and different training regimes.

**Procedural middle tier.** 3DGS initializes from random points and optimizes
directly. We insert a structured procedural phase (reaction-diffusion) before
any image-matching occurs. The hypothesis is that biologically-motivated
initial structure allows gradient descent in Tier 3 to reach lower cell counts
for natural-looking scenes.

---

## 2. The 19-Float Cell Parameterization

Each cell is a contiguous block of 19 floats in this exact order:

| Field | Type | Floats | Description |
|---|---|---|---|
| `position` | vec3 | 3 | world-space center |
| `rotation` | quat | 4 | orientation (wxyz convention) |
| `scale` | vec3 | 3 | per-axis radii (not half-radii, not variances) |
| `falloff` | f32 | 1 | density kernel steepness; 0.5=soft exp, 1.0=Gaussian, 2.0=super-Gaussian |
| `sharpness` | f32 | 1 | edge hardness; 0.0=pure generalized Gaussian, 1.0=near hard ellipsoid |
| `color_inner` | vec3 | 3 | linear RGB color at cell center |
| `color_outer` | vec3 | 3 | linear RGB color at cell edge (d=1 in Mahalanobis distance) |
| `opacity` | f32 | 1 | base transmittance coefficient |

All 19 parameters are continuous floats with no discrete branching. This is a
hard constraint for Tier 3 differentiability: the density kernel and the
compositing equation must be differentiable with respect to all 19 at every
point except a measure-zero set.

### Density Kernel Proposal

Let `r = world_pos - cell.position`, and let `Σ^{-1}` be the inverse covariance:

```
R      = rotation_matrix_from_quaternion(cell.rotation)
Σ      = R * diag(cell.scale²) * R^T
Σ^{-1} = R * diag(1.0 / cell.scale²) * R^T

d_sq   = r^T Σ^{-1} r          (Mahalanobis distance squared)
d      = sqrt(d_sq + 1e-6)      (eps avoids zero-gradient at center for Tier 3)

gaussian = exp(-pow(d, 2.0 * falloff))
    -- falloff=0.5 → exp(-d) soft exponential
    -- falloff=1.0 → exp(-d²) standard Gaussian
    -- falloff=2.0 → exp(-d⁴) flat plateau, sharp edge

crisp    = 1.0 / (1.0 + exp(-8.0 * (1.0 - d)))
    -- sigmoid centered at d=1; near-1 for d<0.8, near-0 for d>1.2
    -- k=8 gives near-step while keeping nonzero gradients everywhere

density  = mix(gaussian, crisp, sharpness)
    -- sharpness=0 → pure generalized Gaussian
    -- sharpness=1 → hard-edged ellipsoid

t        = clamp(d, 0.0, 1.0)
color    = mix(color_inner, color_outer, t)
alpha    = opacity * density
```

**Rationale for two parameters:** `falloff` controls how the density falls off
with distance (Gaussian tail shape). `sharpness` controls whether the outer
boundary is soft or hard. They are independent axes of variation. A cell with
high falloff and high sharpness produces a very flat-topped, hard-edged disc —
think a pancake cross-section. A cell with low falloff and zero sharpness
produces an extended soft cloud. The generalized Gaussian alone cannot produce
a hard edge without sharpness.

**Tier 3 note:** The epsilon in `d = sqrt(d_sq + 1e-6)` ensures `∂d/∂r` is
defined at the origin. The sigmoid in `crisp` has nonzero but potentially small
gradients for `|d - 1| >> 1/8`; if autodiff shows saturation issues in Tier 3,
the scale factor `8.0` can be reduced. Do not replace it with `step()` — that
would kill gradients at the edge.

---

## 3. v1 → Tier 2 File Migration

### Files that survive unchanged

| File | Status | Notes |
|---|---|---|
| `src/camera.rs` | **Unchanged** | OrbitCamera and CameraUniform are fully reusable |

### Files that survive with modifications

| File | Status | Changes needed |
|---|---|---|
| `src/main.rs` | **Modified** | Replace cell/sim imports; add CLI args (`--record`, `--grid`); keep winit event loop structure |
| `Cargo.toml` | **Modified** | Add `clap` for CLI or use `std::env`; no new GPU deps needed |

### Files deleted

| File | Reason |
|---|---|
| `src/sphere.rs` | No more triangle mesh; sphere geometry not needed |
| `src/crowding.rs` | Contact physics replaced by field-driven animation |
| `src/renderer.rs` | Entirely replaced by new compositing pipeline |
| `src/shaders/sphere.wgsl` | Replaced by splat shader |

### New files

| File | Responsibility |
|---|---|
| `src/cell.rs` | `Cell19` struct (19 floats), `CellGrid` (3D array of cells) |
| `src/grid.rs` | 3D indexing helpers, `.myo` binary format read/write |
| `src/rasterizer.rs` | wgpu device/queue/pipeline, billboard instanced rendering, frame submit |
| `src/sort.rs` | Depth sort: back-to-front ordering of cells before upload |
| `src/preprocess.rs` | Per-cell per-frame CPU math: project center, compute 2D covariance, compute quad radius |
| `src/influencer/mod.rs` | `Influencer` trait: `step(&mut CellGrid, dt)` |
| `src/influencer/gray_scott.rs` | Gray-Scott 3D simulation on the cell grid |
| `src/recorder.rs` | ffmpeg pipe management, staging buffer readback |
| `src/shaders/splat.wgsl` | Vertex + fragment for billboarded cell quads |

---

## 4. Module Structure and Responsibilities

```
myocyte/
├── Cargo.toml
├── src/
│   ├── main.rs              CLI, winit event loop, App struct, per-frame orchestration
│   ├── camera.rs            Unchanged from v1
│   │
│   ├── cell.rs              Cell19 struct; CellGrid (flat Vec<Cell19> + xyz dims)
│   ├── grid.rs              (i,j,k) <-> flat index; neighbor access; .myo read/write
│   │
│   ├── preprocess.rs        Per-frame: compute Σ_2D, quad radius, screen center, depth
│   ├── sort.rs              Sort cells back-to-front by view-space depth
│   │
│   ├── rasterizer.rs        wgpu plumbing: device, queue, pipeline, buffers, render()
│   ├── recorder.rs          ffmpeg spawn, staging buffer, per-frame readback + write
│   │
│   ├── influencer/
│   │   ├── mod.rs           Influencer trait
│   │   └── gray_scott.rs    Gray-Scott 3D RD field
│   │
│   └── shaders/
│       └── splat.wgsl       Billboard vertex + fragment shader
└── docs/
    ├── architecture.md      (v1, preserved as reference)
    ├── myocyte_design.md    Full design doc
    └── tier2_plan.md        This file
```

### Data flow through one frame

```
App::update(dt)
  │
  ├── influencer.step(&mut grid, dt)
  │     -- N substeps of Gray-Scott forward Euler
  │     -- map (U,V) values to cell parameters
  │
  ├── preprocess::run(&grid, &camera)
  │     -- for each cell: compute view-space depth, screen center,
  │        2D covariance, quad radius
  │     -- returns Vec<SplatGpuData> (sorted or unsorted)
  │
  ├── sort::by_depth(&mut splat_data)
  │     -- unstable sort by depth descending (back to front)
  │
  └── rasterizer.render(&splat_data, &camera)
        -- upload sorted SplatGpuData to storage buffer
        -- 1 instanced draw call: N_cells instances × 4 verts
        -- fragment shader evaluates 2D Gaussian density
        -- alpha blend: src_alpha, one_minus_src_alpha
        -- present OR (if --record) blit to staging buffer → ffmpeg pipe
```

---

## 5. Rendering Pipeline

### Why billboards, not a tile-dispatch compute shader

The 3DGS CUDA implementation uses a tile-dispatch approach where screen tiles
are processed in parallel by thread blocks. This is an optimization for
millions of Gaussians; it is not a correctness requirement. For Tier 2 at
8³–32³ cells (512–32768), instanced billboard rendering with CPU depth sort
produces identical output with far less implementation complexity.

The billboard approach:
1. Each cell → 1 quad instance (4 vertices, 6 indices, or 2 triangles as a
   full-screen-covering vertex-index pair)
2. Quads are drawn back-to-front (sorted by depth) with alpha blending enabled
3. The fragment shader evaluates the 2D Gaussian density at each pixel and
   computes the compositing contribution

This satisfies "not raymarching" (no ray traversal through a volume grid) and
produces the correct composited image. A tile-based compute upgrade is
documented as a future path in §11 (Risks) if 32³ performance demands it.

### GPU data layout per cell (after preprocess)

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SplatGpuData {
    screen_xy:    [f32; 2],   // NDC center after projection
    depth:        f32,         // view-space z, for sort only (not sent to GPU)
    quad_radius:  f32,         // half-width of bounding quad in NDC units
    inv_cov2d:    [f32; 4],   // [a, b, d, 0] upper tri of 2×2 inv cov in screen px
    color_inner:  [f32; 4],   // rgb + opacity
    color_outer:  [f32; 4],   // rgb + falloff
    sharpness:    f32,
    _pad:         [f32; 3],
}
// Total: 12 × 4 = 48 bytes per cell
// At 32³ = 32768 cells: 32768 × 48 = 1.5 MB → storage buffer (not uniform)
```

The `inv_cov2d` is the inverse of the 2×2 projected covariance Σ_2D. Computing
it on CPU per frame avoids a matrix inversion in the shader (which would be
expensive and less stable).

### Covariance projection (in `preprocess.rs`)

```
// R = rotation matrix from cell.rotation quaternion
// Σ_3D = R * diag(scale²) * R^T  (3×3, symmetric)
// W = upper-left 3×3 of view matrix
// t = view-space center = W * cell.position + view.translation

// Jacobian of perspective projection at t:
// J = [[fx/tz, 0, -tx*fx/tz²],
//      [0, fy/tz, -ty*fy/tz²]]
// where fx,fy = focal lengths in pixels, (tx,ty,tz) = view-space center

// Σ_2D = J * W * Σ_3D * W^T * J^T  (2×2, symmetric)
// Σ_2D += 0.3 * I   (low-pass to avoid degenerate near-zero splats)
// inv_cov2d = inverse(Σ_2D)  (2×2 closed-form inverse)

// quad_radius: max eigenvalue of Σ_2D * 3σ → covers 99.7% of Gaussian mass
```

### Splat fragment shader (in `splat.wgsl`)

```wgsl
// In fragment shader, given interpolated screen-space position relative
// to the splat center (dx, dy) in pixels:

let d2: f32 = inv_cov2d.x * dx*dx
            + 2.0 * inv_cov2d.y * dx*dy
            + inv_cov2d.z * dy*dy;

// d2 here is the 2D Mahalanobis distance squared.
// We apply falloff/sharpness to the 2D distance as a visual approximation
// (the exact 3D→2D projection of a generalized Gaussian has no closed form).

let d: f32 = sqrt(d2 + 1e-6);
let gaussian: f32 = exp(-pow(d, 2.0 * falloff));
let crisp: f32 = 1.0 / (1.0 + exp(-8.0 * (1.0 - d)));
let density: f32 = mix(gaussian, crisp, sharpness);

let t: f32 = clamp(d, 0.0, 1.0);
let color: vec3<f32> = mix(color_inner, color_outer, t);
let alpha: f32 = opacity * density;

// Discard essentially invisible fragments early
if alpha < 1.0 / 255.0 { discard; }

// Premultiplied alpha output for correct GPU blending
return vec4<f32>(color * alpha, alpha);
// Pipeline blend state: src=ONE, dst=ONE_MINUS_SRC_ALPHA (premultiplied)
```

Note: premultiplied alpha compositing requires the pipeline blend state
`src_factor = ONE, dst_factor = ONE_MINUS_SRC_ALPHA`. This is NOT the default
`BlendState::ALPHA_BLENDING` (which uses non-premultiplied). Setting this
correctly is a common mistake in first implementations.

### Why storage buffers, not uniform buffers

wgpu's minimum uniform buffer binding size is 64KB (WebGPU limit); at 32³
cells × 48 bytes = 1.5MB, we exceed this by 24×. Use
`BufferBindingType::Storage { read_only: true }` and declare `var<storage,
read>` in WGSL.

---

## 6. Gray-Scott Reaction-Diffusion

### Why Gray-Scott

The Pearson (1993) parameter map provides a well-charted space of qualitatively
distinct pattern regimes: spots, stripes, labyrinths, worms, self-replication.
This lets us pick a target pattern intentionally rather than hunting in
parameter space. FitzHugh-Nagumo would also work but its parameter space for
3D spatial patterns is less documented, and the two-species formulation of
Gray-Scott maps naturally to our two-color (inner/outer) cell parameterization.

### Equations

```
∂U/∂t = D_U ∇²U  -  U·V²  +  f·(1 - U)
∂V/∂t = D_V ∇²V  +  U·V²  -  (f + k)·V
```

Discretized on a 3D grid with spacing dx=1, forward Euler with dt=1 (RD time
units):

```rust
let laplacian_u = u[x+1][y][z] + u[x-1][y][z]
                + u[x][y+1][z] + u[x][y-1][z]
                + u[x][y][z+1] + u[x][y][z-1]
                - 6.0 * u[x][y][z];
let laplacian_v = /* same with v */;

let reaction = u[x][y][z] * v[x][y][z] * v[x][y][z];  // UV²

let du = D_U * laplacian_u - reaction + f * (1.0 - u[x][y][z]);
let dv = D_V * laplacian_v + reaction - (f + k) * v[x][y][z];

u_next[x][y][z] = (u[x][y][z] + du).clamp(0.0, 1.0);
v_next[x][y][z] = (v[x][y][z] + dv).clamp(0.0, 1.0);
```

Boundary conditions: periodic (wrap with modular indexing).

### Proposed parameters

**Starting point: Pearson spots/coral pattern**

| Parameter | Value | Notes |
|---|---|---|
| D_U | 0.2100 | U diffuses faster than V (required for Turing instability) |
| D_V | 0.1050 | V diffuses slower; ratio D_U/D_V ≈ 2 is typical |
| f | 0.0350 | Feed rate — U replenishment |
| k | 0.0650 | Kill rate — V decay |
| dt_rd | 1.0 | RD time step; stable: dt ≤ dx²/(2·D_U) = 2.38 |
| steps/frame | 5 | RD substeps per rendered frame; tune for animation speed |

**Stability check:** Forward Euler is stable when `dt * D_U / dx² ≤ 0.5` →
`1.0 * 0.21 / 1.0 = 0.21 ≤ 0.5` ✓. The 6-neighbor 3D Laplacian requires the
factor 6 in the denominator: `dt * D_U * 6 / dx² = 1.26`; strictly this
exceeds the 3D stability limit of 1.0. Use `dt_rd = 0.5` if instability
appears (blow-up to NaN is the diagnostic).

**Initialization:** U = 1.0 everywhere, V = 0.0, then seed a small cubic
region at the center with V = 0.5 + uniform noise, U = 0.25. This seeds the
pattern from a single focus and produces expanding rings that self-organize
into spots/stripes depending on (f, k).

**Alternative parameter sets to try after first influencer works:**

| Name | f | k | Pattern |
|---|---|---|---|
| Worms | 0.025 | 0.056 | Long connected filaments |
| Stripes | 0.060 | 0.062 | Parallel bands |
| Mitosis | 0.028 | 0.062 | Spots that divide and replicate |
| Chaos | 0.026 | 0.051 | Unstable, constantly restructuring |

### Mapping RD field to cell parameters

The V concentration (activator species) creates the visible pattern: high V =
active region, low V = background.

**Minimum viable mapping** (implement first):

```rust
// v in [0.0, 1.0]
cell.opacity = 0.15 + 0.85 * v;
cell.color_inner = lerp(color_background, color_active, v);
cell.color_outer = lerp(color_background * 0.6, color_active * 0.7, v);
```

Good starting colors: `color_background = [0.15, 0.20, 0.35]` (deep blue),
`color_active = [0.95, 0.70, 0.30]` (amber). This gives the "bioluminescent
spots on dark water" appearance.

**Extended mapping** (add after minimum viable is working):

```rust
// gradient magnitude |∇V| ≈ max abs difference across neighbors
let grad_v = max_abs_neighbor_diff(v_field, i, j, k);

cell.scale = base_scale * vec3::splat(1.0 + 0.4 * v);
cell.falloff = 0.8 + 0.4 * v;           // active cells are more Gaussian-shaped
cell.sharpness = 0.3 * grad_v.clamp(0.0, 1.0);  // sharp edges at reaction fronts
```

Scale, falloff, and sharpness responding to the field produces structural
variation that makes the 3D cell grid feel volumetrically alive rather than
just a color pattern on a lattice.

### Data layout

Keep U and V as two separate `Vec<f32>` of length `grid_x * grid_y * grid_z`,
indexed the same way as the cell grid. Use double-buffering (front/back) and
swap after each substep. This avoids in-place update artifacts.

Running Gray-Scott on CPU for 32³ per frame: 32768 cells × 6 neighbors × 2
species × 5 substeps = ~2M multiply-adds per frame. At ~1ns/op this is ~2ms.
Acceptable at 30fps. If profiling shows it as the bottleneck, move to a wgpu
compute shader (ping-pong texture approach).

---

## 7. Recording (--record flag)

### Approach

Render to an offscreen `Rgba8UnormSrgb` texture with
`RENDER_ATTACHMENT | COPY_SRC` usage. When `--record` is active:
1. After each frame, issue `copy_texture_to_buffer` to a persistent staging
   buffer (same dimensions as the render texture, row-padded to
   `COPY_BYTES_PER_ROW_ALIGNMENT = 256` bytes)
2. Submit the copy and wait synchronously (`device.poll(Maintain::Wait)`) —
   this is only done when recording
3. Map the staging buffer, read the RGBA bytes, write to ffmpeg stdin pipe
4. Unmap the staging buffer

Synchronous poll introduces ~1 frame of latency and may reduce to ~15fps
during recording. This is acceptable; recording is not a real-time constraint.

### ffmpeg invocation

```rust
let ffmpeg = Command::new("ffmpeg")
    .args(["-y",
           "-f", "rawvideo",
           "-pixel_format", "rgba",
           "-video_size", &format!("{}x{}", width, height),
           "-framerate", "30",
           "-i", "pipe:0",
           "-c:v", "libx264",
           "-pix_fmt", "yuv420p",   // broad player compatibility
           "-crf", "23",
           "output.mp4"])
    .stdin(Stdio::piped())
    .spawn()?;
```

The render texture uses sRGB format so the bytes written to the pipe are
already gamma-corrected. `-pixel_format rgba` tells ffmpeg to interpret them
as 8-bit sRGB RGBA, which is correct.

**Row padding:** wgpu requires rows to be padded to 256-byte alignment. A
960px-wide RGBA8 texture has rows of 960×4 = 3840 bytes (multiple of 256 ✓).
For non-standard resolutions, strip the padding before writing to ffmpeg:

```rust
let row_bytes = width as usize * 4;
let padded_row_bytes = align_to(row_bytes, 256);
for row in 0..height {
    let src_start = row as usize * padded_row_bytes;
    pipe.write_all(&data[src_start..src_start + row_bytes])?;
}
```

### Surface vs. offscreen texture

The swap-chain surface texture does not have `COPY_SRC` usage and cannot be
mapped. The offscreen texture IS readable. For display: either blit the
offscreen texture to the surface, or render twice (once offscreen for
recording, once to surface for display). Blit is cheaper: one `copy_texture_to_texture`
or a fullscreen blit pass.

---

## 8. .myo File Format (Tier 3 Bridge)

The `.myo` format is the handoff between Tier 2 (Rust renderer) and Tier 3
(Python optimizer). Design it now so Tier 3 doesn't need to change the on-disk
format.

```
magic:   [u8; 4]    = b"MYO\x01"
version: u32        = 1
grid_x:  u32
grid_y:  u32
grid_z:  u32
-- cells follow in flat row-major order (z fastest, x slowest) --
cells:   [Cell19Raw; grid_x * grid_y * grid_z]

Cell19Raw (76 bytes, little-endian f32):
  position:    [f32; 3]
  rotation:    [f32; 4]   -- wxyz convention
  scale:       [f32; 3]
  falloff:     f32
  sharpness:   f32
  color_inner: [f32; 3]
  color_outer: [f32; 3]
  opacity:     f32
  _pad:        f32        -- align to 80 bytes total (4-float alignment)
```

Total file size: 20 bytes header + `grid_x * grid_y * grid_z * 80` bytes.
At 32³: 20 + 32768 × 80 = 2.6 MB. Trivial.

The Python side (`load_myo()`) reads this with `numpy.frombuffer`. The padding
float is reserved (write 0, ignore on read).

---

## 9. Incremental Build Order and Validation Checkpoints

Always start at 8³ (512 cells). Never jump grid size until the previous
checkpoint passes cleanly. Always build with `--release` for perf tests.

### CP1 — One cell renders correctly

**Goal:** A single Cell19 with default parameters appears as a soft Gaussian
blob in the center of the screen.

**Setup:** 8³ grid, all cells at opacity=0 except the center cell at (4,4,4).
No influencer. No sort (one cell, order irrelevant).

**What to test:**
- Density kernel math: blob shape matches expected Gaussian profile
- Color gradient: inner/outer color interpolation visible
- Falloff variation: visually confirm soft (falloff=0.5) vs. Gaussian (falloff=1) vs. super-Gaussian (falloff=2)
- Sharpness variation: confirm smooth→hard transition

**Done when:** A clearly soft, colored blob is visible. Changing falloff and
sharpness produces the expected visual changes.

### CP2 — Grid of identical cells

**Goal:** 8³ grid, all 512 cells active and identical, evenly spaced. Depth
sort produces stable ordering.

**What to test:**
- CPU sort correctness: no flickering during camera orbit
- Premultiplied alpha blending: accumulation is correct (center of grid appears
  denser than edge)
- Storage buffer upload: all 512 cells read correctly on GPU
- Instanced draw call: vertex shader reads the correct instance data

**Done when:** Orbiting the camera reveals a 3D lattice of identical blobs
with correct depth ordering.

### CP3 — Varied cells

**Goal:** Each cell in the 8³ grid has randomized parameters within valid
ranges. Visually diverse appearance.

**What to test:**
- Full 19-float parameter range in practice
- Scale anisotropy: elongated ellipsoids in various orientations
- Color gradient variation: different inner/outer color combinations
- No parameter combination produces NaN or Inf in the shader

**Done when:** A randomly-initialized grid displays structural variety with no
GPU errors and no NaN artifacts.

### CP4 — Single Gray-Scott influencer

**Goal:** Gray-Scott field runs on the 8³ grid and drives opacity + color.
Pattern visibly evolves over time.

**What to test:**
- RD initialization: spot seeds at center
- Pattern emergence: spots form and stabilize within ~1000 RD time steps
  (~200 rendered frames at 5 substeps/frame)
- Parameter mapping: opacity and colors track V concentration
- Grid boundary conditions: periodic wrapping, no edge artifacts

**Done when:** A time-lapse of the first 10 seconds shows a Gray-Scott pattern
forming on the cell grid.

### CP5 — Recording

**Goal:** `cargo run --release -- --record` writes `output.mp4`.

**What to test:**
- ffmpeg pipe opens without error
- Frame bytes are correct: playback shows the same visuals as the live window
- Row padding is stripped correctly (test at non-standard window sizes)
- Recording completes cleanly on window close or Ctrl-C

**Done when:** A 10-second MP4 of the Gray-Scott animation plays back
correctly in VLC or a browser.

### CP6 — Scale to 16³, then 32³

**Goal:** Verify performance at production grid sizes.

**16³ target:** 30fps live, recording completes without OOM.

**32³ target:** Acceptable at 20fps or better live. If depth sort is the
bottleneck (profile with `std::time::Instant`), switch to `rayon::sort`
(parallel) before implementing GPU sort.

**What to test:**
- Sort time vs. cell count (measure separately)
- GPU upload time (storage buffer write)
- Fragment shader cost per frame (GPU timestamp queries if available)
- Gray-Scott CPU time (profile inner loop)

**Done when:** 16³ runs at ≥30fps and 32³ runs at ≥15fps on Intel LNL.

### CP7 — Extended parameter mapping + .myo output

**Goal:** Scale and falloff respond to RD field. .myo files can be written and
re-loaded.

**What to test:**
- Scale variation: cells swell at high-V regions
- Falloff variation: sharper edges at reaction fronts
- .myo round-trip: save, reload, render — output is identical
- File size is correct (verify against formula above)

**Done when:** The full 19-float space is exercised by the influencer, and
.myo files serialize/deserialize correctly.

---

## 10. Known Risks and Fallback Plans

### Risk 1: CPU depth sort bottleneck at 32³

At 32768 cells, `sort_unstable_by` on a Vec of f32s takes ~2–5ms on a modern
CPU. This leaves ~28ms for everything else at 30fps. If Gray-Scott also costs
~2ms, the CPU budget is tight.

**Fallback A (likely sufficient):** Replace `sort_unstable_by` with
`rayon::slice::ParallelSliceMut::par_sort_unstable_by`. This uses all CPU cores
and typically gives 4–8× speedup.

**Fallback B (if A insufficient):** Implement GPU radix sort in a wgpu compute
shader. This is ~300–500 lines of compute WGSL with key-value sort on
(depth, instance_index) pairs. Doable but adds ~1 week of work.

**Fallback C (last resort):** Accept 20fps at 32³ for the research demo.
The cell-count-vs-fidelity curve doesn't require real-time rendering.

### Risk 2: Premultiplied alpha blending setup

The blend state `src=ONE, dst=ONE_MINUS_SRC_ALPHA` is NOT wgpu's built-in
`BlendState::ALPHA_BLENDING`. Getting this wrong produces either a white halo
around cells (non-premultiplied with premultiplied blend state) or dark
fringing (premultiplied with non-premultiplied blend state). The fragment
shader must output `vec4(color * alpha, alpha)`, not `vec4(color, alpha)`.

This will look subtly wrong during development and may not be immediately
obvious. Test by compositing a single cell over a colored background and
verifying the edge color matches the background (no halo, no fringe).

### Risk 3: Intel LNL storage buffer size or compute limits

Intel Arc (LNL) Vulkan drivers via ANV are generally reliable. However, wgpu
feature detection should be done explicitly at startup:

```rust
println!("max_storage_buffer_binding_size: {}",
    adapter.limits().max_storage_buffer_binding_size);
```

This should report ≥128MB. If it reports less and causes bind failures, the
fallback is to chunk cells across multiple storage buffers or use a dynamic
offset buffer.

### Risk 4: Gray-Scott 3D numerical instability

The forward Euler discretization of the 3D Gray-Scott equations can blow up
for certain (f, k) values or if the time step is too large. Symptoms: U or V
values reaching NaN or drifting outside [0, 1] (hence the `.clamp(0.0, 1.0)`
in the update). If instability is observed, reduce `dt_rd` from 1.0 to 0.5.
The stability limit for 3D with our parameters is approximately dt ≤ 1/(6·D_U)
= 0.79; `dt_rd = 0.5` is safely within this.

### Risk 5: ffmpeg not installed or wrong pixel format

The recorder assumes `ffmpeg` is on PATH and accepts `-pixel_format rgba`. On
some Linux distributions this requires the `ffmpeg` package rather than
`ffmpeg-minimal`.

**Fallback:** If ffmpeg is unavailable, write raw RGBA frames to
`output_%04d.raw` files and document a conversion command. This is a
reasonable fallback since it doesn't block any rendering work.

### Risk 6: Tier 3 differentiability of the density kernel

The sigmoid in `crisp = 1/(1 + exp(-8*(1-d)))` saturates for `d` far from 1.
For a cell that is mostly background (low opacity), most of its rendered pixels
will have large `d`, where `∂crisp/∂d ≈ 0`. In Tier 3, this means gradient
signals through heavily-transparent cells will be near zero, potentially
stalling optimization.

If Tier 3 reports gradient vanishing for low-opacity cells, the fix is to
reduce the sigmoid steepness from 8.0 to ~4.0, which widens the gradient
support at the cost of a less sharp edge. This is a tunable hyperparameter —
leave it in a named constant (`const EDGE_SHARPNESS_SCALE: f32 = 8.0`) from
day one.

### Risk 7: Covariance projection numerical issues

Very small or very flat cells (scale → 0 along one axis) produce near-singular
Σ_2D. The `+= 0.3 * I` regularization should catch most cases. If a cell has
scale < 0.01 in any axis, clamp it in the cell parameters before projection to
avoid division by very small numbers.

### Risk 8: wgpu 22.x API surface changes

The project uses wgpu 22.1. If wgpu releases a breaking change before Tier 2
is done, updating is straightforward (the API changes are well-documented) but
takes half a day. Pin the version in Cargo.toml and only update intentionally.

---

## 11. Estimated Complexity by Section

Assumes one developer familiar with Rust and moderately familiar with GPU
programming. "Half-day" = ~3–4 focused hours.

| Section | Estimate | Notes |
|---|---|---|
| Cell19 struct, CellGrid, grid.rs | 1 day | Straightforward data structures |
| Camera unchanged | 0 | Done |
| main.rs cleanup + CLI args | 0.5 days | Keep event loop, replace simulation |
| rasterizer.rs skeleton (device, pipeline, 1 draw call) | 1.5 days | New pipeline but familiar wgpu patterns |
| preprocess.rs (covariance projection) | 1 day | 2D covariance math is the hardest part; test with known cases |
| sort.rs | 0.5 days | Wrap standard sort, add Rayon flag |
| splat.wgsl (vertex + fragment) | 1 day | Dense shader math; test visually at CP1 |
| CP1: single cell renders | included above | |
| CP2: grid of identical cells | 0.5 days | Mostly debugging |
| CP3: varied cells | 0.5 days | Parameter exploration |
| Gray-Scott 3D simulation | 1.5 days | Numerics + 3D stencil; test 2D first |
| Field-to-parameter mapping | 0.5 days | Artistic tuning |
| CP4: RD influencer | included above | |
| recorder.rs + ffmpeg pipe | 1.5 days | Staging buffer + row padding fiddly |
| CP5: recording | included above | |
| CP6: scale testing + profiling | 1 day | Benchmark, possibly parallelize sort |
| .myo format + CP7 | 0.5 days | Binary I/O is simple |
| Integration, cleanup, CP7 | 0.5 days | |
| **Total** | **~12 days** | |

At a pace of 2–3 focused days per week, this is 4–6 weeks — consistent with
the design doc's estimate.

---

## 12. Out of Scope Until Tier 3

- Gradient computation through the renderer
- Image-matching loss function
- Python optimization loop
- Multi-view synthesis evaluation
- SH view-dependent color
- More than one influencer type (the second influencer is stretch goal for late Tier 2)
- Audio reactivity
- WASM/WebGPU target
- Mouse interaction with cells

---

## 13. Open Questions Deferred to Implementation

1. **Quad vertex layout:** Screenspace-aligned billboard (faster, ignores
   rotation for quad shape) vs. world-space billboard (more correct for highly
   elongated cells). Start with screenspace-aligned.

2. **Rendering resolution during recording:** Record at window size or at a
   fixed target resolution? If window can be resized, the MP4 resolution
   changes per session. Consider fixing the offscreen render target at 960×720
   regardless of window size.

3. **Grid coordinate system:** Should the 3D grid be centered at the world
   origin, or start at (0,0,0)? Centering at origin makes the camera start
   position work without adjustment. Use center-origin convention:
   `cell_center = (vec3(i,j,k) - (dims-1)/2.0) * cell_spacing`.

4. **Cell spacing:** How far apart are cell centers? The default `cell_spacing`
   should be chosen so that cells at `scale=(0.4, 0.4, 0.4)` just touch their
   neighbors at default parameters. Set `cell_spacing = 1.0` and
   `default_scale = 0.45` to leave a slight gap. Adjust aesthetically.
