# architecture

How a single frame of myocyte runs, from input to pixels.

## The whole picture in one diagram

```
                   ┌─────────────────────┐
                   │  winit event loop   │
                   └──────────┬──────────┘
                              │ RedrawRequested
                              ▼
                   ┌─────────────────────┐
                   │   App::update()     │  CPU
                   │                     │
                   │  for cell in cells: │
                   │   update_script(t)  │  ← cell.rs, the choreography
                   │                     │
                   │  crowding::step()   │  ← crowding.rs, inter-cell physics
                   │                     │
                   │  for cell in cells: │
                   │   integrate(dt)     │  ← cell.rs, position/velocity
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Renderer::render() │
                   │                     │
                   │  upload camera      │  CPU → GPU
                   │  upload 4×cell      │
                   │                     │
                   │  for each cell:     │
                   │   draw_indexed()    │  one call per cell, on the GPU
                   │                     │
                   │     ↓                │
                   │  vertex shader:     │  GPU
                   │   for vertex v:     │
                   │    for cursor c:    │  ← 3K verts × 360 cursors per cell
                   │     v.pos += ...    │
                   │                     │
                   │     ↓                │
                   │  fragment shader:   │  GPU
                   │   lambert lighting  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   present()         │
                   └─────────────────────┘
```

## Why each module exists

**`sphere.rs`** — pure geometry. Generates a sphere mesh (vertices + indices) and produces the Fibonacci-distributed cursor directions. Doesn't know anything about cells, time, or rendering. Just math.

**`cell.rs`** — per-cell state and choreography. A cell knows its position, its current cursor strengths, and how to update its own cursor strengths from time. It doesn't know about its neighbors and doesn't know about rendering.

**`crowding.rs`** — the only thing cells share. Looks at all pairs of cells, detects overlap, and applies displacement forces + compression bias. This is the entire "cellular relationship" — pure geometry through volume conservation. No abstract signaling.

**`camera.rs`** — orbit camera math. Takes input (mouse drag, scroll), produces view-projection matrices.

**`renderer.rs`** — wgpu plumbing. Owns the device, queue, pipelines, buffers. Renders the scene. The most code by volume but the least conceptually interesting — it's the wgpu equivalent of "boilerplate."

**`shaders/sphere.wgsl`** — the work that matters. Vertex shader does the cursor-driven deformation. Fragment shader does soft matte lighting. ~150 lines that produce the entire visual identity.

## Data flow inside the vertex shader

The most conceptually important part of the whole project. For each vertex on the mesh:

```
input:  vertex direction (= position on unit sphere, also = normal)
        cell center, base_radius
        360 cursor directions (shared across cells)
        360 cursor strengths (per cell)
        360 cursor crowd biases (per cell)

algorithm:
    total_displacement = 0
    for each cursor i:
        weight = max(0, dot(vertex_dir, cursor_dir[i])) ^ sharpness
        total_displacement += (strength[i] + crowd_bias[i]) * weight
    total_displacement /= normalization_factor

    radius = base_radius * (1 + total_displacement)
    world_position = cell.center + vertex_dir * radius
```

The clever part is in `weight`. The dot product between a vertex's direction and a cursor's direction is `1` when they point the same way, `0` when perpendicular, `-1` when opposite. We clamp to `[0, 1]` so cursors on the far side of the sphere don't pull. Then we raise to a power (`sharpness = 6`) to make each cursor's influence a tight lobe rather than a broad bulge.

Higher sharpness = pointier deformation; lower = softer. The chosen value (6) gives lobes about a third of the sphere wide — enough that 360 cursors with adjacent lobes cover the surface with reasonable overlap, but not so wide that every cursor influences every vertex significantly.

## How crowding works mechanically

When two cells overlap (distance between centers < sum of radii):

1. **Displacement.** Add an impulse to each cell's velocity along the line between their centers, scaled by overlap depth times a stiffness. The Cell::integrate step the next frame moves the cells apart. Damping prevents oscillation.

2. **Compression.** For every cursor on cell A, compute the dot product between the cursor's direction and the unit vector pointing from A toward B. If positive (cursor faces B), add a negative bias to that cursor's crowd_bias proportional to (dot ^ 3) * overlap_depth.

Step 2 is what gives the visible "two balloons pressed together" look — the surfaces facing each other dimple inward. The cube of the dot product makes the dimple a smooth lobe that fades to zero away from the contact axis.

After crowding, every cell's crowd_bias is decayed by `exp(-3 * dt)` so that bias from prior frames doesn't linger after cells separate.

## Why 4 draw calls instead of instanced rendering

GPU-instanced rendering would let us issue 1 draw call for all 4 spheres, with each "instance" reading a different cell uniform. This is faster — but with only 4 cells the savings are nanoseconds, and the code is much harder to follow (you have to feed per-instance data through either an instance buffer or an indexed uniform array, both of which add complexity).

At higher cell counts (say 50+), instanced rendering would be worth it. For 4, the simple "one draw call per cell with a different bind group" approach is clearer and just as fast in practice.

## Where the performance budget goes

At 4 cells × 360 cursors × ~3000 vertices × 144fps:

- 4 × 360 × 3000 = **4.3 million cursor-vertex computations per frame**
- × 144fps = **~620 million per second**

This is laughable for a GPU — even integrated graphics chew through this. The CPU's job per frame is much smaller: ~1500 sine calls (script updates for 4 cells × 360 cursors), 6 pair comparisons for crowding, a handful of buffer uploads. CPU time per frame should be well under 1 ms.

The actual bottleneck on any modern machine is the display refresh rate. If you turn off vsync, this renderer should hit several hundred fps trivially.

## Things deliberately omitted from v1

These are all *easy* to add but would have bloated v1 without changing the core demonstration:

- **Multiple cursor counts.** Every cell has exactly 360 cursors. Varying this per cell would need either separate shaders or padding logic, both manageable but not free.

- **Cell scripts beyond breathing.** The cursor script in `cell.rs` is one specific pattern. Many more are easy to add — a "percussive contraction" that pulses hard then relaxes, a "lopsided bulge" that only fires one side. These belong in a `scripts.rs` module to be added in v2.

- **Variable cell count.** The number is hardcoded as `NUM_CELLS = 4` in `cell.rs`. Making it dynamic is straightforward (change a few `for` ranges, add a CLI arg), just not needed for "see if it works."

- **Mouse interaction.** You can orbit the camera but can't grab a cell to move it. Adding this needs ray-picking, which is its own little subsystem.

- **Material variety.** Every cell uses the same matte/waxy shader. Translucent, glowing, metallic — each would be a meaningful next look.

- **Audio reactivity.** Routing FFT bins to cursor strengths would be a beautiful addition but pulls in audio dependencies and a bunch of input plumbing.

Add them when they're worth the complexity.
