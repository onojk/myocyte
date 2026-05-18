# myocyte

Four soft spheres in 3D space. Each one is deformed by 360 invisible cursors arrayed across its surface (placed via spherical Fibonacci lattice, as evenly as math allows). The cursors pull and push the surface in and out, making each sphere ripple and bulge like a single muscle cell flexing unevenly. When two cells get close enough to overlap, they crowd each other — pushing apart and visibly squashing where they meet.

There's no central nervous system. The cells don't signal each other. They share only space, and they respect each other's bodies. That's the whole relationship.

## Run it

You need Rust. If you don't have it:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# follow the prompts, then restart your shell
```

Then, in this directory:

```bash
cargo run --release
```

First build takes ~3 minutes (Rust compiles a *lot* of code in `--release`). After that, edits compile in seconds.

A window opens. You see four spheres breathing on a dark background. They're identical, arranged in a 2×2 grid. Each one runs its own slightly offset breathing pattern so they're not in lockstep. When their breathing pushes them into each other, watch the contact surfaces flatten — that's the crowding behavior. Each cell also gets gently shoved away by a swollen neighbor and oscillates back to its starting position.

## Controls

| Action | Input |
|--------|-------|
| Orbit camera | Left-click drag |
| Zoom | Mouse wheel |
| Close | Window close button or Cmd/Ctrl-W |

## What you're looking at

**Each cell** is a deformable sphere with two layers of cursor influence:

1. **Script strengths** — the per-cell choreography that animates each cursor's strength over time. Three sine waves of different frequency are superimposed at each cursor, with the cursor's surface position phase-shifting the result. This is what makes the sphere breathe and ripple.

2. **Crowd bias** — transient negative bias added to cursors that face a neighboring cell when overlap is detected. This is what makes the contact side dimple inward. Decays back to zero with a ~0.23 second half-life when the crowding ends.

**Why 360 cursors?** A divisor of 360 was a design constraint — it gives each cursor a clean "angular slice" of the surface in spherical terms. 360 specifically is dense enough that the cursors blend into smooth deformation but sparse enough that the cell is still tractable to reason about.

**Why Fibonacci lattice?** Mathematically, you cannot evenly distribute an arbitrary number of points on a sphere. The Fibonacci lattice gets as close as is possible in closed form (no iteration, no optimization) and is far more even than a naive latitude/longitude grid (which clusters points at the poles).

## Project layout

```
myocyte/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs          window + event loop
│   ├── renderer.rs      wgpu pipeline, GPU buffers, draw loop
│   ├── sphere.rs        mesh generation, Fibonacci cursor placement
│   ├── cell.rs          per-cell state, cursor scripts
│   ├── crowding.rs      inter-cell physics
│   ├── camera.rs        orbit camera
│   └── shaders/
│       └── sphere.wgsl  vertex deformation + matte lighting
└── docs/
    └── architecture.md  why each piece exists, how data flows
```

Read `docs/architecture.md` for the longer explanation of how data flows through a frame.

## Troubleshooting

**`cargo run` fails with "failed to find a suitable adapter"** — your GPU driver is too old, or you're on a headless machine. wgpu needs Vulkan (Linux/Windows), Metal (macOS), or DirectX 12 (Windows). On Linux you may need `vulkan-tools` and `mesa-vulkan-drivers`.

**Window opens but is solid black** — shader probably failed to compile. Re-run with `RUST_LOG=wgpu_core=error,wgpu_hal=error cargo run --release` and look for shader error messages.

**Stutters or low framerate** — make sure you're using `--release` and not the debug build. Debug Rust is ~10× slower than release, and graphics code particularly hates debug builds.

**Build is taking forever** — first builds genuinely take a few minutes because wgpu's dependency tree is large. Subsequent builds use the cache and complete in seconds.

## What's next

This is v1 — the foundation. The repo is structured so you can layer things on without breaking what's here. Some directions in `docs/roadmap.md`:

- More cursor scripts: percussive contractions, lopsided bulging, sustained holds
- Variable cell count and starting layout
- Translucent material (the most beautiful next visual step, but more shader work)
- WASM target so the same binary runs in a browser via WebGPU
- Audio reactivity — cursor strengths modulated by FFT bins of mic input
- More than 4 cells, larger cluster, larger viewing space
