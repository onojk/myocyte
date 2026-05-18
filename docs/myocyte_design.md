# Myocyte: a volumetric primitive for image representation

A design document.

## What this is

Myocyte is a research project investigating whether a small set of soft volumetric primitives — "cells" — can serve as a universal medium for representing images, in the same sense that pixels are universal for raster images and triangles are universal for surface meshes. The goal is to characterize the relationship between cell count and image fidelity for arbitrary target images, and to build a working differentiable renderer that can find the minimal cell configuration matching any given image to a chosen accuracy threshold.

If the project succeeds, the result is a new image and video codec: representations that are smaller than pixel-based encodings at high fidelity, smoothly interpolable in time and space, continuously resolvable at any rendering resolution, naturally 3D, and compositable in ways pixel-based images can never be. The codec would be most useful for generative art, animation, interactive media, and any domain where the image is a function rather than a recording.

## What this is not

This is not a real-time rendering engine. Real-time rendering of myocyte cells is required as a *playback* system once a representation has been found, but the central contribution is the *finding* — the optimization process that determines, for any target image, the smallest set of cells that produces it within tolerance.

This is not an aesthetic project. The four-spheres demo in v1 was an aesthetic prototype that validated the basic cell-and-cursor metaphor. The project has since reframed around the more general question of representational completeness. Aesthetic outputs remain possible (and beautiful) but the success criterion is no longer "looks cool"; it's "matches arbitrary targets with provably efficient cell counts."

This is not a replacement for triangle graphics. Triangles are the right primitive for authored surfaces meant to be rendered into images. Myocyte is for the inverse problem: given an image, find the minimal volumetric representation. The two domains overlap but are not the same.

## The central distinction: surface vs volume

Triangle graphics treats an image as the result of light interacting with surfaces. The fundamental object is the boundary between solid and empty space. To make an image you build a 3D world of surfaces (triangle meshes) and simulate light hitting them. This is a constructive, surface-based model.

Myocyte treats an image as a density distribution in space, viewed from somewhere. There are no surfaces, only densities — regions of space that are more or less opaque, more or less colored. To make an image you populate 3D space with soft volumetric cells, each contributing color and opacity, and composite along the viewing direction. This is a projective, volume-based model.

The two are not equivalent. Triangle graphics is biased toward boundaries; myocyte is biased toward fields. Real-world imagery contains both, but volumetric phenomena (clouds, flames, hair, skin, atmosphere, depth-of-field, subsurface scattering) are fundamental to natural images and are exactly where triangle graphics has spent decades inventing workarounds. The myocyte primitive matches these phenomena natively.

The practical consequence that makes the project tractable is that volumetric primitives are differentiable in a way surface primitives are not. A cell's contribution to any pixel is a smooth function of its parameters, so gradient descent on cell parameters to match a target image is well-posed. A triangle's contribution to a pixel is a step function (covered or not), so triangle optimization requires heroic engineering of soft rasterizers to work at all. This is why 3D Gaussian Splatting succeeded as a representation-finding technique while triangle-fitting did not.

## The cell specification

A cell is defined by exactly 19 floating-point parameters, 76 bytes:

| field         | type | floats | meaning                                  |
| ------------- | ---- | ------ | ---------------------------------------- |
| position      | vec3 | 3      | centroid in world space                  |
| rotation      | quat | 4      | orientation                              |
| scale         | vec3 | 3      | per-axis extent in cell's local frame    |
| falloff       | f32  | 1      | density profile steepness                |
| sharpness     | f32  | 1      | edge softness (0 = Gaussian, 1 = hard)   |
| color_inner   | vec3 | 3      | color at cell center                     |
| color_outer   | vec3 | 3      | color at cell edge                       |
| opacity       | f32  | 1      | maximum opacity at center                |

All parameters are continuous real numbers. There are no discrete fields, no type enums, no variable-length arrays. This is essential — every parameter must be differentiable for gradient descent to work end-to-end.

The cell's density at a point p in world space is computed as follows. Transform p into the cell's local frame using position, rotation, and scale. Compute the local radius r from the cell's center. The density is a function of r controlled by falloff and sharpness: at low sharpness it's a smooth Gaussian falloff; at high sharpness it approximates a hard-edged ellipsoid. The cell's color at that point is interpolated between color_inner and color_outer by r. The cell's opacity contribution at that point is opacity * density.

This parameterization is deliberately a generalization of the 3D Gaussian Splatting primitive (3DGS uses 14 floats: position, rotation, scale, single color, opacity, plus spherical harmonics for view-dependent color which we omit). The differences are: an explicit sharpness parameter that lets a cell range from soft Gaussian to hard-edged ellipsoid, and a radial color gradient that lets one cell express center-to-edge color variation. These additions are bets — they pay off if they reduce cell counts on natural images by more than they cost in per-cell complexity. This is testable.

Each cell expresses one coherent visual element: a soft blob, a hard ellipsoid, a glowing orb with halo, a flat disk, a needle, a radial gradient, any single anisotropic region with continuous color. Each cell *cannot* express two disconnected components, concave shapes, holes, or branched forms. These limitations are correct — they are exactly the boundary where you should be using more than one cell, and the gradient descent process will allocate cells accordingly.

## Rendering

Cells are rendered by alpha-compositing along the viewing direction. For each pixel of the output image, a ray is cast from the camera through the pixel into the scene. The ray accumulates color and opacity by integrating contributions from all cells it passes through, ordered by depth.

The rendering equation, in alpha-blending form, is:

out_color = sum over cells (in depth order):
    cell_color * cell_alpha * product of (1 - prev_alpha) for cells closer to camera

This is exactly the rendering equation used by 3D Gaussian Splatting and by volume rendering generally. It is differentiable with respect to every cell parameter, so back-propagation through the renderer to obtain gradients on cell parameters is well-defined.

Implementation in v2 will use a tile-based rasterizer in the manner of 3D Gaussian Splatting: each cell is projected to screen space as a 2D footprint, cells are assigned to screen tiles, and each tile is rasterized by alpha-blending its assigned cells in depth order. This achieves real-time framerates at millions of cells on consumer GPUs.

## The cell-fidelity curve

The central research artifact of the project is the cell-fidelity curve: for any target image, a plot of cell count on the x-axis against reconstruction fidelity on the y-axis. Fidelity is measured by some image-quality metric (PSNR, SSIM, or LPIPS — likely all three).

This curve is the image's complexity fingerprint. Simple images (solid colors, single shapes) plateau at high fidelity within a handful of cells. Complex images (photographs, detailed scenes) approach high fidelity asymptotically with cell count. The shape of the curve characterizes the image's structure in a way pixel counts cannot.

The minimal cell count to reach a given fidelity is the image's *information content* in this representation. Two images at the same pixel resolution may have very different cell counts at the same fidelity, and the difference is meaningful — it tells you which image has more visual structure.

The cell-fidelity curve also gives the codec a natural progressive encoding: render with the first 10 cells, get a rough impression; render with 100, get the gist; render with 1000, get good fidelity. Each additional cell adds the most useful next piece of structure, in order of gradient-descent priority.

## The optimization problem

Given a target image I and a cell budget N, find the cell configuration C of N cells that minimizes the rendering loss L(C, I), where L is some image-quality metric.

The basic algorithm is direct gradient descent: initialize C randomly, render to produce image I_hat = render(C), compute loss = L(I_hat, I), backpropagate to get dL/dC, update C, repeat.

The challenges are the same ones 3D Gaussian Splatting solved:

- **Initialization**: random initialization works but poorly; better initializations seed cells at points of high gradient or known structure
- **Cell pruning**: cells that contribute little (low opacity, or whose loss-gradient is near zero) should be removed and their budget reallocated
- **Cell splitting**: cells in regions of high local loss should be split into two cells in slightly different positions
- **Regularization**: terms penalizing cell count, cell size variance, or overly concentrated configurations
- **Learning rate scheduling**: different cell parameters benefit from different learning rates (position needs slow precise moves, color can change quickly)

All of these are established techniques in the 3DGS literature. We adopt them rather than reinvent them.

## Tiered roadmap

The project has three tiers, each shippable on its own and each enabling the next.

### Tier 1: complete (May 2026)

Four cells with cursor-driven deformation, crowding behavior, Rust+wgpu, matte lighting. Validates the basic 3D cell concept and the wgpu rendering pipeline. Source: myocyte/ directory, current state.

### Tier 2: aesthetic engine (target: 4-6 weeks)

A 16³ to 32³ grid of myocyte cells with the 19-float parameterization above, rendered via tile-based alpha compositing. Cells driven by procedural influencers (reaction-diffusion, audio, scripted patterns) — no image matching yet. Outputs to window and MP4 file. The first version of the engine, optimized for visual experimentation and aesthetic exploration. This validates the rendering pipeline and the cell parameterization at scale.

Migration from Tier 1: orbit camera, wgpu render scaffolding, and matte lighting style carry over. The fixed sphere mesh, four-cell hardcoding, cursor system, and crowding math get replaced. Approximate scope: full rewrite of cell.rs, sphere.rs (now cells.rs), crowding.rs (now influencers.rs), renderer.rs (tile-based rasterizer for cells), new record.rs for MP4 output. Keep camera.rs and main.rs largely as-is.

### Tier 3: differentiable codec (target: 3-6 months after Tier 2)

The full image-matching pipeline. Differentiable renderer (likely Python+PyTorch+gsplat or jax+jax3d to leverage existing differentiable splatting work), gradient-descent optimization on cell parameters to match target images and videos, cell-fidelity curve generation, comparison against 3DGS and traditional codecs on benchmark image and video sets.

The Rust engine from Tier 2 becomes the playback runtime. The Python pipeline produces cell configurations as a file format (.myo or similar), which the Rust engine renders. This separation matches the 3DGS ecosystem (training in PyTorch, viewing in many runtimes) and is the right division of labor.

Open question for Tier 3: do we also test polyhedral cells (e.g. soft octahedra with 6 face distances) as a second primitive type? If natural images have a meaningful mix of organic and angular structure, a mixed-primitive system may find more efficient representations than a single-primitive one. This is one of the most interesting research sub-questions and worth a dedicated experiment.

## Why this is worth doing

Three reasons, in increasing order of importance.

First, **the engineering is tractable**. 3D Gaussian Splatting has demonstrated that volumetric primitive optimization works at scale, on consumer hardware, with established techniques. Myocyte is a small generalization of the 3DGS primitive plus a research question about cell-count minimization. The technical risk is bounded.

Second, **the result is a real codec**. If the cell-fidelity curve is favorable (cell counts grow sub-linearly with image complexity, which is the empirical pattern for 3DGS), then myocyte-encoded images and videos are smaller than pixel-encoded ones at matched fidelity, with all the additional properties of volumetric representations (resolution-independence, temporal interpolation, 3D navigation, composability). This is publishable, useful, and a foundation for further work.

Third, **the conceptual reframe matters**. Image representation has been pixel-bound since photography and triangle-bound since computer graphics. Both are surface-of-light models. A volume-of-density model is a different theory of what images are, with different affordances and different uses. Even if the codec doesn't win on every metric, having a fully worked-out volumetric representation of arbitrary imagery is a contribution to how images are thought about.

## What to do next

Tier 2 is the immediate target. Claude Code should:

1. Read this document and confirm the framing.
2. Write a detailed implementation plan for the Tier 2 engine, covering the cell parameterization migration, the tile-based renderer, the influencer system, and the recording pipeline.
3. Implement Tier 2 incrementally, starting with a single cell rendered correctly, then a grid, then influencers, then recording. Validate each step before adding the next.
4. Keep Tier 3 in mind throughout: data structures should be differentiable-friendly (no discrete switches, no shape enums), the renderer should be structured so that a Python optimizer could eventually drive it.

The four-cell v1 stays in the repo as a reference and as a witness to the project's evolution. It also remains the most visually interesting demo for anyone first encountering the project — soft, slow, recognizable as "cells doing something."

This document is the project's source of truth as of its writing. When the framing changes again — and it might — this document should be the first thing updated.
