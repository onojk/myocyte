// sphere.wgsl — vertex deformation and matte lighting in one shader file.
//
// VERTEX SHADER
// For each vertex (a point on the unit sphere mesh):
//   1. Compute its outward direction (= its position on the unit sphere)
//   2. Loop over all 360 cursors; sum their contributions based on angular
//      distance between this vertex's direction and each cursor's direction
//   3. The summed contribution displaces this vertex along its normal,
//      adding (base radius * (1 + displacement)) outward from the cell center
//   4. Output world position, view-space position, world normal for lighting
//
// FRAGMENT SHADER
// Simple Lambertian + soft ambient. Matte and waxy — no specular highlights,
// no rim lighting. The point is for surface curvature to be clearly visible,
// and matte tissue is the cleanest way to read curvature.

// ---------- Uniforms ----------

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye: vec3<f32>,
    _pad: f32,
};

// Cursor directions: 360 entries, packed as vec4 (xyz used, w unused)
struct CursorDirs {
    dirs: array<vec4<f32>, 360>,
};

// Per-cell data — see renderer.rs::CellUniform for layout
struct CellUniform {
    center_radius: vec4<f32>,                // xyz=center, w=base_radius * size_bias
    color: vec4<f32>,                        // rgb=albedo
    strengths: array<vec4<f32>, 90>,         // 360 floats packed 4-per-vec4
    crowd_bias: array<vec4<f32>, 90>,        // ditto
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> cursors: CursorDirs;
@group(1) @binding(0) var<uniform> cell: CellUniform;

// ---------- Vertex stage ----------

struct VsIn {
    @location(0) position: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

fn get_strength(i: u32) -> f32 {
    let block = cell.strengths[i / 4u];
    let lane = i % 4u;
    // Dynamic indexing into a vec4 with a runtime u32 isn't portable across
    // all WGSL backends, so we select explicitly.
    if (lane == 0u) { return block.x; }
    if (lane == 1u) { return block.y; }
    if (lane == 2u) { return block.z; }
    return block.w;
}

fn get_crowd(i: u32) -> f32 {
    let block = cell.crowd_bias[i / 4u];
    let lane = i % 4u;
    if (lane == 0u) { return block.x; }
    if (lane == 1u) { return block.y; }
    if (lane == 2u) { return block.z; }
    return block.w;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // The mesh is a unit sphere — vertex position IS the outward normal.
    let normal = normalize(in.position);

    // Sum contributions from all cursors. For each cursor, the contribution
    // depends on how aligned the vertex direction is with the cursor direction.
    //
    // We use (dot+1)/2 raised to a sharpening power as the influence weight.
    // Higher power = tighter cursor "spot" on the surface; lower = wider blob.
    //
    // The total deformation factor — how much the radius bulges or dimples
    // at this point on the surface — is the sum of (strength * weight) over
    // all cursors, plus the same for crowd_bias.
    var total: f32 = 0.0;
    let sharpness: f32 = 6.0; // smaller = softer cursor lobes, larger = pointier

    for (var i: u32 = 0u; i < 360u; i = i + 1u) {
        let cdir = cursors.dirs[i].xyz;
        let aligned = max(0.0, dot(normal, cdir));
        let weight = pow(aligned, sharpness);
        total = total + (get_strength(i) + get_crowd(i)) * weight;
    }

    // The lobes overlap (each surface point is within reach of ~5-10 cursors).
    // Normalize roughly by dividing by an estimate of the overlap factor so the
    // deformation magnitude doesn't scale wildly with cursor count.
    // This factor was tuned empirically for 360 cursors at sharpness=6.
    total = total / 8.0;

    // Apply: radius = base * (1 + total). World position = center + normal * radius.
    let base_radius = cell.center_radius.w;
    let radius = base_radius * (1.0 + total);
    let world_pos = cell.center_radius.xyz + normal * radius;

    // For lighting, the surface normal is no longer just `normal` — the
    // deformation changes the true surface direction. Computing the exact
    // analytic normal would require differentiating the cursor-sum, which is
    // expensive. The rough approximation `normal` is wrong but reads acceptably
    // because the deformations are small. For v1 this is fine; if we want
    // sharper deformations later, switch to screen-space-derivative normals
    // in the fragment shader.
    let world_normal = normal;

    var out: VsOut;
    out.world_pos = world_pos;
    out.world_normal = world_normal;
    out.color = cell.color.rgb;
    out.clip = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

// ---------- Fragment stage ----------

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);

    // Single soft key light from upper-right. Hardcoded for v1; if we want
    // user-controllable lighting later, this becomes a uniform.
    let light_dir = normalize(vec3<f32>(0.6, 0.8, 0.4));

    // Lambertian diffuse — angle between surface and light.
    // Half-Lambert ramp (lit = dot * 0.5 + 0.5) makes shadows less harsh,
    // which is what gives the waxy/skin look. Pure Lambert would have hard
    // terminators that don't fit soft tissue.
    let n_dot_l = dot(n, light_dir);
    let lit = n_dot_l * 0.5 + 0.5;

    // Soft fill from below — gives the underside a faint warm tone instead
    // of going pure black. Fakes the bounced-light look you get in real life.
    let fill_dir = normalize(vec3<f32>(-0.3, -0.8, 0.2));
    let n_dot_f = max(0.0, dot(n, fill_dir));
    let fill_tint = vec3<f32>(0.35, 0.25, 0.20);

    // Ambient — never let the cell go below this.
    let ambient = 0.18;

    let lighting = vec3<f32>(lit * lit) + fill_tint * n_dot_f * 0.4 + vec3<f32>(ambient);
    let final_color = in.color * lighting;

    // Subtle gamma — output is sRGB target, so we want linear-ish here.
    // The pipeline format is srgb so the framebuffer does conversion for us.
    // But we squash highlights very slightly so bright bulges don't blow out.
    let tone_mapped = final_color / (final_color + vec3<f32>(0.6));

    return vec4<f32>(tone_mapped, 1.0);
}
