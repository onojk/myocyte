// sphere.rs — geometry that lives on every cell.
//
// Two distinct things in here:
//   1. A unit-sphere mesh (vertices + indices) for drawing.
//   2. A set of "cursor anchor" directions on the unit sphere, placed via
//      spherical Fibonacci lattice for as-even-as-mathematically-possible spread.
//
// The mesh is generated once at startup and shared by all four cells —
// per-cell deformation happens in the vertex shader by perturbing each vertex
// along its outward normal based on nearby cursor strengths.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// One vertex on the base sphere mesh.
/// `position` is the unit-sphere position (also doubles as the surface normal
/// for a unit sphere). The vertex shader will scale by radius and apply
/// cursor-driven perturbations along this normal.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub _pad: f32, // align to 16 bytes — keeps GPU happy
}

/// Vertex buffer layout for the render pipeline.
impl Vertex {
    pub fn buffer_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

/// Build a UV sphere with given stack/sector resolution.
///
/// UV sphere is the simplest sphere mesh: imagine horizontal rings (stacks)
/// of vertices, with poles at top and bottom. Not the most uniform topology
/// possible (icosphere is better), but easy to reason about and fine for
/// our purposes — we have ~3k vertices either way and the GPU doesn't care.
pub fn build_sphere(stacks: u32, sectors: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(((stacks + 1) * (sectors + 1)) as usize);
    let mut indices = Vec::with_capacity((stacks * sectors * 6) as usize);

    for i in 0..=stacks {
        // theta: 0 at top pole, PI at bottom pole
        let theta = (i as f32 / stacks as f32) * std::f32::consts::PI;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        for j in 0..=sectors {
            // phi: 0 to 2*PI around the equator
            let phi = (j as f32 / sectors as f32) * std::f32::consts::TAU;
            let sin_p = phi.sin();
            let cos_p = phi.cos();
            let position = [sin_t * cos_p, cos_t, sin_t * sin_p];
            vertices.push(Vertex { position, _pad: 0.0 });
        }
    }

    // Two triangles per quad cell of the (stacks × sectors) grid.
    for i in 0..stacks {
        for j in 0..sectors {
            let a = i * (sectors + 1) + j;
            let b = a + sectors + 1;
            indices.push(a);
            indices.push(b);
            indices.push(a + 1);
            indices.push(a + 1);
            indices.push(b);
            indices.push(b + 1);
        }
    }

    (vertices, indices)
}

/// Spherical Fibonacci lattice — places N points on the unit sphere with
/// near-uniform distribution. Closed form, no iteration.
///
/// The "golden angle" of ~137.5° in 3D maps to a spiral that, when wrapped
/// onto a sphere with cos-distributed latitudes, leaves no large gaps.
/// This is provably about as even as a sphere distribution can get.
pub fn fibonacci_lattice(n: u32) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(n as usize);
    // Golden angle in radians
    let golden = std::f32::consts::PI * (3.0 - (5.0f32).sqrt());
    for i in 0..n {
        // y goes from 1 to -1 evenly (cos of latitude)
        let y = 1.0 - (i as f32 / (n - 1).max(1) as f32) * 2.0;
        let radius_at_y = (1.0 - y * y).sqrt();
        let theta = golden * i as f32;
        let x = theta.cos() * radius_at_y;
        let z = theta.sin() * radius_at_y;
        out.push(Vec3::new(x, y, z));
    }
    out
}
