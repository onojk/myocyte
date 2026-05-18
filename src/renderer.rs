// renderer.rs — wgpu rendering, all the GPU plumbing.
//
// The high-level shape:
//   - One shared sphere mesh on the GPU (vertex + index buffer)
//   - One shared cursor directions buffer (uniform, 360 vec3s)
//   - One camera uniform (updated each frame)
//   - One PER-CELL uniform buffer holding center, radius, color, and per-cursor
//     strength + crowd-bias arrays
//   - One render pipeline that vertex-shades each sphere by perturbing each
//     vertex along its normal based on the cursor data
//   - One depth texture for proper 3D occlusion
//
// We draw 4 spheres by issuing 4 separate draw calls, binding a different
// per-cell uniform each time. This isn't the most efficient possible (instanced
// rendering would do one call) but it keeps the code clear and at N=4 the
// difference is unmeasurable.

use crate::camera::{CameraUniform, OrbitCamera};
use crate::cell::{Cell, CURSORS_PER_CELL, NUM_CELLS};
use crate::sphere::{build_sphere, Vertex};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Per-cell uniform — uploaded every frame. WGSL std140-ish alignment rules
/// require careful padding. Sticking to vec4-aligned blocks keeps it simple.
///
/// Layout:
///   center_radius : vec4<f32>  (xyz=center, w=base_radius * size_bias)
///   color         : vec4<f32>  (xyz=albedo, w=unused)
///   strengths     : array<vec4<f32>, 90>  (4 floats packed per vec4 → 360 entries)
///   crowd_bias    : array<vec4<f32>, 90>
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CellUniform {
    pub center_radius: [f32; 4],
    pub color: [f32; 4],
    pub strengths: [[f32; 4]; CURSORS_PER_CELL / 4],
    pub crowd_bias: [[f32; 4]; CURSORS_PER_CELL / 4],
}

impl CellUniform {
    pub fn from_cell(cell: &Cell) -> Self {
        let mut strengths = [[0.0f32; 4]; CURSORS_PER_CELL / 4];
        let mut crowd_bias = [[0.0f32; 4]; CURSORS_PER_CELL / 4];
        for i in 0..CURSORS_PER_CELL {
            strengths[i / 4][i % 4] = cell.strengths[i];
            crowd_bias[i / 4][i % 4] = cell.crowd_bias[i];
        }
        let r = cell.effective_radius();
        Self {
            center_radius: [cell.center.x, cell.center.y, cell.center.z, r],
            color: [cell.color[0], cell.color[1], cell.color[2], 1.0],
            strengths,
            crowd_bias,
        }
    }
}

/// Cursor directions uniform — one shared buffer for all cells. Same layout
/// trick: packed 4 cursors per vec4 line (only xyz used, w unused).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CursorDirs {
    pub dirs: [[f32; 4]; CURSORS_PER_CELL],
}

impl CursorDirs {
    pub fn from_vecs(v: &[Vec3]) -> Self {
        let mut dirs = [[0.0f32; 4]; CURSORS_PER_CELL];
        for i in 0..CURSORS_PER_CELL {
            dirs[i] = [v[i].x, v[i].y, v[i].z, 0.0];
        }
        Self { dirs }
    }
}

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,

    camera_buffer: wgpu::Buffer,
    cursors_buffer: wgpu::Buffer,
    cell_buffers: Vec<wgpu::Buffer>,

    // Bind groups: 0 = camera + cursors (shared), 1 = per-cell (one per cell)
    shared_bind_group: wgpu::BindGroup,
    cell_bind_groups: Vec<wgpu::BindGroup>,

    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, cursor_dirs: &[Vec3]) -> Self {
        let size = window.inner_size();

        // === 1. Instance + Surface + Adapter ===
        // The Instance is wgpu's entry point. We ask for a surface bound to
        // our window, then ask the OS for a graphics adapter that can render to it.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("request adapter");

        // === 2. Device + Queue ===
        // The Device is what we create resources from; the Queue is what we
        // submit commands to. Both are owned by us for the rest of the program.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("myocyte device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("request device");

        // === 3. Surface configuration ===
        // Picks the swap-chain format and present mode for the window.
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo, // vsync — predictable framerate
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // === 4. Sphere mesh ===
        // 48 stacks × 64 sectors = ~3100 vertices, ~6000 triangles per cell.
        // Smooth enough to hide the underlying topology when deformed.
        let (vertices, indices) = build_sphere(48, 64);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let index_count = indices.len() as u32;

        // === 5. Uniform buffers ===
        // Camera and cursor directions live in "shared" bind group 0.
        // They're set up once with reasonable initial data; the camera one
        // gets rewritten every frame, the cursors one never changes.
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cursors_uniform = CursorDirs::from_vecs(cursor_dirs);
        let cursors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cursor directions"),
            contents: bytemuck::cast_slice(&[cursors_uniform]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let cell_buffers: Vec<wgpu::Buffer> = (0..NUM_CELLS).map(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("cell {} uniform", i)),
                size: std::mem::size_of::<CellUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }).collect();

        // === 6. Bind group layouts ===
        // These describe the SHAPE of bind groups — what each binding slot
        // contains. The actual binding to buffers happens in step 7.
        let shared_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shared bgl"),
            entries: &[
                // 0: camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: cursor directions
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let cell_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cell bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // === 7. Bind groups ===
        let shared_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared bg"),
            layout: &shared_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cursors_buffer.as_entire_binding() },
            ],
        });
        let cell_bind_groups: Vec<wgpu::BindGroup> = cell_buffers.iter().enumerate().map(|(i, buf)| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("cell {} bg", i)),
                layout: &cell_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            })
        }).collect();

        // === 8. Shader + Pipeline ===
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sphere shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sphere.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[&shared_bgl, &cell_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sphere pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // === 9. Depth buffer ===
        let (depth_texture, depth_view) = create_depth_texture(&device, &config);

        Self {
            surface, device, queue, config,
            pipeline, vertex_buffer, index_buffer, index_count,
            camera_buffer, cursors_buffer, cell_buffers,
            shared_bind_group, cell_bind_groups,
            depth_texture, depth_view,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        let (tex, view) = create_depth_texture(&self.device, &self.config);
        self.depth_texture = tex;
        self.depth_view = view;
    }

    /// Render one frame. Called from the event loop.
    pub fn render(&mut self, camera: &OrbitCamera, cells: &[Cell]) -> Result<(), wgpu::SurfaceError> {
        // Upload camera and per-cell uniforms
        let cam_uniform = CameraUniform::from_camera(camera);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[cam_uniform]));
        for (i, cell) in cells.iter().enumerate() {
            let cell_uniform = CellUniform::from_cell(cell);
            self.queue.write_buffer(&self.cell_buffers[i], 0, bytemuck::cast_slice(&[cell_uniform]));
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Soft dark teal background — flat, no gradient. Looks like
                        // a microscope field at low light.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04, g: 0.05, b: 0.07, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.set_bind_group(0, &self.shared_bind_group, &[]);
            // One draw call per cell — bind that cell's uniform, draw the mesh.
            for cell_bg in &self.cell_bind_groups {
                rpass.set_bind_group(1, cell_bg, &[]);
                rpass.draw_indexed(0..self.index_count, 0, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
