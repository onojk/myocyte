// rasterizer.rs — wgpu pipeline for tile-based alpha compositing of splat cells.
//
// One pipeline, no depth test (we sort on CPU), premultiplied alpha blending.
// Each frame: upload sorted SplatGpuData to a storage buffer, issue one
// instanced draw call (6 verts × N_cells).

use std::sync::Arc;

use bytemuck::cast_slice;
use winit::window::Window;

use crate::camera::{CameraUniform, OrbitCamera};
use crate::preprocess::SplatGpuData;

/// Maximum cells we'll ever upload in one frame. Sized for 32³ = 32768.
/// The storage buffer is allocated once at this size; only the filled portion
/// is drawn via instance_count in the draw call.
const MAX_SPLATS: usize = 32768;

pub struct Rasterizer {
    pub surface: wgpu::Surface<'static>,
    pub device:  wgpu::Device,
    pub queue:   wgpu::Queue,
    pub config:  wgpu::SurfaceConfiguration,

    pipeline:        wgpu::RenderPipeline,
    camera_buffer:   wgpu::Buffer,
    splat_buffer:    wgpu::Buffer,   // storage buffer, MAX_SPLATS * sizeof(SplatGpuData)
    shared_bind_group: wgpu::BindGroup,
}

impl Rasterizer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface  = instance.create_surface(window.clone()).expect("create surface");
        let adapter  = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("request adapter");

        log::info!("adapter: {:?}", adapter.get_info().name);
        log::info!(
            "max_storage_buffer_binding_size: {} MB",
            adapter.limits().max_storage_buffer_binding_size / (1024 * 1024)
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("myocyte device"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                    memory_hints:      wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("request device");

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage:                          wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:                          size.width.max(1),
            height:                         size.height.max(1),
            present_mode:                   wgpu::PresentMode::Fifo,
            alpha_mode:                     caps.alpha_modes[0],
            view_formats:                   vec![],
            desired_maximum_frame_latency:  2,
        };
        surface.configure(&device, &config);

        // Camera uniform buffer (64+16 = 80 bytes, updated every frame)
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("camera uniform"),
            size:               std::mem::size_of::<CameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Splat storage buffer — write_buffer each frame with sorted splat data.
        // Storage buffers support up to 128MB+ on any modern GPU; 32768 * 80 = 2.6MB.
        let splat_buffer_size = (MAX_SPLATS * std::mem::size_of::<SplatGpuData>()) as u64;
        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("splat storage"),
            size:               splat_buffer_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("shared bgl"),
            entries: &[
                // 0: camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 1: splat storage buffer (read-only from shader perspective)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let shared_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("shared bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buffer.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("splat shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/splat.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("splat pipeline layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("splat pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:               &shader,
                entry_point:          "vs_main",
                buffers:              &[],    // positions computed from instance index
                compilation_options:  Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Premultiplied alpha blend: src=ONE, dst=ONE_MINUS_SRC_ALPHA.
                    // Fragment must output (color*alpha, alpha), not (color, alpha).
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:           wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face:         wgpu::FrontFace::Ccw,
                // No culling: the "back" of a billboard faces the camera the same as the front.
                cull_mode:          None,
                polygon_mode:       wgpu::PolygonMode::Fill,
                unclipped_depth:    false,
                conservative:       false,
            },
            // No depth test: cells are sorted on CPU, composited in painter's order.
            // A depth buffer would incorrectly cull cells that show through transparent ones.
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        Self { surface, device, queue, config, pipeline, camera_buffer, splat_buffer, shared_bind_group }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width  = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    /// Render one frame: upload camera + sorted splat data, issue one draw call.
    pub fn render(
        &mut self,
        camera:    &OrbitCamera,
        splats:    &[SplatGpuData],
    ) -> Result<(), wgpu::SurfaceError> {
        // Upload camera uniform
        let cam = CameraUniform::from_camera(camera);
        self.queue.write_buffer(&self.camera_buffer, 0, cast_slice(&[cam]));

        // Upload sorted splat data (only the filled portion)
        if !splats.is_empty() {
            self.queue.write_buffer(&self.splat_buffer, 0, cast_slice(splats));
        }

        let output = self.surface.get_current_texture()?;
        let view   = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("render encoder") }
        );
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Dark teal background — same aesthetic as v1.
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.05, b: 0.07, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.shared_bind_group, &[]);
            // 6 vertices per quad, N_splats instances, one draw call total.
            rpass.draw(0..6, 0..splats.len() as u32);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
