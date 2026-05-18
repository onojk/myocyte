// main.rs — application entry point for the Tier 2 myocyte engine.
//
// Orchestrates per-frame work: influencer step → project → sort → render.

mod camera;
mod cell;
mod grid;
mod influencer;
mod preprocess;
mod rasterizer;
mod sort;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use camera::OrbitCamera;
use cell::CellGrid;
use influencer::{gray_scott::GrayScott, Influencer};
use rasterizer::Rasterizer;

/// Grid dimensions for the current checkpoint.
/// CP1: 8³ (512 cells, 1 active). CP2+: 8³ all active. 16³ and 32³ after CP6.
const GRID: u32 = 8;

/// Center-to-center spacing between cells in world units.
const CELL_SPACING: f32 = 1.0;

/// Cell scale for CP2+ (uniform grid). Chosen so midpoint alpha ≈ 5.6%:
/// neighbors are clearly distinct rather than merging into fog.
/// At spacing=1.0: vis_radius=0.699wu, midpoint=0.5wu ≈ 0.72σ from each cell.
const CELL_SCALE_GRID: f32 = 0.30;

/// Camera pull-back distance for an 8³ grid at spacing=1.0.
/// 11.0 wu puts the nearest face 7.5wu ahead, giving 25° half-angle in 60° FOV.
const CAMERA_DISTANCE_8: f32 = 11.0;

struct App {
    window:           Option<Arc<Window>>,
    rasterizer:       Option<Rasterizer>,
    camera:           Option<OrbitCamera>,
    grid:             CellGrid,
    influencer:       GrayScott,
    last_frame:       Instant,
    mouse_pressed:    bool,
    last_mouse_pos:   Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        let dims = [GRID, GRID, GRID];
        let mut g = CellGrid::new(dims, CELL_SPACING);
        grid::place_cells(&mut g);
        grid::activate_all_cells(&mut g, CELL_SCALE_GRID);

        Self {
            window:         None,
            rasterizer:     None,
            camera:         None,
            grid:           g,
            influencer:     GrayScott::new(dims),
            last_frame:     Instant::now(),
            mouse_pressed:  false,
            last_mouse_pos: None,
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32().min(0.05);
        self.last_frame = now;

        self.influencer.step(&mut self.grid, dt);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs  = Window::default_attributes()
            .with_title("myocyte — Tier 2")
            .with_inner_size(winit::dpi::LogicalSize::new(960, 720));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let rasterizer = pollster::block_on(Rasterizer::new(window.clone()));
        let aspect     = rasterizer.config.width as f32 / rasterizer.config.height as f32;
        let mut camera = OrbitCamera::new(aspect);
        camera.distance = CAMERA_DISTANCE_8;

        self.window     = Some(window);
        self.rasterizer = Some(rasterizer);
        self.camera     = Some(camera);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let (Some(r), Some(c)) = (self.rasterizer.as_mut(), self.camera.as_mut()) {
                    r.resize(size.width, size.height);
                    c.set_aspect(size.width as f32 / size.height.max(1) as f32);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if !self.mouse_pressed { self.last_mouse_pos = None; }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        if let Some(c) = self.camera.as_mut() {
                            c.orbit((position.x - lx) as f32, (position.y - ly) as f32);
                        }
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y)  => y,
                    MouseScrollDelta::PixelDelta(p)    => (p.y / 100.0) as f32,
                };
                if let Some(c) = self.camera.as_mut() { c.zoom(scroll); }
            }

            WindowEvent::RedrawRequested => {
                self.update();

                if let (Some(r), Some(cam), Some(w)) = (
                    self.rasterizer.as_mut(),
                    self.camera.as_ref(),
                    self.window.as_ref(),
                ) {
                    // Project all cells, sort back-to-front, render.
                    let mut projected = preprocess::project_grid(&self.grid, cam);
                    let sorted        = sort::sort_back_to_front(&mut projected);

                    match r.render(cam, &sorted) {
                        Ok(())                                                   => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let sz = w.inner_size();
                            r.resize(sz.width, sz.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("out of GPU memory");
                            event_loop.exit();
                        }
                        Err(e) => eprintln!("render error: {:?}", e),
                    }
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn")
    ).init();

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()).expect("run app");
}
