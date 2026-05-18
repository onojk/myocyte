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
use influencer::{
    authored::{Authored, AuthoredShape},
    gray_scott::GrayScott,
    Influencer,
};
use rasterizer::Rasterizer;

// ---- Checkpoint constants ---------------------------------------------------
//
// Update GRID and CAMERA_DISTANCE together when scaling up.
//   CP1/2:  8, 11.0   |  CP3+:  16, 22.0  |  CP6:  32, 42.0

/// Grid dimension per axis. 16³ = 4096 cells for CP3.
const GRID: u32 = 16;

/// Center-to-center spacing between cells in world units.
const CELL_SPACING: f32 = 1.0;

/// Cell scale base for activate_varied_cells.
/// At spacing=1.0, scale=0.30 gives ~5.6% alpha at the midpoint between neighbors.
const CELL_SCALE_GRID: f32 = 0.30;

/// Initial camera distance sized for the current GRID.
/// 16³: grid half-extent 7.5wu, distance 22.0 → 27° half-angle in 60° FOV.
const CAMERA_DISTANCE: f32 = 22.0;

// ---- App state --------------------------------------------------------------

struct App {
    window:          Option<Arc<Window>>,
    rasterizer:      Option<Rasterizer>,
    camera:          Option<OrbitCamera>,
    grid:            CellGrid,
    influencer:      Box<dyn Influencer>,
    mode:            &'static str,   // "rd" | "sphere" | "shell" | "letter-a"
    last_frame:      Instant,
    mouse_pressed:   bool,
    last_mouse_pos:  Option<(f64, f64)>,
    // FPS and simulation-time tracking
    frame_count:     u32,
    fps_timer:       Instant,
    sim_time:        f32,   // accumulated RD simulation time; unused for authored modes
}

impl App {
    fn new(shape: Option<&str>) -> Self {
        let dims = [GRID, GRID, GRID];
        let mut g = CellGrid::new(dims, CELL_SPACING);
        grid::place_cells(&mut g);
        grid::activate_varied_cells(&mut g, CELL_SCALE_GRID);

        let (influencer, mode): (Box<dyn Influencer>, &'static str) = match shape {
            Some("sphere")   => (Box::new(Authored::new(AuthoredShape::Sphere)),  "sphere"),
            Some("shell")    => (Box::new(Authored::new(AuthoredShape::Shell)),   "shell"),
            Some("a") | Some("letter-a") => (Box::new(Authored::new(AuthoredShape::LetterA)), "letter-a"),
            _                => (Box::new(GrayScott::new(dims)),                  "rd"),
        };

        Self {
            window:         None,
            rasterizer:     None,
            camera:         None,
            grid:           g,
            influencer,
            mode,
            last_frame:     Instant::now(),
            mouse_pressed:  false,
            last_mouse_pos: None,
            frame_count:    0,
            fps_timer:      Instant::now(),
            sim_time:       0.0,
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt  = (now - self.last_frame).as_secs_f32().min(0.05);
        self.last_frame = now;
        self.influencer.step(&mut self.grid, dt);
        if self.mode == "rd" {
            self.sim_time += influencer::gray_scott::DT_RD;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs  = Window::default_attributes()
            .with_title("myocyte")
            .with_inner_size(winit::dpi::LogicalSize::new(960, 720));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let rasterizer = pollster::block_on(Rasterizer::new(window.clone()));
        let aspect     = rasterizer.config.width as f32 / rasterizer.config.height as f32;
        let mut camera = OrbitCamera::new(aspect);
        camera.distance = CAMERA_DISTANCE;
        // letter-a: start face-on for legibility; other modes use default orbit angle.
        if self.mode == "letter-a" {
            camera.azimuth   = 0.0;
            camera.elevation = 0.0;
        }

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
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p)   => (p.y / 100.0) as f32,
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
                    let mut projected = preprocess::project_grid(&self.grid, cam);
                    let sorted        = sort::sort_back_to_front(&mut projected);

                    match r.render(cam, &sorted) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let sz = w.inner_size();
                            r.resize(sz.width, sz.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("out of GPU memory");
                            event_loop.exit();
                        }
                        // Timeout is transient (GPU didn't return frame in time); skip silently.
                        Err(wgpu::SurfaceError::Timeout) => {}
                    }

                    // Update window title with fps once per second.
                    self.frame_count += 1;
                    let elapsed = self.fps_timer.elapsed().as_secs_f32();
                    if elapsed >= 1.0 {
                        let fps   = self.frame_count as f32 / elapsed;
                        let title = if self.mode == "rd" {
                            format!("myocyte  {:.0} fps  {}³ ({} cells)  t={:.1}s",
                                fps, GRID, self.grid.len(), self.sim_time)
                        } else {
                            format!("myocyte  {:.0} fps  {}³ ({} cells)  {}",
                                fps, GRID, self.grid.len(), self.mode)
                        };
                        w.set_title(&title);
                        self.frame_count = 0;
                        self.fps_timer   = Instant::now();
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

    // Optional: --shape=sphere | shell | a | letter-a
    // Default (no flag): Gray-Scott reaction-diffusion.
    let shape = std::env::args()
        .find_map(|a| a.strip_prefix("--shape=").map(str::to_owned));

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new(shape.as_deref())).expect("run app");
}
