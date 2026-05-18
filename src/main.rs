// main.rs — application entry point.
//
// Uses winit 0.30's ApplicationHandler trait. The app holds the renderer and
// simulation state; the event loop calls our methods on window/redraw/input
// events.

mod camera;
mod cell;
mod crowding;
mod renderer;
mod sphere;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use camera::OrbitCamera;
use cell::{build_cursor_directions, build_initial_cells, Cell};
use renderer::Renderer;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Option<OrbitCamera>,
    cells: Vec<Cell>,
    cursor_dirs: Vec<glam::Vec3>,
    start_time: Instant,
    last_frame_time: Instant,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            camera: None,
            cells: build_initial_cells(),
            cursor_dirs: build_cursor_directions(),
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.05);
        self.last_frame_time = now;
        let t = (now - self.start_time).as_secs_f32();

        // Per-cell scripts: animate each cell's cursor strengths.
        for cell in &mut self.cells {
            cell.update_script(&self.cursor_dirs, t);
        }

        // Inter-cell crowding physics.
        crowding::step(&mut self.cells, &self.cursor_dirs, dt);

        // Integrate positions (center motion from crowding, damped).
        for cell in &mut self.cells {
            cell.integrate(dt);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Called once at startup (and on Android resume, irrelevant on desktop).
        let attrs = Window::default_attributes()
            .with_title("myocyte — four cells in space")
            .with_inner_size(winit::dpi::LogicalSize::new(960, 720));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let renderer = pollster::block_on(Renderer::new(window.clone(), &self.cursor_dirs));
        let aspect = renderer.config.width as f32 / renderer.config.height as f32;
        let camera = OrbitCamera::new(aspect);

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.camera = Some(camera);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let (Some(r), Some(c)) = (self.renderer.as_mut(), self.camera.as_mut()) {
                    r.resize(size.width, size.height);
                    c.set_aspect(size.width as f32 / size.height.max(1) as f32);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if !self.mouse_pressed {
                        self.last_mouse_pos = None;
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        if let Some(c) = self.camera.as_mut() {
                            c.orbit(dx, dy);
                        }
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 100.0) as f32,
                };
                if let Some(c) = self.camera.as_mut() {
                    c.zoom(scroll);
                }
            }

            WindowEvent::RedrawRequested => {
                self.update();
                if let (Some(r), Some(c), Some(w)) =
                    (self.renderer.as_mut(), self.camera.as_ref(), self.window.as_ref())
                {
                    match r.render(c, &self.cells) {
                        Ok(()) => {}
                        // Surface lost — recreate at current size.
                        Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                            let size = w.inner_size();
                            r.resize(size.width, size.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("out of GPU memory");
                            event_loop.exit();
                        }
                        Err(e) => eprintln!("render error: {:?}", e),
                    }
                    // Request the next frame.
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}
