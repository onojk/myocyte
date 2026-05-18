// camera.rs — orbit camera.
//
// The camera always looks at the origin. Mouse drag changes its azimuth and
// elevation; scroll wheel changes its distance. This is the standard "view a
// thing from any angle" camera and is plenty for v1.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub azimuth: f32,    // around vertical axis, radians
    pub elevation: f32,  // up/down tilt, radians, clamped near poles
    pub fov_y: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl OrbitCamera {
    pub fn new(aspect: f32) -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.5,
            azimuth: 0.5,
            elevation: 0.3,
            fov_y: 60f32.to_radians(),
            aspect,
            z_near: 0.1,
            z_far: 100.0,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let r = self.distance;
        let x = r * self.elevation.cos() * self.azimuth.sin();
        let y = r * self.elevation.sin();
        let z = r * self.elevation.cos() * self.azimuth.cos();
        self.target + Vec3::new(x, y, z)
    }

    pub fn view_proj(&self) -> Mat4 {
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far);
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        proj * view
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * 0.005;
        self.elevation += dy * 0.005;
        // Clamp elevation so we don't flip past the poles (looks bad with up=Y).
        let limit = std::f32::consts::FRAC_PI_2 - 0.05;
        self.elevation = self.elevation.clamp(-limit, limit);
    }

    pub fn zoom(&mut self, delta: f32) {
        // Multiplicative zoom feels right — closer = smaller steps.
        self.distance *= (1.0 - delta * 0.1).clamp(0.5, 2.0);
        self.distance = self.distance.clamp(0.8, 30.0);
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }
}

/// GPU-side camera uniform. 64 bytes (one Mat4) — must match WGSL layout.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub eye: [f32; 3],
    pub _pad: f32,
}

impl CameraUniform {
    pub fn from_camera(cam: &OrbitCamera) -> Self {
        Self {
            view_proj: cam.view_proj().to_cols_array_2d(),
            eye: cam.eye().to_array(),
            _pad: 0.0,
        }
    }
}
