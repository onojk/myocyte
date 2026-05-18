// gray_scott.rs — 3D Gray-Scott reaction-diffusion influencer.
//
// Stub for CP4. The struct exists so main.rs can construct it; step() is a
// no-op until CP4. This lets the module tree compile cleanly at CP1.

use crate::cell::CellGrid;
use crate::influencer::Influencer;

pub struct GrayScott {
    // Field state will be added at CP4.
    _dims: [u32; 3],
}

impl GrayScott {
    pub fn new(dims: [u32; 3]) -> Self {
        Self { _dims: dims }
    }
}

impl Influencer for GrayScott {
    fn step(&mut self, _grid: &mut CellGrid, _dt: f32) {
        // No-op until CP4.
    }
}
