// influencer/mod.rs — trait for procedural drivers of cell parameters.
//
// An Influencer runs a simulation step and writes results back into the
// CellGrid. Tier 2 ships with Gray-Scott (gray_scott.rs). Other influencers
// (audio FFT, scripted keyframes) can be added without changing the trait.

pub mod authored;
pub mod gray_scott;

use crate::cell::CellGrid;

pub trait Influencer {
    /// Advance the influencer's internal state by `dt` seconds (real time),
    /// then write derived values into `grid`.
    fn step(&mut self, grid: &mut CellGrid, dt: f32);
}
