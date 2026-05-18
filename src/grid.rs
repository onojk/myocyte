// grid.rs — grid initialization helpers.
//
// Keeps CellGrid construction out of cell.rs, which only defines the data shape.

use crate::cell::{Cell19, CellGrid};
use glam::Vec3;

/// Assign each cell's position field based on its (x,y,z) index.
/// Call this once after creating a new CellGrid before rendering.
pub fn place_cells(grid: &mut CellGrid) {
    for x in 0..grid.dims[0] {
        for y in 0..grid.dims[1] {
            for z in 0..grid.dims[2] {
                let idx = grid.idx(x, y, z);
                grid.cells[idx].position = grid.world_pos(x, y, z);
            }
        }
    }
}

/// Activate the single center cell with default visible parameters.
/// CP1 reference — not called in CP2+ but kept for debugging single-cell issues.
#[allow(dead_code)]
pub fn activate_center_cell(grid: &mut CellGrid) {
    let cx = grid.dims[0] / 2;
    let cy = grid.dims[1] / 2;
    let cz = grid.dims[2] / 2;
    let idx = grid.idx(cx, cy, cz);
    let pos = grid.cells[idx].position;
    grid.cells[idx] = Cell19 {
        position: pos,
        ..Cell19::visible_default()
    };
}

/// Activate every cell in the grid at the given scale, all other parameters
/// from Cell19::visible_default(). Used for CP2: uniform grid of identical cells.
///
/// Scale is passed explicitly because the right value depends on spacing:
///   scale=0.30 at spacing=1.0 → ~5.6% alpha at midpoint (clearly separate)
///   scale=0.45 at spacing=1.0 → ~26% alpha at midpoint (merges into fog)
pub fn activate_all_cells(grid: &mut CellGrid, scale: f32) {
    for x in 0..grid.dims[0] {
        for y in 0..grid.dims[1] {
            for z in 0..grid.dims[2] {
                let idx = grid.idx(x, y, z);
                let pos = grid.cells[idx].position;
                grid.cells[idx] = Cell19 {
                    position: pos,
                    scale: Vec3::splat(scale),
                    ..Cell19::visible_default()
                };
            }
        }
    }
}
