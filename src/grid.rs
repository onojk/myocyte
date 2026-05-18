// grid.rs — grid initialization helpers.
//
// Keeps CellGrid construction out of cell.rs, which only defines the data shape.

use crate::cell::{Cell19, CellGrid};

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
/// Used for CP1: one cell on screen, all others invisible.
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
