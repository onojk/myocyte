// crowding.rs — the only thing the four cells share: each other's space.
//
// No central nervous system, no shared field, no signaling. Cells "talk" by
// physically pressing on each other. Two effects:
//
//   1. Displacement: if two cells overlap, push them apart along the line
//      between their centers, proportional to overlap depth. This is just
//      soft-body collision.
//
//   2. Compression: where two cells touch, the surface cursors facing the
//      neighbor get a transient NEGATIVE bias added. That makes the surface
//      on the contact side dimple inward, so cells visibly flatten where they
//      meet — like two balloons pressed together.
//
// This is "organically but limited": cells affect each other through pure
// geometry. A bigger cell crowds neighbors more. A smaller one barely touches.
// No telegraph wires. Just bodies in space.

use crate::cell::Cell;
use glam::Vec3;

/// Run one timestep of inter-cell crowding for the whole cluster.
/// O(N²) over cells; with N=4 this is 6 pairs and negligible.
pub fn step(cells: &mut [Cell], cursors: &[Vec3], dt: f32) {
    let n = cells.len();
    // We need to read all cells' positions/radii while writing to one at a time.
    // Snapshot the read-only quantities up front to keep the borrow checker happy.
    let snapshot: Vec<(Vec3, f32)> = cells.iter()
        .map(|c| (c.center, c.effective_radius()))
        .collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let (ci, ri) = snapshot[i];
            let (cj, rj) = snapshot[j];
            let delta = cj - ci;
            let dist = delta.length();
            let sum_r = ri + rj;

            if dist >= sum_r || dist < 1e-6 {
                continue; // not touching, or perfectly coincident (degenerate)
            }

            // How much they overlap, as a fraction of summed radii.
            let overlap = sum_r - dist;
            let overlap_frac = overlap / sum_r;
            let axis = delta / dist; // unit vector from i toward j

            // --- 1. Displacement: push apart ---
            // Stiffness controls how aggressively crowding repels. Too high and
            // cells bounce off each other like beach balls; too low and they
            // pass through. 6.0 is a reasonable middle ground for our scale.
            let stiffness = 6.0;
            let force = axis * overlap * stiffness;
            cells[i].velocity -= force * dt;
            cells[j].velocity += force * dt;

            // --- 2. Compression: dimple the contact side of each cell ---
            // For each cursor on cell i, if it points toward cell j (positive
            // dot with axis), add a negative bias to its crowd_bias proportional
            // to how directly it faces j and how deep the overlap is.
            //
            // This is the "visible squashing" half of the effect — without it,
            // cells just nudge each other but stay perfectly spherical.
            let compression_depth = overlap_frac * 0.7; // tuned for visible-but-not-extreme
            apply_directional_compression(&mut cells[i], cursors, axis, compression_depth);
            // For cell j, the axis points the other way.
            apply_directional_compression(&mut cells[j], cursors, -axis, compression_depth);
        }
    }

    // Decay any leftover compression bias so it doesn't haunt cells after
    // they separate.
    for cell in cells.iter_mut() {
        cell.decay_crowd_bias(dt);
    }
}

/// Add negative bias to every cursor on `cell` whose direction faces `axis`.
/// The bias is strongest at exact alignment (dot = 1) and falls off via a
/// cosine-shaped lobe so the dimple has a soft, rounded boundary instead of
/// a sharp edge.
fn apply_directional_compression(
    cell: &mut Cell,
    cursors: &[Vec3],
    axis: Vec3,
    depth: f32,
) {
    for (i, dir) in cursors.iter().enumerate() {
        let alignment = dir.dot(axis);
        if alignment > 0.0 {
            // Cosine raised to a power makes the lobe tighter. Power=3 = soft
            // dimple about a third of the sphere wide.
            let lobe = alignment.powi(3);
            // Negative because we're pushing the surface IN, not out.
            cell.crowd_bias[i] -= lobe * depth;
        }
    }
}
