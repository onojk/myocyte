// cell.rs — one myocyte cell.
//
// A Cell owns:
//   - A center position in world space
//   - A base radius (the size it tends to relax toward)
//   - A current "size bias" (how inflated/deflated it currently is overall)
//   - A velocity for its center (so it can be displaced and settle back)
//   - An array of cursor strengths, one per Fibonacci direction
//
// Each frame, the cell:
//   1. Runs its cursor-strength script — sine waves with phase offsets that
//      animate the surface deformation pattern.
//   2. Applies external influences (crowding from other cells — handled in
//      crowding.rs, this module just stores the resulting nudges).
//   3. Decays its position velocity toward zero so it eventually settles.
//
// We hold cursor directions in a single shared Vec (since they're identical
// across cells — same Fibonacci lattice on every unit sphere). Only the
// strengths vary per cell.

use crate::sphere::fibonacci_lattice;
use glam::Vec3;

/// Number of surface cursors per cell. Divisor of 360 per the design rule.
/// 360 is the chosen number — dense enough for smooth deformation, sparse
/// enough that you can see individual cursors when they fire hard.
pub const CURSORS_PER_CELL: usize = 360;

/// Number of cells in the scene.
pub const NUM_CELLS: usize = 4;

/// One cell's mutable state.
pub struct Cell {
    pub center: Vec3,             // world-space position of the sphere's center
    pub velocity: Vec3,           // damped — only nonzero during crowding events
    pub base_radius: f32,         // size it wants to be
    pub size_bias: f32,           // current global inflation factor (1.0 = normal)
    pub strengths: Vec<f32>,      // per-cursor signed strengths, len = CURSORS_PER_CELL
    pub crowd_bias: Vec<f32>,     // additional per-cursor strength from being crowded
    pub color: [f32; 3],          // RGB albedo for this cell
    pub phase: f32,               // per-cell phase offset, so the four cells don't
                                  // breathe in perfect unison
}

impl Cell {
    pub fn new(center: Vec3, base_radius: f32, color: [f32; 3], phase: f32) -> Self {
        Self {
            center,
            velocity: Vec3::ZERO,
            base_radius,
            size_bias: 1.0,
            strengths: vec![0.0; CURSORS_PER_CELL],
            crowd_bias: vec![0.0; CURSORS_PER_CELL],
            color,
            phase,
        }
    }

    /// Run the per-cell cursor script for one timestep.
    ///
    /// The script produces a slow, organic breathing pattern: each cursor's
    /// strength is a sine wave whose phase depends on the cursor's position
    /// on the sphere AND on the cell's overall phase. This creates a "wave
    /// traveling across the surface" rather than every cursor firing in unison.
    ///
    /// Signed strengths — positive bulges the surface outward, negative dimples
    /// it inward. Net energy averages to zero so the cell stays around its
    /// base radius (no drift toward over-inflation).
    pub fn update_script(&mut self, cursors: &[Vec3], t: f32) {
        // Three frequencies superimposed gives a less mechanical feel than one.
        let f1 = 0.7;
        let f2 = 1.3;
        let f3 = 0.45;

        for (i, dir) in cursors.iter().enumerate() {
            // Use the cursor's direction on the sphere to phase-modulate the wave.
            // dir.y controls "wave from top to bottom", dir.x controls "wave around".
            let p1 = (t * f1 + self.phase + dir.y * 2.0).sin();
            let p2 = (t * f2 + self.phase * 1.5 + dir.x * 3.0).sin();
            let p3 = (t * f3 + dir.dot(Vec3::ONE) * 1.2).sin();
            // Combine the three. Magnitude kept modest so net deformation is
            // a percentage of base_radius, not a multiple of it.
            let s = (p1 * 0.5 + p2 * 0.3 + p3 * 0.2) * 0.18;
            self.strengths[i] = s;
        }
    }

    /// Decay the crowding bias each frame so it doesn't accumulate forever.
    /// Crowding events deposit transient strength on cursors facing a neighbor;
    /// without decay, that strength would stay even after the neighbor moved away.
    pub fn decay_crowd_bias(&mut self, dt: f32) {
        let decay = (-3.0 * dt).exp(); // half-life ~0.23s
        for s in &mut self.crowd_bias {
            *s *= decay;
        }
    }

    /// Apply velocity to center, then damp it.
    pub fn integrate(&mut self, dt: f32) {
        self.center += self.velocity * dt;
        // Damping — without this the cells would oscillate forever.
        self.velocity *= (-4.0 * dt).exp();
        // Also restore center toward origin gently so the cluster doesn't drift away.
        let restore = -self.center * 0.5;
        self.velocity += restore * dt;
    }

    /// The "effective" radius of this cell right now — the average across the
    /// surface after cursor deformation. Used by the crowding code to decide
    /// if cells are overlapping. We approximate by base_radius * size_bias.
    pub fn effective_radius(&self) -> f32 {
        self.base_radius * self.size_bias
    }
}

/// Build the initial set of four cells, identical and arranged in a square row.
///
/// They're placed close enough that natural cursor-driven breathing won't make
/// them touch most of the time, but a strong inflation will cause crowding.
/// This is the "all four spheres identical, lined up in a square" arrangement.
pub fn build_initial_cells() -> Vec<Cell> {
    let r = 0.5;
    let gap = 1.15; // center-to-center spacing as multiple of (2 * r)
    let d = 2.0 * r * gap;
    let positions = [
        Vec3::new(-d * 0.5, -d * 0.5, 0.0),
        Vec3::new(d * 0.5, -d * 0.5, 0.0),
        Vec3::new(-d * 0.5, d * 0.5, 0.0),
        Vec3::new(d * 0.5, d * 0.5, 0.0),
    ];
    // Soft tissue colors — subtle variation so cells are visually distinguishable
    // without looking like four different species.
    let colors = [
        [0.92, 0.78, 0.72], // warm pink
        [0.88, 0.82, 0.75], // peach
        [0.85, 0.75, 0.70], // dusty rose
        [0.90, 0.80, 0.73], // cream
    ];
    // Phase offsets — quarter-cycle apart so the four cells stagger their breathing.
    let phases = [0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI,
                  std::f32::consts::PI + std::f32::consts::FRAC_PI_2];

    positions.iter().enumerate().map(|(i, &p)| {
        Cell::new(p, r, colors[i], phases[i])
    }).collect()
}

/// Build the shared cursor direction lattice. Same for every cell.
pub fn build_cursor_directions() -> Vec<Vec3> {
    fibonacci_lattice(CURSORS_PER_CELL as u32)
}
