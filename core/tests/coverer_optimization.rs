// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Torgeir Børresen <tb@starkad.no>
// Written for this crate; not ported from upstream S2.

//! Correctness tests that guard `RegionCoverer` performance work.
//!
//! The checks rely on geometric invariants, not on exact cell ids. Tied
//! candidate priorities, a different heap layout, or a different time to
//! normalize the result can change which equally good covering comes back.
//! Every covering must still:
//!
//! - be valid (sorted and non-overlapping) and obey `min_level`, `max_level`
//!   and `level_mod`;
//! - cover the region: this mirrors C++ `CheckCovering` from
//!   `s2region_coverer_test.cc`, and also tests sampled points;
//! - for an interior covering, contain only cells that the region contains;
//! - be the same on every call.
//!
//! Budget behavior is checked against baseline golden outputs separately.
//! This port does not obey the stronger C++ cell-budget invariant, even
//! without level constraints when initial candidates are already terminal.
//!
//! Fixed-level coverings (`min_level == max_level`) have only one correct
//! answer, so those tests compare exact sets against a brute-force search.
//! Other tests compare `Cell::children` with `Cell::from_cell_id` on every
//! face and level.

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use s2rst::r1;
use s2rst::r3::Vector;
use s2rst::s1;
use s2rst::s1::Angle;
use s2rst::s2::polyline::Polyline;
use s2rst::s2::region_coverer::RegionCoverer;
use s2rst::s2::shape_index_region::ShapeIndexRegion;
use s2rst::s2::{Cap, Cell, CellId, CellUnion, LatLng, Loop, Point, Polygon, Rect, Region};

const MAX_LEVEL: u8 = 30;

/// Upper bound on the estimated cell count of one fixture covering. It keeps
/// debug-build run time small.
const MAX_ESTIMATED_CELLS: f64 = 6000.0;

/// Highest `max_level` used for lines (zero-area regions that are not points).
///
/// An interior covering of a line has no cells, so no candidate becomes
/// terminal and the `max_cells` budget never stops the search. The coverer
/// then subdivides every cell along the line down to `max_level`: about
/// `2^level` cells per level. C++ `GetInteriorCovering` does the same; it
/// is a cost of the algorithm, not of these tests. Points are cheap at any
/// level, because only the 1 to 4 cells around the point intersect it.
const LINE_MAX_LEVEL: u8 = 10;

/// These tests use the polygon's index. The direct
/// `Polygon::contains_cell` only checks vertices and can miss an enclosed hole.
#[derive(Debug)]
struct IndexedPolygon(Polygon);

impl IndexedPolygon {
    fn region(&self) -> ShapeIndexRegion<'_> {
        ShapeIndexRegion::new(self.0.shape_index())
    }
}

impl Region for IndexedPolygon {
    fn cap_bound(&self) -> Cap {
        self.region().cap_bound()
    }
    fn rect_bound(&self) -> Rect {
        self.region().rect_bound()
    }
    fn cell_union_bound(&self) -> Vec<CellId> {
        self.region().cell_union_bound()
    }
    fn contains_cell(&self, cell: &Cell) -> bool {
        self.region().contains_cell(cell)
    }
    fn intersects_cell(&self, cell: &Cell) -> bool {
        self.region().intersects_cell(cell)
    }
    fn contains_point(&self, p: &Point) -> bool {
        Region::contains_point(&self.0, p)
    }
}

fn ll(lat: f64, lng: f64) -> Point {
    LatLng::from_degrees(lat, lng).to_point()
}

fn deg_rect(lat_lo: f64, lat_hi: f64, lng_lo: f64, lng_hi: f64) -> Rect {
    Rect::new(
        r1::Interval::new(lat_lo.to_radians(), lat_hi.to_radians()),
        s1::Interval::new(lng_lo.to_radians(), lng_hi.to_radians()),
    )
}

fn deg_cap(center: Point, radius_degrees: f64) -> Cap {
    Cap::from_center_angle(center, Angle::from_degrees(radius_degrees))
}

/// Returns a star-shaped CCW loop around `center`, with one radius in degrees
/// per vertex. The vertices sit at equal angles around the center.
fn star_loop(center: Point, radii_degrees: &[f64]) -> Loop {
    let center_vector = center.vector().normalize();
    let tangent = center_vector.ortho().normalize();
    // tangent x bitangent == center_vector, so increasing theta goes CCW seen from outside the sphere.
    let bitangent = center_vector.cross(tangent);
    let vertex_count = radii_degrees.len() as f64;
    let mut vertices = Vec::with_capacity(radii_degrees.len());
    for (vertex_index, &radius) in radii_degrees.iter().enumerate() {
        let theta = 2.0 * PI * vertex_index as f64 / vertex_count;
        let radius_radians = radius.to_radians();
        let direction: Vector = tangent * theta.cos() + bitangent * theta.sin();
        vertices.push(Point::new(
            (center_vector * radius_radians.cos() + direction * radius_radians.sin()).normalize(),
        ));
    }
    Loop::new(vertices)
}

// ─── Covering checks ────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Options {
    min_level: u8,
    max_level: u8,
    level_mod: u8,
    max_cells: usize,
}

impl Options {
    const fn new(min_level: u8, max_level: u8, level_mod: u8, max_cells: usize) -> Self {
        Options {
            min_level,
            max_level,
            level_mod,
            max_cells,
        }
    }

    /// Returns these options with `max_level` at most `level`, or `None` if
    /// `min_level` is above `level`.
    fn with_max_level_at_most(self, level: u8) -> Option<Self> {
        (self.min_level <= level).then_some(Options {
            max_level: self.max_level.min(level),
            ..self
        })
    }

    fn coverer(self) -> RegionCoverer {
        RegionCoverer::new()
            .min_level(self.min_level)
            .max_level(self.max_level)
            .level_mod(self.level_mod)
            .max_cells(self.max_cells)
    }
}

/// Mirrors C++ `CheckCovering(region, covering, check_tight, id)`.
///
/// Every cell that the region contains must be contained by the covering.
/// When `check_tight` is set, a cell that the region does not intersect must
/// not intersect the covering. The search only descends where the covering
/// does not already contain the cell, so its cost follows the covering size.
fn check_covers_region(
    region: &dyn Region,
    covering: &CellUnion,
    check_tight: bool,
    id: CellId,
    label: &str,
) {
    let cell = Cell::from_cell_id(id);
    if !region.intersects_cell(&cell) {
        if check_tight {
            assert!(
                !covering.intersects_cell_id(id),
                "{label}: covering intersects {id:?}, which the region does not intersect"
            );
        }
    } else if !covering.contains_cell_id(id) {
        assert!(
            !region.contains_cell(&cell),
            "{label}: region contains {id:?}, but the covering does not"
        );
        assert!(
            !id.is_leaf(),
            "{label}: region intersects leaf {id:?}, but the covering does not"
        );
        for child in id.children() {
            check_covers_region(region, covering, check_tight, child, label);
        }
    }
}

fn check_covers_region_from_faces(
    region: &dyn Region,
    covering: &CellUnion,
    check_tight: bool,
    label: &str,
) {
    // contains_cell_id tests for a single containing member. A denormalized
    // union can cover a parent collectively with its children, so normalize
    // before asking it whether it contains a whole cell.
    let mut covering = covering.clone();
    covering.normalize();
    for face in 0..6u8 {
        check_covers_region(
            region,
            &covering,
            check_tight,
            CellId::from_face(face),
            label,
        );
    }
}

/// Reports whether `id` is a cell the region contains, or a union of such
/// cells no finer than `max_level`.
///
/// An interior covering may hold a parent made by merging four contained
/// children during normalization (C++ merges in the same way). The region's
/// `contains_cell` can still be false for that parent: for example,
/// `Region for Cap` tests the cell's bounding cap, which is conservative.
fn is_union_of_contained_cells(region: &dyn Region, id: CellId, max_level: u8) -> bool {
    if region.contains_cell(&Cell::from_cell_id(id)) {
        return true;
    }
    id.level().as_u8() < max_level
        && id
            .children()
            .into_iter()
            .all(|child| is_union_of_contained_cells(region, child, max_level))
}

/// How `max_cells` limits a covering that has more cells than requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BudgetRule {
    /// C++: the extra cells are allowed only when `min_level` forces them.
    Cpp,
    /// Check geometric/level invariants; budget parity is covered by the
    /// golden/differential corpus. Initial terminal cells and later
    /// denormalization can both violate the stronger C++ budget rule.
    Port,
}

/// Checks validity, levels, and the `max_cells` budget, as C++
/// `CheckCovering(options, region, covering, interior)` does.
fn check_levels_and_budget(options: Options, ids: &[CellId], rule: BudgetRule, label: &str) {
    assert!(
        CellUnion::from_verbatim(ids.to_vec()).is_valid(),
        "{label}: result is not sorted and non-overlapping: {ids:?}"
    );
    let mut min_level_ancestors: HashMap<CellId, usize> = HashMap::new();
    for &id in ids {
        let level = id.level().as_u8();
        assert!(
            (options.min_level..=options.max_level).contains(&level),
            "{label}: {id:?} has level {level} outside the allowed range"
        );
        assert_eq!(
            (level - options.min_level) % options.level_mod,
            0,
            "{label}: {id:?} has level {level}, which does not satisfy level_mod"
        );
        *min_level_ancestors
            .entry(id.parent_at_level(options.min_level))
            .or_default() += 1;
    }
    // More than `max_cells` cells is allowed only when `min_level` forces it.
    // Then no two cells may share an ancestor at `min_level`, because that
    // ancestor could have replaced them.
    if rule == BudgetRule::Cpp && ids.len() > options.max_cells {
        for (ancestor, count) in &min_level_ancestors {
            assert_eq!(
                *count,
                1,
                "{label}: {} cells exceed max_cells {}, but {count} of them share the \
                 min_level ancestor {ancestor:?}",
                ids.len(),
                options.max_cells
            );
        }
    }
}

/// Runs every coverer entry point on `region` and checks the results.
fn check_region(options: Options, region: &dyn Region, samples: &[Point], label: &str) {
    check_region_with_rule(options, region, samples, BudgetRule::Port, label);
}

fn check_region_with_rule(
    options: Options,
    region: &dyn Region,
    samples: &[Point],
    rule: BudgetRule,
    label: &str,
) {
    let label = format!("{label} {options:?}");
    let label = label.as_str();
    let coverer = options.coverer();

    let covering = coverer.covering(region);
    check_levels_and_budget(options, covering.cell_ids(), rule, label);
    check_covers_region_from_faces(region, &covering, true, label);
    for &p in samples {
        if region.contains_point(&p) {
            assert!(
                covering.contains_point(p),
                "{label}: covering misses region point {p:?}"
            );
        }
    }
    assert_eq!(
        covering,
        coverer.covering(region),
        "{label}: covering is not deterministic"
    );

    let interior = coverer.interior_covering(region);
    check_levels_and_budget(
        options,
        interior.cell_ids(),
        rule,
        &format!("{label} interior"),
    );
    for &id in interior.cell_ids() {
        let cell = Cell::from_cell_id(id);
        assert!(
            is_union_of_contained_cells(region, id, options.max_level),
            "{label}: interior cell {id:?} is not inside the region"
        );
        assert!(
            region.contains_point(&cell.center()),
            "{label}: center of interior cell {id:?} is not inside the region"
        );
    }
    let mut normalized_covering = covering.clone();
    normalized_covering.normalize();
    assert!(
        normalized_covering.contains_union(&interior),
        "{label}: covering does not contain the interior covering"
    );
    assert_eq!(
        interior,
        coverer.interior_covering(region),
        "{label}: interior covering is not deterministic"
    );

    // `cell_union` skips the final denormalization, so it must be normalized
    // and must cover the same leaf cells as `covering`.
    let union = coverer.cell_union(region);
    assert!(
        union.is_normalized(),
        "{label}: cell_union result is not normalized"
    );
    assert_eq!(
        union, normalized_covering,
        "{label}: cell_union and normalized covering differ"
    );

    let fast = coverer.fast_covering(region);
    assert!(
        fast.is_valid(),
        "{label}: fast covering is not sorted and non-overlapping"
    );
    check_covers_region_from_faces(region, &fast, false, &format!("{label} fast"));
}

// ─── Fixtures ───────────────────────────────────────────────────────────

struct Fixture {
    name: &'static str,
    region: Box<dyn Region>,
    /// Points to test for containment, in addition to random samples.
    extra_points: Vec<Point>,
    /// Set for lines, whose coverings are limited to [`LINE_MAX_LEVEL`].
    is_line: bool,
}

impl Fixture {
    fn polygon(name: &'static str, polygon: Polygon) -> Self {
        assert!(polygon.find_validation_error().is_none(), "{name}");
        Self::new(name, IndexedPolygon(polygon))
    }
    fn new(name: &'static str, region: impl Region + 'static) -> Self {
        Fixture {
            name,
            region: Box::new(region),
            extra_points: Vec::new(),
            is_line: false,
        }
    }

    fn line(name: &'static str, region: impl Region + 'static) -> Self {
        Fixture {
            is_line: true,
            ..Fixture::new(name, region)
        }
    }

    /// Returns `options` limited as this fixture requires, or `None` if the
    /// fixture must skip them.
    fn limit(&self, options: Options) -> Option<Options> {
        if self.is_line {
            options.with_max_level_at_most(LINE_MAX_LEVEL)
        } else {
            Some(options)
        }
    }

    fn with_points(mut self, points: Vec<Point>) -> Self {
        self.extra_points = points;
        self
    }
}

fn fixtures() -> Vec<Fixture> {
    let face_corner = Point::from_coords(1.0, 1.0, 1.0).normalize();
    let north_pole = Point::from_coords(0.0, 0.0, 1.0);
    let south_pole = Point::from_coords(0.0, 0.0, -1.0);

    let shell = Loop::new(vec![
        ll(-20.0, -20.0),
        ll(-20.0, 20.0),
        ll(20.0, 20.0),
        ll(20.0, -20.0),
    ]);
    let hole = Loop::new(vec![
        ll(-5.0, -5.0),
        ll(-5.0, 5.0),
        ll(5.0, 5.0),
        ll(5.0, -5.0),
    ]);
    let nested_shell = star_loop(ll(10.0, 100.0), &[30.0; 12]);
    let nested_hole = star_loop(ll(10.0, 100.0), &[15.0; 12]);
    let nested_island = star_loop(ll(10.0, 100.0), &[5.0; 12]);

    let cell_10 = Cell::from_cell_id(CellId::from_point(&ll(48.85, 2.35)).parent_at_level(10));

    vec![
        // Caps.
        Fixture::new("empty cap", Cap::empty()),
        Fixture::new("full cap", Cap::full()),
        Fixture::new("small cap", deg_cap(ll(40.0, -74.0), 1.0)),
        Fixture::new("north pole cap", deg_cap(north_pole, 5.0)).with_points(vec![north_pole]),
        Fixture::new("south pole cap", deg_cap(south_pole, 30.0)).with_points(vec![south_pole]),
        Fixture::new("hemisphere cap", deg_cap(ll(10.0, 20.0), 90.0)),
        Fixture::new(
            "cap larger than hemisphere",
            deg_cap(ll(-35.0, 140.0), 150.0),
        ),
        Fixture::new("dateline cap", deg_cap(ll(0.0, 180.0), 3.0))
            .with_points(vec![ll(0.0, 179.0), ll(0.0, -179.0)]),
        Fixture::new("face center cap", deg_cap(ll(0.0, 0.0), 0.1)),
        Fixture::new("face edge cap", deg_cap(ll(0.0, 45.0), 0.5)),
        Fixture::new("face corner tiny cap", deg_cap(face_corner, 1e-7))
            .with_points(vec![face_corner]),
        Fixture::new("point cap", Cap::from_point(ll(12.3, 45.6)))
            .with_points(vec![ll(12.3, 45.6)]),
        Fixture::new("face corner point cap", Cap::from_point(face_corner))
            .with_points(vec![face_corner]),
        // Rects.
        Fixture::new("empty rect", Rect::empty()),
        Fixture::new("full rect", Rect::full()),
        Fixture::new("small rect", deg_rect(10.0, 20.0, 20.0, 30.0)).with_points(vec![
            ll(10.0, 20.0),
            ll(20.0, 30.0),
            ll(10.0, 30.0),
            ll(20.0, 20.0),
        ]),
        Fixture::new("dateline rect", deg_rect(-10.0, 10.0, 170.0, -170.0)).with_points(vec![
            ll(0.0, 180.0),
            ll(-10.0, 170.0),
            ll(10.0, -170.0),
        ]),
        Fixture::new("north polar rect", deg_rect(80.0, 90.0, -180.0, 180.0))
            .with_points(vec![north_pole]),
        Fixture::new("south polar wedge", deg_rect(-90.0, -85.0, 0.0, 10.0))
            .with_points(vec![south_pole]),
        Fixture::new("equatorial band", deg_rect(-5.0, 5.0, -180.0, 180.0)),
        Fixture::line("zero height rect", deg_rect(10.0, 10.0, 0.0, 30.0)).with_points(vec![
            ll(10.0, 0.0),
            ll(10.0, 15.0),
            ll(10.0, 30.0),
        ]),
        Fixture::new("point rect", deg_rect(-33.0, -33.0, 151.0, 151.0))
            .with_points(vec![ll(-33.0, 151.0)]),
        // Polygons.
        Fixture::polygon("empty polygon", Polygon::empty()),
        Fixture::polygon("full polygon", Polygon::full()),
        Fixture::polygon("quad polygon", Polygon::from_loops(vec![shell.clone()])),
        Fixture::polygon("polygon with hole", Polygon::from_loops(vec![shell, hole]))
            .with_points(vec![ll(0.0, 0.0), ll(15.0, 0.0), ll(-5.0, -5.0)]),
        Fixture::polygon(
            "shell, hole and island",
            Polygon::from_loops(vec![nested_shell, nested_hole, nested_island]),
        )
        .with_points(vec![ll(10.0, 100.0), ll(10.0, 110.0)]),
        Fixture::polygon(
            "concave star polygon",
            Polygon::from_loops(vec![star_loop(
                ll(-40.0, -60.0),
                &[12.0, 3.0, 12.0, 3.0, 12.0, 3.0, 12.0, 3.0, 12.0, 3.0],
            )]),
        ),
        Fixture::polygon(
            "north pole polygon",
            Polygon::from_loops(vec![Loop::new(
                [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
                    .iter()
                    .map(|&lng| ll(70.0, lng))
                    .collect(),
            )]),
        )
        .with_points(vec![north_pole]),
        Fixture::polygon(
            "dateline polygon",
            Polygon::from_loops(vec![Loop::new(vec![
                ll(-10.0, 170.0),
                ll(-10.0, -170.0),
                ll(10.0, -170.0),
                ll(10.0, 170.0),
            ])]),
        )
        .with_points(vec![ll(0.0, 180.0), ll(0.0, -175.0), ll(0.0, 175.0)]),
        Fixture::polygon(
            "face corner polygon",
            Polygon::from_loops(vec![star_loop(
                face_corner,
                &[4.0, 6.0, 4.0, 6.0, 4.0, 6.0],
            )]),
        )
        .with_points(vec![face_corner]),
        Fixture::polygon(
            "tiny triangle",
            Polygon::from_loops(vec![Loop::new(vec![
                ll(1.0, 1.0),
                ll(1.0, 1.000001),
                ll(1.000001, 1.0),
            ])]),
        ),
        Fixture::polygon("cell polygon", Polygon::from_cell(&cell_10)),
        // Cells.
        Fixture::new("face cell", Cell::from_face(3u8)),
        Fixture::new("level 10 cell", cell_10).with_points(vec![cell_10.center()]),
        Fixture::new(
            "leaf cell",
            Cell::from_cell_id(CellId::from_point(&ll(-12.0, 77.0))),
        ),
        // Zero-area region.
        Fixture::line(
            "dateline polyline",
            Polyline::new(vec![ll(-5.0, 175.0), ll(0.0, -178.0), ll(5.0, 179.0)]),
        )
        .with_points(vec![ll(-5.0, 175.0), ll(0.0, -178.0), ll(5.0, 179.0)]),
    ]
}

/// Coverer settings used with every fixture. Settings with high minimum
/// levels are skipped for large regions (see [`is_affordable`]).
const FIXTURE_OPTIONS: [Options; 20] = [
    Options::new(0, MAX_LEVEL, 1, 8),
    Options::new(0, MAX_LEVEL, 1, 1),
    Options::new(0, MAX_LEVEL, 1, 3),
    Options::new(0, MAX_LEVEL, 1, 50),
    Options::new(0, 12, 1, 30),
    Options::new(0, MAX_LEVEL, 2, 8),
    Options::new(0, MAX_LEVEL, 3, 8),
    Options::new(0, MAX_LEVEL, 2, 40),
    Options::new(0, MAX_LEVEL, 3, 40),
    Options::new(1, 29, 2, 20),
    Options::new(2, 20, 3, 12),
    Options::new(3, 9, 2, 16),
    Options::new(5, 14, 3, 10),
    Options::new(4, 4, 1, 8),
    Options::new(5, 5, 1, 8),
    Options::new(4, 13, 3, 4),
    Options::new(10, 30, 1, 8),
    Options::new(16, 16, 1, 8),
    Options::new(20, 26, 2, 8),
    Options::new(27, MAX_LEVEL, 3, 8),
];

/// Reports whether a covering at `min_level` is cheap enough to check.
fn is_affordable(region: &dyn Region, min_level: u8) -> bool {
    region.cap_bound().area() / Cell::average_area_for_level(min_level) <= MAX_ESTIMATED_CELLS
}

/// Returns `count` random points from the region's bounding rectangle, plus
/// the rectangle corners. Points outside the region are kept: the checks only
/// use points for which `contains_point` is true.
fn sample_points(region: &dyn Region, rng: &mut ChaCha8Rng, count: usize) -> Vec<Point> {
    let bound = region.rect_bound();
    if bound.is_empty() {
        return Vec::new();
    }
    let lat = bound.lat;
    let lng = bound.lng;
    let lng_hi = if lng.is_inverted() {
        lng.hi + 2.0 * PI
    } else {
        lng.hi
    };
    let mut points = Vec::with_capacity(count + 4);
    for lat_value in [lat.lo, lat.hi] {
        for lng_value in [lng.lo, lng.hi] {
            points.push(LatLng::from_radians(lat_value, lng_value).to_point());
        }
    }
    for _ in 0..count {
        let lat_value = rng.gen_range(lat.lo..=lat.hi);
        let lng_value = rng.gen_range(lng.lo..=lng_hi);
        points.push(LatLng::from_radians(lat_value, lng_value).to_point());
    }
    points
}

#[test]
fn fixtures_satisfy_covering_invariants() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_c0e7);
    for fixture in fixtures() {
        let region = fixture.region.as_ref();
        let mut samples = sample_points(region, &mut rng, 48);
        samples.extend_from_slice(&fixture.extra_points);
        for options in FIXTURE_OPTIONS {
            if let Some(options) = fixture.limit(options)
                && is_affordable(region, options.min_level)
            {
                check_region(options, region, &samples, fixture.name);
            }
        }
    }
}

// ─── Exact results ──────────────────────────────────────────────────────

/// Returns every cell at `level` that the region intersects (or contains,
/// when `interior` is set), below the coverer's start cells. Like the
/// coverer, it starts from a fast covering of the region with at most
/// `min(max_cells, 4)` cells. A search from the six faces would differ:
/// `intersects_cell` is conservative, so for example a small cap
/// "intersects" face cells next to the one it lies in. It uses
/// `Cell::from_cell_id` and skips subtrees that the region does not
/// intersect.
fn brute_force_cells(
    region: &dyn Region,
    level: u8,
    interior: bool,
    max_cells: usize,
) -> Vec<CellId> {
    fn visit(region: &dyn Region, id: CellId, level: u8, interior: bool, out: &mut Vec<CellId>) {
        let cell = Cell::from_cell_id(id);
        if !region.intersects_cell(&cell) {
            return;
        }
        if id.level().as_u8() == level {
            if !interior || region.contains_cell(&cell) {
                out.push(id);
            }
            return;
        }
        for child in id.children() {
            visit(region, child, level, interior, out);
        }
    }
    let start = RegionCoverer::new()
        .max_level(level)
        .max_cells(max_cells.min(4))
        .fast_covering(region);
    let mut out = Vec::new();
    for &id in start.cell_ids() {
        visit(region, id, level, interior, &mut out);
    }
    out
}

/// With `min_level == max_level`, the covering is every intersecting cell at
/// that level, and the interior covering is every contained cell at that
/// level. The exact sets do not depend on the queue order or the time of
/// normalization. They also check that `Cell::children` gives the same
/// geometry as `Cell::from_cell_id`.
#[test]
fn fixed_level_coverings_match_brute_force() {
    for fixture in fixtures() {
        let region = fixture.region.as_ref();
        for level in [0u8, 1, 3, 6, 9, 14, 21, 30] {
            if !is_affordable(region, level) || (fixture.is_line && level > LINE_MAX_LEVEL) {
                continue;
            }
            let label = format!("{} at level {level}", fixture.name);
            for max_cells in [1usize, 8, 1_000_000] {
                let coverer = RegionCoverer::new()
                    .min_level(level)
                    .max_level(level)
                    .max_cells(max_cells);
                let expected = brute_force_cells(region, level, false, max_cells);
                assert_eq!(
                    coverer.covering(region).cell_ids(),
                    expected.as_slice(),
                    "{label}, max_cells {max_cells}: covering"
                );
                if max_cells >= expected.len() {
                    let expected_interior = brute_force_cells(region, level, true, max_cells);
                    assert_eq!(
                        coverer.interior_covering(region).cell_ids(),
                        expected_interior.as_slice(),
                        "{label}, max_cells {max_cells}: interior covering"
                    );
                }
            }
        }
    }
}

/// C++ `RandomCells`: a single cell covered with `max_cells == 1` gives back
/// that cell when its level is allowed. This holds on every face, on many
/// levels, and with every `level_mod`.
#[test]
fn single_cell_is_its_own_covering() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for face in 0..6u8 {
        for level in 0..=MAX_LEVEL {
            let pos = rng.r#gen::<u64>() & ((1u64 << 61) - 1);
            let id = CellId::from_face_pos_level(face, pos, level);
            let cell = Cell::from_cell_id(id);
            for level_mod in 1..=3u8 {
                for min_level in [0, level.saturating_sub(level_mod), level] {
                    if (level - min_level) % level_mod != 0 {
                        continue;
                    }
                    let coverer = RegionCoverer::new()
                        .min_level(min_level)
                        .level_mod(level_mod)
                        .max_cells(1);
                    let label = format!("{id:?} min_level {min_level} level_mod {level_mod}");
                    assert_eq!(
                        coverer.covering(&cell).cell_ids(),
                        [id],
                        "{label}: covering"
                    );
                    assert_eq!(
                        coverer.interior_covering(&cell).cell_ids(),
                        [id],
                        "{label}: interior covering"
                    );
                }
            }
        }
    }
}

#[test]
fn empty_and_full_regions() {
    let faces: Vec<CellId> = (0..6u8).map(CellId::from_face).collect();
    let empty_regions: [Box<dyn Region>; 3] = [
        Box::new(Cap::empty()),
        Box::new(Rect::empty()),
        Box::new(Polygon::empty()),
    ];
    let full_regions: [Box<dyn Region>; 3] = [
        Box::new(Cap::full()),
        Box::new(Rect::full()),
        Box::new(Polygon::full()),
    ];
    for max_cells in [1usize, 6, 100] {
        let coverer = RegionCoverer::new().max_cells(max_cells);
        for region in &empty_regions {
            assert!(coverer.covering(region.as_ref()).is_empty(), "{region:?}");
            assert!(
                coverer.interior_covering(region.as_ref()).is_empty(),
                "{region:?}"
            );
        }
        for region in &full_regions {
            // The normalized union of the whole sphere is the six faces, for
            // any budget: a covering may not drop part of the region.
            assert_eq!(
                coverer.covering(region.as_ref()).cell_ids(),
                faces,
                "{region:?} max_cells {max_cells}: covering"
            );
            // An interior covering may stop at the budget. Every face is
            // contained, so a face is never split; the result is a non-empty
            // set of faces, and all six when the budget allows.
            let interior = coverer.interior_covering(region.as_ref());
            assert!(
                !interior.is_empty() && interior.cell_ids().iter().all(|id| id.is_face()),
                "{region:?} max_cells {max_cells}: interior {interior:?}"
            );
            if max_cells >= faces.len() {
                assert_eq!(
                    interior.cell_ids(),
                    faces,
                    "{region:?} max_cells {max_cells}: interior"
                );
            }
        }
    }
}

/// Regions with no area have an empty interior covering, and their covering
/// still holds every point of the region. Points are checked down to leaf
/// cells; lines only to [`LINE_MAX_LEVEL`].
#[test]
fn zero_area_regions_have_empty_interior() {
    let corner = Point::from_coords(1.0, 1.0, 1.0).normalize();
    let regions: [(&str, Box<dyn Region>, Vec<Point>, u8); 4] = [
        (
            "point cap",
            Box::new(Cap::from_point(corner)),
            vec![corner],
            MAX_LEVEL,
        ),
        (
            "point rect",
            Box::new(deg_rect(45.0, 45.0, 90.0, 90.0)),
            vec![ll(45.0, 90.0)],
            MAX_LEVEL,
        ),
        (
            "zero width rect",
            Box::new(deg_rect(-30.0, 30.0, 179.0, 179.0)),
            vec![ll(-30.0, 179.0), ll(0.0, 179.0), ll(30.0, 179.0)],
            LINE_MAX_LEVEL,
        ),
        (
            "polyline",
            Box::new(Polyline::new(vec![
                ll(0.0, 0.0),
                ll(0.0, 10.0),
                ll(5.0, 15.0),
            ])),
            vec![ll(0.0, 0.0), ll(0.0, 10.0), ll(5.0, 15.0)],
            LINE_MAX_LEVEL,
        ),
    ];
    for (name, region, points, max_level) in &regions {
        let option_list = [
            Options::new(0, *max_level, 1, 8),
            Options::new(0, *max_level, 3, 20),
            Options::new(8, *max_level, 2, 4),
        ];
        for options in option_list {
            let coverer = options.coverer();
            assert!(
                coverer.interior_covering(region.as_ref()).is_empty(),
                "{name} {options:?}"
            );
            let covering = coverer.covering(region.as_ref());
            for &p in points {
                assert!(
                    covering.contains_point(p),
                    "{name} {options:?}: misses {p:?}"
                );
            }
        }
    }
}

/// C++ documents that a tiny convex region at a cube corner can need three
/// cells even when `max_cells` is 1.
#[test]
fn tiny_cap_at_cube_corner_uses_at_most_three_cells() {
    let corner = Point::from_coords(-1.0, 1.0, -1.0).normalize();
    let cap = deg_cap(corner, 1e-9);
    for max_cells in [1usize, 2] {
        let covering = RegionCoverer::new().max_cells(max_cells).covering(&cap);
        assert!(
            (1..=3).contains(&covering.num_cells()),
            "max_cells {max_cells}: {covering:?}"
        );
        assert!(covering.contains_point(corner));
    }
}

/// The hole and the area around the island stay out of fine coverings.
#[test]
fn holes_are_excluded() {
    let center = ll(10.0, 100.0);
    let polygon = Polygon::from_loops(vec![
        star_loop(center, &[30.0; 12]),
        star_loop(center, &[15.0; 12]),
        star_loop(center, &[5.0; 12]),
    ]);
    let in_hole = star_loop(center, &[10.0; 12]).vertex(0);
    let coverer = RegionCoverer::new().min_level(6).max_level(6);
    let covering = coverer.covering(&polygon);
    assert!(
        covering.contains_point(center),
        "island center is not covered"
    );
    assert!(!covering.contains_point(in_hole), "hole point is covered");

    let interior = RegionCoverer::new()
        .max_level(10)
        .max_cells(500)
        .interior_covering(&polygon);
    assert!(
        !interior.contains_point(in_hole),
        "hole point is in the interior"
    );
    assert!(!interior.is_empty());
}

// ─── Random regions ─────────────────────────────────────────────────────

#[test]
fn indexed_polygon_rejects_cell_with_enclosed_hole() {
    let cell = Cell::from_cell_id(CellId::from_point(&ll(40.0, -74.0)).parent_at_level(10));
    let center = cell.center();
    let polygon = Polygon::from_loops(vec![
        Loop::make_regular(center, Angle::from_degrees(1.0), 32),
        Loop::make_regular(center, Angle::from_degrees(0.001), 8),
    ]);
    assert!(polygon.find_validation_error().is_none());
    assert!((0..4).all(|i| Region::contains_point(&polygon, &cell.vertex(i))));
    assert!(!Region::contains_point(&polygon, &center));
    let indexed = ShapeIndexRegion::new(polygon.shape_index());
    assert!(!indexed.contains_cell(&cell));
    let interior = RegionCoverer::new()
        .min_level(10)
        .max_level(10)
        .max_cells(256)
        .interior_covering(&indexed);
    assert!(!interior.contains_point(center));
}

fn random_point(rng: &mut ChaCha8Rng) -> Point {
    loop {
        let vector = Vector {
            x: rng.gen_range(-1.0..=1.0),
            y: rng.gen_range(-1.0..=1.0),
            z: rng.gen_range(-1.0..=1.0),
        };
        let norm2 = vector.dot(vector);
        if norm2 > 1e-6 && norm2 <= 1.0 {
            return Point::new(vector.normalize());
        }
    }
}

/// Like C++ `RandomCaps`: random settings, with the region area limited so
/// that `min_level` cannot force too many cells.
fn random_options(rng: &mut ChaCha8Rng) -> Options {
    let mut min_level = rng.gen_range(0..=MAX_LEVEL);
    let mut max_level = rng.gen_range(0..=MAX_LEVEL);
    if min_level > max_level {
        std::mem::swap(&mut min_level, &mut max_level);
    }
    // Skewed toward small budgets, as in C++ `SkewedInt(10)`.
    let max_cells = 1 + (rng.r#gen::<u32>() & ((1u32 << rng.gen_range(0..=10)) - 1)) as usize;
    Options::new(min_level, max_level, rng.gen_range(1..=3), max_cells)
}

fn max_area_for(options: Options) -> f64 {
    (4.0 * PI)
        .min((3 * options.max_cells + 1) as f64 * Cell::average_area_for_level(options.min_level))
}

fn log_uniform(rng: &mut ChaCha8Rng, lo: f64, hi: f64) -> f64 {
    (rng.gen_range(lo.ln()..=hi.ln())).exp()
}

#[test]
fn random_caps_satisfy_covering_invariants() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xc0ffee);
    let min_area = 0.1 * Cell::average_area_for_level(MAX_LEVEL);
    for i in 0..300 {
        let options = random_options(&mut rng);
        let area = log_uniform(&mut rng, min_area, max_area_for(options).max(min_area));
        let cap = Cap::from_center_area(random_point(&mut rng), area);
        let samples = sample_points(&cap, &mut rng, 16);
        check_region(options, &cap, &samples, &format!("random cap {i}"));
    }
}

#[test]
fn random_polygons_satisfy_covering_invariants() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xbad5eed);
    for i in 0..80 {
        let options = random_options(&mut rng);
        // Radius of a cap with the allowed area, kept below 60 degrees so the
        // star stays a simple loop.
        let max_area = max_area_for(options);
        let max_radius = (1.0 - max_area / (2.0 * PI))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let radius = log_uniform(&mut rng, 1e-6, max_radius.clamp(2e-6, 60.0));
        let center = random_point(&mut rng);
        let num_vertices = rng.gen_range(3..=10);
        let outer: Vec<f64> = (0..num_vertices)
            .map(|_| radius * rng.gen_range(0.3..=1.0))
            .collect();
        let mut loops = vec![star_loop(center, &outer)];
        if num_vertices > 4 && rng.gen_bool(0.5) {
            let hole_radius = 0.25 * outer.iter().copied().fold(f64::INFINITY, f64::min);
            loops.push(star_loop(center, &[hole_radius; 5]));
        }
        let polygon = Polygon::from_loops(loops);
        assert!(polygon.find_validation_error().is_none());
        let polygon = IndexedPolygon(polygon);
        let mut samples = sample_points(&polygon, &mut rng, 16);
        samples.push(center);
        check_region(options, &polygon, &samples, &format!("random polygon {i}"));
    }
}

#[test]
fn random_rects_satisfy_covering_invariants() {
    check_random_rects(BudgetRule::Port);
}

/// Known deviation from C++, so ignored. C++ counts the raw accepted cells
/// against `max_cells`, and their normalized-then-denormalized union is never
/// larger. This port counts cells of the normalized union, where four
/// siblings count as one parent. With `level_mod > 1`, denormalizing can
/// split that parent back into up to `4^(level_mod - 1)` cells, so the
/// covering can exceed `max_cells` while cells still share a `min_level`
/// ancestor (seen: 349 cells for `max_cells` 259 with `level_mod` 3). The
/// optimized code keeps this behavior on purpose.
#[test]
#[ignore = "known deviation from C++: the budget counts normalized cells"]
fn level_mod_budget_matches_cpp() {
    check_random_rects(BudgetRule::Cpp);
}

fn check_random_rects(rule: BudgetRule) {
    let mut rng = ChaCha8Rng::seed_from_u64(0x7ec7);
    for i in 0..120 {
        let options = random_options(&mut rng);
        let max_area = max_area_for(options);
        // Square-ish rect: side in radians from the allowed area.
        let side = log_uniform(&mut rng, 1e-8, max_area.sqrt().clamp(2e-8, PI));
        let lat_center = rng.gen_range(-PI / 2.0..=PI / 2.0);
        let lng_lo = rng.gen_range(-PI..=PI);
        let lat = r1::Interval::new(
            (lat_center - side / 2.0).max(-PI / 2.0),
            (lat_center + side / 2.0).min(PI / 2.0),
        );
        // The interval may cross the antimeridian (lo > hi after wrapping).
        let mut lng_hi = lng_lo + side.min(2.0 * PI - 1e-9);
        if lng_hi > PI {
            lng_hi -= 2.0 * PI;
        }
        let rect = Rect::new(lat, s1::Interval::new(lng_lo, lng_hi));
        let samples = sample_points(&rect, &mut rng, 16);
        check_region_with_rule(options, &rect, &samples, rule, &format!("random rect {i}"));
    }
}

// ─── Cell::children ─────────────────────────────────────────────────────

fn assert_same_cell(actual: Cell, expected: Cell, context: &str) {
    let id = expected.id();
    assert_eq!(actual.id(), id, "{context}: id");
    assert_eq!(actual.face(), expected.face(), "{context} {id:?}: face");
    assert_eq!(actual.level(), expected.level(), "{context} {id:?}: level");
    assert_eq!(
        actual.orientation(),
        expected.orientation(),
        "{context} {id:?}: orientation"
    );
    let (a, e) = (actual.bound_uv(), expected.bound_uv());
    assert!(
        a.x.lo == e.x.lo && a.x.hi == e.x.hi && a.y.lo == e.y.lo && a.y.hi == e.y.hi,
        "{context} {id:?}: bound_uv {a:?} != {e:?}"
    );
    for k in 0..4 {
        assert_eq!(
            actual.vertex(k),
            expected.vertex(k),
            "{context} {id:?}: vertex {k}"
        );
    }
    assert_eq!(
        actual.center(),
        expected.center(),
        "{context} {id:?}: center"
    );
}

fn check_children(id: CellId) {
    let cell = Cell::from_cell_id(id);
    let Some(children) = cell.children() else {
        assert!(id.is_leaf(), "{id:?}: non-leaf cell has no children");
        return;
    };
    assert!(!id.is_leaf(), "{id:?}: leaf cell has children");
    for (child, child_id) in children.into_iter().zip(id.children()) {
        assert_same_cell(child, Cell::from_cell_id(child_id), "Cell::children");
    }
}

/// Every cell on every face at levels 0 to 4.
#[test]
fn cell_children_match_cell_from_id_on_coarse_levels() {
    for face in 0..6u8 {
        let mut level_cells = vec![CellId::from_face(face)];
        for _ in 0..=4 {
            let mut next = Vec::with_capacity(level_cells.len() * 4);
            for &id in &level_cells {
                check_children(id);
                next.extend(id.children());
            }
            level_cells = next;
        }
    }
}

/// Random descents to the leaves on every face, so each level 0..=30 and
/// each child position and orientation is reached many times.
#[test]
fn cell_children_match_cell_from_id_on_every_level() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xce11);
    for face in 0..6u8 {
        for _ in 0..64 {
            let mut id = CellId::from_face(face);
            loop {
                check_children(id);
                if id.is_leaf() {
                    break;
                }
                id = id.children()[rng.gen_range(0..4)];
            }
        }
    }
    // Cells at the corners and edges of each face, where uv rounding is
    // most likely to differ.
    for face in 0..6u8 {
        for level in 0..=MAX_LEVEL {
            let first = CellId::from_face(face).child_begin_at_level(level);
            let last = CellId::from_face(face).child_end_at_level(level).prev();
            check_children(first);
            check_children(last);
        }
    }
}
