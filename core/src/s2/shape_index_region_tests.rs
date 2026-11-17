// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Torgeir Børresen <tb@starkad.no>
// Written for this crate; not ported from upstream S2.

use super::*;
use crate::s2::LatLng;
use crate::s2::lax_loop::LaxLoop;
use crate::s2::point_vector::PointVector;

fn p(lat: f64, lng: f64) -> Point {
    LatLng::from_degrees(lat, lng).to_point()
}

#[test]
fn combined_relation_matches_both_predicates_with_overlapping_shapes() {
    let point = p(0.25, 0.25);
    let mut index = ShapeIndex::new();
    index.add(Box::new(PointVector::new(vec![point])));
    index.add(Box::new(LaxLoop::new(vec![
        p(1.0, 1.0),
        p(1.0, -1.0),
        p(-1.0, -1.0),
        p(-1.0, 1.0),
    ])));
    index.build();

    let region = ShapeIndexRegion::new(&index);
    let point_cell = Cell::from_cell_id(CellId::from_point(&point).parent_at_level(20));
    assert_eq!(
        region.cell_relation(&point_cell),
        RegionCellRelation::Contains
    );
    for point in [point, p(1.0, 0.0), p(50.0, 90.0)] {
        let id = CellId::from_point(&point);
        for level in 0..=30 {
            let cell = Cell::from_cell_id(id.parent_at_level(level));
            let expected = if !region.intersects_cell(&cell) {
                RegionCellRelation::Disjoint
            } else if region.contains_cell(&cell) {
                RegionCellRelation::Contains
            } else {
                RegionCellRelation::Intersects
            };
            assert_eq!(region.cell_relation(&cell), expected);
        }
    }
}
