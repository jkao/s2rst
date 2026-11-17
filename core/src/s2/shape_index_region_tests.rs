// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Torgeir Børresen <tb@starkad.no>
// Written for this crate; not ported from upstream S2.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::ShapeIndexRegion;
use crate::s1::Angle;
use crate::s2::lax_loop::LaxLoop;
use crate::s2::point_vector::PointVector;
use crate::s2::polyline::Polyline;
use crate::s2::shape::{Chain, ChainPosition, Dimension, Edge, ReferencePoint, Shape};
use crate::s2::shape_index::{CellRelation, ShapeIndex};
use crate::s2::{Cell, CellId, LatLng, Loop, Point, Region, RegionCellRelation};

fn p(lat: f64, lng: f64) -> Point {
    LatLng::from_degrees(lat, lng).to_point()
}

fn check_combined_relation(region: &ShapeIndexRegion<'_>, id: CellId) -> RegionCellRelation {
    let cell = Cell::from_cell_id(id);
    let expected = if !region.intersects_cell(&cell) {
        RegionCellRelation::Disjoint
    } else if region.contains_cell(&cell) {
        RegionCellRelation::Contains
    } else {
        RegionCellRelation::Intersects
    };
    assert_eq!(region.cell_relation(&cell), expected, "cell {id:?}");
    expected
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
    for point in [point, p(1.0, 0.0), p(10.0, 10.0), p(50.0, 90.0)] {
        let id = CellId::from_point(&point);
        for level in 0..=30 {
            check_combined_relation(&region, id.parent_at_level(level));
        }
    }
}

#[test]
fn combined_relation_matches_both_predicates_on_subdivided_index() {
    let mut index = ShapeIndex::new();
    index.add(Box::new(Loop::make_regular(
        p(10.0, 10.0),
        Angle::from_degrees(8.0),
        64,
    )));
    index.add(Box::new(Polyline::new(vec![p(0.0, 0.0), p(15.0, 15.0)])));
    index.build();

    let region = ShapeIndexRegion::new(&index);
    let mut cells = index.iter();
    let mut locate = index.iter();
    let mut exact_contains = false;
    let mut exact_intersects = false;
    let mut subdivided = false;
    while !cells.done() {
        let id = cells.cell_id();
        let relation = check_combined_relation(&region, id);
        exact_contains |= relation == RegionCellRelation::Contains;
        exact_intersects |= relation == RegionCellRelation::Intersects;
        if !id.is_face() {
            let parent = id.parent();
            subdivided |= locate.locate_cell_id(parent) == CellRelation::Subdivided;
            check_combined_relation(&region, parent);
        }
        if !id.is_leaf() {
            for child in id.children() {
                check_combined_relation(&region, child);
            }
        }
        cells.next();
    }
    assert!(exact_contains, "expected an index cell inside the polygon");
    assert!(exact_intersects, "expected an index cell with edges");
    assert!(subdivided, "expected a parent subdivided into index cells");
}

#[test]
fn combined_relation_skips_polyline_edges_after_an_intersection() {
    let point = p(0.25, 0.25);
    let edge_reads = Arc::new(AtomicUsize::new(0));
    let mut index = ShapeIndex::new();
    index.add(Box::new(PointVector::new(vec![point])));
    index.add(Box::new(CountingPolyline {
        line: Polyline::new(vec![p(0.1, 0.25), p(0.4, 0.25)]),
        edge_reads: Arc::clone(&edge_reads),
    }));
    index.build();
    edge_reads.store(0, Ordering::Relaxed);

    let region = ShapeIndexRegion::new(&index);
    let point_cell = Cell::from_cell_id(CellId::from_point(&point).parent_at_level(20));
    assert_eq!(
        region.cell_relation(&point_cell),
        RegionCellRelation::Intersects
    );
    assert_eq!(edge_reads.load(Ordering::Relaxed), 0);

    let line_point = p(0.3, 0.25);
    let line_cell = Cell::from_cell_id(CellId::from_point(&line_point).parent_at_level(20));
    assert_eq!(
        region.cell_relation(&line_cell),
        RegionCellRelation::Intersects
    );
    assert!(edge_reads.load(Ordering::Relaxed) > 0);
}

#[derive(Debug)]
struct CountingPolyline {
    line: Polyline,
    edge_reads: Arc<AtomicUsize>,
}

impl Shape for CountingPolyline {
    fn num_edges(&self) -> usize {
        self.line.num_edges()
    }

    fn edge(&self, id: usize) -> Edge {
        self.edge_reads.fetch_add(1, Ordering::Relaxed);
        self.line.edge(id)
    }

    fn reference_point(&self) -> ReferencePoint {
        self.line.reference_point()
    }

    fn num_chains(&self) -> usize {
        self.line.num_chains()
    }

    fn chain(&self, chain_id: usize) -> Chain {
        self.line.chain(chain_id)
    }

    fn chain_edge(&self, chain_id: usize, offset: usize) -> Edge {
        self.line.chain_edge(chain_id, offset)
    }

    fn chain_position(&self, edge_id: usize) -> ChainPosition {
        self.line.chain_position(edge_id)
    }

    fn dimension(&self) -> Dimension {
        self.line.dimension()
    }
}
