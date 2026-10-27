// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Torgeir Børresen <tb@starkad.no>
// Written for this crate; not ported from upstream S2.

use std::fmt::Write as _;

use super::*;
use crate::s1::Angle;
use crate::s2::polyline::Polyline;
use crate::s2::{Cap, LatLng, Loop, Rect};

/// `SplitMix64`: a small deterministic generator, so the golden cases below do
/// not depend on the version of an external RNG crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Returns a value uniformly distributed in `[0, 1)`.
    #[expect(clippy::cast_precision_loss, reason = "53-bit mantissa is intended")]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    fn below(&mut self, upper_bound: u64) -> u64 {
        self.next_u64() % upper_bound
    }

    fn lat_lng(&mut self) -> LatLng {
        LatLng::from_degrees(self.range_f64(-89.0, 89.0), self.range_f64(-180.0, 180.0))
    }

    /// Returns an angle log-uniformly distributed in `[1e-6, 30]` degrees.
    fn radius(&mut self) -> Angle {
        Angle::from_degrees(10f64.powf(self.range_f64(-6.0, 30f64.log10())))
    }
}

/// FNV-1a over the cell ids, so a golden case fits on one line.
fn digest(cells: &CellUnion) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for id in cells.cell_ids() {
        for byte in id.id().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

/// Builds the region for golden case `case_index`.
fn golden_region(case_index: usize, rng: &mut SplitMix64) -> Box<dyn Region> {
    match case_index % 5 {
        0 | 1 => Box::new(Cap::from_center_angle(
            rng.lat_lng().to_point(),
            rng.radius(),
        )),
        2 => {
            let center = rng.lat_lng();
            let size = rng.radius().degrees();
            Box::new(Rect::from_center_size(
                center,
                LatLng::from_degrees(size, 2.0 * size),
            ))
        }
        3 => {
            let num_vertices = 3 + rng.below(8) as usize;
            Box::new(Loop::make_regular(
                rng.lat_lng().to_point(),
                rng.radius(),
                num_vertices,
            ))
        }
        _ => {
            let start = rng.lat_lng();
            let step = rng.radius().degrees();
            let vertices = (0..4)
                .map(|vertex_index| {
                    let offset = f64::from(vertex_index);
                    LatLng::from_degrees(
                        (start.lat.degrees() + offset * step).clamp(-89.0, 89.0),
                        start.lng.degrees() + 1.5 * offset * step,
                    )
                    .to_point()
                })
                .collect();
            Box::new(Polyline::new(vertices))
        }
    }
}

/// Builds randomized coverer options for a golden case.
fn golden_coverer(rng: &mut SplitMix64) -> RegionCoverer {
    const MAX_CELLS: [usize; 7] = [1, 3, 4, 8, 20, 50, 200];
    let min_level = rng.below(5) as u8;
    let max_level = (min_level + 4 + rng.below(27) as u8).min(30);
    RegionCoverer::new()
        .min_level(min_level)
        .max_level(max_level)
        .level_mod(1 + rng.below(3) as u8)
        .max_cells(MAX_CELLS[rng.below(MAX_CELLS.len() as u64) as usize])
}

/// Golden `(covering len, covering digest, interior len, interior digest)`
/// per case, recorded from the implementation that normalized the result on
/// every append. Any change here is a change in covering output.
const GOLDEN: &[(usize, u64, usize, u64)] = &[
    (1, 0x5bda4155e237c28c, 3, 0x0590dd6f03f6144c),
    (9, 0xd495ce51d59b6434, 0, 0xcbf29ce484222325),
    (18, 0x6beafafa5df8ccc0, 32, 0xb8ef1a77ffbe7f4c),
    (1, 0x477c34574c09ad6a, 0, 0xcbf29ce484222325),
    (1, 0xcc07ad5797da056e, 0, 0xcbf29ce484222325),
    (1, 0xfdd806bd4a8a23b5, 0, 0xcbf29ce484222325),
    (3, 0x306981cc40e7c068, 1, 0xd1d7eed5bc58d6fc),
    (1, 0xa990833228c4a383, 0, 0xcbf29ce484222325),
    (1, 0xa891483227eb7f45, 0, 0xcbf29ce484222325),
    (1, 0x9beacf058732e032, 0, 0xcbf29ce484222325),
    (1, 0x7cdeff30eccb9ebe, 0, 0xcbf29ce484222325),
    (1, 0xd623fe76e156670d, 0, 0xcbf29ce484222325),
    (1, 0xf65d17697a3729cd, 0, 0xcbf29ce484222325),
    (2, 0x57c5fb89125aedee, 0, 0xcbf29ce484222325),
    (1, 0x8ac19f8810e5f1cd, 0, 0xcbf29ce484222325),
    (2, 0xfc2c0e2dd9651093, 1, 0xa152a111c450d09c),
    (1, 0xa8d618322826ad81, 1, 0x098184313d074ab3),
    (2, 0xe614939c738c1ca9, 3, 0x2e33462f97738f3d),
    (20, 0x252c6443c8b9e06d, 20, 0xd41b5231fb6f61d2),
    (1, 0x9706a334627532e9, 0, 0xcbf29ce484222325),
    (8, 0x9f1d050c8b9ec28a, 8, 0x5da9a8f86acfdb6b),
    (15, 0x9084c1693c890f10, 7, 0x69369ed6f57bf3d3),
    (53, 0x295df8dcde246ccc, 125, 0xee8ab8473fa184a9),
    (2, 0x45c90db7749ed42c, 3, 0xcfed93a201f59ba8),
    (1, 0x0f5d83fc19b081e7, 0, 0xcbf29ce484222325),
    (3, 0xbf55cc25695160ef, 0, 0xcbf29ce484222325),
    (1, 0xa9a16b3228d2edee, 3, 0x9c3381803cb42ea1),
    (1, 0xaa0096322923cb83, 0, 0xcbf29ce484222325),
    (2, 0x8b1c3ddb584f338c, 4, 0xe8a620e18e532f31),
    (3, 0x7b22f8a82fb408a5, 0, 0xcbf29ce484222325),
    (1, 0xfbef1b3fa8a7f3f7, 0, 0xcbf29ce484222325),
    (347, 0x52f20e1d61a4200a, 449, 0xf59da4b047b09bc2),
    (20, 0xe114b6c8fdd657f0, 20, 0xd43f543ecab1a0d1),
    (1, 0xa843d43227aa4bd8, 0, 0xcbf29ce484222325),
    (1, 0xeb238a79dbb11215, 0, 0xcbf29ce484222325),
    (2, 0x6ed0be534c30ab09, 4, 0xc07e9b6f190588b2),
    (59, 0x62b4cba93f059bb7, 116, 0x1361d94d4df289aa),
    (47, 0xe6a4286cb7581e21, 50, 0xdb999e5d24cb48e1),
    (1, 0xa9a12e3228d28647, 0, 0xcbf29ce484222325),
    (1, 0x165f3b30b325ac45, 0, 0xcbf29ce484222325),
    (4, 0xdc12204bb198f737, 4, 0x4be4b9b1e0407925),
    (1, 0x7d5cf233c1a68edf, 0, 0xcbf29ce484222325),
    (2, 0x6baed72c4a62ea39, 1, 0xf755956660d5fd54),
    (1, 0xbb1c58dad4502059, 0, 0xcbf29ce484222325),
    (1, 0x36f909bf9e628e21, 0, 0xcbf29ce484222325),
    (4, 0xbe0a43a1ddd18769, 3, 0x48cad9964f26f670),
    (1, 0x929005638a7f44c9, 0, 0xcbf29ce484222325),
    (3, 0x153ce8add9db4d9c, 3, 0xb34f825c28e95ca0),
    (22, 0x8c587bcf9def3973, 29, 0x9baf2174d22449ed),
    (8, 0x1b620f31fcfadc65, 0, 0xcbf29ce484222325),
    (9, 0x6c9c38d1a47d8e78, 1, 0x81687733e30fe73c),
    (23, 0x9d9d5eb8ff68c433, 29, 0x30fd5fc81eb437c1),
    (1, 0xa8c7883228197b75, 4, 0x2c40690c1ef7384d),
    (20, 0xa386ed6ca5ba0532, 20, 0x30c77b7c0737ad32),
    (2, 0x1dead266cab0d2e3, 0, 0xcbf29ce484222325),
    (3, 0x83a157022c3a221e, 3, 0x7fac82d83752ce28),
    (399, 0xf70aadb9619b39e9, 413, 0x349ebbbf39272d79),
    (2, 0xd17c3d67efc5b1d1, 0, 0xcbf29ce484222325),
    (4, 0x1281483e9c0583f3, 4, 0xe99c52ed092d38ee),
    (3, 0xc4bd50d226782a28, 0, 0xcbf29ce484222325),
    (50, 0x6eb499ebed6e3fd6, 50, 0x230e09b5438574cf),
    (7, 0xf10cd45f656d830d, 0, 0xcbf29ce484222325),
    (1, 0x7fd80930ef5206dc, 1, 0x677560f39d101e42),
    (2, 0xd56d1f1bc597d20e, 1, 0xc1e4308d78826197),
    (30, 0x437f6332583fc9a7, 0, 0xcbf29ce484222325),
    (3, 0x7b17306c7296002b, 0, 0xcbf29ce484222325),
    (2, 0x3639b96d0e8d7523, 0, 0xcbf29ce484222325),
    (2, 0xfc2ed712159e5517, 0, 0xcbf29ce484222325),
    (1, 0x575264328b54aedf, 0, 0xcbf29ce484222325),
    (1, 0xd030dc9eae0cbce6, 0, 0xcbf29ce484222325),
    (1, 0x523a1fedc5e872ff, 0, 0xcbf29ce484222325),
    (1, 0x7e40fe30edf8da63, 0, 0xcbf29ce484222325),
    (1, 0xa37ebef571122e2d, 0, 0xcbf29ce484222325),
    (3, 0x60e18a01da2c5fec, 8, 0x4403a408ebe1d07d),
    (1, 0x22e5ea0c4da2e1b4, 0, 0xcbf29ce484222325),
    (1, 0x13247f7a128f07df, 0, 0xcbf29ce484222325),
    (22, 0xb16e584eeed26e5c, 23, 0xb0009338d0faf1a1),
    (115, 0x876c436c8277402e, 72, 0x05796f8c8d1a5eac),
    (381, 0x4603d081f8075f0f, 401, 0x302962ffeb757ef0),
    (1, 0x3a4bd9fefc94f6ba, 0, 0xcbf29ce484222325),
    (3, 0x1b48856cc800c6ed, 0, 0xcbf29ce484222325),
    (1, 0x7d532f30ed2ee0dc, 0, 0xcbf29ce484222325),
    (9, 0xb072d86b7df7f6f4, 3, 0x3f81c52f14bc8bb4),
    (4, 0x2a950f9022b7f7c8, 4, 0x92d8887cdec81721),
    (2, 0xca69c1e07885a089, 0, 0xcbf29ce484222325),
    (7, 0x3adf26eb5d7a4e66, 8, 0xa7c293951c54af2d),
    (1, 0xd2f3e89ffc6583c4, 0, 0xcbf29ce484222325),
    (361, 0x98d52e190170679f, 473, 0xdf768b59a6e8acba),
    (4, 0xdbbbaa37de103a25, 0, 0xcbf29ce484222325),
    (20, 0x67af9d042e31b971, 0, 0xcbf29ce484222325),
    (200, 0x217926414817b86c, 200, 0x70c58ac74f1a0e48),
    (6, 0x4958aaa055ac62d6, 0, 0xcbf29ce484222325),
    (1, 0xa8c84732281ac002, 1, 0xa994103228c7c8f1),
    (1, 0x3abc24339caae054, 1, 0xf1776da4949739aa),
    (1, 0xaabf5d3229c646b0, 0, 0xcbf29ce484222325),
    (20, 0xb137a7940a9729aa, 20, 0x57752480090ae6ec),
    (1, 0x97f0fc8ab3099e46, 0, 0xcbf29ce484222325),
    (1, 0xe1ce5858f13e0fde, 0, 0xcbf29ce484222325),
    (8, 0x1e8c429905ae7a83, 8, 0xcc99d098abc5839e),
    (20, 0xf7262c3cbc5f6003, 0, 0xcbf29ce484222325),
    (2, 0xbade2fa24b8a2ddb, 0, 0xcbf29ce484222325),
    (66, 0xee0d14a820a15ad0, 68, 0xaabbb24e08f0e84c),
    (1, 0x1dd81d2128329963, 0, 0xcbf29ce484222325),
    (3, 0xd04bf0702fffd4a1, 3, 0x947c22210396558a),
    (13, 0x2dbef6aa3557a764, 0, 0xcbf29ce484222325),
    (2, 0xf477d5347b8587c5, 0, 0xcbf29ce484222325),
    (14, 0xe8783ae92bdf6c8f, 29, 0xec6cb946466e3d93),
    (4, 0xac53bdb68e4ed681, 2, 0x9aea9d8fdfc2e395),
    (1, 0xb6b9ac020daf39f6, 0, 0xcbf29ce484222325),
    (1, 0x6cc4ff4122cb532c, 0, 0xcbf29ce484222325),
    (8, 0x88d10f0c7f3c89c4, 8, 0x89f9074b971927e0),
    (2, 0xe127e7e7db10d11d, 0, 0xcbf29ce484222325),
    (1, 0x5beedf1eb2dafbcd, 4, 0x960c1ae3686b773c),
    (1, 0x48d3013313d3babc, 0, 0xcbf29ce484222325),
    (1, 0x959a7c358382fb96, 0, 0xcbf29ce484222325),
    (6, 0xdf584b90672180dd, 8, 0xb2f276dc132e3405),
    (17, 0xfe6508a95110e72c, 32, 0x987b623bdac453ad),
    (4, 0xf4aa22a78401dc9d, 8, 0x2debf5e2361b555e),
    (391, 0xe4483c6a0ce69f4f, 446, 0xb0b2ed0bec555d10),
    (8, 0x4fb26a3abac39d33, 0, 0xcbf29ce484222325),
];

const NUM_GOLDEN_CASES: usize = 120;

/// Returns the region and coverer of every golden case, in order.
fn golden_cases() -> impl Iterator<Item = (Box<dyn Region>, RegionCoverer)> {
    let mut rng = SplitMix64(0x5eed_c0fe_u64);
    (0..NUM_GOLDEN_CASES).map(move |i| {
        let region = golden_region(i, &mut rng);
        (region, golden_coverer(&mut rng))
    })
}

fn tokens(cells: &CellUnion) -> Vec<String> {
    cells.iter().map(|id| id.to_token()).collect()
}

#[test]
fn test_covering_matches_golden_outputs() {
    let actual: Vec<_> = golden_cases()
        .map(|(region, coverer)| {
            let covering = coverer.covering(region.as_ref());
            let interior = coverer.interior_covering(region.as_ref());
            (
                covering.len(),
                digest(&covering),
                interior.len(),
                digest(&interior),
            )
        })
        .collect();
    let mut rendered = String::new();
    for (covering_len, covering_digest, interior_len, interior_digest) in &actual {
        writeln!(rendered, "    ({covering_len}, {covering_digest:#018x}, {interior_len}, {interior_digest:#018x}),").expect("writing to a String cannot fail");
    }
    assert!(
        actual == GOLDEN,
        "covering outputs changed; actual table:\n{rendered}"
    );
}

// The covering budget counts the cells of the normalized result, so four
// accepted siblings count as one cell. Upstream C++ and Go instead count every
// accepted cell and normalize only at the end. The two tests below pin this
// crate's behaviour where the two differ.

#[test]
fn test_interior_covering_budget_counts_normalized_cells() {
    let (region, coverer) = golden_cases().nth(95).expect("golden case 95 exists");
    assert_eq!(coverer.level_mod, 1);
    assert_eq!(coverer.max_cells, 20);
    let interior = coverer.interior_covering(region.as_ref());
    // Counting raw accepted cells in this Rust traversal instead yields 17 cells.
    assert_eq!(
        tokens(&interior),
        [
            "a837b6e7bd23254",
            "a837b6e7bd2334",
            "a837b6e7bd2339",
            "a837b6e7bd233b",
            "a837b6e7bd233c4",
            "a837b6e7bd235",
            "a837b6e7bd24a5c",
            "a837b6e7bd24a7",
            "a837b6e7bd24a9",
            "a837b6e7bd24ab",
            "a837b6e7bd3b55",
            "a837b6e7bd3b564",
            "a837b6e7bd3ca54",
            "a837b6e7bd3cab",
            "a837b6e7bd3cad",
            "a837b6e7bd3caf",
            "a837b6e7bd3cb4",
            "a837b6e7bd3cc9",
            "a837b6e7bd3ccb",
            "a837b6e7bd3ccd",
        ]
    );
}

#[test]
fn test_interior_covering_level_mod_can_exceed_max_cells() {
    // With level_mod > 1, normalizing merges siblings into a parent at a
    // level that level_mod excludes. The merged parent counts as one cell,
    // but the final denormalization expands it again, so the result can
    // exceed max_cells. Raw counting in this Rust traversal yields 20 cells.
    let (region, coverer) = golden_cases().nth(106).expect("golden case 106 exists");
    assert_eq!(coverer.level_mod, 3);
    assert_eq!(coverer.max_cells, 20);
    let interior = coverer.interior_covering(region.as_ref());
    assert_eq!(
        tokens(&interior),
        [
            "bae1df", "bae1e1", "bae1e3", "bae1e5", "bae1f7", "bb02c", "bb034", "bb04c", "bb054",
            "bb06a1", "bb06a7", "bb06a9", "bb06ab", "bb0f63", "bb0f79", "bb0f7b", "bb0f7d",
            "bb0f7f", "bb194", "bb19c", "bb1a4", "bb1ac", "bb1b4", "bb1bc", "bb1c4", "bb1cc",
            "bb1d4", "bb1dc", "bb1ec",
        ]
    );
}

/// Appends to `out` a random set of disjoint cells inside `id`, often
/// including complete sibling groups.
fn random_disjoint_cells(id: CellId, rng: &mut SplitMix64, out: &mut Vec<CellId>) {
    match rng.below(8) {
        _ if id.level() >= 10 => out.push(id),
        0 => {}
        1 | 2 => out.push(id),
        _ => {
            for child in id.children() {
                random_disjoint_cells(child, rng, out);
            }
        }
    }
}

#[test]
fn test_covering_result_tracks_normalized_len() {
    let mut rng = SplitMix64(7);
    for _ in 0..200 {
        let root = CellId::from_face_pos_level(rng.below(6) as u8, rng.next_u64() >> 3, 6);
        let mut cells = Vec::new();
        random_disjoint_cells(root, &mut rng, &mut cells);
        // Shuffle, so sibling groups complete in arbitrary order.
        for i in (1..cells.len()).rev() {
            cells.swap(i, rng.below(i as u64 + 1) as usize);
        }

        let mut result = CoveringResult::default();
        for (i, &id) in cells.iter().enumerate() {
            result.push(id);
            let expected = CellUnion::from_cell_ids(cells[..=i].to_vec());
            assert_eq!(result.len(), expected.len(), "after {} cells", i + 1);
        }
        assert_eq!(
            result.into_cell_union(),
            CellUnion::from_cell_ids(cells.clone())
        );
    }
}

#[test]
fn test_cell_children_match_cells_from_child_ids() {
    // `Cell::children` derives each
    // child's bounds from its parent's; they must match `Cell::from_cell_id`
    // bit for bit, or region predicates could answer differently.
    let mut rng = SplitMix64(11);
    for _ in 0..2000 {
        let level = rng.below(30) as u8;
        let id = CellId::from_face_pos_level(rng.below(6) as u8, rng.next_u64() >> 3, level);
        let children = Cell::from_cell_id(id)
            .children()
            .expect("cells below level 30 have children");
        for (child, child_id) in children.into_iter().zip(id.children()) {
            let expected = Cell::from_cell_id(child_id);
            assert_eq!(child.id(), expected.id());
            assert_eq!(child.face(), expected.face());
            assert_eq!(child.level(), expected.level());
            assert_eq!(child.orientation(), expected.orientation());
            let actual_bounds = child.bound_uv();
            let expected_bounds = expected.bound_uv();
            for (actual_value, expected_value) in [
                (actual_bounds.x.lo, expected_bounds.x.lo),
                (actual_bounds.x.hi, expected_bounds.x.hi),
                (actual_bounds.y.lo, expected_bounds.y.lo),
                (actual_bounds.y.hi, expected_bounds.y.hi),
            ] {
                assert_eq!(
                    actual_value.to_bits(),
                    expected_value.to_bits(),
                    "cell {child_id:?}"
                );
            }
        }
    }
}
