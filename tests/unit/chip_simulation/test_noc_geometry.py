"""N1 — one home for the SANA-FE mesh conventions, with delegation pins.

Three conventions decide every NoC number the record seals: core→(tile, local)
assignment (``idx // cores_per_tile``), tile→(x, y) placement (``i // mesh_height,
i % mesh_height``), and the XY route walk (x first, then y; one edge per mesh
hop). They were spelled out in three files; the candidate NoC estimator makes a
fourth reader, so the rules move to ``noc_geometry`` and every reader delegates.
"""

from mimarsinan.chip_simulation.sanafe.noc_geometry import (
    legacy_mesh,
    most_square_exact_dims,
    replicated_mesh,
    tile_and_local_of_core,
    xy_of_tile,
    xy_route_edges,
    xy_route_hops,
)


class TestTheRouteWalk:
    def test_x_first_then_y(self):
        assert xy_route_edges((0, 0), (2, 1)) == [
            (0, 0, 1, 0), (1, 0, 2, 0), (2, 0, 2, 1),
        ]

    def test_negative_direction_steps(self):
        assert xy_route_edges((2, 1), (0, 0)) == [
            (2, 1, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0),
        ]

    def test_local_message_traverses_no_edges(self):
        assert xy_route_edges((3, 3), (3, 3)) == []

    def test_hops_equal_manhattan_distance(self):
        assert xy_route_hops((0, 0), (2, 1)) == 3
        assert xy_route_hops((5, 2), (5, 2)) == 0


class TestCoreAndTilePlacement:
    def test_sequential_tile_fill(self):
        assert tile_and_local_of_core(0, 4) == (0, 0)
        assert tile_and_local_of_core(5, 4) == (1, 1)
        assert tile_and_local_of_core(7, 4) == (1, 3)

    def test_nonpositive_cores_per_tile_means_one_tile(self):
        assert tile_and_local_of_core(9, 0) == (0, 9)

    def test_tile_xy_is_column_major_over_mesh_height(self):
        assert xy_of_tile(0, 2) == (0, 0)
        assert xy_of_tile(1, 2) == (0, 1)
        assert xy_of_tile(2, 2) == (1, 0)
        assert xy_of_tile(5, 2) == (2, 1)


class TestMeshDims:
    """The two ``derive_arch_spec`` regimes, mirrored one-to-one."""

    def test_most_square_exact_factorization(self):
        assert most_square_exact_dims(1) == (1, 1)
        assert most_square_exact_dims(3) == (3, 1)   # prime: exact, not padded
        assert most_square_exact_dims(12) == (4, 3)  # width >= height

    def test_declared_floorplan_replicates_rows_until_the_pack_fits(self):
        # 2x2 tiles x 4 cores = 16 slots < 20 packed -> 2 replicas -> 2x4 mesh.
        assert replicated_mesh(
            packed_cores=20, cores_per_tile=4, rows=2, cols=2,
        ) == (8, 2, 4)

    def test_declared_floorplan_that_already_fits_is_untouched(self):
        assert replicated_mesh(
            packed_cores=6, cores_per_tile=4, rows=2, cols=2,
        ) == (4, 2, 2)

    def test_legacy_mesh_matches_the_packed_count_derivation(self):
        # packed=6, cpt=2 -> 3 tiles -> 3x1 mesh (exact factorization).
        assert legacy_mesh(packed_cores=6, cores_per_tile=2) == (2, 3, 3, 1)

    def test_legacy_mesh_derives_cores_per_tile_when_unset(self):
        # isqrt(6)=2, 2*2 < 6 -> 3 cores/tile, 2 tiles, 2x1 mesh.
        assert legacy_mesh(packed_cores=6, cores_per_tile=0) == (3, 2, 2, 1)


class TestTheReadersDelegate:
    """The SSOT pins: every prior reader must answer THROUGH the shared rules."""

    def test_trace_route_walk_is_the_shared_walk(self):
        from mimarsinan.chip_simulation.sanafe.analysis.noc import _xy_route_edges

        for src, dst in [((0, 0), (2, 1)), ((3, 1), (0, 0)), ((1, 1), (1, 1))]:
            ev = {"src_x": src[0], "src_y": src[1],
                  "dest_x": dst[0], "dest_y": dst[1]}
            assert _xy_route_edges(ev) == xy_route_edges(src, dst)

    def test_trace_walk_still_skips_unplaced_messages(self):
        from mimarsinan.chip_simulation.sanafe.analysis.noc import _xy_route_edges

        assert _xy_route_edges({"src_x": -1, "src_y": 0,
                                "dest_x": 1, "dest_y": 0}) == []

    def test_net_synth_tile_fill_is_the_shared_rule(self):
        from mimarsinan.chip_simulation.sanafe.net_synth.build import (
            _pack_tile_index,
        )

        for idx in range(10):
            for cpt in (0, 1, 4):
                assert _pack_tile_index(idx, cpt) == tile_and_local_of_core(idx, cpt)

    def test_floorplan_factorization_is_the_shared_rule(self):
        from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import (
            _mesh_dims,
        )

        for n in (1, 2, 3, 6, 12, 17):
            assert _mesh_dims(n) == most_square_exact_dims(n)

    def test_arch_synth_and_runner_delegate_in_source(self):
        """Structural drift guards: the geometry readers must go THROUGH
        noc_geometry (behavioral equality would need full preset fixtures)."""
        import inspect

        import mimarsinan.chip_simulation.sanafe.arch_synth.spec as spec_module
        import mimarsinan.chip_simulation.sanafe.runner.core as runner_core

        spec_src = inspect.getsource(spec_module)
        assert "replicated_mesh" in spec_src and "legacy_mesh" in spec_src, (
            "arch_synth/spec.py stopped delegating its mesh geometry to "
            "noc_geometry"
        )
        assert "xy_of_tile" in inspect.getsource(runner_core), (
            "runner/core.py stopped delegating tile placement to "
            "noc_geometry.xy_of_tile"
        )
