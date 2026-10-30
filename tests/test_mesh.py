# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.geo.mesh import (
    icon_domain_boundary,
    neighbour_cells,
    neighbour_cells_diamond,
    outside_icon_domain,
)


def _lattice(n):
    """N x n squares, each split into 2 triangles; topologically an ICON mesh (6 cells per vertex)."""
    cells = []
    for j in range(n):
        for i in range(n):
            a, b = j * (n + 1) + i, j * (n + 1) + i + 1
            c, d = b + n + 1, a + n + 1
            cells += [(a, b, c), (a, c, d)]
    vertex_of_cell = np.array(cells).T
    n_vertices = (n + 1) ** 2

    cells_of_vertex = np.full((6, n_vertices), -1)
    fill = np.zeros(n_vertices, dtype=int)
    for c, verts in enumerate(cells):
        for v in verts:
            cells_of_vertex[fill[v], v] = c
            fill[v] += 1

    by_edge = {}
    for c, (a, b, d) in enumerate(cells):
        for e in ((a, b), (b, d), (d, a)):
            by_edge.setdefault(tuple(sorted(e)), []).append(c)
    neighbor_cell_index = np.full((3, len(cells)), -1)
    for c, (a, b, d) in enumerate(cells):
        for k, e in enumerate(((a, b), (b, d), (d, a))):
            other = [x for x in by_edge[tuple(sorted(e))] if x != c]
            if other:
                neighbor_cell_index[k, c] = other[0]
    return vertex_of_cell, cells_of_vertex, neighbor_cell_index


def _rows(table):
    return [sorted(r[r >= 0].tolist()) for r in table]


def _bfs(cell, order, step):
    cells, frontier = {cell}, {cell}
    for _ in range(order):
        frontier = {n for c in frontier for n in step(c)} - cells
        cells |= frontier
    return sorted(cells)


def test_neighbour_cells_diamond_hand_table():
    # 2 x 2 squares, 8 triangles
    _, _, neighbor_cell_index = _lattice(2)
    table = neighbour_cells_diamond(neighbor_cell_index, 1)
    assert table.dtype == np.int32
    assert table.shape == (8, 4)
    assert _rows(table) == [
        [0, 1, 3],
        [0, 1, 4],
        [2, 3],
        [0, 2, 3, 6],
        [1, 4, 5, 7],
        [4, 5],
        [3, 6, 7],
        [4, 6, 7],
    ]


def test_neighbour_cells_square_order1():
    vertex_of_cell, cells_of_vertex, _ = _lattice(2)
    table = neighbour_cells(vertex_of_cell, cells_of_vertex, 1)
    # cell 0 = vertices (0, 1, 4); vertex 4 is the centre of the mesh
    assert _rows(table)[0] == [0, 1, 2, 3, 4, 6, 7]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_neighbour_cells_against_bfs(order):
    vertex_of_cell, cells_of_vertex, neighbor_cell_index = _lattice(8)

    def by_vertex(c):
        return {int(x) for v in vertex_of_cell[:, c] for x in cells_of_vertex[:, v] if x >= 0}

    def by_edge(c):
        return {int(x) for x in neighbor_cell_index[:, c] if x >= 0}

    n_cells = vertex_of_cell.shape[1]
    square = neighbour_cells(vertex_of_cell, cells_of_vertex, order)
    diamond = neighbour_cells_diamond(neighbor_cell_index, order)
    assert _rows(square) == [_bfs(c, order, by_vertex) for c in range(n_cells)]
    assert _rows(diamond) == [_bfs(c, order, by_edge) for c in range(n_cells)]

    # left-packed, sorted, no duplicates
    for table in (square, diamond):
        for r in table:
            valid = r[r >= 0]
            assert (r[valid.size :] == -1).all()
            assert (np.diff(valid) > 0).all()


@pytest.mark.parametrize("order, square_size, diamond_size", [(1, 13, 4), (2, 37, 10), (3, 73, 19)])
def test_neighbour_cells_ring_sizes(order, square_size, diamond_size):
    vertex_of_cell, cells_of_vertex, neighbor_cell_index = _lattice(10)
    square = neighbour_cells(vertex_of_cell, cells_of_vertex, order)
    diamond = neighbour_cells_diamond(neighbor_cell_index, order)
    assert int((square >= 0).sum(axis=1).max()) == square_size
    assert int((diamond >= 0).sum(axis=1).max()) == diamond_size
    # a larger order grows every ring
    if order > 1:
        smaller = neighbour_cells(vertex_of_cell, cells_of_vertex, order - 1)
        assert ((square >= 0).sum(axis=1) >= (smaller >= 0).sum(axis=1)).all()
        assert int((square >= 0).sum(axis=1).max()) > int((smaller >= 0).sum(axis=1).max())


@pytest.mark.parametrize("order", [0, -1, 1.5])
def test_neighbour_cells_bad_order(order):
    vertex_of_cell, cells_of_vertex, neighbor_cell_index = _lattice(2)
    with pytest.raises(ValueError, match="order"):
        neighbour_cells(vertex_of_cell, cells_of_vertex, order)
    with pytest.raises(ValueError, match="order"):
        neighbour_cells_diamond(neighbor_cell_index, order)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_neighbour_cells_packing_and_dtype(order):
    vertex_of_cell, cells_of_vertex, neighbor_cell_index = _lattice(6)
    for table in (
        neighbour_cells(vertex_of_cell, cells_of_vertex, order),
        neighbour_cells_diamond(neighbor_cell_index, order),
    ):
        assert table.dtype == np.int32
        valid = table >= 0
        # padding only on the right, every row contains its own cell, rows sorted and unique
        assert not (valid[:, 1:] & ~valid[:, :-1]).any()
        assert (table == np.arange(len(table))[:, None]).any(axis=1).all()
        for row in table:
            row = row[row >= 0]
            assert (np.diff(row) > 0).all()


def test_neighbour_cells_edge_rows_have_fewer_entries():
    vertex_of_cell, cells_of_vertex, _ = _lattice(6)
    table = neighbour_cells(vertex_of_cell, cells_of_vertex)
    counts = (table >= 0).sum(axis=1)
    assert counts.max() == 13 and counts.min() < 13


def test_neighbour_cells_blocks_give_same_table(monkeypatch):
    from earthkit.geo import mesh

    vertex_of_cell, cells_of_vertex, neighbor_cell_index = _lattice(6)
    ref = neighbour_cells(vertex_of_cell, cells_of_vertex, 3)
    ref_d = neighbour_cells_diamond(neighbor_cell_index, 3)
    monkeypatch.setattr(mesh, "_BLOCK", 7)
    np.testing.assert_array_equal(neighbour_cells(vertex_of_cell, cells_of_vertex, 3), ref)
    np.testing.assert_array_equal(neighbour_cells_diamond(neighbor_cell_index, 3), ref_d)


def test_neighbour_cells_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        neighbour_cells(np.zeros((2, 4), dtype=int), np.zeros((6, 4), dtype=int))
    with pytest.raises(ValueError, match="shape"):
        neighbour_cells_diamond(np.zeros((2, 4), dtype=int))


def _four_triangles():
    # a unit square split into 4 triangles around its centre vertex 4
    vlat = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
    vlon = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
    vertex_of_cell = np.array([[0, 1, 2, 3], [1, 2, 3, 0], [4, 4, 4, 4]])
    return vlat, vlon, vertex_of_cell


def test_icon_domain_boundary_ring():
    vlat, vlon, vertex_of_cell = _four_triangles()
    lat, lon = icon_domain_boundary(vlat, vlon, vertex_of_cell)
    assert lat.size == lon.size == 4
    # each step, including last -> first, is one side of the square
    steps = np.abs(np.diff(np.append(lat, lat[0]))) + np.abs(np.diff(np.append(lon, lon[0])))
    assert steps.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert sorted(zip(lat.tolist(), lon.tolist())) == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


def test_icon_domain_boundary_global_mesh():
    # a tetrahedron has no boundary edge
    vertex_of_cell = np.array([[0, 0, 0, 1], [1, 2, 3, 2], [2, 3, 1, 3]])
    with pytest.raises(ValueError, match="no boundary edges"):
        icon_domain_boundary(np.zeros(4), np.zeros(4), vertex_of_cell)


def test_icon_domain_boundary_two_rings():
    # two triangles that share no vertex: two boundary rings
    vertex_of_cell = np.array([[0, 3], [1, 4], [2, 5]])
    with pytest.raises(ValueError, match="one ring"):
        icon_domain_boundary(np.zeros(6), np.zeros(6), vertex_of_cell)


def test_icon_domain_boundary_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        icon_domain_boundary(np.zeros(4), np.zeros(4), np.zeros((2, 4), dtype=int))


def test_outside_icon_domain():
    vlat, vlon, vertex_of_cell = _four_triangles()
    # inside a cell, on the interior edge between two cells, outside
    lat = np.array([0.3, 0.25, 1.5])
    lon = np.array([0.5, 0.25, 0.5])
    assert outside_icon_domain(vlat, vlon, vertex_of_cell, lat, lon).tolist() == [False, False, True]
    res = outside_icon_domain(vlat, vlon, vertex_of_cell, lat.reshape(3, 1), lon.reshape(3, 1))
    assert res.shape == (3, 1)


def test_outside_icon_domain_without_matplotlib(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib.path", None)
    vlat, vlon, vertex_of_cell = _four_triangles()
    with pytest.raises(ImportError, match="matplotlib"):
        outside_icon_domain(vlat, vlon, vertex_of_cell, [0.5], [0.5])
