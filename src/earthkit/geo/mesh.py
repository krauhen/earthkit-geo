# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Neighbourhood tables on an unstructured triangular mesh such as ICON's.

The functions take the connectivity arrays of an ICON grid file, made
**0-based** (subtract 1 from the file's values, so a missing entry is -1),
and return for every cell the indices of the cells around it. A neighbourhood
average is then the mean over the valid (``>= 0``) entries of each row.
"""

import numpy as np

_BLOCK = 50_000  # rows processed at once, to bound the memory of the intermediate tables


def _check_order(order):
    if int(order) != order or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")
    return int(order)


def _pack_rows(candidates):
    """Per row: the sorted unique non-negative entries, left-packed and padded with -1."""
    srt = np.sort(candidates, axis=1)
    keep = srt >= 0
    keep[:, 1:] &= srt[:, 1:] != srt[:, :-1]
    # stable sort of the "drop" flag moves the kept entries to the left, in order
    order = np.argsort(~keep, axis=1, kind="stable")
    keep = np.take_along_axis(keep, order, axis=1)
    packed = np.where(keep, np.take_along_axis(srt, order, axis=1), -1).astype(np.int32)
    return packed[:, : keep.sum(axis=1).max(initial=0)]


def _grow(table, one_step):
    """Union, per row of ``table``, of the one-step neighbourhoods of its members."""
    blocks = []
    for start in range(0, table.shape[0], _BLOCK):
        rows = table[start : start + _BLOCK]
        grown = one_step[np.where(rows >= 0, rows, 0)]  # (rows, members, step_width)
        grown[rows < 0] = -1
        blocks.append(_pack_rows(grown.reshape(len(rows), -1)))
    width = max(b.shape[1] for b in blocks)
    return np.concatenate([np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=-1) for b in blocks])


def neighbour_cells(vertex_of_cell, cells_of_vertex, order=1):
    """Return the cells within ``order`` vertex-sharing steps of every cell.

    One step adds every cell that shares a vertex with a cell already in the
    set. On a regular part of an ICON mesh this gives 13, 37 and 73 cells
    (centre included) for order 1, 2 and 3.

    Parameters
    ----------
    vertex_of_cell: array-like
        The three vertex indices of each cell, shape ``(3, n_cells)``, 0-based.
    cells_of_vertex: array-like
        The cells around each vertex, shape ``(k, n_vertices)`` (``k = 6`` on
        ICON), 0-based, -1 where a vertex has fewer cells.
    order: int, optional
        Number of steps, 1 or more. The default is 1.

    Returns
    -------
    ndarray of int32
        Shape ``(n_cells, width)``. Row ``i`` holds the sorted cell indices
        of the neighbourhood of cell ``i``, the cell itself included, padded
        with -1. Near the domain edge a row has fewer valid entries.

    Examples
    --------
    >>> import numpy as np
    >>> from earthkit.geo.mesh import neighbour_cells
    >>> vertex_of_cell = np.array([[0, 1], [1, 3], [2, 2]])
    >>> cells_of_vertex = np.array([[0, 0, 0, 1], [-1, 1, 1, -1]])
    >>> neighbour_cells(vertex_of_cell, cells_of_vertex)
    array([[0, 1],
           [0, 1]], dtype=int32)
    """
    order = _check_order(order)
    vertex_of_cell = np.asarray(vertex_of_cell)
    cells_of_vertex = np.asarray(cells_of_vertex)
    if vertex_of_cell.ndim != 2 or vertex_of_cell.shape[0] != 3:
        raise ValueError("vertex_of_cell must have shape (3, n_cells)")
    n_cells = vertex_of_cell.shape[1]

    # every cell's 3 vertices, each vertex's k cells: (n_cells, 3 * k)
    table = one_step = _pack_rows(cells_of_vertex[:, vertex_of_cell].reshape(-1, n_cells).T)
    for _ in range(order - 1):
        table = _grow(table, one_step)
    return table


def neighbour_cells_diamond(neighbor_cell_index, order=1):
    """Return the cells within ``order`` edge-sharing steps of every cell.

    One step adds every cell that shares an edge with a cell already in the
    set. On a regular part of an ICON mesh this gives 4, 10 and 19 cells
    (centre included) for order 1, 2 and 3.

    Parameters
    ----------
    neighbor_cell_index: array-like
        The three edge neighbours of each cell, shape ``(3, n_cells)``. This is
        the ICON grid-file variable ``neighbor_cell_index``, made 0-based, with
        -1 where there is none.
    order: int, optional
        Number of steps, 1 or more. The default is 1.

    Returns
    -------
    ndarray of int32
        Shape ``(n_cells, width)``. Row ``i`` holds the sorted cell indices
        of the neighbourhood of cell ``i``, the cell itself included, padded
        with -1.

    Examples
    --------
    >>> import numpy as np
    >>> from earthkit.geo.mesh import neighbour_cells_diamond
    >>> neighbor_cell_index = np.array([[1, 0, 1], [-1, 2, -1], [-1, -1, -1]])
    >>> neighbour_cells_diamond(neighbor_cell_index)
    array([[ 0,  1, -1],
           [ 0,  1,  2],
           [ 1,  2, -1]], dtype=int32)
    """
    order = _check_order(order)
    nb = np.asarray(neighbor_cell_index)
    if nb.ndim != 2 or nb.shape[0] != 3:
        raise ValueError("neighbor_cell_index must have shape (3, n_cells)")
    nb = nb.T.astype(np.int32)
    n_cells = nb.shape[0]

    table = one_step = _pack_rows(np.concatenate([np.arange(n_cells, dtype=np.int32)[:, None], nb], axis=1))
    for _ in range(order - 1):
        table = _grow(table, one_step)
    return table
