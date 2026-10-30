# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Reproduce the nearest-neighbour selection of the ICON tool ``icontools``.

The functions work on plain numpy arrays in ``(lat, lon)`` order. Angles are
in **radians**: the selection rule is exact to the last bit, and a round trip
through degrees would move a point by ~1e-16 rad, enough to flip a near-tie.
"""

import math

import numpy as np

# re-exported: the mesh-outline functions live in mesh.py, the ports import them from here
from .mesh import (
    icon_domain_boundary,  # noqa: F401
    outside_icon_domain,  # noqa: F401
)

PI_180 = math.pi / 180.0
_UMICRO = 1_000_000

_TREE_CACHE = {}


def icontools_target_points(area, grid, north_pole_lat=None, north_pole_lon=None):
    """Build the cell centres of a (rotated) regular lat/lon target grid as icontools does.

    Parameters
    ----------
    area: list or tuple
        ``[north, west, south, east]`` of the target grid in degrees, in its own
        (rotated, if a pole is given) coordinates. Same order as MIR's ``area`` option.
    grid: list or tuple
        ``[dlon, dlat]`` increments in degrees. Same order as MIR's ``grid`` option.
    north_pole_lat: float, optional
        Latitude of the rotated north pole (degrees). ``None`` for a regular grid.
    north_pole_lon: float, optional
        Longitude of the rotated north pole (degrees). ``None`` for a regular grid.

    Returns
    -------
    tuple of ndarray
        Geographic latitudes and longitudes in **radians**, both with shape
        ``(nj, ni)``, rows from north to south (the order MIR writes them).

    Notes
    -----
    The arithmetic follows fieldextra ``generate_regular_grid`` and icontools
    ``rotate_latlon_points_internal`` step by step: axis values are
    ``1e-6 * float(integer micro-degrees)``, a longitude above 180 loses 360,
    the pole is rounded to micro-degrees, and a rotated point goes degrees to
    radians, is rotated, then goes back to degrees and to radians again.
    A different but mathematically equal route can flip a near-tie in
    :func:`nearest_cell_icontools`.

    Examples
    --------
    >>> import numpy as np
    >>> from earthkit.geo.icon import icontools_target_points
    >>> lat, lon = icontools_target_points([47.0, 7.0, 46.99, 7.01], [0.01, 0.01])
    >>> lat.shape
    (2, 2)
    >>> np.rad2deg(lat[:, 0]).round(6)
    array([47.  , 46.99])
    >>> np.rad2deg(lon[0]).round(6)
    array([7.  , 7.01])
    """
    if (north_pole_lat is None) != (north_pole_lon is None):
        raise ValueError("north_pole_lat and north_pole_lon must be given together")

    north, west, south, east = area
    dlon, dlat = grid
    di, dj = round(dlon * _UMICRO), round(dlat * _UMICRO)
    swi = round(west * _UMICRO) % (360 * _UMICRO)
    swj = round(south * _UMICRO)
    ni = round((east - west) * _UMICRO / di) + 1
    nj = round((north - south) * _UMICRO / dj) + 1
    x = 1e-6 * (swi + di * np.arange(ni)).astype("f8")
    x = np.where(x > 180.0, x - 360.0, x)
    y = 1e-6 * (swj + dj * np.arange(nj)).astype("f8")

    if north_pole_lat is None:
        lon = np.broadcast_to(x, (nj, ni))
        lat = np.broadcast_to(y[:, None], (nj, ni))
    else:
        pole_lon = 1e-6 * round(north_pole_lon * _UMICRO)
        pole_lat = 1e-6 * round(north_pole_lat * _UMICRO)
        sin_x, cos_x = np.sin(x * PI_180)[None, :], np.cos(x * PI_180)[None, :]
        sin_y, cos_y = np.sin(y * PI_180)[:, None], np.cos(y * PI_180)[:, None]
        sin_pl, cos_pl = math.sin(pole_lon * PI_180), math.cos(pole_lon * PI_180)
        sin_pb, cos_pb = math.sin(pole_lat * PI_180), math.cos(pole_lat * PI_180)
        arg1 = cos_y * sin_x
        arg2 = sin_pb * cos_y * cos_x - sin_y * cos_pb
        lon = np.arctan2(-1.0 * sin_pl * arg2 - cos_pl * arg1, -1.0 * cos_pl * arg2 + sin_pl * arg1)
        lat = np.arcsin(sin_y * sin_pb + cos_y * cos_x * cos_pb)
        lon, lat = lon / PI_180, lat / PI_180

    return np.asarray(lat * PI_180)[::-1], np.asarray(lon * PI_180)[::-1]


def _unit_xyz(lat, lon):
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _source_tree(src_lat, src_lon):
    """Return a KD-tree over the source points on the unit sphere.

    The last tree is kept when both arrays are read-only and are passed again
    as the same objects, so a large mesh is indexed only once.
    """
    from scipy.spatial import cKDTree

    cached = _TREE_CACHE.get("last")
    if cached is not None and cached[0] is src_lat and cached[1] is src_lon:
        return cached[2]

    tree = cKDTree(_unit_xyz(src_lat, src_lon))
    if not (src_lat.flags.writeable or src_lon.flags.writeable):
        _TREE_CACHE["last"] = (src_lat, src_lon, tree)
    return tree


def nearest_cell_icontools(src_lat, src_lon, tgt_lat, tgt_lon, candidates=8):
    r"""Find the source cell icontools' nearest-neighbour rule picks for each target point.

    Parameters
    ----------
    src_lat: ndarray
        Latitudes of the source cell centres (radians), 1-D.
    src_lon: ndarray
        Longitudes of the source cell centres (radians), 1-D.
    tgt_lat: array-like
        Latitudes of the target points (radians). Flattened.
    tgt_lon: array-like
        Longitudes of the target points (radians). Flattened.
    candidates: int, optional
        Number of nearest neighbours a KD-tree preselects before the exact
        distance decides. The default is 8.

    Returns
    -------
    ndarray
        For each target point, the index of the selected source cell.

    Notes
    -----
    The rule is the one of icontools 2.7.0 (``query_nnb_list`` and its GNAT search):
    the distance is :math:`\arccos(\sin\phi_1 \sin\phi_2 + \cos\phi_1 \cos\phi_2
    \cos(\lambda_1 - \lambda_2))` in float64, clipped to ``[-1, 1]``, the
    smallest distance wins, and an exact tie goes to the lowest source index.
    Two cells 1e-10 rad apart are not a tie. MIR's own nearest-neighbour search
    can break such near-ties differently.

    The tree is kept between calls when ``src_lat`` and ``src_lon`` are
    read-only arrays passed again as the same objects.

    Examples
    --------
    >>> import numpy as np
    >>> from earthkit.geo.icon import nearest_cell_icontools
    >>> src_lat = np.deg2rad([0.0, 0.0, 10.0])
    >>> src_lon = np.deg2rad([-1.0, 1.0, 0.0])
    >>> nearest_cell_icontools(src_lat, src_lon, [0.0, np.deg2rad(9.0)], [0.0, 0.0], candidates=3)
    array([0, 2])
    """
    src_lat = np.asarray(src_lat, dtype="f8")
    src_lon = np.asarray(src_lon, dtype="f8")
    tgt_lat = np.ravel(np.asarray(tgt_lat, dtype="f8"))
    tgt_lon = np.ravel(np.asarray(tgt_lon, dtype="f8"))
    if src_lat.shape != src_lon.shape or src_lat.ndim != 1:
        raise ValueError("src_lat and src_lon must be 1-D arrays of the same shape")
    if tgt_lat.shape != tgt_lon.shape:
        raise ValueError("tgt_lat and tgt_lon must have the same size")

    k = min(candidates, src_lat.size)
    _, near = _source_tree(src_lat, src_lon).query(_unit_xyz(tgt_lat, tgt_lon), k=k)
    near = near.reshape(tgt_lat.size, k)

    value = np.sin(src_lat[near]) * np.sin(tgt_lat)[:, None] + np.cos(src_lat[near]) * np.cos(tgt_lat)[
        :, None
    ] * np.cos(src_lon[near] - tgt_lon[:, None])
    dist = np.arccos(np.clip(value, -1.0, 1.0))
    best = np.where(dist == dist.min(axis=1)[:, None], near, np.iinfo(np.int64).max)
    return best.min(axis=1)
