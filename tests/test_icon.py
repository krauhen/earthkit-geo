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

from earthkit.geo.icon import icontools_target_points, nearest_cell_icontools
from earthkit.geo.rotate import rotate


def _brute_force(src_lat, src_lon, tgt_lat, tgt_lon):
    value = np.sin(src_lat)[None, :] * np.sin(tgt_lat)[:, None] + np.cos(src_lat)[None, :] * np.cos(tgt_lat)[
        :, None
    ] * np.cos(src_lon[None, :] - tgt_lon[:, None])
    # argmin returns the first, i.e. the lowest index, on an exact tie
    return np.argmin(np.arccos(np.clip(value, -1.0, 1.0)), axis=1)


def test_nearest_cell_icontools_tie_lowest_index():
    # target at (0, 0); sources 1 and 3 are exactly equidistant, both nearer than 0 and 2
    src_lat = np.deg2rad([0.0, 0.0, 5.0, 0.0])
    src_lon = np.deg2rad([3.0, 1.0, 0.0, -1.0])
    res = nearest_cell_icontools(src_lat, src_lon, [0.0], [0.0])
    assert res.tolist() == [1]

    # same points, swapped order: the lower index still wins
    res = nearest_cell_icontools(src_lat[::-1], src_lon[::-1], [0.0], [0.0])
    assert res.tolist() == [0]


def test_nearest_cell_icontools_brute_force():
    rng = np.random.default_rng(42)
    src_lat = np.arcsin(rng.uniform(-1.0, 1.0, 2000))
    src_lon = rng.uniform(-np.pi, np.pi, 2000)
    tgt_lat = np.arcsin(rng.uniform(-1.0, 1.0, 1000))
    tgt_lon = rng.uniform(-np.pi, np.pi, 1000)

    res = nearest_cell_icontools(src_lat, src_lon, tgt_lat, tgt_lon)
    ref = _brute_force(src_lat, src_lon, tgt_lat, tgt_lon)
    assert res.dtype.kind == "i"
    np.testing.assert_array_equal(res, ref)


def test_nearest_cell_icontools_cached_tree():
    rng = np.random.default_rng(1)
    src_lat = np.arcsin(rng.uniform(-1.0, 1.0, 300))
    src_lon = rng.uniform(-np.pi, np.pi, 300)
    src_lat.setflags(write=False)
    src_lon.setflags(write=False)
    tgt_lat = np.arcsin(rng.uniform(-1.0, 1.0, 50))
    tgt_lon = rng.uniform(-np.pi, np.pi, 50)

    ref = _brute_force(src_lat, src_lon, tgt_lat, tgt_lon)
    for _ in range(2):
        np.testing.assert_array_equal(nearest_cell_icontools(src_lat, src_lon, tgt_lat, tgt_lon), ref)


def test_nearest_cell_icontools_bad_shape():
    with pytest.raises(ValueError):
        nearest_cell_icontools(np.zeros(3), np.zeros(2), [0.0], [0.0])


def test_icontools_target_points_hand_list():
    lat, lon = icontools_target_points([46.02, 7.0, 46.0, 7.01], [0.01, 0.01])
    assert lat.shape == lon.shape == (3, 2)

    # rows run north to south, columns west to east
    ref_lat = [[46.02, 46.02], [46.01, 46.01], [46.0, 46.0]]
    ref_lon = [[7.0, 7.01], [7.0, 7.01], [7.0, 7.01]]
    np.testing.assert_allclose(np.rad2deg(lat), ref_lat, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(lon), ref_lon, rtol=0, atol=1e-12)

    # exact construction: 1e-6 * integer micro-degrees, then * pi/180
    assert lat[2, 0] == (1e-6 * 46_000_000.0) * (np.pi / 180.0)
    assert lon[0, 1] == (1e-6 * 7_010_000.0) * (np.pi / 180.0)


def test_icontools_target_points_wraps_longitude():
    _, lon = icontools_target_points([0.0, 179.99, 0.0, 180.01], [0.01, 0.01])
    np.testing.assert_allclose(np.rad2deg(lon[0]), [179.99, 180.0, -179.99], rtol=0, atol=1e-9)


def test_icontools_target_points_rotated():
    # COSMO/ICON-CH style rotated grid: north pole at (43 N, 190 E) = south pole at (43 S, 10 E)
    area = [0.1, -0.2, -0.1, 0.2]
    lat, lon = icontools_target_points(area, [0.1, 0.1], north_pole_lat=43.0, north_pole_lon=-170.0)
    assert lat.shape == (3, 5)

    r_lat = np.repeat([0.1, 0.0, -0.1], 5)
    r_lon = np.tile([-0.2, -0.1, 0.0, 0.1, 0.2], 3)
    ref_lat, ref_lon = rotate(r_lat, r_lon, -43.0, 10.0)
    np.testing.assert_allclose(np.rad2deg(lat).ravel(), ref_lat, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.rad2deg(lon).ravel(), ref_lon, rtol=0, atol=1e-9)


def test_icontools_target_points_pole_needs_both():
    with pytest.raises(ValueError):
        icontools_target_points([1.0, 0.0, 0.0, 1.0], [1.0, 1.0], north_pole_lat=43.0)
