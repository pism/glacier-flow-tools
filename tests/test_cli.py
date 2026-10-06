# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
#
# This file is part of glacier-flow-tools.
#
# GLACIER-FLOW-TOOLS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# GLACIER-FLOW-TOOLS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with glacier-flow-tools; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Tests for the console scripts.
"""

import importlib
from importlib.metadata import entry_points

import geopandas as gp
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString

from glacier_flow_tools import compute_pathlines

SCRIPTS = ["compute_pathlines", "compute_profiles"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_entry_point_resolves(name):
    """
    The installed console script points at a callable.

    Parameters
    ----------
    name : str
        Name of the console script.
    """
    (entry_point,) = [e for e in entry_points(group="console_scripts") if e.name == name]
    assert callable(entry_point.load())


@pytest.mark.parametrize("name", SCRIPTS)
def test_help(name, monkeypatch, capsys):
    """
    Check that ``--help`` prints usage and exits with status 0.

    Parameters
    ----------
    name : str
        Name of the console script.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    capsys : pytest.CaptureFixture
        Used to capture the printed usage.
    """
    module = importlib.import_module(f"glacier_flow_tools.{name}")
    monkeypatch.setattr("sys.argv", [name, "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out


@pytest.fixture(name="pathline_inputs")
def fixture_pathline_inputs(tmp_path):
    """
    Write a uniform eastward velocity field of 100 m/yr and a short starting line.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    tuple of pathlib.Path
        The raster and vector file names.
    """
    x = np.arange(0.0, 10_000.0, 100.0)
    y = np.arange(0.0, 10_000.0, 100.0)
    shape = (len(y), len(x))
    ds = xr.Dataset(
        {"vx": (("y", "x"), np.full(shape, 100.0)), "vy": (("y", "x"), np.zeros(shape))},
        coords={"x": x, "y": y},
    )
    raster = tmp_path / "velocity.nc"
    ds.to_netcdf(raster)
    vector = tmp_path / "start.gpkg"
    gp.GeoDataFrame(
        {"id": [1], "name": ["a"]}, geometry=[LineString([(5000, 5000), (5100, 5000)])], crs="EPSG:3413"
    ).to_file(vector)
    return raster, vector


@pytest.mark.parametrize(
    "extra_args, output_type, direction",
    [
        ([], "point", 1.0),
        (["--reverse"], "point", -1.0),
        ([], "line", 1.0),
    ],
)
def test_compute_pathlines(pathline_inputs, tmp_path, monkeypatch, extra_args, output_type, direction):
    """
    Compute pathlines in a uniform flow and check where they end.

    Parameters
    ----------
    pathline_inputs : tuple of pathlib.Path
        The raster and vector file names.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    extra_args : list of str
        Additional command line arguments.
    output_type : str
        Either ``point`` or ``line``.
    direction : float
        Direction of the flow in x: 1 for eastward, -1 for westward (the speed is 100 m/yr).
    """
    raster, vector = pathline_inputs
    outfile = tmp_path / "out" / "pathlines.gpkg"
    argv = [
        "compute_pathlines",
        "--raster_url",
        str(raster),
        "--vector_url",
        str(vector),
        "--n_jobs",
        "1",
        "--end_time",
        "10",
        "--output_type",
        output_type,
        *extra_args,
        str(outfile),
    ]
    monkeypatch.setattr("sys.argv", argv)
    compute_pathlines.main()

    result = gp.read_file(outfile)
    assert set(result["pathline_id"]) == {0, 1}

    # One pathline per end point of the starting line, advected by 100 m/yr in x.
    for pathline_id, start_x in [(0, 5000.0), (1, 5100.0)]:
        pathline = result[result["pathline_id"] == pathline_id]
        if output_type == "point":
            xs = np.array([g.x for g in pathline.geometry])
            ys = np.array([g.y for g in pathline.geometry])
        else:
            xs, ys = (np.array(c) for c in pathline.geometry.iloc[0].xy)
        assert xs[0] == start_x
        assert np.allclose(ys, 5000.0)
        assert np.allclose(np.diff(xs), np.sign(direction) * 100.0)
        assert abs(xs[-1] - xs[0]) >= 900.0
