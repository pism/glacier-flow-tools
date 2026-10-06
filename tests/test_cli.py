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

import dask
import geopandas as gp
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from glacier_flow_tools import compute_pathlines, compute_profiles

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
        {
            "vx": (("y", "x"), np.full(shape, 100.0), {"units": "m/yr"}),
            "vy": (("y", "x"), np.zeros(shape), {"units": "m/yr"}),
        },
        coords={"x": ("x", x, {"units": "m"}), "y": ("y", y, {"units": "m"})},
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


@pytest.fixture(name="profile_inputs")
def fixture_profile_inputs(tmp_path):
    """
    Write observations, two simulations and two profiles for a uniform northward flow.

    The observed speed is 100 m/yr. The simulated speeds are 90 m/yr and 110 m/yr.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    tuple
        The observation file name, the list of simulation file names and the profiles file name.
    """
    x = np.arange(0.0, 10_000.0, 100.0)
    y = np.arange(0.0, 10_000.0, 100.0)
    shape = (len(y), len(x))

    obs_values = {"vx": 0.0, "vy": 100.0, "v": 100.0, "vx_err": 1.0, "vy_err": 1.0, "v_err": 1.0}
    obs_values |= {"rock": 0.0, "count": 1.0, "ocean": 0.0, "ice": 1.0}
    obs = tmp_path / "obs.nc"
    xr.Dataset({k: (("y", "x"), np.full(shape, v)) for k, v in obs_values.items()}, coords={"x": x, "y": y}).to_netcdf(
        obs, engine="h5netcdf"
    )

    sims = []
    for exp_id, speed in enumerate([90.0, 110.0]):
        sim = tmp_path / f"sim_id_{exp_id}_0_50.nc"
        xr.Dataset(
            {
                "uvelsurf": (("time", "y", "x"), np.zeros((1, *shape)), {"units": "m/yr"}),
                "vvelsurf": (("time", "y", "x"), np.full((1, *shape), speed), {"units": "m/yr"}),
            },
            coords={"time": [0.0], "x": x, "y": y},
        ).to_netcdf(sim, engine="h5netcdf")
        sims.append(sim)

    profiles = tmp_path / "profiles.gpkg"
    gp.GeoDataFrame(
        {"id": [1, 2], "name": ["a", "b"]},
        geometry=[LineString([(2000, 5000), (8000, 5000)]), LineString([(2000, 3000), (8000, 3000)])],
        crs="EPSG:3413",
    ).to_file(profiles)
    return obs, sims, profiles


def test_compute_profiles(profile_inputs, tmp_path, monkeypatch):
    """
    Extract two profiles from two simulations and check the statistics and figures.

    Parameters
    ----------
    profile_inputs : tuple
        The observation file name, the list of simulation file names and the profiles file name.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    """
    obs, sims, profiles = profile_inputs
    result_dir = tmp_path / "results"
    argv = [
        "compute_profiles",
        "--n_jobs",
        "1",
        "--segmentize",
        "500",
        "--profiles_url",
        str(profiles),
        "--velocity_url",
        str(obs),
        "--result_dir",
        str(result_dir),
        *[str(s) for s in sims],
    ]
    monkeypatch.setattr("sys.argv", argv)
    # The workers import dask_geopandas through the script when it is run from the command line.
    # Here pytest is the main program, so they are told to import it.
    with dask.config.set({"distributed.worker.preload": ["dask_geopandas"]}):
        compute_profiles.main()

    stats = gp.read_file(result_dir / "files" / "stats.gpkg")
    assert len(stats) == 4
    assert set(zip(stats["profile_id"], stats["exp_id"])) == {(1, 0), (1, 1), (2, 0), (2, 1)}
    # Simulated speeds of 90 and 110 m/yr are both 10 m/yr off the observed 100 m/yr.
    assert np.allclose(stats["rmsd"], 10.0)
    # Flux through the 6 km long profiles scales with the speed.
    assert np.allclose(np.abs(stats["obs_flux"]), 100.0 * 6000.0)
    for exp_id, speed in [(0, 90.0), (1, 110.0)]:
        assert np.allclose(np.abs(stats.loc[stats["exp_id"] == exp_id, "sim_flux"]), speed * 6000.0)

    assert sorted(p.name for p in (result_dir / "figures").iterdir()) == ["a_profile.pdf", "b_profile.pdf"]


def test_compute_pathlines_densify(tmp_path, monkeypatch):
    """
    With ``--densify`` a pathline starts every 500 m along the line.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    """
    x = np.arange(0.0, 10_000.0, 100.0)
    y = np.arange(0.0, 10_000.0, 100.0)
    shape = (len(y), len(x))
    raster = tmp_path / "velocity.nc"
    xr.Dataset(
        {
            "vx": (("y", "x"), np.full(shape, 100.0), {"units": "m/yr"}),
            "vy": (("y", "x"), np.zeros(shape), {"units": "m/yr"}),
        },
        coords={"x": ("x", x, {"units": "m"}), "y": ("y", y, {"units": "m"})},
    ).to_netcdf(raster)
    # A 1.2 km long line across the flow: points at y = 4000, 4500 and 5000.
    vector = tmp_path / "start.gpkg"
    gp.GeoDataFrame(
        {"id": [1], "name": ["a"]}, geometry=[LineString([(2000, 4000), (2000, 5200)])], crs="EPSG:3413"
    ).to_file(vector)
    outfile = tmp_path / "pathlines.gpkg"

    argv = ["compute_pathlines", "--raster_url", str(raster), "--vector_url", str(vector)]
    argv += ["--n_jobs", "1", "--end_time", "5", "--densify", "500m", str(outfile)]
    monkeypatch.setattr("sys.argv", argv)
    compute_pathlines.main()

    result = gp.read_file(outfile)
    starts = result.groupby("pathline_id").first()
    assert len(starts) == 3
    assert [(g.x, g.y) for g in starts.geometry] == [(2000.0, 4000.0), (2000.0, 4500.0), (2000.0, 5000.0)]


def test_compute_pathlines_densify_rejects_bad_distance(monkeypatch, capsys):
    """
    An invalid distance ends with a usage error that explains the problem.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    capsys : pytest.CaptureFixture
        Used to capture the error message.
    """
    monkeypatch.setattr("sys.argv", ["compute_pathlines", "--densify", "abc", "out.gpkg"])
    with pytest.raises(SystemExit) as exc:
        compute_pathlines.main()
    assert exc.value.code == 2
    assert "is not a distance" in capsys.readouterr().err


def write_velocity(path, vx, velocity_units=None, x_units=None, scale_x=1.0):
    """
    Write a uniform eastward velocity field on a 10 km by 10 km grid.

    Parameters
    ----------
    path : pathlib.Path
        The file to write.
    vx : float
        The velocity in x.
    velocity_units : str, optional
        The units attribute of the velocities. If None (default), they have no units.
    x_units : str, optional
        The units attribute of the coordinates. If None (default), they have no units.
    scale_x : float, optional
        Factor applied to the coordinates, which are in meters before scaling.
    """
    x = np.arange(0.0, 10_000.0, 100.0)
    shape = (len(x), len(x))
    v_attrs = {} if velocity_units is None else {"units": velocity_units}
    x_attrs = {} if x_units is None else {"units": x_units}
    xr.Dataset(
        {"vx": (("y", "x"), np.full(shape, vx), v_attrs), "vy": (("y", "x"), np.zeros(shape), v_attrs)},
        coords={"x": ("x", x * scale_x, x_attrs), "y": ("y", x * scale_x, x_attrs)},
    ).to_netcdf(path)


def run_compute_pathlines(tmp_path, monkeypatch, raster, extra_args=()):
    """
    Run compute_pathlines for 5 years from the point (2000, 5000) and return the result.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    raster : pathlib.Path
        The velocity file.
    extra_args : sequence of str, optional
        Additional command line arguments.

    Returns
    -------
    numpy.ndarray
        The x coordinates of the points along the pathline.
    """
    vector = tmp_path / "start.gpkg"
    gp.GeoDataFrame({"id": [1], "name": ["a"]}, geometry=[Point(2000, 5000)], crs="EPSG:3413").to_file(vector)
    outfile = tmp_path / "pathlines.gpkg"
    argv = ["compute_pathlines", "--raster_url", str(raster), "--vector_url", str(vector)]
    argv += ["--n_jobs", "1", "--end_time", "5", "--hmin", "1", "--hmax", "1", *extra_args, str(outfile)]
    monkeypatch.setattr("sys.argv", argv)
    compute_pathlines.main()
    result = gp.read_file(outfile)
    return np.array([g.x for g in result[result["pathline_id"] == 0].geometry])


SECONDS_PER_YEAR = 365.25 * 86400.0
EXPECTED_X = [2000.0, 2100.0, 2200.0, 2300.0, 2400.0]


@pytest.mark.parametrize(
    "vx, velocity_units",
    [(100.0, "m/yr"), (100.0, "m year-1"), (0.1, "km/yr"), (100.0 / SECONDS_PER_YEAR, "m/s")],
)
def test_compute_pathlines_converts_velocity_units(tmp_path, monkeypatch, vx, velocity_units):
    """
    Get the same pathline for the same flow of 100 m/yr, whatever units the file uses.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    vx : float
        The velocity in x, in the units of the file.
    velocity_units : str
        The units of the velocities in the file.
    """
    raster = tmp_path / "velocity.nc"
    write_velocity(raster, vx, velocity_units=velocity_units, x_units="m")
    assert np.allclose(run_compute_pathlines(tmp_path, monkeypatch, raster), EXPECTED_X)


def test_compute_pathlines_converts_coordinate_units(tmp_path, monkeypatch):
    """
    Convert grid coordinates given in km to meters.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    """
    raster = tmp_path / "velocity.nc"
    write_velocity(raster, 100.0, velocity_units="m/yr", x_units="km", scale_x=1e-3)
    assert np.allclose(run_compute_pathlines(tmp_path, monkeypatch, raster), EXPECTED_X)


def test_compute_pathlines_assumes_units_with_a_warning(tmp_path, monkeypatch):
    """
    Assume m/yr and m, with a warning, for a file without units.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    """
    raster = tmp_path / "velocity.nc"
    write_velocity(raster, 100.0)
    with pytest.warns(UserWarning) as record:
        result = run_compute_pathlines(tmp_path, monkeypatch, raster)
    messages = [str(w.message) for w in record]
    assert "'vx' has no units, assuming 'm/yr'." in messages
    assert "'x' has no units, assuming 'm'." in messages
    assert np.allclose(result, EXPECTED_X)


def test_compute_pathlines_velocity_units_option(tmp_path, monkeypatch):
    """
    Use ``--velocity_units`` in place of the units in the file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    """
    raster = tmp_path / "velocity.nc"
    # 'm/y' is not understood, so the file alone would be rejected.
    write_velocity(raster, 100.0 / 365.25, velocity_units="m/y", x_units="m")
    result = run_compute_pathlines(tmp_path, monkeypatch, raster, extra_args=["--velocity_units", "m/d"])
    assert np.allclose(result, EXPECTED_X)


@pytest.mark.parametrize(
    "velocity_units, message",
    [("m/y", "The units 'm/y' of 'vx' are not understood."), ("m", "cannot be converted to 'm/yr'")],
)
def test_compute_pathlines_rejects_bad_units(tmp_path, monkeypatch, capsys, velocity_units, message):
    """
    End with a usage error for velocity units that are not understood or are not a velocity.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Used to set the command line arguments.
    capsys : pytest.CaptureFixture
        Used to capture the error message.
    velocity_units : str
        The units of the velocities in the file.
    message : str
        Part of the expected error message.
    """
    raster = tmp_path / "velocity.nc"
    write_velocity(raster, 100.0, velocity_units=velocity_units, x_units="m")
    with pytest.raises(SystemExit) as exc:
        run_compute_pathlines(tmp_path, monkeypatch, raster)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
