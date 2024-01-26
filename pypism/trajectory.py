# Copyright (C) 2023 Andy Aschwanden, Constantine Khroulev
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-RAGIS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Module provides functions for calculating trajectories
"""


from pathlib import Path
from typing import Tuple, Union

import geopandas as gp
import numpy as np
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame
from numpy import ndarray
from osgeo import ogr, osr
from shapely import Point
from tqdm.auto import tqdm
from xarray import DataArray

from .interpolation import interpolate_rkf, velocity_at_point

# Need to figure out how to make hooks so we can detect and propagate how we use TQDM
# https://github.com/benbovy/xarray-simlab/blob/master/xsimlab/monitoring.py


def compute_trajectory(
    point: Point,
    Vx: Union[ndarray, DataArray],
    Vy: Union[ndarray, DataArray],
    x: Union[ndarray, DataArray],
    y: Union[ndarray, DataArray],
    dt: float = 0.1,
    total_time: float = 1000,
    reverse: bool = False,
) -> Tuple[list[Point], list]:
    """
    Compute trajectory

    Computes a trajectory using Runge-Kutta-Fehlberg integration. Routine is
    unit-agnostic, requiring the user to ensure consistency of units. For example
    if the velocity field is given in m/yr, the `dt` and `total_time` are assumed
    to be in years.

    Parameters
    ----------
    Point : shapely.Point
        Starting point of the trajectory
    Vx : numpy.ndarray or xarray.DataArray
        x-component of velocity
    Vy : numpy.ndarray or xarray.DataArray
        y-component of velocity
    x : numpy.ndarray or xarray.DataArray
        coordinates in x direction
    y : numpy.ndarray or xarray.DataArray
        coordinates in y direction
    dt : float
        integration time step
    dt : float
        total integration time

    Returns
    ----------
    pts: list[shapely.Point]
        `dt`-spaced points along trajectory from 0 to `total_time`.
    pts_error_estim: list[
        error estimate at `dt`-spaced points along trajectory
        from 0 to `total_time`.

    Examples
    ----------

    Create data:

    >>>    import numpy as np
    >>>    from shapely import Point

    >>>    nx = 201
    >>>    ny = 401
    >>>    x = np.linspace(-100e3, 100e3, nx)
    >>>    y = np.linspace(-100e3, 100e3, ny)
    >>>    X, Y = np.meshgrid(x, y)

    >>>    vx = -Y / np.sqrt(X**2 + Y**2) * 250
    >>>    vy = X / np.sqrt(X**2 + Y**2) * 250
    >>>    p = Point(0, -50000)

    >>>    pts, pts_error_estim = compute_trajectory(p, vx, vx, x, y, dt=1, total_time=10)
    >>>    pts
    [<POINT (0 -50000)>,
     <POINT (249.994 -49750.006)>,
     <POINT (499.975 -49500.025)>,
     <POINT (749.943 -49250.057)>,
     <POINT (999.897 -49000.103)>,
     <POINT (1249.825 -48750.175)>,
     <POINT (1499.713 -48500.287)>,
     <POINT (1749.56 -48250.44)>,
     <POINT (1999.364 -48000.636)>,
     <POINT (2249.113 -47750.887)>,
     <POINT (2498.79 -47501.21)>,
     <POINT (2748.394 -47251.606)>]
    """
    if reverse:
        Vx = -Vx
        Vy = -Vy
    pts = [point]
    pts_error_estim = [0.0]
    time = 0.0
    with tqdm(desc="Integration trajectory", total=total_time) as pbar:
        while abs(time) <= (total_time):
            point, point_error_estim = interpolate_rkf(
                Vx, Vy, x, y, point, delta_time=dt
            )
            if (point is None) or (point_error_estim is None):
                break
            pts.append(point)
            pts_error_estim.append(point_error_estim)
            time += dt
            pbar.update(dt)
    pbar.refresh()
    return pts, pts_error_estim


def compute_perturbation(
    data_url: Union[str, Path],
    ogr_url: Union[str, Path],
    perturbation: int = 0,
    sample: Union[list, ndarray] = [0.5, 0.5],
    sigma: float = 1,
    total_time: float = 10_000,
    dt: float = 1,
    reverse: bool = False,
) -> GeoDataFrame:
    """
    Compute a perturbed trajectory.

    It appears OGR objects cannot be pickled by joblib hence we load it here.

    Parameters
    ----------
    url : string or pathlib.Path
        Path to an ogr data set
    VX_min : numpy.ndarray or xarray.DataArray
        Minimum
    VX_min : dict-like, optional
        Another mapping in similar form as the `data_vars` argument,
        except the each item is saved on the dataset as a "coordinate".
        These variables have an associated meaning: they describe
        constant/fixed/independent quantities, unlike the
        varying/measured/dependent quantities that belong in
        `variables`. Coordinates values may be given by 1-dimensional
        arrays or scalars, in which case `dims` do not need to be
        supplied: 1D arrays will be assumed to give index values along
        the dimension with the same name.

        The following notations are accepted:

        - mapping {coord name: DataArray}
        - mapping {coord name: Variable}
        - mapping {coord name: (dimension name, array-like)}
        - mapping {coord name: (tuple of dimension names, array-like)}
        - mapping {dimension name: array-like}
          (the dimension name is implicitly set to be the same as the
          coord name)

        The last notation implies that the coord name is the same as
        the dimension name.

    attrs : dict-like, optional
        Global attributes to save on this dataset.

    Examples
    --------
    Create data:

    >>> np.random.seed(0)
    >>> temperature = 15 + 8 * np.random.randn(2, 2, 3)
    >>> precipitation = 10 * np.random.rand(2, 2, 3)
    >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
    >>> lat = [[42.25, 42.21], [42.63, 42.59]]
    >>> time = pd.date_range("2014-09-06", periods=3)
    >>> reference_time = pd.Timestamp("2014-09-05")

    Initialize a dataset with multiple dimensions:

    >>> ds = xr.Dataset(
    ...     data_vars=dict(
    ...         temperature=(["x", "y", "time"], temperature),
    ...         precipitation=(["x", "y", "time"], precipitation),
    ...     ),
    ...     coords=dict(
    ...         lon=(["x", "y"], lon),
    ...         lat=(["x", "y"], lat),
    ...         time=time,
    ...         reference_time=reference_time,
    ...     ),
    ...     attrs=dict(description="Weather related data."),
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:         (x: 2, y: 2, time: 3)
    Coordinates:
        lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
        lat             (x, y) float64 42.25 42.21 42.63 42.59
      * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Dimensions without coordinates: x, y
    Data variables:
        temperature     (x, y, time) float64 29.11 18.2 22.83 ... 18.28 16.15 26.63
        precipitation   (x, y, time) float64 5.68 9.256 0.7104 ... 7.992 4.615 7.805
    Attributes:
        description:  Weather related data.

    Find out where the coldest temperature was and what values the
    other variables had:

    >>> ds.isel(ds.temperature.argmin(...))
    <xarray.Dataset>
    Dimensions:         ()
    Coordinates:
        lon             float64 -99.32
        lat             float64 42.21
        time            datetime64[ns] 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Data variables:
        temperature     float64 7.182
        precipitation   float64 8.326
    Attributes:
        description:  Weather related data.


    """

    ds = xr.open_dataset(data_url)

    VX = np.squeeze(ds["vx"].to_numpy())
    VY = np.squeeze(ds["vy"].to_numpy())
    VX_e = np.squeeze(ds["vx_err"].to_numpy())
    VY_e = np.squeeze(ds["vy_err"].to_numpy())
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()

    Vx, Vy = get_perturbed_velocities(VX, VY, VX_e, VY_e, sample=sample, sigma=sigma)
    ogr.UseExceptions()
    if isinstance(ogr_url, Path):
        ogr_url = str(ogr_url.absolute())
    in_ds = ogr.Open(ogr_url)

    layer = in_ds.GetLayer(0)
    srs_geo = osr.SpatialReference()
    srs_geo.ImportFromEPSG(3413)

    all_glaciers = []
    with tqdm(enumerate(layer), total=len(layer), leave=False) as pbar:
        for _, feature in pbar:
            geometry = feature.GetGeometryRef()
            geometry.TransformTo(srs_geo)
            points = geometry.GetPoints()
            points = [Point(p) for p in points]
            attrs = feature.items()
            attrs["perturbation"] = perturbation
            glacier_name = attrs["name"]
            pbar.set_description(f"""Processing {glacier_name}""")
            trajs = []
            for p in points:
                traj, _ = compute_trajectory(
                    p, Vx, Vy, x, y, total_time=total_time, dt=dt, reverse=reverse
                )
                trajs.append(traj)
            df = trajectories_to_geopandas(trajs, Vx, Vy, x, y, attrs=attrs)
            all_glaciers.append(df)
            pbar.refresh()
    return pd.concat(all_glaciers)


def get_perturbed_velocities(
    VX, VY, VX_e, VY_e, sample, sigma: float = 1.0
) -> Tuple[Union[ndarray, DataArray], Union[ndarray, DataArray]]:
    """
    Return perturbed velocity field
    """
    VX_min, VX_max = VX - sigma * VX_e, VX + sigma * VX_e
    VY_min, VY_max = VY - sigma * VY_e, VY + sigma * VY_e

    Vx = VX_min + sample[0] * (VX_max - VX_min)
    Vy = VY_min + sample[1] * (VY_max - VY_min)

    return Vx, Vy


def trajectories_to_geopandas(
    trajs: list,
    Vx: np.ndarray,
    Vy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    attrs: dict = {},
) -> gp.GeoDataFrame:
    """Convert trajectory to GeoDataFrame"""
    dfs = []
    for traj_id, traj in enumerate(trajs):
        vx, vy = velocity_at_point(Vx, Vy, x, y, traj)
        v = np.sqrt(vx**2 + vy**2)
        d = [0] + [traj[k].distance(traj[k - 1]) for k in range(1, len(traj))]
        traj_data = {
            "vx": vx,
            "vy": vy,
            "v": v,
            "trai_id": traj_id,
            "traj_pt": range(len(traj)),
            "distance": d,
            "distance_from_origin": np.cumsum(d),
        }
        for k, v in attrs.items():
            traj_data[k] = v
        df = gp.GeoDataFrame.from_dict(traj_data, geometry=traj, crs="EPSG:3413")
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)
