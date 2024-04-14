# Copyright (C) 2023-24 Andy Aschwanden, Constantine Khroulev
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PYPISM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Module provides functions for calculating pathlines (trajectories)
"""


from pathlib import Path
from typing import Tuple, Union

import geopandas as gp
import numpy as np
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from numpy import ndarray
from shapely.geometry import Point
from tqdm import tqdm
from xarray import DataArray

from pypism.gaussian_random_fields import distrib_normal, generate_field, power_spectrum
from pypism.interpolation import (
    interpolate_at_point,
    interpolate_rkf,
    interpolate_rkf_np,
    velocity_at_point,
)
from pypism.utils import tqdm_joblib


def compute_trajectory(
    point: Point,
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
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
    >>>    from pypism.geom import Point

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
    while abs(time) <= (total_time):
        point, point_error_estim = interpolate_rkf(Vx, Vy, x, y, point, delta_time=dt)
        if (point is None) or (point_error_estim is None) or (point.is_empty):
            break
        pts.append(point)
        pts_error_estim.append(point_error_estim)
        time += dt
    return pts, pts_error_estim


def compute_pathline(
    point: Union[list, ndarray, Point],
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    dt: float = 0.1,
    total_time: float = 1000,
    reverse: bool = False,
) -> Tuple[ndarray, ndarray, ndarray]:
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
    total_time : float
        total integration time

    Returns
    ----------
    pts: np.ndarray
        `dt`-spaced points along trajectory from 0 to `total_time`.
    pts_error_estim: np.ndarray
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

    if isinstance(point, Point):
        point = np.squeeze(np.array(point.coords.xy).reshape(1, -1))

    nt = int(total_time / dt) + 2

    pts = np.zeros((nt, 2))
    pts[0, :] = point

    pts_error_estimate = np.zeros((nt, 1))
    pts_error_estimate[0] = 0.0

    velocities = np.zeros((nt, 2))
    velocities[0, :] = interpolate_at_point(Vx, Vy, x, y, *point)

    time = 0.0
    k = 0
    while abs(time) <= (total_time):
        k += 1
        point, v, error_estimate = interpolate_rkf_np(Vx, Vy, x, y, point, delta_time=dt)

        if np.any(np.isnan(point)) or np.any(np.isnan(error_estimate)):
            break
        pts[k, :] = point
        velocities[k, :] = v
        pts_error_estimate[k] = error_estimate
        time += dt
    return pts, velocities, pts_error_estimate


def compute_pathlines(
    raster_url: Union[str, Path],
    vector_url: Union[str, Path],
    perturbation: int = 0,
    dt: float = 1,
    total_time: float = 10_000,
    x_var: str = "vx",
    y_var: str = "vy",
    reverse: bool = False,
    n_jobs: int = 4,
    tolerance: float = 0.1,
    crs: str = "EPSG:3413",
) -> GeoDataFrame:
    """
    Compute a pathline (pathlines).

    """

    ds = xr.open_dataset(raster_url, decode_times=False)

    Vx = np.squeeze(ds[x_var].to_numpy())
    Vy = np.squeeze(ds[y_var].to_numpy())
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()

    pts_gp = gp.read_file(vector_url).to_crs(crs).reset_index(drop=True)
    geom = pts_gp.simplify(tolerance)
    pts_gp = gp.GeoDataFrame(pts_gp, geometry=geom)

    n_pts = len(pts_gp)

    def compute_pathline_gp(
        index, pts_gp, Vx, Vy, x, y, dt=dt, total_time=total_time, reverse=reverse
    ) -> gp.GeoDataFrame:
        pts = pts_gp[pts_gp.index == index].reset_index(drop=True)
        if len(pts.geometry) > 0:
            points = [Point(p) for p in pts.geometry[0].coords]
            attrs = pts.to_dict()
            attrs = {key: value[0] for key, value in attrs.items() if isinstance(value, dict)}
            attrs["perturbation"] = perturbation
            pathlines = []
            for p in points:
                pathline, _ = compute_trajectory(p, Vx, Vy, x, y, total_time=total_time, dt=dt, reverse=reverse)
                pathlines.append(pathline)
            df = pathlines_to_geopandas(pathlines, Vx, Vy, x, y, attrs=attrs)
        else:
            df = gp.GeoDataFrame()
        return df

    with tqdm_joblib(
        tqdm(desc="Processing Pathlines", total=n_pts, leave=True, position=0)
    ) as progress_bar:  # pylint: disable=unused-variable
        result = Parallel(n_jobs=n_jobs)(
            delayed(compute_pathline_gp)(
                index,
                pts_gp,
                Vx,
                Vy,
                x,
                y,
                dt=dt,
                total_time=total_time,
                reverse=reverse,
            )
            for index in range(n_pts)
        )
        results = pd.concat(result).reset_index(drop=True)
        return results


def get_grf_perturbed_velocities(
    VX: Union[ndarray, DataArray],
    VY: Union[ndarray, DataArray],
    VX_e: Union[ndarray, DataArray],
    VY_e: Union[ndarray, DataArray],
    pl_exp: float,
    perturbation: int,
    sigma: float = 1.0,
) -> Tuple[Union[ndarray, DataArray], Union[ndarray, DataArray]]:
    """
    Return perturbed velocity field
    """

    d_x, d_y = distrib_normal(VX_e, sigma=sigma, seed=perturbation), distrib_normal(
        VY_e, sigma=sigma, seed=perturbation
    )

    Vx_grf = generate_field(d_x, power_spectrum, pl_exp)
    Vy_grf = generate_field(d_y, power_spectrum, pl_exp)

    Vx = VX + Vx_grf
    Vy = VY + Vy_grf

    Vx = VX
    Vy = VY

    return Vx, Vy


def pathlines_to_geopandas(
    pathlines: list,
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    attrs: dict = {},
) -> gp.GeoDataFrame:
    """Convert pathlines to GeoDataFrame"""

    dfs = []
    for pathline_id, pathline in enumerate(pathlines):
        vx, vy = velocity_at_point(Vx, Vy, x, y, pathline)
        v = np.sqrt(vx**2 + vy**2)
        d = [0] + [pathline[k].distance(pathline[k - 1]) for k in range(1, len(pathline))]
        pathline_data = {
            "vx": vx,
            "vy": vy,
            "v": v,
            "pathline_id": pathline_id,
            "pathline_pt": range(len(pathline)),
            "distance": d,
            "distance_from_origin": np.cumsum(d),
        }
        for k, v in attrs.items():
            pathline_data[k] = v
        df = gp.GeoDataFrame.from_dict(pathline_data, geometry=pathline, crs="EPSG:3413")
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def pathline_to_geopandas_dataframe(
    points: Union[list[Point], np.ndarray], attrs: Union[dict, None] = None, crs="EPSG:3413"
) -> gp.GeoDataFrame:
    """
    Convert a list of shapely.Point or np.ndarray to a geopandas.GeoDataFrame
    Compute trajectory

    Computes a trajectory using Runge-Kutta-Fehlberg integration. Routine is
    unit-agnostic, requiring the user to ensure consistency of units. For example
    if the velocity field is given in m/yr, the `dt` and `total_time` are assumed
    to be in years.

    Parameters
    ----------
    points : list[shapely.Point] or np.ndarray
        Points along pathlines
    attributes : dict
        Dictionary of attributes to be added. E.g. {"pathline_id": 0}
    crs : str
        Coordinate reference system, e.g. "EPSG:3413:

    Returns
    ----------
    df: gp.GeoDataFrame
        Geopandas dataframe of pathline

    Examples
    ----------

    Create data:

    >>>    import numpy as np

    >>>    nx = 201
    >>>    ny = 401
    >>>    x = np.linspace(-100e3, 100e3, nx)
    >>>    y = np.linspace(-100e3, 100e3, ny)
    >>>    X, Y = np.meshgrid(x, y)

    >>>    vx = -Y / np.sqrt(X**2 + Y**2) * 250
    >>>    vy = X / np.sqrt(X**2 + Y**2) * 250
    >>>    p = [0, -50000

    Compute pathline:

    >>>    pts, v, _ = compute_pathline(p, vx, vx, x, y, dt=1, total_time=10)

    Convert to geopandas.GeoDataFrame:

    >>>    pathline_to_geopandas_dataframe(pts)

    Convert to geopandas.GeoDataFrame, add attributes:

    >>>    attributes = {"pathline_id": 0}
    >>>    pl = pathline_to_geopandas_dataframe(pts, attributes)
    >>>    pl.to_dict()
    >>>        {'geometry': {0: <POINT (0 -50000)>,
    >>>      1: <POINT (249.994 -49750.006)>,
    >>>      2: <POINT (499.975 -49500.025)>,
    >>>      3: <POINT (749.943 -49250.057)>,
    >>>      4: <POINT (999.897 -49000.103)>,
    >>>      5: <POINT (1249.825 -48750.175)>,
    >>>      6: <POINT (1499.713 -48500.287)>,
    >>>      7: <POINT (1749.56 -48250.44)>,
    >>>      8: <POINT (1999.364 -48000.636)>,
    >>>      9: <POINT (2249.113 -47750.887)>,
    >>>      10: <POINT (2498.79 -47501.21)>,
    >>>      11: <POINT (2748.394 -47251.606)>},
    >>>     'pathline_id': {0: 0,
    >>>      1: 0,
    >>>      2: 0,
    >>>      3: 0,
    >>>      4: 0,
    >>>      5: 0,
    >>>      6: 0,
    >>>      7: 0,
    >>>      8: 0,
    >>>      9: 0,
    >>>      10: 0,
    >>>      11: 0}}
    """
    if isinstance(points, np.ndarray):
        geom = [Point(pt) for pt in points]
    else:
        geom = points
    pathline_dict = {"geometry": geom}

    if attrs is not None:
        pathline_dict.update(attrs)
    return gp.GeoDataFrame.from_dict(pathline_dict, crs=crs)
