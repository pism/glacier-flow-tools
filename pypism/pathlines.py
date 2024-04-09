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
Module provides functions for calculating pathlines (trajectories)
"""


from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import geopandas as gp
import numpy as np
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame
from joblib import Parallel, delayed
from numpy import ndarray
from shapely import Point
from tqdm import tqdm
from xarray import DataArray

# from pypism.geom import Point
from pypism.interpolation import interpolate_rkf, velocity_at_point
from pypism.utils import tqdm_joblib

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


def compute_pathlines(
    data_url: Union[str, Path],
    ogr_url: Union[str, Path],
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

    ds = xr.open_dataset(data_url, decode_times=False)

    Vx = np.squeeze(ds[x_var].to_numpy())
    Vy = np.squeeze(ds[y_var].to_numpy())
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()

    pts_gp = gp.read_file(ogr_url).to_crs(crs).reset_index(drop=True)
    geom = pts_gp.simplify(tolerance)
    pts_gp = gp.GeoDataFrame(pts_gp, geometry=geom)

    n_pts = len(pts_gp)

    def compute_pathline(index, pts_gp, Vx, Vy, x, y, dt=dt, total_time=total_time, reverse=reverse) -> gp.GeoDataFrame:
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
            delayed(compute_pathline)(
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


def compute_perturbation(
    data_url: Union[str, Path],
    ogr_url: Union[str, Path],
    perturbation: int = 0,
    pl_exp: float = 1.0,
    sigma: float = 1,
    total_time: float = 10_000,
    dt: float = 1,
    reverse: bool = False,
    crs: str = "EPSG:3413",
) -> GeoDataFrame:
    """
    Compute a perturbed pathlines.


    Parameters
    ----------
    url : string or pathlib.Path
        Path to an ogr data set

    Examples
    --------



    """

    ds = xr.open_dataset(data_url, decode_times=False)

    VX = np.squeeze(ds["vx"].to_numpy())
    VY = np.squeeze(ds["vy"].to_numpy())
    VX_e = np.squeeze(ds["vx_err"].fillna(0).to_numpy())
    VY_e = np.squeeze(ds["vy_err"].fillna(0).to_numpy())
    VX_e = np.where(VX_e < 0, 0, VX_e)
    VY_e = np.where(VY_e < 0, 0, VY_e)
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()

    Vx, Vy = get_grf_perturbed_velocities(VX, VY, VX_e, VY_e, pl_exp, perturbation, sigma=sigma)

    pts_gp = gp.read_file(ogr_url).to_crs(crs).reset_index(drop=True)

    all_glaciers = []
    with tqdm(enumerate(pts_gp), total=len(pts_gp), leave=False) as pbar:
        for index, _ in pbar:
            pts = pts_gp[pts_gp.index == index].reset_index(drop=True)
            if len(pts.geometry) > 0:
                points = [Point(p) for p in pts.geometry[0].coords]
                attrs = pts.to_dict()
                attrs = {key: value[0] for key, value in attrs.items() if isinstance(value, dict)}
                attrs["perturbation"] = perturbation
                glacier_name = attrs["name"]
                pbar.set_description(f"""Processing {glacier_name}""")
                pathlines = []
                for p in points:
                    pathline, _ = compute_trajectory(p, Vx, Vy, x, y, total_time=total_time, dt=dt, reverse=reverse)
                    pathlines.append(pathline)
                df = pathlines_to_geopandas(pathlines, Vx, Vy, x, y, attrs=attrs)
                all_glaciers.append(df)
                pbar.refresh()
    return pd.concat(all_glaciers)


def get_perturbed_velocities(
    VX: Union[ndarray, DataArray],
    VY: Union[ndarray, DataArray],
    VX_e: Union[ndarray, DataArray],
    VY_e: Union[ndarray, DataArray],
    sample,
    sigma: float = 1.0,
) -> Tuple[Union[ndarray, DataArray], Union[ndarray, DataArray]]:
    """
    Return perturbed velocity field
    """
    VX_min, VX_max = VX - sigma * VX_e, VX + sigma * VX_e
    VY_min, VY_max = VY - sigma * VY_e, VY + sigma * VY_e

    Vx = VX_min + sample[0] * (VX_max - VX_min)
    Vy = VY_min + sample[1] * (VY_max - VY_min)

    return Vx, Vy


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

    # Generates power-law power spectrum - structures of all sizes and fractal sub-structures
    def plPk(n):
        def Pk(k):
            return np.power(k, -n)

        return Pk

    d_x, d_y = distrib_normal(VX_e, sigma=sigma, seed=perturbation), distrib_normal(
        VY_e, sigma=sigma, seed=perturbation
    )

    Vx_grf = generate_field(d_x, plPk(pl_exp))
    Vy_grf = generate_field(d_y, plPk(pl_exp))

    Vx = VX + Vx_grf
    Vy = VY + Vy_grf

    return Vx, Vy


def distrib_normal(da: Union[xr.DataArray, np.ndarray], sigma: float = 1.0, seed: int = 0, n: float = 1):
    """
    Generates a complex normal distribution
    """
    rng = np.random.default_rng(seed=seed)
    a = rng.normal(
        loc=0,
        scale=(sigma * da) ** n,
    )
    b = rng.normal(
        loc=0,
        scale=(sigma * da) ** n,
    )
    return a + 1j * b


def generate_field(
    fftfield: np.ndarray,
    power_spectrum: Callable[[np.ndarray], np.ndarray],
    unit_length: float = 1,
    fft: Any = np.fft,
    fft_args: Dict[str, Any] = {},
) -> np.ndarray:
    """
    Generates a field given a statistic and a power_spectrum.
    """

    try:
        fftfreq = fft.fftfreq
    except AttributeError:
        # Fallback on numpy for the frequencies
        fftfreq = np.fft.fftfreq
    else:
        fftfreq = fft.fftfreq

    # Compute the k grid
    shape = fftfield.shape
    all_k = [fftfreq(s, d=unit_length) for s in shape]

    kgrid = np.meshgrid(*all_k, indexing="ij")
    knorm = np.hypot(*kgrid)

    power_k = np.zeros_like(knorm)
    mask = knorm > 0
    power_k[mask] = np.sqrt(power_spectrum(knorm[mask]))
    fftfield *= power_k

    return np.real(fft.ifftn(fftfield, **fft_args))


def pathlines_to_geopandas(
    pathlines: list,
    Vx: Union[ndarray, DataArray],
    Vy: Union[ndarray, DataArray],
    x: Union[ndarray, DataArray],
    y: Union[ndarray, DataArray],
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
