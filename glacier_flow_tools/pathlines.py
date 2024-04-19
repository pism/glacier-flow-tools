# Copyright (C) 2023-24 Andy Aschwanden, Constantine Khroulev
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
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Module provides functions for calculating pathlines (trajectories).
"""

from typing import Dict, Tuple, Union

import geopandas as gp
import numpy as np
from numpy import ndarray
from shapely.geometry import Point
from xarray import DataArray

from glacier_flow_tools.gaussian_random_fields import (
    distrib_normal,
    generate_field,
    power_spectrum,
)
from glacier_flow_tools.geom import distances
from glacier_flow_tools.interpolation import interpolate_at_point, interpolate_rkf_np


def compute_pathline(
    point: Union[list, ndarray, Point],
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    dt: float = 0.1,
    total_time: float = 1000,
    reverse: bool = False,
    notebook: bool = False,
    progress: bool = False,
    progress_kwargs: Dict = {"leave": False, "position": 0},
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Compute a pathline using Runge-Kutta-Fehlberg integration.

    This function computes a pathline, which is a trajectory traced by a particle in a fluid flow. The pathline is computed by integrating the velocity field using the Runge-Kutta-Fehlberg method. The function is unit-agnostic, requiring the user to ensure consistency of units. For example, if the velocity field is given in m/yr, the `dt` and `total_time` are assumed to be in years.

    Parameters
    ----------
    point : list, ndarray, or shapely.Point
        Starting point of the pathline.
    Vx : ndarray
        The x-component of the velocity field.
    Vy : ndarray
        The y-component of the velocity field.
    x : ndarray
        The coordinates in the x direction.
    y : ndarray
        The coordinates in the y direction.
    dt : float, optional
        The integration time step. Default is 0.1.
    total_time : float, optional
        The total integration time. Default is 1000.
    reverse : bool, optional
        If True, the velocity field is reversed before computing the pathline. Default is False.
    notebook : bool, optional
        If True, a progress bar is displayed in a Jupyter notebook. Default is False.
    progress : bool, optional
        If True, a progress bar is displayed. Default is False.
    progress_kwargs : dict, optional
        Additional keyword arguments for the progress bar. Default is {"leave": False, "position": 0}.

    Returns
    -------
    pts : ndarray
        The `dt`-spaced points along the pathline from 0 to `total_time`.
    velocities : ndarray
        The velocity at each point along the pathline.
    pts_error_estimate : ndarray
        Error estimate at each point along the pathline.

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

    >>>    pts, pts_error_estim = compute_pathline(p, vx, vx, x, y, dt=1, total_time=10)
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

    if progress:
        if notebook:
            from tqdm.notebook import tqdm  # pylint: disable=reimported

        else:
            from tqdm import tqdm  # pylint: disable=reimported

    if reverse:
        Vx = -Vx
        Vy = -Vy

    if isinstance(point, Point):
        point = np.squeeze(np.array(point.coords.xy).reshape(1, -1))

    nt = int(total_time / dt) + 1

    pts = np.zeros((nt, 2))
    pts[0, :] = point

    pts_error_estimate = np.zeros((nt, 1))
    pts_error_estimate[0] = 0.0

    velocities = np.zeros((nt, 2))
    velocities[0, :] = interpolate_at_point(Vx, Vy, x, y, *point)

    time = 0.0
    k = 0
    if progress:
        p_bar = tqdm(desc="Integrating pathline", total=total_time, **progress_kwargs)
    while abs(time) <= (total_time):
        point, v, error_estimate = interpolate_rkf_np(Vx, Vy, x, y, point, delta_time=dt)
        if np.any(np.isnan(point)) or np.any(np.isnan(error_estimate)):
            break
        pts[k, :] = point
        velocities[k, :] = v
        pts_error_estimate[k] = error_estimate
        time += dt
        k += 1
        if progress:
            p_bar.update(dt)
    return pts, velocities, pts_error_estimate


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
    Return perturbed velocity field using Gaussian Random Fields (GRF).

    This function generates a perturbed velocity field by adding a GRF to the original velocity field. The GRF is generated using a power-law spectrum.

    Parameters
    ----------
    VX : ndarray or DataArray
        Original velocity field in the x-direction.
    VY : ndarray or DataArray
        Original velocity field in the y-direction.
    VX_e : ndarray or DataArray
        Error estimates for the velocity field in the x-direction.
    VY_e : ndarray or DataArray
        Error estimates for the velocity field in the y-direction.
    pl_exp : float
        Exponent for the power-law spectrum used to generate the GRF.
    perturbation : int
        Seed for the random number generator used to generate the GRF.
    sigma : float, optional
        Standard deviation for the normal distribution used to generate the GRF. Default is 1.0.

    Returns
    -------
    Vx : ndarray or DataArray
        Perturbed velocity field in the x-direction.
    Vy : ndarray or DataArray
        Perturbed velocity field in the y-direction.

    Notes
    -----
    The function `distrib_normal` is used to generate a normal distribution with mean 0 and standard deviation `sigma`. This distribution is used as input to the function `generate_field`, which generates the GRF using a power-law spectrum.
    """
    d_x, d_y = distrib_normal(VX_e, sigma=sigma, seed=perturbation), distrib_normal(
        VY_e, sigma=sigma, seed=perturbation
    )

    Vx_grf = generate_field(d_x, power_spectrum, pl_exp)
    Vy_grf = generate_field(d_y, power_spectrum, pl_exp)

    Vx = VX + Vx_grf
    Vy = VY + Vy_grf

    return Vx, Vy


def pathline_to_geopandas_dataframe(
    points: Union[list[Point], np.ndarray], attrs: Union[dict, None] = None, crs="EPSG:3413"
) -> gp.GeoDataFrame:
    """
    Convert a list of shapely.Point or np.ndarray to a geopandas.GeoDataFrame.

    This function takes a list of points, optionally with associated attributes, and converts it into a GeoDataFrame.
    The points represent a pathline, which is a trajectory traced by a particle in a fluid flow.

    Parameters
    ----------
    points : list[shapely.Point] or np.ndarray
        Points along pathlines.
    attrs : dict, optional
        Dictionary of attributes to be added to the GeoDataFrame. E.g. {"pathline_id": 0}.
    crs : str, optional
        Coordinate reference system to be used for the GeoDataFrame. The default is "EPSG:3413", which represents the WGS 84 / NSIDC Sea Ice Polar Stereographic North coordinate system.

    Returns
    ----------
    df: gp.GeoDataFrame
        A GeoDataFrame where each row represents a point in the pathline. If attributes are provided, they are added as columns in the GeoDataFrame.

    Examples
    ----------

    Create data:

    >>>    import numpy as np
    >>>    from glacier_flow_tools.pathlines import compute_pathline, pathline_to_geopandas_dataframe

    >>>    np.seterr(all="ignore")

    >>>    nx = 201
    >>>    ny = 401
    >>>    x = np.linspace(-100e3, 100e3, nx)
    >>>    y = np.linspace(-100e3, 100e3, ny)
    >>>    X, Y = np.meshgrid(x, y)

    >>>    vx = -Y / np.sqrt(X**2 + Y**2) * 250
    >>>    vy = X / np.sqrt(X**2 + Y**2) * 250
    >>>    p = [0, -50000]

    Compute pathline:

    >>>    pts, v, _ = compute_pathline(p, vx, vx, x, y, dt=1, total_time=10)

    Convert to geopandas.GeoDataFrame:

    >>>    pl = pathline_to_geopandas_dataframe(pts)
    >>>    plt.to_dict()
    >>>    {'geometry': {0: <POINT (249.994 -49750.006)>,
    >>>      1: <POINT (499.975 -49500.025)>,
    >>>      2: <POINT (749.943 -49250.057)>,
    >>>      3: <POINT (999.897 -49000.103)>,
    >>>      4: <POINT (1249.825 -48750.175)>,
    >>>      5: <POINT (1499.713 -48500.287)>,
    >>>      6: <POINT (1749.56 -48250.44)>,
    >>>      7: <POINT (1999.364 -48000.636)>,
    >>>      8: <POINT (2249.113 -47750.887)>,
    >>>      9: <POINT (2498.79 -47501.21)>,
    >>>      10: <POINT (2748.394 -47251.606)>}}

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

    Convert to geopandas.GeoDataFrame, add more attributes:

    >>>    from glacier_flow_tools.geom import distances

    >>>    vx = v[:, 0]
    >>>    vy = v[:, 1]
    >>>    speed = np.sqrt(vx**2 + vy**2)

    >>>    d = distances(pts)
    >>>    pathline_data = {
    >>>        "vx": vx,
    >>>        "vy": vy,
    >>>        "v": speed,
    >>>        "distance": d,
    >>>        "distance_from_origin": np.cumsum(d),
    >>>    }
    >>>    attributes.update(pathline_data)

    >>>      {'geometry': {0: <POINT (249.994 -49750.006)>,
    >>>    1: <POINT (499.975 -49500.025)>,
    >>>    2: <POINT (749.943 -49250.057)>,
    >>>    3: <POINT (999.897 -49000.103)>,
    >>>    4: <POINT (1249.825 -48750.175)>,
    >>>    5: <POINT (1499.713 -48500.287)>,
    >>>    6: <POINT (1749.56 -48250.44)>,
    >>>    7: <POINT (1999.364 -48000.636)>,
    >>>    8: <POINT (2249.113 -47750.887)>,
    >>>    9: <POINT (2498.79 -47501.21)>,
    >>>    10: <POINT (2748.394 -47251.606)>},
    >>>      'pathline_id': {0: 0,
    >>>    1: 0,
    >>>    2: 0,
    >>>    3: 0,
    >>>    4: 0,
    >>>    5: 0,
    >>>    6: 0,
    >>>    7: 0,
    >>>    8: 0,
    >>>    9: 0,
    >>>    10: 0},
    >>>    'vx': {0: 249.98737724611084,
    >>>    1: 249.97450152124216,
    >>>    2: 249.9613611799404,
    >>>    3: 249.94796017540068,
    >>>    4: 249.90805496694236,
    >>>    5: 249.86733950159993,
    >>>    6: 249.82577024672722,
    >>>    7: 249.78337719458116,
    >>>    8: 249.71298833558737,
    >>>    9: 249.64113140550904,
    >>>    10: 249.56773625309486},
    >>>      'vy': {0: 249.98737724611084,
    >>>    1: 249.97450152124216,
    >>>    2: 249.9613611799404,
    >>>    3: 249.94796017540068,
    >>>    4: 249.90805496694236,
    >>>    5: 249.86733950159993,
    >>>    6: 249.82577024672722,
    >>>    7: 249.78337719458116,
    >>>    8: 249.71298833558737,
    >>>    9: 249.64113140550904,
    >>>    10: 249.56773625309486},
    >>>      'v': {0: 353.53553932352924,
    >>>    1: 353.51733029879455,
    >>>    2: 353.4987470499114,
    >>>    3: 353.4797951675419,
    >>>    4: 353.4233606805308,
    >>>    5: 353.3657803172452,
    >>>    6: 353.3069925132265,
    >>>    7: 353.2470396839311,
    >>>    8: 353.1474948049021,
    >>>    9: 353.0458737598349,
    >>>    10: 352.9420773398783},
    >>>      'distance': {0: 0.0,
    >>>    1: 353.526464621023,
    >>>    2: 353.50806939740113,
    >>>    3: 353.48930182481575,
    >>>    4: 353.4516779458979,
    >>>    5: 353.3946646996867,
    >>>    6: 353.33648355384537,
    >>>    7: 353.2771131453923,
    >>>    8: 353.1974692722979,
    >>>    9: 353.09684911801355,
    >>>    10: 352.99414582417967},
    >>>      'distance_from_origin': {0: 0.0,
    >>>    1: 353.526464621023,
    >>>    2: 707.0345340184242,
    >>>    3: 1060.52383584324,
    >>>    4: 1413.975513789138,
    >>>    5: 1767.3701784888246,
    >>>    6: 2120.70666204267,
    >>>    7: 2473.9837751880623,
    >>>    8: 2827.18124446036,
    >>>    9: 3180.278093578374,
    >>>    10: 3533.2722394025536}}
    """
    if isinstance(points, np.ndarray):
        geom = [Point(pt) for pt in points]
    else:
        geom = points
    pathline_dict = {"geometry": geom}

    if attrs is not None:
        pathline_dict.update(attrs)
    return gp.GeoDataFrame.from_dict(pathline_dict, crs=crs)


def series_to_pathline_geopandas_dataframe(series: gp.GeoSeries, pathline: Tuple) -> gp.GeoDataFrame:
    """
    Convert a  pathline tuple and a geopandas.GeoSeries to a geopandas.GeoDataFrame.

    Convert a geopandas.GeoSeries single point and a pathline to a geopandas.GeoDataFrame.

    Parameters
    ----------
    series : geopandas.GeoSeries
        A series with attributes.
    pathline : Tuple
        Tuple of (List[Points], List[vx, vy], List[error]).

    Returns
    ----------
    df: gp.GeoDataFrame
        Geopandas dataframe of pathline.

    Examples
    ----------

    Create data:

    >>>    import numpy as np
    >>>    import geopandas as gp
    >>>    from glacier_flow_tools.pathlines import series_to_pathline_geopandas_dataframe

    >>>    series = gp.GeoSeries([Point([0, -5000])])
    >>>    pathline = (np.array([[   249.99370971, -49750.00629029],
    >>>            [   499.97467017, -49500.02532983],
    >>>            [   749.94262324, -49250.05737676],
    >>>            [   999.89730564, -49000.10269436],
    >>>            [  1249.82538394, -48750.17461606],
    >>>            [  1499.71314778, -48500.28685222],
    >>>            [  1749.55977134, -48250.44022866],
    >>>            [  1999.36441369, -48000.63558631],
    >>>            [  2249.11273931, -47750.88726069],
    >>>            [  2498.78991573, -47501.21008427],
    >>>            [  2748.39446997, -47251.60553003]]),
    >>>     np.array([[249.98737725, 249.98737725],
    >>>            [249.97450152, 249.97450152],
    >>>            [249.96136118, 249.96136118],
    >>>            [249.94796018, 249.94796018],
    >>>            [249.90805497, 249.90805497],
    >>>            [249.8673395 , 249.8673395 ],
    >>>            [249.82577025, 249.82577025],
    >>>            [249.78337719, 249.78337719],
    >>>            [249.71298834, 249.71298834],
    >>>            [249.64113141, 249.64113141],
    >>>            [249.56773625, 249.56773625]]),
    >>>     np.array([[2.84217094e-14],
    >>>            [0.00000000e+00],
    >>>            [7.58982048e-12],
    >>>            [0.00000000e+00],
    >>>            [4.19384810e-08],
    >>>            [2.27373675e-13],
    >>>            [1.87642803e-10],
    >>>            [2.27373675e-13],
    >>>            [2.69376538e-07],
    >>>            [4.54747351e-13],
    >>>            [2.25377857e-09]]))

    >>>    pathline_df = series_to_pathline_geopandas_dataframe(series, pathline).to_dict()
    >>>    pathline_df.to_dict()
    >>>   {'geometry': {0: <POINT (249.994 -49750.006)>,
    >>>     1: <POINT (499.975 -49500.025)>,
    >>>     2: <POINT (749.943 -49250.057)>,
    >>>     3: <POINT (999.897 -49000.103)>,
    >>>     4: <POINT (1249.825 -48750.175)>,
    >>>     5: <POINT (1499.713 -48500.287)>,
    >>>     6: <POINT (1749.56 -48250.44)>,
    >>>     7: <POINT (1999.364 -48000.636)>,
    >>>     8: <POINT (2249.113 -47750.887)>,
    >>>     9: <POINT (2498.79 -47501.21)>,
    >>>     10: <POINT (2748.394 -47251.606)>},
    >>>    0: {0: <POINT (0 -5000)>,
    >>>     1: <POINT (0 -5000)>,
    >>>     2: <POINT (0 -5000)>,
    >>>     3: <POINT (0 -5000)>,
    >>>     4: <POINT (0 -5000)>,
    >>>     5: <POINT (0 -5000)>,
    >>>     6: <POINT (0 -5000)>,
    >>>     7: <POINT (0 -5000)>,
    >>>     8: <POINT (0 -5000)>,
    >>>     9: <POINT (0 -5000)>,
    >>>     10: <POINT (0 -5000)>},
    >>>    'vx': {0: 249.98737725,
    >>>     1: 249.97450152,
    >>>     2: 249.96136118,
    >>>     3: 249.94796018,
    >>>     4: 249.90805497,
    >>>     5: 249.8673395,
    >>>     6: 249.82577025,
    >>>     7: 249.78337719,
    >>>     8: 249.71298834,
    >>>     9: 249.64113141,
    >>>     10: 249.56773625},
    >>>    'vy': {0: 249.98737725,
    >>>     1: 249.97450152,
    >>>     2: 249.96136118,
    >>>     3: 249.94796018,
    >>>     4: 249.90805497,
    >>>     5: 249.8673395,
    >>>     6: 249.82577025,
    >>>     7: 249.78337719,
    >>>     8: 249.71298834,
    >>>     9: 249.64113141,
    >>>     10: 249.56773625},
    >>>    'v': {0: 353.53553932902935,
    >>>     1: 353.51733029703786,
    >>>     2: 353.4987470499957,
    >>>     3: 353.4797951740463,
    >>>     4: 353.42336068485497,
    >>>     5: 353.3657803149826,
    >>>     6: 353.3069925178549,
    >>>     7: 353.2470396774524,
    >>>     8: 353.14749481114256,
    >>>     9: 353.04587376618605,
    >>>     10: 352.9420773355015},
    >>>    'pathline_id': {0: None,
    >>>     1: None,
    >>>     2: None,
    >>>     3: None,
    >>>     4: None,
    >>>     5: None,
    >>>     6: None,
    >>>     7: None,
    >>>     8: None,
    >>>     9: None,
    >>>     10: None},
    >>>    'distance': {0: 0.0,
    >>>     1: 353.5264646175854,
    >>>     2: 353.5080693902363,
    >>>     3: 353.48930182873886,
    >>>     4: 353.4516779497044,
    >>>     5: 353.3946646936138,
    >>>     6: 353.33648355167645,
    >>>     7: 353.2771131551291,
    >>>     8: 353.1974692717807,
    >>>     9: 353.0968491081801,
    >>>     10: 352.99414583630073},
    >>>    'distance_from_origin': {0: 0.0,
    >>>     1: 353.5264646175854,
    >>>     2: 707.0345340078218,
    >>>     3: 1060.5238358365607,
    >>>     4: 1413.975513786265,
    >>>     5: 1767.3701784798789,
    >>>     6: 2120.7066620315554,
    >>>     7: 2473.9837751866844,
    >>>     8: 2827.1812444584652,
    >>>     9: 3180.278093566645,
    >>>     10: 3533.272239402946}}
    """

    k = series.name
    attributes = series.drop(columns="geometry").to_dict()
    if hasattr(attributes, "geometry"):
        del attributes["geometry"]

    points = pathline[0]
    v = pathline[1]
    vx = v[:, 0]
    vy = v[:, 1]
    speed = np.sqrt(vx**2 + vy**2)

    d = distances(points)
    pathline_data = {
        "vx": vx,
        "vy": vy,
        "v": speed,
        "pathline_id": k,
        "distance": d,
        "distance_from_origin": np.cumsum(d),
    }
    attributes.update(pathline_data)
    return pathline_to_geopandas_dataframe(points, attributes).reset_index(drop=True)
