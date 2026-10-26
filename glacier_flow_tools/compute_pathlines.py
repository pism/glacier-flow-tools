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
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Calculate pathlines (trajectories).
"""

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from pathlib import Path

import geopandas as gp
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from glacier_flow_tools.geom import (
    geopandas_dataframe_densify_lines,
    geopandas_dataframe_shorten_lines,
    parse_distance,
)
from glacier_flow_tools.interpolation import velocity
from glacier_flow_tools.pathlines import (
    StepSizeWarning,
    compute_pathline,
    pathline_to_line_geopandas_dataframe,
    series_to_pathline_geopandas_dataframe,
)
from glacier_flow_tools.utils import to_numpy_in_units, tqdm_joblib


def compute_pathline_and_collect_warnings(*args, **kwargs):
    """
    Compute a pathline and return its step size warning instead of printing it.

    A warning printed by a worker would break up the progress bar. This returns the warning to the caller,
    which can report all of them together.

    Parameters
    ----------
    *args
        Positional arguments for `compute_pathline`.
    **kwargs
        Keyword arguments for `compute_pathline`.

    Returns
    -------
    pathline : tuple
        The result of `compute_pathline`.
    step_size_warning : dict or None
        The number of steps taken at the minimum step size with an error above the tolerance (``n_steps``),
        the time of the first one (``first_time``) and the largest error estimate (``max_error``).
        None if the pathline met the tolerance everywhere.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pathline = compute_pathline(*args, **kwargs)

    step_size_warning = None
    for w in caught:
        if isinstance(w.message, StepSizeWarning):
            m = w.message
            step_size_warning = {"n_steps": m.n_steps, "first_time": m.first_time, "max_error": m.max_error}
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return pathline, step_size_warning


def step_size_summary(step_size_warnings: list, hmin: float, tol: float, max_listed: int = 5) -> str:
    """
    Summarize the step size warnings of all pathlines in a few lines.

    Parameters
    ----------
    step_size_warnings : list
        One entry per pathline, as returned by `compute_pathline_and_collect_warnings`.
    hmin : float
        The minimum step size, in years.
    tol : float
        The error tolerance.
    max_listed : int, optional
        The largest number of pathlines listed individually. Default is 5.

    Returns
    -------
    str
        The summary, or an empty string if no pathline has a warning.
    """
    affected = [(k, w) for k, w in enumerate(step_size_warnings) if w is not None]
    if not affected:
        return ""
    lines = [
        f"Warning: {len(affected)} of {len(step_size_warnings)} pathlines did not meet the tolerance tol={tol:g} "
        f"everywhere, even at the minimum time step hmin={hmin:g} yr.",
        "The affected steps were taken at the minimum time step.",
    ]
    for k, w in affected[:max_listed]:
        steps = "step" if w["n_steps"] == 1 else "steps"
        lines.append(
            f"  pathline {k}: {w['n_steps']} {steps}, first at t={w['first_time']:.4g} yr, "
            f"largest error estimate {w['max_error']:.2g}"
        )
    if len(affected) > max_listed:
        lines.append(f"  ... and {len(affected) - max_listed} more")
    lines.append("This is common where the velocity jumps, such as at an ice margin or a data gap.")
    lines.append("Use a smaller --hmin or a larger --tol to change this.")
    return "\n".join(lines)


def distance_argument(value: str) -> float:
    """
    Convert the value of a command line option to a distance in meters.

    Parameters
    ----------
    value : str
        The text given on the command line, such as ``500m``.

    Returns
    -------
    float
        The distance in meters.

    Raises
    ------
    ArgumentTypeError
        If the text is not a positive distance.
    """
    try:
        return parse_distance(value)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


def main() -> None:
    """
    Command line interface: compute pathlines (forward/backward) from a velocity field and starting points.
    """
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute pathlines (forward/backward) given a velocity field (xr.Dataset) and starting points (geopandas.GeoDataFrame)."
    parser.add_argument("--raster_url", help="""Path to raster dataset.""", default=None)
    parser.add_argument("--vector_url", help="""Path to vector dataset.""", default=None)
    parser.add_argument(
        "--densify",
        help="""Start a pathline every DENSIFY along each line of the vector dataset, for example 500m or 0.5km.
        The distance is measured along the line, in the units of the dataset's coordinate reference system,
        which must be meters. Points are kept as they are. Without this option, a pathline starts near each
        end of a line.""",
        type=distance_argument,
        default=None,
        metavar="DENSIFY",
    )
    parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
    parser.add_argument(
        "--velocity_units",
        help="""Units of the velocities vx and vy in the raster dataset, for example 'm/s'. By default the units
        are read from the dataset; without units there, m/yr is assumed. The velocities are converted to m/yr.""",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hmin",
        help="""Minimum time step in years for adaptive time stepping. Default=0.01. If hmin=hmax then a fixed time step is used""",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--hmax",
        help="""Maximum time step in years for adaptive time stepping. Default=1.0. If hmin=hmax then a fixed time step is used""",
        type=float,
        default=1.0,
    )
    parser.add_argument("--tol", help="""Adaptive time stepping tolerance. Default=1e-3""", type=float, default=1e-3)
    parser.add_argument(
        "--start_time",
        help="""Start time in years. Default=0.0""",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--end_time",
        help="""End time in years. Default=1000.0""",
        type=float,
        default=1_000.0,
    )
    parser.add_argument(
        "--reverse",
        help="""Reverse velocity field to calculate backward pathlines.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output_type",
        help="""Save result as Points or LineStrings.""",
        choices=["point", "line"],
        default="point",
    )

    parser.add_argument(
        "--v_threshold",
        help="""Threshold velocity in m/yr below which solver stops Default is 0.0.""",
        default=0.0,
        type=float,
    )
    parser.add_argument("outfile", nargs=1, help="Geopandas output file", default="pathlines.gpkg")

    options = parser.parse_args()

    p = Path(options.outfile[-1])
    p.parent.mkdir(parents=True, exist_ok=True)

    df = gp.read_file(options.vector_url)
    if options.densify is not None:
        starting_points_df = geopandas_dataframe_densify_lines(df, options.densify)
    else:
        starting_points_df = geopandas_dataframe_shorten_lines(df).convert.to_points()

    # The solver does not track units: velocities are converted to m/yr and coordinates to m,
    # so that times are in years and distances in meters.
    ds = xr.open_dataset(options.raster_url)
    try:
        Vx = np.squeeze(to_numpy_in_units(ds["vx"], "m/yr", assume="m/yr", override=options.velocity_units))
        Vy = np.squeeze(to_numpy_in_units(ds["vy"], "m/yr", assume="m/yr", override=options.velocity_units))
        x = to_numpy_in_units(ds["x"], "m", assume="m")
        y = to_numpy_in_units(ds["y"], "m", assume="m")
    except ValueError as exc:
        parser.error(f"{options.raster_url}: {exc}")

    if options.reverse:
        Vx = -Vx
        Vy = -Vy

    n_pts = len(starting_points_df)

    start = time.time()

    with tqdm_joblib(
        tqdm(desc="Processing Pathlines", total=n_pts, leave=True, position=0)
    ) as progress_bar:  # pylint: disable=unused-variable
        results = Parallel(n_jobs=options.n_jobs)(
            delayed(compute_pathline_and_collect_warnings)(
                [*df.geometry.coords[0]],
                velocity,
                f_args=(Vx, Vy, x, y),
                hmin=options.hmin,
                hmax=options.hmax,
                tol=options.tol,
                start_time=options.start_time,
                end_time=options.end_time,
                v_threshold=options.v_threshold,
                progress=False,
            )
            for index, df in starting_points_df.iterrows()
        )
    pathlines = [pathline for pathline, _ in results]
    time_elapsed = time.time() - start
    print(f"Time elapsed {time_elapsed:.0f}s\n")

    summary = step_size_summary([w for _, w in results], hmin=options.hmin, tol=options.tol)
    if summary:
        print(summary + "\n")

    print(f"Saving {p}")

    if options.output_type == "point":
        ps = [
            series_to_pathline_geopandas_dataframe(s.drop("geometry", errors="ignore"), pathlines[k])
            for k, s in starting_points_df.iterrows()
        ]
    else:
        ps = [
            pathline_to_line_geopandas_dataframe(
                pathlines[k][0], attrs={"pathline_id": [k], "id": df["id"], "name": df["name"]}
            )
            for k, df in starting_points_df.iterrows()
        ]
    result = pd.concat(ps).reset_index(drop=True)
    result.to_file(p, mode="w")


if __name__ == "__main__":
    main()
