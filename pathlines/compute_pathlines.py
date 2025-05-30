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
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import geopandas as gp
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from glacier_flow_tools.geom import geopandas_dataframe_shorten_lines
from glacier_flow_tools.interpolation import velocity
from glacier_flow_tools.pathlines import (
    compute_pathline,
    pathline_to_line_geopandas_dataframe,
    series_to_pathline_geopandas_dataframe,
)
from glacier_flow_tools.utils import tqdm_joblib


if __name__ == "__main__":
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute pathlines (forward/backward) given a velocity field (xr.Dataset) and starting points (geopandas.GeoDataFrame)."
    parser.add_argument("--raster_url", help="""Path to raster dataset.""", default=None)
    parser.add_argument("--vector_url", help="""Path to vector dataset.""", default=None)
    parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
    parser.add_argument(
        "--hmin",
        help="""Minimum time step for adaptive time stepping. Default=0.01. If hmin=hmax then a fixed time step is used""",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--hmax",
        help="""Maximum time step for adaptive time stepping. Default=1.0. If hmin=hmax then a fixed time step is used""",
        type=float,
        default=1.0,
    )
    parser.add_argument("--tol", help="""Adaptive time stepping tolerance. Default=1e-3""", type=float, default=1e-3)
    parser.add_argument(
        "--start_time",
        help="""Start time. Default=0.0""",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--end_time",
        help="""End time. Default=1000.0""",
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
        help="""Threshold velocity below which solver stops Default is 0.0.""",
        default=0.0,
        type=float,
    )
    parser.add_argument("outfile", nargs=1, help="Geopandas output file", default="pathlines.gpkg")

    options = parser.parse_args()

    p = Path(options.outfile[-1])
    p.parent.mkdir(parents=True, exist_ok=True)

    df = gp.read_file(options.vector_url)
    starting_points_df = geopandas_dataframe_shorten_lines(df).convert.to_points()

    ds = xr.open_dataset(options.raster_url)
    Vx = np.squeeze(ds["vx"].to_numpy())
    Vy = np.squeeze(ds["vy"].to_numpy())

    if options.reverse:
        Vx = -Vx
        Vy = -Vy

    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()

    n_pts = len(starting_points_df)

    start = time.time()

    with tqdm_joblib(
        tqdm(desc="Processing Pathlines", total=n_pts, leave=True, position=0)
    ) as progress_bar:  # pylint: disable=unused-variable
        pathlines = Parallel(n_jobs=options.n_jobs)(
            delayed(compute_pathline)(
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
    time_elapsed = time.time() - start
    print(f"Time elapsed {time_elapsed:.0f}s\n")

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
