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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import geopandas as gp
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from glacier_flow_tools.pathlines import (
    compute_pathline,
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
    parser.add_argument("--dt", help="""Time step. Default=1.0""", type=float, default=1.0)
    parser.add_argument(
        "--total_time",
        help="""Total time. Default=1_000""",
        type=float,
        default=1_000.0,
    )
    parser.add_argument(
        "--reverse",
        help="""Reverse velocity field to calculate backward pathlines.""",
        action="store_true",
        default=False,
    )
    parser.add_argument("outfile", nargs=1, help="Geopandas output file", default="pathlines.gpkg")

    options = parser.parse_args()

    p = Path(options.outfile[-1])
    p.parent.mkdir(parents=True, exist_ok=True)

    starting_points_df = gp.read_file(options.vector_url).convert.to_points()

    ds = xr.open_dataset(options.raster_url)
    Vx = np.squeeze(ds["vx"].to_numpy())
    Vy = np.squeeze(ds["vy"].to_numpy())
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()

    n_pts = len(starting_points_df)

    with tqdm_joblib(
        tqdm(desc="Processing Pathlines", total=n_pts, leave=True, position=0)
    ) as progress_bar:  # pylint: disable=unused-variable
        pathlines = Parallel(n_jobs=options.n_jobs)(
            delayed(compute_pathline)(
                [*df.geometry.coords[0]],
                Vx,
                Vy,
                x,
                y,
                dt=options.dt,
                total_time=options.total_time,
                reverse=options.reverse,
                progress=True,
                progress_kwargs={"leave": False, "position": 1},
            )
            for index, df in starting_points_df.iterrows()
        )
    result = pd.concat(
        list(
            starting_points_df.reset_index().apply(
                series_to_pathline_geopandas_dataframe, pathline=next(iter(pathlines)), axis=1
            )
        )
    ).reset_index(drop=True)

    result.to_file(p, mode="w")
