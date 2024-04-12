# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
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
Calculate pathlines (trajectories)
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from pypism.pathlines import compute_pathlines

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

    result = compute_pathlines(
        options.raster_url,
        options.vector_url,
        dt=options.dt,
        total_time=options.total_time,
        n_jobs=options.n_jobs,
        reverse=options.reverse,
    )

    result.to_file(p, mode="w")
