#!/usr/bin/env python3

"""
Module to calcuate drainage basins from surface DEM.
"""
# pylint: disable=redefined-outer-name

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import xarray as xr

import drainage_basin_generator as dbg


def load_data(input_file, thickness_varname: str = "thickness", surface_varname: str = "surface", coarsen: int = 1):
    """
    Load data from an input file.

    The input file should contain variables 'x', 'y', and the specified thickness and surface variables.
    The data is coarsened by the specified factor and the mean is taken.

    Parameters
    ----------
    input_file : str
        The path to the input file.
    thickness_varname : str, optional
        The name of the variable representing thickness in the dataset, by default "thickness".
    surface_varname : str, optional
        The name of the variable representing surface in the dataset, by default "surface".
    coarsen : int, optional
        The factor by which to coarsen the data, by default 1.

    Returns
    -------
    tuple
        A tuple containing the x, y, surface, and thickness data.

    Raises
    ------
    AssertionError
        If the specified thickness or surface variable is not found in the dataset.
    """
    ds = xr.open_dataset(input_file).coarsen(x=coarsen, y=coarsen, boundary="pad").mean()  # type: ignore[attr-defined]
    assert thickness_varname in ds.data_vars, f"{thickness_varname} not found"
    assert surface_varname in ds.data_vars, f"{surface_varname} not found"

    x, y = ds["x"], ds["y"]
    z = ds[surface_varname]
    thickness = ds[thickness_varname]

    return (x, y, z, thickness)


def initialize_mask(thk, x, y, terminus):
    """
    Initialize a mask based on thickness data and optional terminus coordinates.

    The mask is initialized using the PISM (Parallel Ice Sheet Model) debug module.
    If terminus coordinates are provided, the mask is updated to mark these areas.

    Parameters
    ----------
    thk : xarray.DataArray
        The thickness data.
    x : xarray.DataArray
        The x coordinates.
    y : xarray.DataArray
        The y coordinates.
    terminus : tuple or None
        The terminus coordinates in the format (x_min, x_max, y_min, y_max). If None, no terminus is set.

    Returns
    -------
    numpy.ndarray
        The initialized mask.
    """
    mask = dbg.initialize_mask(thk.to_numpy().astype(float))  # type: ignore[attr-defined]

    if terminus is not None:
        x_min, x_max, y_min, y_max = terminus
        mask = xr.DataArray(data=mask, dims=["y", "x"], coords=thk.coords, name="mask")
        # Create 2D conditions using broadcasting
        x_cond = (x >= x_min) & (x <= x_max)
        y_cond = (y >= y_min) & (y <= y_max)
        cond = y_cond * x_cond

        mask = xr.where(cond, 2, mask)
        mask = xr.where((mask > 0) & (mask != 2), 1, mask).to_numpy()

    return mask


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Computes the drainage basin mask given a DEM and a terminus location."

    parser.add_argument("-x", "--x_range", dest="x_range", help="x_min,x_max", nargs=2, type=float)
    parser.add_argument("-y", "--y_range", dest="y_range", help="y_min,y_max", nargs=2, type=float)
    parser.add_argument("-c", "--coarsen", dest="coarsen", help="Integer to coarsen the dataset", type=int, default=1)
    parser.add_argument("-i", dest="input", help="input file name")
    parser.add_argument("-o", dest="output", help="output file name")

    opts = parser.parse_args()

    terminus = [*opts.x_range, *opts.y_range]

    sys.stderr.write(f"Loading data from {opts.input}...")
    x, y, z, thickness = load_data(opts.input, coarsen=opts.coarsen)
    sys.stderr.write("done.\n")

    sys.stderr.write("Initializing the mask...")
    mask = initialize_mask(thickness, x, y, terminus)
    sys.stderr.write("done.\n")

    sys.stderr.write("Computing the drainage basin mask...")
    db = dbg.upslope_area(x.to_numpy(), y.to_numpy(), z.to_numpy().astype(float), mask)  # type: ignore[attr-defined]

    db = xr.DataArray(data=db, dims=["y", "x"], coords=thickness.coords, name="mask")
    db.to_netcdf(opts.output)
