"""
Extract the velocities around Jakobshavn Isbræ from an ITS_LIVE mosaic.

Cuts ``vx`` and ``vy`` out of a Greenland-wide velocity mosaic, such as
``GRE_G0240_0000.nc``, for the bounding box of a set of pathlines plus a
margin. The result is the small velocity file used by the Jakobshavn example
in the documentation, ``glacier_flow_tools/data/jib_velocities.nc``.

Usage::

    python docs/make_data/extract_jib_velocities.py GRE_G0240_0000.nc flowlines.gpkg

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import geopandas as gp
import xarray as xr

DEFAULT_OUTFILE = Path(__file__).parents[2] / "glacier_flow_tools" / "data" / "jib_velocities.nc"


def extract(velocity_file: Path, pathlines_file: Path, outfile: Path, margin: float = 10_000.0) -> xr.Dataset:
    """
    Cut ``vx`` and ``vy`` to the bounding box of the pathlines and save them.

    Parameters
    ----------
    velocity_file : Path
        The velocity mosaic, with the variables ``vx`` and ``vy`` on the coordinates ``x`` and ``y``.
    pathlines_file : Path
        The pathlines, in the projection of the velocity mosaic.
    outfile : Path
        The file to write.
    margin : float, optional
        The distance added on every side of the bounding box, in meters. Default is 10 km.

    Returns
    -------
    xr.Dataset
        The dataset that was written.
    """
    x_min, y_min, x_max, y_max = gp.read_file(pathlines_file).total_bounds
    with xr.open_dataset(velocity_file) as ds:
        # The y coordinate of a mosaic usually decreases, so order the slice to match.
        x_slice = slice(x_min - margin, x_max + margin)
        y_slice = slice(y_min - margin, y_max + margin)
        if ds["x"][0] > ds["x"][-1]:
            x_slice = slice(x_slice.stop, x_slice.start)
        if ds["y"][0] > ds["y"][-1]:
            y_slice = slice(y_slice.stop, y_slice.start)
        # Keep the variable that describes the projection, if the velocities name one.
        names = ["vx", "vy"]
        grid_mapping = ds["vx"].attrs.get("grid_mapping")
        if grid_mapping in ds:
            names.append(grid_mapping)
        subset = ds[names].sel(x=x_slice, y=y_slice).load()

    subset.attrs["history"] = (
        f"vx and vy cut from {Path(velocity_file).name} to the bounding box of the pathlines "
        f"plus {margin:g} m with docs/make_data/extract_jib_velocities.py"
    )
    # Keep the packing of the source (data type, scale and fill value) and compress.
    encoding = {}
    for name in ["vx", "vy"]:
        source = subset[name].encoding
        encoding[name] = {k: source[k] for k in ["dtype", "scale_factor", "add_offset", "_FillValue"] if k in source}
        encoding[name] |= {"zlib": True, "complevel": 5}
    subset.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")
    return subset


def main() -> None:
    """
    Command line interface.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Cut vx and vy out of a velocity mosaic for the bounding box of a set of pathlines."
    parser.add_argument("velocity_file", help="Velocity mosaic, for example GRE_G0240_0000.nc.", type=Path)
    parser.add_argument("pathlines_file", help="Pathlines computed with compute_pathlines.", type=Path)
    parser.add_argument("--outfile", help="File to write.", type=Path, default=DEFAULT_OUTFILE)
    parser.add_argument(
        "--margin", help="Distance added on every side of the bounding box, in meters.", type=float, default=10_000.0
    )
    options = parser.parse_args()

    subset = extract(options.velocity_file, options.pathlines_file, options.outfile, options.margin)
    size_mb = options.outfile.stat().st_size / 1e6
    print(f"Wrote {options.outfile} ({size_mb:.1f} MB)")
    print(f"  grid: {subset.sizes['x']} x {subset.sizes['y']} cells")
    print(f"  x: {float(subset.x.min()):.0f} to {float(subset.x.max()):.0f}")
    print(f"  y: {float(subset.y.min()):.0f} to {float(subset.y.max()):.0f}")
    for name in ["vx", "vy"]:
        print(f"  {name}: units '{subset[name].attrs.get('units', '')}'")


if __name__ == "__main__":
    main()
