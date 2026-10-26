# Pathlines at Jakobshavn Isbræ

This example traces the ice that passes through the flux gate of Jakobshavn
Isbræ (Sermeq Kujalleq) back to where it came from, using observed surface
velocities, and plots the result:

```{image} ../_static/jib_pathlines.png
:alt: Backward pathlines from the Jakobshavn Isbræ flux gate over the observed speed.
:width: 90%
:align: center
```

The black lines are pathlines, traced upstream for 1000 years from points 5 km
apart on the flux gate, which is drawn in white. The colors show the observed
speed.

## Data

Both input files are in the repository, in `glacier_flow_tools/data/`:

`jib_velocities.nc`
: The velocity components `vx` and `vy` in m/yr on a 240 m grid, for a
  227 km by 152 km region around the glacier. They are cut from the ITS_LIVE
  Greenland velocity mosaic `GRE_G0240_0000.nc`
  ([ITS_LIVE](https://its-live.jpl.nasa.gov/), A. S. Gardner, NASA JPL).
  The projection is NSIDC Sea Ice Polar Stereographic North (EPSG:3413).

`jib-flux-gates.gpkg`
: The flux gate of Jakobshavn Isbræ, a line about 55 km long.

## Compute the pathlines

Run this from the root of the repository:

```bash
compute_pathlines \
    --raster_url glacier_flow_tools/data/jib_velocities.nc \
    --vector_url glacier_flow_tools/data/jib-flux-gates.gpkg \
    --densify 5km \
    --reverse \
    flowlines.gpkg
```

`--densify 5km`
: Starts a pathline every 5 km along the gate. That gives 11 pathlines.

`--reverse`
: Reverses the velocities, so that the pathlines run upstream, from the gate
  to where the ice came from.

The other options keep their defaults: the pathlines cover 1000 years
(`--end_time`), with a time step between 0.01 and 1 year (`--hmin`, `--hmax`).
See {doc}`../features/pathlines` and {doc}`../reference/cli`.

It takes about ten seconds and prints:

```text
Processing Pathlines: 100%|██████████| 11/11 [00:09<00:00,  1.19it/s]
Time elapsed 9s

Warning: 2 of 11 pathlines did not meet the tolerance tol=0.001 everywhere, even at the minimum time step hmin=0.01 yr.
The affected steps were taken at the minimum time step.
  pathline 6: 1 step, first at t=1.281 yr, largest error estimate 0.0012
  pathline 7: 44 steps, first at t=0.0172 yr, largest error estimate 0.045
This is common where the velocity jumps, such as at an ice margin or a data gap.
Use a smaller --hmin or a larger --tol to change this.

Saving flowlines.gpkg
```

The warning is expected here. Pathline 7 starts in the trunk of the glacier,
where the observed speed is above 5000 m/yr and changes quickly from one grid
cell to the next. The pathlines are computed and saved all the same.

`flowlines.gpkg` holds one point per time step, about 12,750 in total. Each
point has the attributes of the gate, the `pathline_id` (0 to 10), the `time`
in years, the velocity (`vx`, `vy`, `v`) and the distance travelled.

## Plot

`docs/make_data/plot_jib_pathlines.py` makes the figure at the top of this
page:

```bash
python docs/make_data/plot_jib_pathlines.py flowlines.gpkg jib_pathlines.png
```

It draws the speed with the colormap `speed_colorblind`, which ships with the
package and is made available by
{func}`~glacier_flow_tools.utils.register_colormaps`, then the pathlines and
the gate on top:

```{literalinclude} ../../make_data/plot_jib_pathlines.py
:language: python
:pyobject: plot
```

The script needs [cartopy](https://scitools.org.uk/cartopy/) for the map
projection and the graticule. It is among the requirements of the package.

## Starting from the full mosaic

The pathlines can also be computed from the Greenland-wide mosaic. Only the
velocity file changes:

```bash
compute_pathlines \
    --raster_url GRE_G0240_0000.nc \
    --vector_url glacier_flow_tools/data/jib-flux-gates.gpkg \
    --densify 5km \
    --reverse \
    flowlines.gpkg
```

This takes longer, about 30 seconds, because the whole mosaic is read. The
pathlines agree with those from `jib_velocities.nc` to within about 40 m over
their length of 100 to 190 km. The small differences come from the adaptive
time stepping, which reacts to rounding.

`compute_pathlines` converts the velocities to m/yr, see
{ref}`the section on units <units>`. If it stops because it does not
understand the units in the mosaic, give them with `--velocity_units`:

```bash
compute_pathlines --velocity_units "m/yr" ...
```

`jib_velocities.nc` was made from the mosaic and these pathlines with:

```bash
python docs/make_data/extract_jib_velocities.py GRE_G0240_0000.nc flowlines.gpkg
```

It cuts `vx` and `vy` to the bounding box of the pathlines plus 10 km on every
side.
