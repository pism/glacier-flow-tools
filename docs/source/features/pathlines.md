# Pathlines

A pathline is the trajectory of a particle through a velocity field.
glacier-flow-tools integrates pathlines with an adaptive
Runge-Kutta-Fehlberg method.

## In Python

{func}`~glacier_flow_tools.pathlines.compute_pathline` needs a starting point
and a function `f(point, time, *f_args)` that returns the velocity at that
point. For a gridded velocity field,
{func}`~glacier_flow_tools.interpolation.velocity` interpolates bilinearly and
takes the arguments `(Vx, Vy, x, y)`:

```python
from glacier_flow_tools.interpolation import velocity
from glacier_flow_tools.pathlines import compute_pathline

result = compute_pathline(
    [x0, y0],
    velocity,
    f_args=(Vx, Vy, x, y),
    start_time=0.0,
    end_time=1000.0,
    hmin=0.01,
    hmax=1.0,
    tol=1e-3,
)
points, velocities = result[0], result[1]
```

The options that control the integration are:

`start_time`, `end_time`
: The time span. The package does not track units. With velocities in m/yr,
  times are in years.

`hmin`, `hmax`
: The smallest and largest time step. If they are equal, the time step is
  fixed.

`tol`
: The error tolerance of the adaptive time stepping.

The solver shortens the time step until the error estimate is below `tol`, but
never below `hmin`. Where the tolerance cannot be met even at `hmin`, the step
is taken at `hmin` anyway. This happens where the velocity jumps, for example
at an ice margin or a data gap. The pathline then carries a
{class}`~glacier_flow_tools.pathlines.StepSizeWarning`, issued once when the
pathline is complete. It gives the number of such steps, the time of the first
one and the largest error estimate. The error estimate of every step is the
last element of the result.

`v_threshold`
: The solver stops where the speed drops below this value.

To trace a pathline backward in time, reverse the sign of both velocity
components.

### Saving pathlines

Two helpers turn a result into a {class}`geopandas.GeoDataFrame`, which can
then be written to a file:

- {func}`~glacier_flow_tools.pathlines.pathline_to_geopandas_dataframe` and
  {func}`~glacier_flow_tools.pathlines.series_to_pathline_geopandas_dataframe`
  give one point geometry per step.
- {func}`~glacier_flow_tools.pathlines.pathline_to_line_geopandas_dataframe`
  gives one line geometry per pathline.

### Uncertain velocities

{func}`~glacier_flow_tools.pathlines.get_grf_perturbed_velocities` perturbs a
velocity field with a Gaussian random field scaled by the velocity errors, to
study how the uncertainty of the velocities affects the pathlines.

## From the command line

`compute_pathlines` computes one pathline for each starting point in a vector
file.

```bash
compute_pathlines \
    --raster_url velocity.nc \
    --vector_url starting_points.gpkg \
    --end_time 100 \
    --n_jobs 4 \
    pathlines.gpkg
```

The velocity file must have the variables `vx` and `vy` on the coordinates `x`
and `y`. Add `--reverse` for backward pathlines, and `--output_type line` to
save lines instead of points. All options are listed in
{doc}`../reference/cli`.

### Messages about the time step

If pathlines did not meet the tolerance everywhere, `compute_pathlines` prints
one summary after the progress bar:

```text
Warning: 3 of 44 pathlines did not meet the tolerance tol=0.001 everywhere, even at the minimum time step hmin=0.01 yr.
The affected steps were taken at the minimum time step.
  pathline 12: 1 step, first at t=1.129 yr, largest error estimate 0.012
  ...
This is common where the velocity jumps, such as at an ice margin or a data gap.
Use a smaller --hmin or a larger --tol to change this.
```

The pathlines are still computed and saved. The numbers are the positions of
the pathlines in the output, the `pathline_id`.

### Units

`compute_pathlines` reads the units of `vx`, `vy`, `x` and `y` from their
`units` attributes. It converts the velocities to m/yr and the coordinates to
m, with [pint-xarray](https://pint-xarray.readthedocs.io/) and the unit
definitions of [cf-xarray](https://cf-xarray.readthedocs.io/). Times
(`--start_time`, `--end_time`, `--hmin`, `--hmax`) are therefore in years, and
`--v_threshold` is in m/yr.

- Common spellings are understood, for example `m/yr`, `m year-1`,
  `meter/year`, `m/s`, `m s-1`, `m day-1` and `km/yr`. A year is 365.25 days.
- If a variable has no `units` attribute, m/yr (for velocities) or m (for
  coordinates) is assumed, with a warning.
- If the units are not understood, or are not a velocity or a length, the
  command stops with an error that names the variable.
- `--velocity_units` gives the units of `vx` and `vy` and takes precedence
  over the file. Use it when the file has units that are wrong or not
  understood, such as `m/y`:

  ```bash
  compute_pathlines --velocity_units "m/yr" ...
  ```

The coordinates of the starting points are not converted. They must be in
meters, in the same projection as the velocity grid.

In Python, {func}`~glacier_flow_tools.utils.to_numpy_in_units` does the
conversion for one {class}`xarray.DataArray`.
{func}`~glacier_flow_tools.pathlines.compute_pathline` itself does not track
units.

### Starting points from lines

The vector file may hold points, lines, or both. Points are used as they are.
For lines there are two modes:

Default
: Two pathlines per line, one near each end. The start points lie about 200 m
  in from the ends of the line.

`--densify`
: A pathline every given distance along each line, for example every 500 m:

  ```bash
  compute_pathlines \
      --raster_url velocity.nc \
      --vector_url flux_gates.gpkg \
      --densify 500m \
      pathlines.gpkg
  ```

  The distance takes the unit `m` or `km`; a plain number is in meters. It is
  measured along the line, starting at the first vertex. The end of the line
  only gets a point if it falls on a multiple of the distance, so a 1.2 km
  line with `--densify 500m` gives points at 0, 500 and 1000 m. The lines are
  not shortened in this mode. The coordinate reference system of the file
  must be in meters.

In Python, {func}`~glacier_flow_tools.geom.geopandas_dataframe_densify_lines`
does the same for a {class}`geopandas.GeoDataFrame`.

## Example

{doc}`../auto_examples/plot_pathlines` computes pathlines in a rotating flow
and checks them against the exact solution.
