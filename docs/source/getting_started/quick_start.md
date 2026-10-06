# Quick start

This page computes one pathline in Python, and then many pathlines from the
command line.

## A pathline in Python

{func}`~glacier_flow_tools.pathlines.compute_pathline` takes a starting point
and a function that returns the velocity at a point. The package does not
track units, so keep them consistent: with a velocity in m/yr, times are in
years and coordinates in metres.

```python
import numpy as np

from glacier_flow_tools.interpolation import velocity
from glacier_flow_tools.pathlines import compute_pathline

# A uniform eastward flow of 100 m/yr on a 10 km by 10 km grid.
x = np.arange(0.0, 10_000.0, 100.0)
y = np.arange(0.0, 10_000.0, 100.0)
Vx = np.full((len(y), len(x)), 100.0)
Vy = np.zeros((len(y), len(x)))

result = compute_pathline(
    [1_000.0, 5_000.0],
    velocity,
    f_args=(Vx, Vy, x, y),
    start_time=0.0,
    end_time=10.0,
    hmin=1.0,
    hmax=1.0,
)
points = result[0]
print(points[0], points[-1])
```

The first element of the result holds the points along the pathline. With a
fixed step of one year, it has ten points, from `x = 1000` to `x = 1900`,
100 m apart.

The {doc}`gallery <../auto_examples/index>` has a complete example with a
figure.

## Many pathlines from the command line

`compute_pathlines` reads the velocity from a NetCDF file with the variables
`vx` and `vy` on coordinates `x` and `y`, and the starting points from a
vector file such as a GeoPackage:

```bash
compute_pathlines \
    --raster_url velocity.nc \
    --vector_url starting_points.gpkg \
    --end_time 100 \
    --n_jobs 4 \
    pathlines.gpkg
```

See {doc}`../reference/cli` for all options.
