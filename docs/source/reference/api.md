# API reference

Hand-grouped by module. Each table is built with `autosummary` and generates
one page per symbol under `generated/`.

## Pathlines

```{eval-rst}
.. currentmodule:: glacier_flow_tools.pathlines

.. autosummary::
    :toctree: generated/

    compute_pathline
    compute_pathline_with_progress
    get_grf_perturbed_velocities
    pathline_to_geopandas_dataframe
    pathline_to_line_geopandas_dataframe
    series_to_pathline_geopandas_dataframe
```

## Profiles

```{eval-rst}
.. currentmodule:: glacier_flow_tools.profiles

.. autosummary::
    :toctree: generated/

    extract_profile
    extract_profile_simple
    process_profile
    normal
    tangential
    compute_normals
    compute_tangentials
    plot_obs_sims_profile
    plot_glacier
    ProfilesMethods
    FluxMethods
```

## Interpolation

```{eval-rst}
.. currentmodule:: glacier_flow_tools.interpolation

.. autosummary::
    :toctree: generated/

    InterpolationMatrix
    interpolate_at_point
    velocity
    velocity_steady
```

## Geometry

```{eval-rst}
.. currentmodule:: glacier_flow_tools.geom

.. autosummary::
    :toctree: generated/

    distance
    distances
    linestring_to_points
    multilinestring_to_points
    convert_to_point_geometry_dataframe
    densify_line
    geopandas_dataframe_densify_lines
    geopandas_dataframe_shorten_lines
    parse_distance
    shorten_line
    GeometryConverter
```

## Gaussian random fields

```{eval-rst}
.. currentmodule:: glacier_flow_tools.gaussian_random_fields

.. autosummary::
    :toctree: generated/

    generate_field
    power_spectrum
    distrib_normal
```

## Utilities

```{eval-rst}
.. currentmodule:: glacier_flow_tools.utils

.. autosummary::
    :toctree: generated/

    preprocess_nc
    register_colormaps
    qgis2cmap
    blend_multiply
    get_dataarray_extent
    figure_extent
    merge_on_intersection_dask
    merge_on_intersection_pandas
    tqdm_joblib
```
