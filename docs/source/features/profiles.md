# Profiles

A profile is a line, for example a flux gate across an outlet glacier.
glacier-flow-tools extracts observed and simulated velocities along profiles,
computes the component normal to the profile and, optionally, the ice flux
through it, and compares observations with simulations.

## From the command line

`compute_profiles` processes all profiles in a vector file:

```bash
compute_profiles \
    --profiles_url flux_gates.gpkg \
    --velocity_url observed_velocity.nc \
    --result_dir results \
    --n_jobs 4 \
    simulation_id_0_.nc simulation_id_1_.nc
```

It needs:

Profiles
: A vector file with line geometries and the columns `id` and `name`.
  `--segmentize` sets the spacing of the points along each profile, in metres.

Observed velocities
: A NetCDF file, given with `--velocity_url`. The variable names are set in
  the project file.

Simulations
: One or more PISM output files. The experiment id is read from each file name
  with a regular expression, by default `id_(.+?)_`.

It writes:

- `files/stats.gpkg` in the result directory, with one row per profile and
  experiment: the root-mean-square difference and the Pearson correlation
  between observations and simulation, and the observed and simulated flux.
- `figures/<name>_profile.pdf`, one figure per profile.

All options are listed in {doc}`../reference/cli`.

### Project file

A project file in [TOML](https://toml.io) names the variables to use.
`--project_file` selects it. The default, `default.toml`, compares velocities:

```{literalinclude} ../../../glacier_flow_tools/data/default.toml
:language: toml
```

`flux.toml`, which ships with the package next to `default.toml`, compares
ice fluxes instead. This also requires an ice thickness dataset, given with
`--thickness_url`.

### Flux gates

The package ships flux gates for Greenland as GeoPackage files:
`greenland-flux-gates.gpkg`, `greenland-flux-gates-29.gpkg` and
`greenland-flux-gates-5.gpkg`. They can be located with:

```python
from importlib.resources import files

gates = files("glacier_flow_tools.data").joinpath("greenland-flux-gates-29.gpkg")
```

## In Python

Importing `glacier_flow_tools.profiles` registers two accessors on
{class}`xarray.Dataset`:

`ds.profiles`
: {class}`~glacier_flow_tools.profiles.ProfilesMethods` extracts a profile,
  adds the normal component, computes statistics and plots.

`ds.fluxes`
: {class}`~glacier_flow_tools.profiles.FluxMethods` adds the ice mass flux and
  its error to a velocity dataset.

{func}`~glacier_flow_tools.profiles.extract_profile` combines these steps for
one profile, an observation dataset and a simulation dataset.
