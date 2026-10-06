# About

glacier-flow-tools is a Python package for analysing glacier flow. It is
developed alongside the [Parallel Ice Sheet Model (PISM)](https://www.pism.io)
and is under active development.

## What it does

Pathlines
: Trace the path of a particle through a gridded velocity field, forward or
  backward in time. See {doc}`../features/pathlines`.

Profiles
: Extract velocities, and optionally ice fluxes, along profiles such as flux
  gates, from observations and from simulations, and compare the two. See
  {doc}`../features/profiles`.

Supporting tools
: Bilinear interpolation on regular grids, Gaussian random fields for
  perturbing velocities, and helpers for geometries and colormaps. See the
  {doc}`../reference/api`.

## License

glacier-flow-tools is free software, distributed under the terms of the GNU
General Public License, version 3 or later. See the `LICENSE` file in the
repository.

## Authors

Andy Aschwanden and Constantine Khroulev.
