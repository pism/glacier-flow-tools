# glacier-flow-tools

**glacier-flow-tools** helps to analyse glacier flow. It computes pathlines
(trajectories) in a velocity field, extracts velocities and ice fluxes along
profiles (flux gates), and compares observations with simulations of the
[Parallel Ice Sheet Model (PISM)](https://www.pism.io).

The structure of this documentation follows
[pism-terra](https://pism-terra.readthedocs.io), which in turn was inspired by
Romain Hugonnet's [xDEM](https://xdem.readthedocs.io).

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Quick start
:link: getting_started/quick_start
:link-type: doc

Install the package and compute a first pathline.
:::

:::{grid-item-card} {octicon}`book` Features
:link: features/pathlines
:link-type: doc

Pathlines, and profiles across flux gates.
:::

:::{grid-item-card} {octicon}`code-square` API reference
:link: reference/api
:link-type: doc

The public functions, grouped by module.
:::
::::

```{toctree}
:caption: Getting started
:hidden:

getting_started/about
getting_started/installation
getting_started/quick_start
```

```{toctree}
:caption: Features
:hidden:

features/pathlines
features/profiles
```

```{toctree}
:caption: Gallery of examples
:hidden:

examples/jakobshavn
auto_examples/index
```

```{toctree}
:caption: Developer's Corner
:hidden:

developer/documentation
```

```{toctree}
:caption: Reference
:hidden:

reference/api
reference/cli
reference/release_notes
```
