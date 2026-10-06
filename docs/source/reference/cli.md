# Command-line interface

Both commands are installed as console scripts by `pip install`, and are also
runnable as `python -m glacier_flow_tools.<command>`. They are declared in
`pyproject.toml` under `[project.scripts]`.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Command
  - Purpose
* - `compute_pathlines`
  - Compute pathlines, forward or backward, from a velocity field and
    starting points. See {doc}`../features/pathlines`.
* - `compute_profiles`
  - Extract profiles from observations and simulations, compute statistics
    and plot them. See {doc}`../features/profiles`.
```

## compute_pathlines

```{program-output} compute_pathlines --help
```

## compute_profiles

```{program-output} compute_profiles --help
```
