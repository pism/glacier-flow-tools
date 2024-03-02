#!/usr/bin/env python3
# Copyright (C) 2016-2023 Andy Aschwanden, Constantine Khroulev
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PYPISM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import operator
import os
from argparse import ArgumentParser
from collections.abc import (
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import Any, Dict, List, Optional, Union

import cf_units
import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
import numpy as np
import pandas as pa
import pylab as plt
import statsmodels.api as sm
from netCDF4 import Dataset as NC
from unidecode import unidecode


def reverse_enumerate(iterable):
    """
    Enumerate over an iterable in reverse order while retaining proper indexes
    """

    return zip(reversed(range(len(iterable))), reversed(iterable))


def get_rmsd(a, b, w=None):
    """
    Returns the (weighted) root mean square differences between a and b.

    Parameters
    ----------
    a, b : array_like
    w : weights

    Returns
    -------
    rmsd : scalar
    """

    if w is None:
        w = np.ones_like(a)
    # If observations or errors have missing values, don't count them
    c = (a.ravel() - b.ravel()) / w.ravel()
    if isinstance(c, np.ma.MaskedArray):
        # count all non-masked elements
        N = c.count()
        # reduce to non-masked elements
        return np.sqrt(np.linalg.norm(np.ma.compressed(c), 2) ** 2.0 / N), N
    else:
        N = c.shape[0]
        return np.sqrt(np.linalg.norm(c, 2) ** 2.0 / N), N


class FluxGate(object):
    """
    A class for FluxGates.

    Parameters
    ----------
    pos_id: int, pos of gate in array
    gate_name: string, name of flux gate
    gate_id: int, gate identification
    profile_axis: 1-d array, profile axis values
    profile_axis_units: string, udunits unit of axis
    profile_axis_name: string, descriptive name of axis

    """

    def __init__(
        self,
        pos_id,
        gate_name: str,
        gate_id: float,
        profile_axis,
        profile_axis_units: str,
        profile_axis_name: str,
        *args,
        **kwargs,
    ):
        self.pos_id = pos_id
        self.gate_name = gate_name
        self.gate_id = gate_id
        self.profile_axis = profile_axis
        self.profile_axis_units = profile_axis_units
        self.profile_axis_name = profile_axis_name
        self.best_rmsd_exp_id: str
        self.best_rmsd: float
        self.best_corr_exp_id: int
        self.best_corr: float
        self.corr: Dict[Any, Any]
        self.corr_units: str
        self.experiments: List[Any] = []
        self.exp_counter: int = 0
        self.has_observations: bool = False
        self.has_fluxes: bool = False
        self.has_stats: bool = False
        self.linear_trend: float
        self.linear_bias: float
        self.linear_r2: float
        self.linear_p: float
        self.N_corr: Dict[Any, Any]
        self.N_rmsd: Dict[Any, Any]
        self.observed_flux: float
        self.observed_flux_units: str
        self.observed_flux_error: float
        self.observed_mean: float
        self.observed_mean_units: Optional[str]
        self.p_ols: Dict[Any, Any]
        self.r2: Dict[Any, Any]
        self.r2_units: str
        self.rmsd: Dict[Any, Any]
        self.rmsd_units: str
        self.sigma_obs: Dict[Any, Any]
        self.sigma_obs_N: int
        self.sigma_obs_units: Optional[str]
        self.varname: Union[str, None] = None
        self.varname_units: Union[str, None] = None

    def __repr__(self):
        return f"{self.gate_id}: {self.gate_name}"

    def __iter__(self) -> Iterator[Hashable]:
        return (key for key in self.experiments)

    def add_experiment(self, data):
        """
        Add an experiment to FluxGate

        """

        print(f"      adding experiment to flux gate {self.gate_name}")
        pos_id = self.pos_id
        fg_exp = FluxGateExperiment(data, pos_id)
        self.experiments.append(fg_exp)
        if self.varname is None:
            self.varname = data.varname
        if self.varname_units is None:
            self.varname_units = data.varname_units
        self.exp_counter += 1

    def add_observations(self, data):
        """
        Add observations to FluxGate

        """

        print(f"      adding observations to flux gate {self.gate_name}")
        pos_id = self.pos_id
        fg_obs = FluxGateObservations(data, pos_id)
        self.observations = fg_obs
        if self.has_observations is not None:
            gate_name = self.gate_name
            print(f"Flux gate {gate_name} already has observations, overriding")
        self.has_observations = True

    def calculate_fluxes(self):
        """
        Calculate fluxes

        """

        if self.has_observations:
            self._calculate_observed_flux()
        self._calculate_experiment_fluxes()
        self.has_fluxes = True

    def calculate_stats(self):
        """
        Calculate statistics
        """

        if not self.has_fluxes:
            self.calculate_fluxes()
        corr: Dict[Any, Any] = {}
        N_rmsd: Dict[Any, Any] = {}
        p_ols: Dict[Any, Any] = {}
        r2: Dict[Any, Any] = {}
        rmsd: Dict[Any, Any] = {}
        for exp in self.experiments:
            m_id = exp.m_id
            x = np.squeeze(self.profile_axis)
            obs_vals = np.squeeze(self.observations.values)
            # mask values where obs is zero
            obs_vals = np.ma.masked_where(obs_vals == 0, obs_vals)
            if isinstance(obs_vals, np.ma.MaskedArray):
                obs_vals = obs_vals.filled(0)
            exp_vals = np.squeeze(self.experiments[m_id].values)
            # Calculate root mean square difference (RMSD), convert units
            my_rmsd, my_N_rmsd = get_rmsd(exp_vals, obs_vals)
            i_units = self.varname_units
            o_units = var_dict[varname]["v_o_units"]
            i_units_cf = cf_units.Unit(i_units)
            o_units_cf = cf_units.Unit(o_units)
            rmsd[m_id] = i_units_cf.convert(my_rmsd, o_units_cf)
            N_rmsd[m_id] = my_N_rmsd
            obsS = pa.Series(data=obs_vals, index=x)
            expS = pa.Series(data=exp_vals, index=x)
            p_ols[m_id] = sm.OLS(expS, sm.add_constant(obsS), missing="drop").fit()
            r2[m_id] = p_ols[m_id].rsquared
            corr[m_id] = obsS.corr(expS)
        best_rmsd_exp_id = sorted(p_ols, key=lambda x: rmsd[x], reverse=False)[0]
        best_corr_exp_id = sorted(p_ols, key=lambda x: corr[x], reverse=True)[0]
        self.p_ols = p_ols
        self.best_rmsd_exp_id = best_rmsd_exp_id
        self.best_rmsd = rmsd[best_rmsd_exp_id]
        self.best_corr_exp_id = best_corr_exp_id
        self.best_corr = corr[best_corr_exp_id]
        self.rmsd = rmsd
        self.rmsd_units = var_dict[varname]["v_o_units"]
        self.N_rmsd = N_rmsd
        self.r2 = r2
        self.r2_units = "1"
        self.corr = corr
        self.corr_units = "1"
        self.has_stats = True
        self.observed_mean = np.mean(self.observations.values)
        self.observed_mean_units = self.varname_units

    def _calculate_observed_flux(self):
        """
        Calculate observed flux

        Calculate observed flux using trapezoidal rule. If observations have
        asscociated errors, the error in observed flux is calculated as well.
        """

        x = self.profile_axis
        y = self.observations.values
        x_units = self.profile_axis_units
        y_units = self.varname_units
        int_val = self._line_integral(y, x)
        # Here we need to directly access udunits2 since we want to
        # multiply units
        if vol_to_mass:
            i_units_cf = (
                cf_units.Unit(x_units)
                * cf_units.Unit(y_units)
                * cf_units.Unit(ice_density_units)
            )
        else:
            i_units_cf = cf_units.Unit(x_units) * cf_units.Unit(y_units)
        o_units_cf = cf_units.Unit(var_dict[varname]["v_flux_o_units"])
        o_units_str = var_dict[varname]["v_flux_o_units_str"]
        o_val = i_units_cf.convert(int_val, o_units_cf)
        observed_flux = o_val
        observed_flux_units = o_units_str
        if self.observations.has_error:
            y = self.observations.error
            int_val = self._line_integral(y, x)
            i_error = int_val
            o_error = i_units_cf.convert(i_error, o_units_cf)
            error_norm, N = get_rmsd(y, np.zeros_like(y, dtype="float32"))
            self.sigma_obs = error_norm
            self.sigma_obs_N = N
            self.sigma_obs_units = self.varname_units
            self.observed_flux_error = o_error

        self.observed_flux = observed_flux
        self.observed_flux_units = observed_flux_units

    def _calculate_experiment_fluxes(self):
        """
        Calculate experiment fluxes

        Calculated experiment fluxes using trapeziodal rule.

        """

        experiment_fluxes = {}
        experiment_fluxes_units = {}
        for exp in self.experiments:
            m_id = exp.m_id
            x = self.profile_axis
            y = exp.values
            x_units = self.profile_axis_units
            y_units = self.varname_units
            int_val = self._line_integral(y, x)
            # Here we need to directly access udunits2 since we want to
            # multiply units
            if vol_to_mass:
                i_units = (
                    cf_units.Unit(x_units)
                    * cf_units.Unit(y_units)
                    * cf_units.Unit(ice_density_units)
                )
            else:
                i_units = cf_units.Unit(x_units) * cf_units.Unit(y_units)
            o_units = cf_units.Unit(var_dict[varname]["v_flux_o_units"])
            o_units_str = var_dict[varname]["v_flux_o_units_str"]
            o_val = i_units.convert(int_val, o_units)
            experiment_fluxes[m_id] = o_val
            experiment_fluxes_units[m_id] = o_units_str
        self.experiment_fluxes = experiment_fluxes
        self.experiment_fluxes_units = experiment_fluxes_units

    def length(self):
        """
        Return length of the profile, rounded to the nearest meter.
        """

        return np.around(self.profile_axis.max())

    def _line_integral(self, y, x):
        """
        Return line integral using the composite trapezoidal rule

        Parameters
        ----------
        y: 1-d array_like
           Input array to integrate
        x: 1-d array_like
           Spacing between elements

        Returns
        -------
        trapz : float
                Definite integral as approximated by trapezoidal rule.
        """

        # Due to the variable length of profiles, we have masked arrays, with
        # masked values at the profile end. We can assume zero error here,
        # since it's not used for the computation

        if isinstance(y, np.ma.MaskedArray):
            x = x.filled(0)
        if isinstance(y, np.ma.MaskedArray):
            y = y.filled(0)

        y_int = float(np.squeeze(np.trapz(y, x)))

        return y_int

    def make_obs_label(self, label_type, **kwargs):
        obs = self.observations
        if label_type == "long":
            label = f"obs: {self.observed_flux:6.1f}"
            if obs.has_error:
                label += f"$\pm${self.observed_flux_error:4.1f}"
        elif label_type in ("short", "regress", "exp"):
            label = "observed"
        elif label_type in ("attr"):
            label = "AERODEM"
        else:
            label = f"obs: {self.observed_flux:6.1f}"
            if obs.has_error:
                label += f"$\pm${self.observed_flux_error:4.1f}"
        return label

    def make_exp_label(self, exp, label_type, params, **kwargs):

        exp_str = self.make_exp_str(exp, params)
        config = exp.config
        if self.has_observations:
            if label_type == "long":
                label = " ".join(
                    [
                        ": ".join(
                            [
                                exp_str,
                                f"{self.experiment_fluxes[exp.m_id]:6.1f}",
                            ]
                        ),
                        f"(r={self.corr[exp.m_id]:1.2f})",
                    ]
                )
            elif label_type == "attr":
                [key for key in params]
            elif label_type == "exp":
                exp_str = ", ".join(
                    [
                        "=".join(
                            [
                                params_dict[key]["abbr"],
                                params_dict[key]["format"].format(config.get(key)),
                            ]
                        )
                        for key in params
                    ]
                )
                label = exp_str
            else:
                label = f"r={self.corr[m_id]:1.2f}"
            if label_type == "regress":
                label = (
                    f"{config['grm_id_dx_meters']:4.0f}m, r={self.corr[exp.m_id]:1.2f}"
                )
            elif label_type == "short":
                label = f"r={self.corr[m_id]:1.2f}"
            else:
                pass
        else:
            label = config.get("grid_dx_meters")
        return label

    def make_exp_str(self, exp, params, **kwargs):
        config = exp.config
        exp_str = ", ".join(
            ["=".join([params_dict[key]["abbr"], config.get(key)]) for key in params]
        )
        return exp_str

    def make_line_plot(self, **kwargs):
        """
        Make a plot.

        Make a line plot along a flux gate.
        """

        gate_name = self.gate_name
        experiments = self.experiments
        profile_axis = self.profile_axis
        profile_axis_name = self.profile_axis_name
        profile_axis_units = self.profile_axis_units
        i_units_cf = cf_units.Unit(profile_axis_units)
        o_units_cf = cf_units.Unit(profile_axis_out_units)
        profile_axis_out = i_units_cf.convert(profile_axis, o_units_cf)
        varname = self.varname
        v_units = self.varname_units
        has_observations = self.has_observations
        if not self.has_fluxes:
            self.calculate_fluxes()
        if has_observations:
            self.calculate_stats()

        labels = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if has_observations:
            obs = self.observations
            label = self.make_obs_label("long")
            has_error = obs.has_error
            i_vals = obs.values
            i_units_cf = cf_units.Unit(v_units)
            o_units_cf = cf_units.Unit(var_dict[varname]["v_o_units"])
            obs_o_vals = i_units_cf.convert(i_vals, o_units_cf)
            obs_max = np.max(obs_o_vals)
            if has_error:
                i_vals = obs.error
                obs_error_o_vals = i_units_cf.convert(i_vals, o_units_cf)
                ax.fill_between(
                    profile_axis_out,
                    obs_o_vals - obs_error_o_vals,
                    obs_o_vals + obs_error_o_vals,
                    color="0.85",
                )

            ax.plot(profile_axis_out, obs_o_vals, "-", color="0.5")
            if not simple_plot:
                ax.plot(
                    profile_axis_out,
                    obs_o_vals,
                    dash_style,
                    color=obscolor,
                    markersize=markersize,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    label=label,
                )

        lines_d = []
        lines_c = []
        # We need to carefully reverse list and properly order
        # handles and labels to have first experiment plotted on top
        for k, exp in enumerate(reversed(experiments)):
            i_vals = exp.values
            i_units_cf = cf_units.Unit(v_units)
            o_units_cf = cf_units.Unit(var_dict[varname]["v_o_units"])
            exp_o_vals = i_units_cf.convert(i_vals, o_units_cf)
            if normalize:
                exp_max = np.max(exp_o_vals)
                exp_o_vals *= obs_max / exp_max
            if "label_param_list" in list(kwargs.keys()):
                params = kwargs["label_param_list"]
                label = self.make_exp_label(exp, "long", params)
                labels.append(label)
            my_color = my_colors[k]

            (line_c,) = ax.plot(
                profile_axis_out, exp_o_vals, "-", color=my_color, alpha=0.5
            )
            if not simple_plot:
                (line_d,) = ax.plot(
                    profile_axis_out,
                    exp_o_vals,
                    dash_style,
                    color=my_color,
                    markersize=markersize,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    label=label,
                )
            labels.append(label)
            lines_d.append(line_d)
            lines_c.append(line_c)

        ax.set_xlim(0, np.max(profile_axis_out))
        xlabel = f"{profile_axis_name} ({profile_axis_out_units})"
        ax.set_xlabel(xlabel)
        v_name: Optional[str]
        if varname in list(var_name_dict.keys()):
            v_name = var_name_dict[varname]
        else:
            v_name = varname
        ylabel = f"{v_name} ({var_dict[varname]['v_o_units_str']})"
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = handles[:0:-1]
        ordered_labels = labels[:0:-1]
        ordered_handles.insert(0, handles[0])
        ordered_labels.insert(0, labels[0])
        if legend != "none":
            if (legend == "short") or (legend == "regress") or (legend == "attr"):
                lg = ax.legend(
                    ordered_handles,
                    ordered_labels,
                    loc="upper right",
                    shadow=True,
                    numpoints=numpoints,
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=plt.gcf().transFigure,
                )
            else:
                lg = ax.legend(
                    ordered_handles,
                    ordered_labels,
                    loc="upper right",
                    title=f"{var_dict[varname]['flux_type']} ({self.experiment_fluxes_units[0]})",
                    shadow=True,
                    numpoints=numpoints,
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=plt.gcf().transFigure,
                )
            fr = lg.get_frame()
            fr.set_lw(legend_frame_width)
        # Replot observations
        if has_observations:
            if simple_plot:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.35")
            else:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.5")
                ax.plot(
                    profile_axis_out,
                    obs_o_vals,
                    dash_style,
                    color=obscolor,
                    markersize=markersize,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                )
        if plot_title:
            plt.title(gate_name, loc="left")

        gate_unicode = gate.gate_name.decode("utf-8", "strict")
        if normalize:
            gate_name = "_".join([gate_unicode, varname, "normalized", "profile"])
        else:
            gate_name = "_".join([gate_unicode, varname, "profile"])
        outname = os.path.join(odir, ".".join([gate_name, "pdf"]).replace(" ", "_"))
        print(f"Saving {outname}")
        fig.tight_layout()
        fig.savefig(outname)
        plt.close(fig)


class FluxGateExperiment(object):
    def __init__(self, data, pos_id, *args, **kwargs):
        super(FluxGateExperiment, self).__init__(*args, **kwargs)
        self.values = data.values[pos_id, Ellipsis]
        self.config = data.config
        self.m_id = data.m_id

    def __repr__(self):
        return f"{self.m_id}"


class FluxGateObservations(object):
    def __init__(self, data, pos_id, *args, **kwargs):
        super(FluxGateObservations, self).__init__(*args, **kwargs)
        self.has_error = None
        self.values = data.values[pos_id, Ellipsis]
        if data.has_error:
            self.error = data.error[pos_id, Ellipsis]
            self.has_error = True

    def __repr__(self):
        return "FluxGateObservations"


class Dataset(object):
    """
    A base class for Experiments or Observations.

    Constructor opens netCDF file, attaches pointer to nc instance.

    """

    def __init__(self, filename, varname, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        print(("  opening NetCDF file %s ..." % filename))
        try:
            nc = NC(filename, "r")
        except FileNotFoundError:
            print(
                (
                    "ERROR:  file '%s' not found or not NetCDF format ... ending ..."
                    % filename
                )
            )
            import sys

            sys.exit(1)

        for name in nc.variables:
            v = nc.variables[name]
            if getattr(v, "standard_name", "") == varname:
                print((f"variabe {name} found by its standard_name {varname}"))
                varname = name

        self.values = np.squeeze(nc.variables[varname][:])
        self.varname_units = nc.variables[varname].units
        self.varname = varname
        self.nc = nc

    def __repr__(self):
        return "Dataset"

    def __del__(self):
        # Close open file
        self.nc.close()


class ExperimentDataset(Dataset):
    """
    A derived class for experiments

    A derived class for handling PISM experiments.

    Experiments are identified by id. Config and run_stats
    attributes are attached to object as config dictionary.

    """

    def __init__(self, m_id, *args, **kwargs):
        super(ExperimentDataset, self).__init__(*args, **kwargs)

        print(f"Experiment {m_id}")
        self.m_id = m_id
        self.config = dict()
        for v in ["pism_config", "run_stats", "config"]:
            if v in self.nc.variables:
                ncv = self.nc.variables[v]
                for attr in ncv.ncattrs():
                    self.config[attr] = getattr(ncv, attr)
            else:
                print(f"Variable {v} not found")

    def __repr__(self):
        return "ExperimentDataset"


class ObservationsDataset(Dataset):
    """
    A derived class for observations.

    A derived class for handling observations.

    """

    def __init__(self, *args, **kwargs):
        super(ObservationsDataset, self).__init__(*args, **kwargs)
        self.has_error = None
        varname = self.varname
        error_varname = "_".join([varname, "error"])
        if error_varname in list(self.nc.variables.keys()):
            print(f"Observational uncertainty found in {error_varname}.")
            self.error = self.nc.variables[error_varname][:]
            self.has_error = True

    def __repr__(self):
        return "ObservationsDataset"


def export_csv_from_dict(filename, mdict, header=None, fmt=["%i", "%4.2f"]):
    """
    Creates a CSV file from a dictionary.

    Parameters
    ----------
    filename: string
    mdict: dictionary with id and data

    """

    ids = [x for x in mdict.keys()]
    values = [x for x in mdict.values()]
    data = np.vstack((ids, values))
    np.savetxt(
        filename, np.transpose(data), fmt=["%i", "%4.2f"], delimiter=",", header=header
    )


def make_correlation_figure(filename):
    """
    Create a Pearson R correlation plot.

    Create a correlation plot for a given experiment, sorted by
    decreasing correlation

    Parameters
    ----------
    filename: string
    exp: FluxGateExperiment

    """
    corrs = {}
    for gate in flux_gates:
        m_id = gate.pos_id
        r = gate.corr[exp.m_id]
        if not np.isnan(r):
            corrs[m_id] = r
    fig = plt.figure(figsize=[6.4, 12])
    ax = fig.add_subplot(111)
    y = np.arange(len(list(corrs.keys()))) + 1.25
    for k, corr in enumerate(corrs_sorted):
        if corr < 0.5:
            colorVal = "#d7191c"
        elif (corr >= pearson_r_threshold_low) and (corr < pearson_r_threshold_high):
            colorVal = "#ff7f00"
        else:
            colorVal = "#33a02c"
        ax.hlines(y[k], -1, corr, colors=colorVal, linestyle="dotted")
        ax.plot(corr, y[k], "o", markersize=5, color=colorVal)

    corr_median = np.nanmedian(list(corrs.values()))
    ax.vlines(corr_median, 0, y[-1], linestyle="dotted", color="0.5")
    print(f"median correlation: {corr_median:1.2f}")
    plt.yticks(
        y,
        [f"{flux_gates[x].gate_name} ({flux_gates[x].gate_id})" for x in sort_order],
    )
    ax.set_xlabel("r (-)", labelpad=0.2)
    ax.set_xlim(-1, 1.1)
    ax.set_ylim(0, y[-1] + 1)
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    # Only draw spine between the y-ticks
    ax.spines["left"].set_bounds(y[0], y[-1])
    ax.spines["bottom"].set_bounds(-1, 1)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    fig.tight_layout()
    outname = os.path.join(odir, filename)
    print(f"Saving {outname}")
    fig.savefig(outname)
    plt.close("all")
    return corrs_dict


def make_correlation_figure_sorted(filename, exp):
    """
    Create a Pearson R correlation plot.

    Create a correlation plot for a given experiment, sorted by
    decreasing correlation

    Parameters
    ----------
    filename: string
    exp: FluxGateExperiment

    """
    corrs = {}
    for gate in flux_gates:
        m_id = gate.pos_id
        r = gate.corr[exp.m_id]
        if not np.isnan(r):
            corrs[m_id] = r
    sort_order = sorted(corrs, key=lambda x: corrs[x])
    corrs_sorted = [corrs[x] for x in sort_order]
    gate_id_sorted = [flux_gates[x].gate_id for x in sort_order]
    corrs_dict = dict(zip(gate_id_sorted, corrs_sorted))
    fig = plt.figure(figsize=[6.4, 12])
    ax = fig.add_subplot(111)
    y = np.arange(len(list(corrs.keys()))) + 1.25
    for k, corr in enumerate(corrs_sorted):
        if corr < 0.5:
            colorVal = "#d7191c"
        elif (corr >= pearson_r_threshold_low) and (corr < pearson_r_threshold_high):
            colorVal = "#ff7f00"
        else:
            colorVal = "#33a02c"
        ax.hlines(y[k], -1, corr, colors=colorVal, linestyle="dotted")
        ax.plot(corr, y[k], "o", markersize=5, color=colorVal)

    corr_median = np.nanmedian(list(corrs.values()))
    ax.vlines(corr_median, 0, y[-1], linestyle="dotted", color="0.5")
    print(f"median correlation: {corr_median:1.2f}")
    plt.yticks(
        y,
        [f"{flux_gates[x].gate_name} ({flux_gates[x].gate_id})" for x in sort_order],
    )
    ax.set_xlabel("r (-)", labelpad=0.2)
    ax.set_xlim(-1, 1.1)
    ax.set_ylim(0, y[-1] + 1)
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    # Only draw spine between the y-ticks
    ax.spines["left"].set_bounds(y[0], y[-1])
    ax.spines["bottom"].set_bounds(-1, 1)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    fig.tight_layout()
    outname = os.path.join(odir, filename)
    print(f"Saving {outname}")
    fig.savefig(outname)
    plt.close("all")
    return corrs_dict


def make_regression(gate):
    grid_dx_meters = [x.config["grid_dx_meters"] for x in gate.experiments]
    for gate in flux_gates:

        # RMSD
        rmsd_data = list(gate.rmsd.values())
        rmsdS = pa.Series(data=rmsd_data, index=list(gate.rmsd.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
        model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
        # Calculate PISM trends and biases (intercepts)
        bias, trend = model.params
        # Calculate r-squared value
        r2 = model.rsquared
        # make x lims from 0 to 5000 m
        xmin, xmax = 0, 5000
        # Create figures
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            grid_dx_meters,
            rmsd_data,
            dash_style,
            color="0.2",
            markeredgewidth=markeredgewidth,
        )
        ax.set_xticks(grid_dx_meters)
        ax.set_xlabel("grid resolution (m)")
        ax.set_ylabel(f"$\chi$ ({var_dict[varname]['v_o_units_str']})")
        ax.set_xlim(xmin, xmax)
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(30)
        plt.title(gate.gate_name)
        fig.tight_layout()
        gate_name = "_".join([unidecode(gate.gate_name), varname, "rmsd", "regress"])
        outname = os.path.join(odir, ".".join([gate_name, "pdf"]).replace(" ", "_"))
        print(f"Saving {outname}")
        fig.savefig(outname)
        plt.close("all")

    grid_dx_meters = [x.config["grid_dx_meters"] for x in gate.experiments]

    # Create RMSD figure
    fig = plt.figure()
    # make x lims from 450 to 5000 m
    xmin, xmax = 0, 5000
    ax = fig.add_subplot(111)
    legend_handles = []
    for n, gate in enumerate(flux_gates):
        m_id = gate.pos_id

        # RMSD
        rmsd_data = list(gate.rmsd.values())
        rmsdS = pa.Series(data=rmsd_data, index=list(gate.rmsd.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
        model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
        # trend and bias (intercept)
        bias, trend = model.params
        # r-squared value
        r2 = model.rsquared
        # p-value
        p = model.f_pvalue

        gate.linear_trend = trend
        gate.linear_bias = bias
        gate.linear_r2 = r2
        gate.linear_p = p

        # select glaciers that don't have a significant (95%) trend
        # and denote them by a dashed line
        if p >= 0.05:
            ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend,
                linestyle="dashed",
                color="0.7",
                linewidth=0.5,
            )
        else:
            ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend,
                color="0.7",
                linewidth=0.5,
            )
        ax.plot(
            grid_dx_meters,
            rmsd_data,
            dash_style,
            color="0.7",
            markeredgewidth=markeredgewidth,
            markeredgecolor="0.7",
            markersize=1.75,
        )

    for m_id in (1, 11, 19, 23):

        gate = flux_gates[m_id]

        # RMSD
        rmsd_data = list(gate.rmsd.values())
        rmsdS = pa.Series(data=rmsd_data, index=list(gate.rmsd.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
        model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
        # trend and bias (intercept)
        bias_selected, trend_selected = model.params
        p_selected = model.f_pvalue

        if m_id == 1:  # Jakobshavn
            colorVal = "#54278f"
        elif m_id == 11:  # Kong Oscar
            colorVal = "#006d2c"
        elif m_id == 19:  # Kangerdlugssuaq
            colorVal = "#08519c"
        elif m_id == 23:  # Koge Bugt S
            colorVal = "#a50f15"
        else:
            print("How did I get here?")

        if p_selected >= 0.05:
            (line_l,) = ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias_selected
                + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_selected,
                linestyle="dashed",
                color=colorVal,
                linewidth=0.5,
            )
        else:
            (line_l,) = ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias_selected
                + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_selected,
                color=colorVal,
                linewidth=0.5,
            )
        ax.plot(
            grid_dx_meters,
            rmsd_data,
            dash_style,
            color=colorVal,
            markeredgewidth=markeredgewidth * 0.8,
            markeredgecolor="0.2",
            markersize=1.75,
        )
        legend_handles.append(line_l)

    # global RMSD
    rmsd_data = list(rmsd_cum_dict.values())
    rmsdS = pa.Series(data=rmsd_data, index=list(rmsd_cum_dict.keys()))
    gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
    model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
    # Calculate PISM trends and biases (intercepts)
    bias_global, trend_global = model.params
    p_global = model.f_pvalue
    if p_global > 0.05:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_global
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_global,
            color="0.2",
            linewidth=1,
            linestyle="dashed",
        )
    else:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_global
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_global,
            color="0.2",
            linewidth=1,
        )
    ax.plot(
        grid_dx_meters,
        rmsd_data,
        dash_style,
        color="0.4",
        markeredgewidth=markeredgewidth,
    )
    legend_handles.append(line_l)

    # Create correlation figures
    fig = plt.figure()
    # make x lims from 0 to 5000 m
    xmin, xmax = 0, 5000

    jet = plt.get_cmap("jet")
    cNorm = mplcolors.Normalize(vmin=0, vmax=15)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    ax = fig.add_subplot(111)
    for n, gate in enumerate(flux_gates):
        # correlation
        corr_data = list(gate.corr.values())
        corrS = pa.Series(data=corr_data, index=list(gate.corr.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.corr.keys()))
        model = pa.ols(x=gridS, y=corrS)
        # Calculate PISM trends and biases (intercepts)
        trend, bias = model.beta
        # Calculate r-squared value
        r2 = model.r2

        colorVal = scalarMap.to_rgba(n)
        if corr_data[0] >= 0.85:
            ax.plot(
                grid_dx_meters,
                corr_data,
                dash_style,
                color=colorVal,
                markeredgewidth=markeredgewidth,
                markeredgecolor="k",
                markersize=2,
            )

    ax.set_xticks(grid_dx_meters)
    ax.set_xlabel("grid resolution (m)")
    ax.set_ylabel("correlation coefficient (-)")
    ax.set_xlim(500, 2000)
    ax.set_ylim(0.85, 1)

    ticklabels = ax.get_xticklabels()
    for tick in ticklabels:
        tick.set_rotation(40)

    fig.tight_layout()
    outname = os.path.join(
        odir, ".".join(["pearson_r_regression", "pdf"]).replace(" ", "_")
    )
    print("Saving {outname}")
    fig.savefig(outname)
    plt.close("all")


def setup_flux_gates():
    # Open first file
    filename = args[0]
    print(("  opening NetCDF file %s ..." % filename))
    try:
        nc0 = NC(filename, "r")
    except FileNotFoundError:
        print(
            (
                "ERROR:  file '%s' not found or not NetCDF format ... ending ..."
                % filename
            )
        )
        import sys

        sys.exit(1)

    # Get profiles from first file
    # All experiments have to contain the same profiles
    # Create flux gates
    profile_names = nc0.variables["profile_name"][:]
    flux_gates = []
    for pos_id, profile_name in enumerate(profile_names):
        profile_axis = nc0.variables["profile_axis"][pos_id]
        profile_axis_units = nc0.variables["profile_axis"].units
        profile_axis_name = nc0.variables["profile_axis"].long_name
        profile_id = int(nc0.variables["profile_id"][pos_id])
        flux_gate = FluxGate(
            pos_id,
            profile_name,
            profile_id,
            profile_axis,
            profile_axis_units,
            profile_axis_name,
        )
        flux_gates.append(flux_gate)
    nc0.close()
    return flux_gates


var_dict: Dict[Optional[str], Dict] = {
    "velsurf_mag": {
        "flux_type": "flux",
        "v_o_units": "m yr-1",
        "v_o_units_str": "m yr$^\mathregular{{-1}}$",
        "v_o_units_str_tex": "m\,yr$^{-1}$",
        "v_flux_o_units": "km2 yr-1",
        "v_flux_o_units_str": "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$",
        "v_flux_o_units_str_tex": "km$^2$\,yr$^{-1}$",
    },
    "velbase_mag": {
        "flux_type": "flux",
        "v_o_units": "m yr-1",
        "v_o_units_str": "m yr$^\mathregular{{-1}}$",
        "v_o_units_str_tex": "m\,yr$^{-1}$",
        "v_flux_o_units": "km2 yr-1",
        "v_flux_o_units_str": "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$",
        "v_flux_o_units_str_tex": "km$^2$\,yr$^{-1}$",
    },
    "velsurf_normal": {
        "flux_type": "flux",
        "v_o_units": "m yr-1",
        "v_o_units_str": "m yr$^\mathregular{{-1}}$",
        "v_o_units_str_tex": "m\,yr$^{-1}$",
        "v_flux_o_units": "km2 yr-1",
        "v_flux_o_units_str": "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$",
        "v_flux_o_units_str_tex": "km$^2$\,yr$^{-1}$",
    },
    "v": {
        "flux_type": "flux",
        "v_o_units": "m yr-1",
        "v_o_units_str": "m yr$^\mathregular{{-1}}$",
        "v_o_units_str_tex": "m\,yr$^{-1}$",
        "v_flux_o_units": "km2 yr-1",
        "v_flux_o_units_str": "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$",
        "v_flux_o_units_str_tex": "km$^2$\,yr$^{-1}$",
    },
    "flux_mag": {
        "flux_type": "flux",
        "v_o_units": "km2 yr-1",
        "v_o_units_str": "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$",
        "v_o_units_str_tex": "km$^2$\,yr$^{-1}$",
        "v_flux_o_units": "Gt yr-1",
        "v_flux_o_units_str": "Gt yr$^\mathregular{{-1}}$",
        "v_flux_o_units_str_tex": "Gt\,yr$^{-1}$",
        "vol_to_mass": True,
    },
}

params_dict: Dict[Optional[str], Dict] = {
    "dataset": {"abbr": "Dataset", "format": "{}"},
    "dem": {"abbr": "DEM", "format": "{}"},
    "bed": {"abbr": "bed", "format": "{}"},
    "init": {"abbr": "INIT", "format": "{}"},
    "surface.pdd.factor_ice": {
        "abbr": "$f_{\mathregular{i}}$",
        "format": "{:1.0f}",
    },
    "surface.pdd.factor_snow": {
        "abbr": "$f_{\mathregular{s}}$",
        "format": "{:1.0f}",
    },
    "basal_resistance.pseudo_plastic.q": {"abbr": "$q$", "format": "{:1.2f}"},
    "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": {
        "abbr": "$\\delta$",
        "format": "{:1.4f}",
    },
    "stress_balance.sia.enhancement_factor": {
        "abbr": "$E_{\mathregular{SIA}}$",
        "format": "{:1.2f}",
    },
    "stress_balance.ssa.enhancement_factor": {
        "abbr": "$E_{\mathregular{SSA}}$",
        "format": "{:1.2f}",
    },
    "stress_balance.ssa.Glen_exponent": {
        "abbr": "$n_{\mathregular{SSA}}$",
        "format": "{:1.2f}",
    },
    "stress_balance.sia.Glen_exponent": {
        "abbr": "$n_{\mathregular{SIA}}$",
        "format": "{:1.2f}",
    },
    "grid_dx_meters": {"abbr": "ds", "format": "{:.0f}"},
    "flow_law.gpbld.water_frac_observed_limit": {
        "abbr": "$\omega$",
        "format": "{:1.2}",
    },
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": {
        "abbr": "$\phi_{\mathregular{min}}$",
        "format": "{:4.2f}",
    },
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": {
        "abbr": "$\phi_{\mathregular{max}}$",
        "format": "{:4.2f}",
    },
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": {
        "abbr": "$z_{\mathregular{min}}$",
        "format": "{:1.0f}",
    },
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": {
        "abbr": "$z_{\mathregular{max}}$",
        "format": "{:1.0f}",
    },
}

var_long = (
    "velsurf_mag",
    "velbase_mag",
    "velsurf_normal",
    "flux_mag",
    "flux_normal",
    "surface",
    "usurf",
    "surface_altitude" "thk",
    "thickness",
    "land_ice_thickness",
)
var_short = (
    "speed",
    "sliding speed",
    "speed",
    "flux",
    "flux",
    "altitude",
    "altitude",
    "altitude",
    "ice thickness",
    "ice thickness",
    "ice thickness",
)
var_name_dict = dict(list(zip(var_long, var_short)))

# ##############################################################################
# MAIN
# ##############################################################################

if __name__ == "__main__":

    __spec__ = None

    # Set up the option parser
    parser = ArgumentParser()
    parser.description = "Analyze flux gates."
    parser.add_argument("FILE", nargs="*")
    parser.add_argument(
        "--aspect_ratio",
        dest="aspect_ratio",
        type=float,
        help='''Plot aspect ratio"''',
        default=0.8,
    )
    parser.add_argument(
        "--colormap",
        dest="colormap",
        nargs=1,
        help="""Name of matplotlib colormap""",
        default="tab20c",
    )
    parser.add_argument(
        "--label_params",
        dest="label_params",
        help='''comma-separated list of parameters that appear in the legend,
                      e.g. "sia_enhancement_factor"''',
        default="bed",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Normalize experiments by muliplying with max(obs)/max(experiment)",
        default=False,
    )
    parser.add_argument(
        "--obs_file",
        dest="obs_file",
        help="""Profile file with observations. Default is None""",
        default=None,
    )
    parser.add_argument(
        "--no_figures",
        dest="make_figures",
        action="store_false",
        help="Do not make profile figures",
        default=True,
    )
    parser.add_argument(
        "--do_regress",
        dest="do_regress",
        action="store_true",
        help="Make grid resolution regression plots",
        default=False,
    )
    parser.add_argument(
        "--legend",
        dest="legend",
        choices=["default", "none", "long", "short", "regress", "exp", "attr"],
        help="Controls the legend, options are: \
                        'default' (default), 'none, 'short', long', 'regress'",
        default="default",
    )
    parser.add_argument(
        "--plot_title",
        dest="plot_title",
        action="store_true",
        help="Plots the flux gate name as title",
        default=False,
    )
    parser.add_argument(
        "--simple_plot",
        dest="simple_plot",
        action="store_true",
        help="Make simple line plot",
        default=False,
    )
    parser.add_argument(
        "--no_legend",
        dest="plot_legend",
        action="store_false",
        help="Don't plot a legend",
        default=True,
    )
    parser.add_argument(
        "-r",
        "--output_resolution",
        dest="out_res",
        help="""
                      Graphics resolution in dots per inch (DPI), default
                      = 300""",
        default=300,
    )
    parser.add_argument(
        "--y_lim", dest="y_lim", nargs=2, help="""Y lims""", default=[None, None]
    )
    parser.add_argument(
        "-v",
        "--variable",
        dest="varname",
        help="""Variable to plot, default = 'velsurf_mag'.""",
        default="velsurf_mag",
    )
    parser.add_argument(
        "-o",
        "--o_dir",
        dest="odir",
        help="""Directory to put results""",
        default="results",
    )

    options = parser.parse_args()
    args = options.FILE

    np.seterr(all="warn")
    aspect_ratio = options.aspect_ratio
    tol = 1e-6
    normalize = options.normalize
    obs_file = options.obs_file
    out_res = int(options.out_res)
    varname = options.varname
    label_params = list(options.label_params.split(","))
    plot_title = options.plot_title
    legend = options.legend
    do_regress = options.do_regress
    make_figures = options.make_figures
    odir = options.odir
    simple_plot = options.simple_plot
    y_lim_min, y_lim_max = options.y_lim
    odir = options.odir

    if not os.path.isdir(odir):
        os.makedirs(odir)

    ice_density = 910.0
    ice_density_units = "910 kg m-3"
    vol_to_mass = False
    profile_axis_out_units = "km"
    pearson_r_threshold_high = 0.85
    pearson_r_threshold_low = 0.50

    if y_lim_min is not None:
        y_lim_min = float(y_lim_min)
    if y_lim_max is not None:
        y_lim_max = float(y_lim_max)

    if varname not in var_dict.keys():
        print(f"variable {varname} not supported")

    na = len(args)
    shade = 0.15
    colormap = options.colormap
    # FIXME: make option
    cstride = 2
    my_colors = plt.get_cmap(colormap).colors[::cstride]

    alpha = 0.75
    dash_style = "o"
    numpoints = 1
    legend_frame_width = 0.25
    markeredgewidth = 0.15
    markersize = 4
    markeredgecolor = "k"
    obscolor = "0.4"

    flow_types = {0: "isbr{\\ae}", 1: "ice-stream", 2: "undefined"}
    glacier_types = {0: "ffmt", 1: "lvmt", 2: "ist", 3: "lt"}

    flux_gates = setup_flux_gates()

    # If observations are provided, load observations
    if obs_file:
        obs = ObservationsDataset(obs_file, varname)
        for flux_gate in flux_gates:
            flux_gate.add_observations(obs)
        del obs

    # Add experiments to flux gates
    for k, filename in enumerate(args):
        m_id = k
        experiment = ExperimentDataset(m_id, filename, varname)
        for flux_gate in flux_gates:
            flux_gate.add_experiment(experiment)

        del experiment

    ne = len(flux_gates[0].experiments)
    ng = len(flux_gates)

    # make figure for each flux gate
    for gate in flux_gates:
        if make_figures:
            gate.make_line_plot(label_param_list=label_params)
        else:
            if not gate.has_fluxes:
                gate.calculate_fluxes()
            if gate.has_observations:
                gate.calculate_stats()

    if obs_file:
        # write rmsd and pearson r tables per gate
        for gate in flux_gates:
            gate_name = "_".join([unidecode(gate.gate_name), "pearson_r", varname])
            outname = ".".join([gate_name, "csv"]).replace(" ", "_")
            ids = sorted(gate.p_ols, key=lambda x: gate.corr[x], reverse=True)
            corrs = [gate.corr[x] for x in ids]
            corrs_dict = dict(zip(ids, corrs))
            export_csv_from_dict(outname, corrs_dict, header="id,correlation")
        # write rmsd and person r figure per experiment
        for exp in flux_gates[0].experiments:
            exp_str = "_".join(["pearson_r_experiment", str(exp.m_id), varname])
            outname = ".".join([exp_str, "pdf"])
            corrs = make_correlation_figure_sorted(outname, exp)
            exp_str = "_".join(["coors_experiment", str(exp.m_id), varname])
            outname = os.path.join(odir, ".".join([exp_str, "csv"]))
            export_csv_from_dict(outname, corrs, header="id,correlation")
        # outname = os.path.join(odir, "corrs.pdf")
        # make_correlation_figure()
        experiments_df = []
        glaciers_above_threshold = []
        lengths = [gate.length() for gate in flux_gates]
        for exp in range(ne):
            names = [gate.gate_name for gate in flux_gates]
            corrs = [gate.corr[exp] for gate in flux_gates]
            rmsds = [gate.rmsd[exp] for gate in flux_gates]
            Ns = [gate.N_rmsd[exp] for gate in flux_gates]
            d = {
                "name": names,
                "correlation": corrs,
                "rmsd": rmsds,
                "N": Ns,
                "length": lengths,
            }
            df = pa.DataFrame(d)
            # Select glaciers with correlation above threshold
            # df = df[df["correlation"] > pearson_r_threshold_high]
            exp_rmsd_cum = np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            experiments_df.append(df)

            no_glaciers_above_threshold = len(
                df[df["correlation"] > pearson_r_threshold_high]
            )

            print(f"Experiment {exp}")
            print(
                "  Number of glaciers with r(all) > {}: {}".format(
                    pearson_r_threshold_high, no_glaciers_above_threshold
                )
            )
            glaciers_above_threshold.append(no_glaciers_above_threshold)
            rmsdd = np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            corr_median = df["correlation"].median()
            print(f"  RMS difference {rmsdd:4.0f}")
            print(f"  median(pearson r(all): {corr_median:1.2f}")

        # Calculate cumulative values of rms differences of all glaciers together
        rmsd_cum_dict = {}
        keys = range(ne)
        rmsd_cum = [
            np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            for df in experiments_df
        ]
        rmsd_cum_dict = dict(zip(keys, rmsd_cum))
        rmsd_cum_dict_sorted = sorted(
            iter(rmsd_cum_dict.items()), key=operator.itemgetter(1)
        )
        outname = os.path.join(
            odir, ".".join(["rmsd_sorted_{}".format(varname), "csv"])
        )
        print(f"  - saving {outname}")
        export_csv_from_dict(outname, dict(rmsd_cum_dict_sorted), header="id,rmsd")

        corr = [df["correlation"].median() for df in experiments_df]
        corr_dict = dict(zip(keys, corr))
        corr_dict_sorted = sorted(
            iter(corr_dict.items()), key=operator.itemgetter(1), reverse=True
        )

        outname = os.path.join(
            odir, ".".join(["pearson_r_sorted_{}".format(varname), "csv"])
        )
        print(("  - saving {0}".format(outname)))
        export_csv_from_dict(outname, dict(corr_dict_sorted), header="id,correlation")

        glaciers_dict = dict(zip(keys, glaciers_above_threshold))
        glaciers_dict_sorted = sorted(
            iter(glaciers_dict.items()), key=operator.itemgetter(1), reverse=True
        )

        outname = os.path.join(
            odir, ".".join(["glaciers_above_threshold_{}".format(varname), "csv"])
        )
        print(("  - saving {0}".format(outname)))
        export_csv_from_dict(
            outname,
            dict(glaciers_dict_sorted),
            header="id,no_glaciers",
            fmt=["%i", "%i"],
        )

    gate = flux_gates[0]
    # make a global regression figure
    if do_regress:
        make_regression(gate)
