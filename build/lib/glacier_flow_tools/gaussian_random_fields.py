# Copyright (C) 2024 Andy Aschwanden, Greg Guillet
#
# This file is part of glacier-flow-tools.
#
# GLACIER-FLOW-TOOLS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-RAGIS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Module for calculating Gaussian Random Fields.
"""

from typing import Any, Callable, Dict, Union

import numpy as np
import xarray as xr


def distrib_normal(
    da: Union[xr.DataArray, np.ndarray], sigma: float = 1.0, seed: int = 0, n: float = 1.0
) -> Union[xr.DataArray, np.ndarray]:
    """
    Generate complex normal distribution.

    Generate complex normal distribution from numpy.ndarray or xarray.DataArray.

    Parameters
    ----------
    da : Union[xr.DataArray, np.ndarray]
        Data array.
    sigma : float
        Standard deviation.
    seed : int
        Seed to np.randdom.default_rng.
    n : float
        Power.

    Returns
    -------

    Union[xr.DataArray, np.ndarray]
        Complex normal distribution.

    Examples
    --------
    >>>    da = np.array([[0., 1.],[1.0, 0.]])
    >>>    distrib_normal(da)
    >>>    array([[ 0.        +0.j        , -0.13210486+0.36159505j],
    >>>    [ 0.64042265+1.30400005j,  0.        +0.j        ]])
    """

    rng = np.random.default_rng(seed=seed)
    a = rng.normal(
        loc=0,
        scale=(sigma * da) ** n,
    )
    b = rng.normal(
        loc=0,
        scale=(sigma * da) ** n,
    )
    return a + 1j * b


def power_spectrum(field: Union[xr.DataArray, np.ndarray], n: float) -> Union[xr.DataArray, np.ndarray]:
    """
    Generate a power spectrum.

    Generate a power spectrum for a given field.

    Parameters
    ----------
    field : xr.DataArray or np.ndarray
        Field.
    n : float
        Power.

    Returns
    -------

    xr.DataArray or np.ndarray
        Power spectrum.

    Examples
    --------

    >>>    da = np.array([[0., 1.],[1.0, 0.]])
    >>>    power_spectrum(da)
    >>>    array([[inf,  1.],
    >>>    [ 1., inf]])
    """
    return np.power(field, -n)


def generate_field(
    fftfield: Union[np.ndarray, xr.DataArray],
    spectrum_function: Callable,
    n: float = 1.0,
    unit_length: float = 1,
    fft: Any = np.fft,
    fft_args: Dict[str, Any] = {},
) -> np.ndarray:
    """
    Generate a field given a statistic and a power_spectrum.

    Generate a field given a statistic and a power spectrum.

    Parameters
    ----------
    fftfield : Union[np.ndarray, xr.DataArray]
        Field to transform.
    spectrum_function : Callable
        A spectrum function.
    n : float
        Power.
    unit_length : float
        Unit length.
    fft : Any
        FFT function.
    fft_args : Dict[str, Any]
        Args to fft.

    Returns
    -------
    np.ndarray
        Returns a FFT array.

    Examples
    --------
    FIXME: Add docs.
    """

    try:
        fftfreq = fft.fftfreq
    except AttributeError:
        # Fallback on numpy for the frequencies
        fftfreq = np.fft.fftfreq
    else:
        fftfreq = fft.fftfreq

    # Compute the k grid
    shape = fftfield.shape
    all_k = [fftfreq(s, d=unit_length) for s in shape]

    kgrid = np.meshgrid(*all_k, indexing="ij")
    knorm = np.hypot(*kgrid)

    power_k = np.zeros_like(knorm)
    mask = knorm > 0
    power_k[mask] = np.sqrt(spectrum_function(knorm[mask], n))
    fftfield *= power_k

    return np.real(fft.ifftn(fftfield, **fft_args))
