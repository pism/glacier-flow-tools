# Copyright (C) 2024 Andy Aschwanden, Greg Guillet
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
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
Module for calculating Gaussian Random Fields
"""

from typing import Any, Callable, Dict, Union

import numpy as np
import xarray as xr


def distrib_normal(da: Union[xr.DataArray, np.ndarray], sigma: float = 1.0, seed: int = 0, n: float = 1.0):
    """
    Generates a complex normal distribution
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


def power_spectrum(field: np.ndarray, n: float) -> np.ndarray:
    """
    Generate a power spectrum
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
    Generates a field given a statistic and a power_spectrum.
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
