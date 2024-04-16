# Copyright (C) 2024 Andy Aschwanden
#
# This file is part of glacier-flow-tools.
#
# GLACIER-FLOW-TOOLS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# GLACIER-FLOW-TOOLS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Tests for Gaussian Random Fields.
"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from glacier_flow_tools.gaussian_random_fields import (
    distrib_normal,
    generate_field,
    power_spectrum,
)


@pytest.fixture(name="field_numpy")
def fixture_create_field() -> np.ndarray:
    """
    Create test data as numpy.ndarray.
    """

    x = y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X**2.0 + Y**2.0))


@pytest.fixture(name="field_datarray")
def fixture_create_field_da() -> xr.DataArray:
    """
    Create test data as xarray.DataArray.
    """

    x = y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    field = np.exp(-(X**2.0 + Y**2.0))
    return xr.DataArray(
        field,
        dims=["x", "y"],
        coords={"x": x, "y": y},
        name="power",
    )


def test_power_spectrum(field_numpy, field_datarray):
    """
    Test power spectrum function
    """

    true_n_1 = np.array(
        [
            [7.3890561, 3.49034296, 2.71828183, 3.49034296, 7.3890561],
            [3.49034296, 1.64872127, 1.28402542, 1.64872127, 3.49034296],
            [2.71828183, 1.28402542, 1.0, 1.28402542, 2.71828183],
            [3.49034296, 1.64872127, 1.28402542, 1.64872127, 3.49034296],
            [7.3890561, 3.49034296, 2.71828183, 3.49034296, 7.3890561],
        ]
    )

    ps = power_spectrum(field_numpy, 1.0)
    assert_array_almost_equal(ps, true_n_1, decimal=4)

    true_n_2 = np.array(
        [
            [54.59815003, 12.18249396, 7.3890561, 12.18249396, 54.59815003],
            [12.18249396, 2.71828183, 1.64872127, 2.71828183, 12.18249396],
            [7.3890561, 1.64872127, 1.0, 1.64872127, 7.3890561],
            [12.18249396, 2.71828183, 1.64872127, 2.71828183, 12.18249396],
            [54.59815003, 12.18249396, 7.3890561, 12.18249396, 54.59815003],
        ]
    )

    ps = power_spectrum(field_numpy, 2.0)
    assert_array_almost_equal(ps, true_n_2, decimal=3)

    ps = power_spectrum(field_datarray, 2.0)
    assert_array_almost_equal(ps, true_n_2, decimal=3)


def test_distrib_normal(field_numpy, field_datarray):
    """
    Test complex normal distribution
    """

    complex_normal_dist_sigma_1_n_1_seed_0 = np.array(
        [
            [
                0.01701574 + 0.01272318j,
                -0.03784868 - 0.2130161j,
                0.23559833 - 0.33908382j,
                0.03005439 - 0.13114064j,
                -0.07249497 + 0.02980017j,
            ],
            [
                0.10359872 - 0.28926045j,
                0.79091601 - 0.1268714j,
                0.7375874 - 0.12400456j,
                -0.426837 + 0.32803943j,
                -0.36254932 + 0.06150087j,
            ],
            [
                -0.22928986 + 0.13073431j,
                0.03218471 - 0.50920223j,
                -2.32503077 - 0.12961363j,
                -0.17039512 + 0.61056071j,
                -0.45834502 + 0.54940262j,
            ],
            [
                -0.20979811 - 0.36072831j,
                -0.33010976 + 0.91824119j,
                -0.24633481 + 1.04816883j,
                0.24966654 + 0.47388932j,
                0.29868508 + 0.07576781j,
            ],
            [
                -0.01739528 - 0.04248483j,
                0.39149834 + 0.41772992j,
                -0.24471144 + 0.72113873j,
                0.10070932 + 0.51617703j,
                0.12227139 + 0.17797994j,
            ],
        ]
    )

    complex_normal_dist_sigma_1_n_1_seed_1 = np.array(
        [
            [
                0.04676973 - 0.25565014j,
                0.23539754 - 0.05007304j,
                0.12156101 - 0.15531517j,
                -0.3733608 + 0.06120974j,
                0.12252659 + 0.02941133j,
            ],
            [
                0.12788846 + 0.60677096j,
                -0.3256786 - 0.67447469j,
                0.45257523 - 0.29407908j,
                0.22112434 + 1.23900361j,
                0.08427037 + 0.18528351j,
            ],
            [
                0.01045596 + 0.24392738j,
                0.4257805 - 0.40030856j,
                -0.73645409 - 1.64807517j,
                -0.1268744 + 0.13042167j,
                -0.17736178 + 0.04010404j,
            ],
            [
                0.17157231 - 0.35164225j,
                0.02409268 - 0.41439792j,
                -0.22776555 - 0.05610767j,
                -0.47425146 - 0.57302083j,
                -0.07368681 - 0.02815482j,
            ],
            [
                0.00110192 + 0.01292222j,
                -0.07896155 + 0.01019563j,
                0.47605947 - 0.18625429j,
                0.28843135 + 0.17011167j,
                -0.36691594 + 0.12060633j,
            ],
        ]
    )

    complex_normal_dist_sigma_5_n_2_seed_0 = np.array(
        [
            [
                5.75707332e-02 + 0.04304738j,
                -2.71095688e-01 - 1.52575337j,
                2.16679452e00 - 3.11854912j,
                2.15268149e-01 - 0.93931059j,
                -2.45278170e-01 + 0.10082536j,
            ],
            [
                7.42038240e-01 - 2.07186268j,
                1.19928702e01 - 1.92378484j,
                1.43608410e01 - 2.41437126j,
                -6.47224313e00 + 4.97414929j,
                -2.59680299e00 + 0.44050734j,
            ],
            [
                -2.10877565e00 + 1.20236166j,
                6.26636838e-01 - 9.91417745j,
                -5.81257694e01 - 3.24034084j,
                -3.31759631e00 + 11.88762898j,
                -4.21539277e00 + 5.05284818j,
            ],
            [
                -1.50270412e00 - 2.58375981j,
                -5.00554226e00 + 13.92353581j,
                -4.79614356e00 + 20.40786772j,
                3.78576029e00 + 7.18571004j,
                2.13936771e00 + 0.542696j,
            ],
            [
                -5.88548618e-02 - 0.14374242j,
                2.80415380e00 + 2.99204065j,
                -2.25060774e00 + 6.63230286j,
                7.21342590e-01 + 3.6971799j,
                4.13690840e-01 + 0.60217414j,
            ],
        ]
    )

    result = distrib_normal(field_numpy, sigma=1.0, n=1.0, seed=0)
    assert_array_almost_equal(result, complex_normal_dist_sigma_1_n_1_seed_0)

    result = distrib_normal(field_datarray, sigma=1.0, n=1.0, seed=0)
    assert_array_almost_equal(result, complex_normal_dist_sigma_1_n_1_seed_0)

    result = distrib_normal(field_numpy, sigma=1.0, n=1.0, seed=1)
    assert_array_almost_equal(result, complex_normal_dist_sigma_1_n_1_seed_1)

    result = distrib_normal(field_numpy, sigma=5.0, n=2.0, seed=0)
    assert_array_almost_equal(result, complex_normal_dist_sigma_5_n_2_seed_0)


def test_gaussian_random_field():
    """
    Test Gaussian Random Fields
    """

    field = np.array(
        [
            [0.13533528, 0.2865048, 0.36787944, 0.2865048, 0.13533528],
            [0.2865048, 0.60653066, 0.77880078, 0.60653066, 0.2865048],
            [0.36787944, 0.77880078, 1.0, 0.77880078, 0.36787944],
            [0.2865048, 0.60653066, 0.77880078, 0.60653066, 0.2865048],
            [0.13533528, 0.2865048, 0.36787944, 0.2865048, 0.13533528],
        ]
    )

    result_true = np.array(
        [
            [0.66941416, -0.10928725, -0.02762629, -0.02762629, -0.10928725],
            [-0.10928725, -0.00353034, -0.01745481, -0.01224363, 0.0333994],
            [-0.02762629, -0.01745481, -0.016339, -0.01501318, -0.01224363],
            [-0.02762629, -0.01224363, -0.01501318, -0.016339, -0.01745481],
            [-0.10928725, 0.0333994, -0.01224363, -0.01745481, -0.00353034],
        ]
    )

    result = generate_field(field, power_spectrum)
    assert_array_almost_equal(result, result_true)

    result_true = np.array(
        [
            [1.77697673, -0.00231109, -0.15587765, -0.15587765, -0.00231109],
            [-0.00231109, -0.01493004, -0.11896727, -0.08750106, 0.09666002],
            [-0.15587765, -0.11896727, -0.12653393, -0.1143703, -0.08750106],
            [-0.15587765, -0.08750106, -0.1143703, -0.12653393, -0.11896727],
            [-0.00231109, 0.09666002, -0.08750106, -0.11896727, -0.01493004],
        ]
    )

    result = generate_field(field, power_spectrum, 2.0)
    assert_array_almost_equal(result, result_true)
