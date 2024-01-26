# Copyright (C) 2023 Andy Aschwanden
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
Module provides utility functions that do not fit anywhere else.
"""

from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import colors


def qgis2cmap(filename: Union[Path, str], N: int = 256, name: str = "my colormap"):
    """
    Reads a colormaps exported from QGIS rasters layers and
    returns a matplotlib.colors.LinearSegmentedColormap

    Parameters
    ----------
    filename : str or Path
      The path to the QGIS colormap.
    N : int
        The number of RGB quantization levels.

    Returns
    ----------
    cmap: matplotlib.colors.LinearSegmentedColormap
        Matplotlib colormap
    """
    m_data = np.loadtxt(filename, skiprows=2, delimiter=",")[:, :-1]
    values_scaled = (m_data[:, 0] - np.min(m_data[:, 0])) / (
        np.max(m_data[:, 0]) - np.min(m_data[:, 0])
    )
    colors_scaled = m_data[:, 1::] / 255.0
    m_colors = [(values_scaled[k], colors_scaled[k]) for k in range(len(values_scaled))]
    cmap = colors.LinearSegmentedColormap.from_list(name, m_colors, N=N)

    return cmap
