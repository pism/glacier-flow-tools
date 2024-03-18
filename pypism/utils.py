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

import contextlib
import re
from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
from matplotlib import colors


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """TQDM Callback"""

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def blend_multiply(rgb: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """
    Combine an RGB image with an intensity map using "overlay" blending.

    Parameters
    ----------
    rgb : `~numpy.ndarray`
        An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
        An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

    Returns
    -------
    ndarray
        An (M, N, 3) RGB array representing the combined images.
    """

    alpha = rgb[..., -1, np.newaxis]
    img_scaled = np.clip(rgb[..., :3] * intensity, 0.0, 1.0)
    return img_scaled * alpha + intensity * (1.0 - alpha)


def qgis2cmap(
    filename: Union[Path, str], N: int = 256, name: str = "my colormap"
) -> colors.LinearSegmentedColormap:
    """
    Reads a colormaps exported from QGIS rasters layers and
    returns a matplotlib.colors.LinearSegmentedColormap

    Parameters
    ----------
    filename : str or Path
      The path to the QGIS colormap.
    N : int
        The number of RGB quantization levels.
    name : str
        Name of the colormap.

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


def preprocess_nc(
    ds,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    drop_vars: Union[List[str], None] = None,
    drop_dims: List[str] = ["nv4"],
):
    """
    Add experiment 'exp_id'
    """
    m_id_re = re.search(regexp, ds.encoding["source"])
    ds.expand_dims(dim)
    assert m_id_re is not None
    m_id: Union[str, int]
    try:
        m_id = int(m_id_re.group(1))
    except:
        m_id = str(m_id_re.group(1))
    ds[dim] = m_id
    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(
        drop_dims, errors="ignore"
    )
