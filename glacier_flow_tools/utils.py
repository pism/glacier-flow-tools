# Copyright (C) 2023 Andy Aschwanden
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
Module provides utility functions that do not fit anywhere else.
"""

import contextlib
import re
from pathlib import Path
from typing import (  # pylint: disable=deprecated-class
    Callable,
    Hashable,
    Iterable,
    List,
    Union,
)

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.

    Parameters
    ----------

    tqdm_object : object
      TQDM Object.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """
        TQDM Callback.
        """

        def __call__(self, *args, **kwargs) -> None:
            """
            Update the tqdm object and call the superclass's __call__ method.

            This method updates the tqdm object with the batch size and then calls the __call__ method of the superclass.

            Parameters
            ----------
            *args
                Variable length argument list.
            **kwargs
                Arbitrary keyword arguments.

            Returns
            -------
            None
              Return None.
            """
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

    This function combines an RGB image with an intensity map using "overlay" blending. The RGB image
    and the intensity map are combined by multiplying the RGB values by the intensity values. The resulting
    image is then scaled to have values between 0 and 1.

    Parameters
    ----------
    rgb : np.ndarray
        An (M, N, 3) RGB array of floats ranging from 0 to 1. This represents the color image.
    intensity : np.ndarray
        An (M, N, 1) array of floats ranging from 0 to 1. This represents the grayscale image.

    Returns
    -------
    np.ndarray
        An (M, N, 3) RGB array representing the combined images. The values in the array range from 0 to 1.
    """

    alpha = rgb[..., -1, np.newaxis]
    img_scaled = np.clip(rgb[..., :3] * intensity, 0.0, 1.0)
    return img_scaled * alpha + intensity * (1.0 - alpha)


def preprocess_nc(
    ds: xr.Dataset,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    drop_vars: Union[str, Iterable[Hashable], Callable[[xr.Dataset], Union[str, Iterable[Hashable]]]] = ["nv4"],
    drop_dims: List[str] = ["nv4"],
) -> xr.Dataset:
    """
    Add experiment 'exp_id' to the dataset and drop specified variables and dimensions.

    This function adds an experiment id ('exp_id') to the dataset, extracted from the source encoding
    using the provided regular expression. It then drops the specified variables and dimensions from the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to be preprocessed.
    regexp : str, optional
        The regular expression used to extract the experiment id from the source encoding, by default "id_(.+?)_".
    dim : str, optional
        The name of the dimension to be added to the dataset, by default "exp_id".
    drop_vars : Union[List[str], None], optional
        The variables to be dropped from the dataset, by default None.
    drop_dims : List[str], optional
        The dimensions to be dropped from the dataset, by default ["nv4"].

    Returns
    -------
    xr.Dataset
        The preprocessed dataset.
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
    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def merge_on_intersection(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two pandas DataFrames on intersection keys.

    This function merges two pandas DataFrames based on the intersection of their columns.
    The intersection of the columns is used as the keys for the merge operation.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame to be merged.
    df2 : pd.DataFrame
        The second DataFrame to be merged.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.
    """
    intersection_keys = list(set(df1.columns) & set(df2.columns))
    return pd.merge(df1, df2, on=intersection_keys)


def qgis2cmap(filename: Union[Path, str], N: int = 256, name: str = "my colormap") -> colors.LinearSegmentedColormap:
    """
    Read a colormap exported from QGIS raster layers and returns a matplotlib.colors.LinearSegmentedColormap.

    This function reads a colormap file exported from QGIS raster layers and converts it into a
    matplotlib colormap. The colormap is quantized into `N` levels.

    Parameters
    ----------
    filename : Union[Path, str]
        The path to the QGIS colormap file.
    N : int, optional
        The number of RGB quantization levels, by default 256.
    name : str, optional
        The name of the colormap, by default "my colormap".

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The matplotlib colormap.
    """
    m_data = np.loadtxt(filename, skiprows=2, delimiter=",")[:, :-1]
    values_scaled = (m_data[:, 0] - np.min(m_data[:, 0])) / (np.max(m_data[:, 0]) - np.min(m_data[:, 0]))
    colors_scaled = m_data[:, 1::] / 255.0
    m_colors = [(values_scaled[k], colors_scaled[k]) for k in range(len(values_scaled))]
    cmap = colors.LinearSegmentedColormap.from_list(name, m_colors, N=N)

    return cmap
