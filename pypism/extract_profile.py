#!/usr/bin/env python
# Copyright (C) 2015, 2016, 2018, 2021, 2023 Constantine Khroulev and Andy Aschwanden
#

# nosetests --with-coverage --cover-branches --cover-html
# --cover-package=extract_profiles scripts/extract_profiles.py

# pylint -d C0301,C0103,C0325,W0621
# --msg-template="{path}:{line}:[{msg_id}({symbol}), {obj}] {msg}"
# extract_profiles.py > lint.txt

"""This script containts tools for extracting 'profiles', that is
sampling 2D and 3D fields on a regular grid at points along a flux
gate or a any kind of profile.
"""

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Union

import numpy as np
from osgeo import gdal, ogr, osr

from .interpolation import InterpolationMatrix

gdal.UseExceptions()

# from pyproj import Proj


profiledim = "profile"
stationdim = "station"


def normal(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """Compute the unit normal vector orthogonal to (point1-point0),
    pointing 'to the right' of (point1-point0).

    """

    a = point0 - point1
    if a[1] != 0.0:
        n = np.array([1.0, -a[0] / a[1]])
        n = n / np.linalg.norm(n)  # normalize
    else:
        n = np.array([0, 1])

    # flip direction if needed:
    if np.cross(a, n) < 0:
        n = -1.0 * n

    return n


def tangential(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """Compute the unit tangential vector to (point1-point0),
    pointing 'to the right' of (point1-point0).

    """

    a = point1 - point0
    norm = np.linalg.norm(a)
    # protect from division by zero
    if norm > 0.0:
        return a / norm
    else:
        return a


class Profile:

    """Collects information about a profile, that is a sequence of points
    along a flux gate or a flightline.

    """

    def __init__(
        self,
        profile_id: int,
        name: str,
        lat: Union[float, np.ndarray, list],
        lon: Union[float, np.ndarray, list],
        center_lat: float,
        center_lon: float,
        flightline: int,
        glaciertype: int,
        flowtype: int,
        projection,
        flip: bool = False,
    ):
        self.profile_id = profile_id
        self.name = name
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.flightline = flightline
        self.glaciertype = glaciertype
        self.flowtype = flowtype

        if isinstance(lon, float):
            lon = [lon]
        else:
            lon[0]

        if isinstance(lat, float):
            lat = [lat]
        else:
            lat[0]

        assert len(lon) == len(lat)

        if flip:
            self.lat = lat[::-1]
            self.lon = lon[::-1]
        else:
            self.lat = lat
            self.lon = lon
        self.x, self.y = projection(lon, lat)

        self.distance_from_start = self._distance_from_start()
        self.nx, self.ny = self._compute_normals()
        self.tx, self.ty = self._compute_tangentials()

    def _compute_normals(self):
        """
        Compute normals to a flux gate described by 'p'. Normals point 'to
        the right' of the path.
        """

        p = np.vstack((self.x, self.y)).T

        if len(p) < 2:
            return [0], [0]

        ns = np.zeros_like(p)
        ns[0] = normal(p[0], p[1])
        for j in range(1, len(p) - 1):
            ns[j] = normal(p[j - 1], p[j + 1])

        ns[-1] = normal(p[-2], p[-1])

        return ns[:, 0], ns[:, 1]

    def _compute_tangentials(self):
        """
        Compute tangetials to a flux gate described by 'p'.
        """

        p = np.vstack((self.x, self.y)).T

        if len(p) < 2:
            return [0], [0]

        ts = np.zeros_like(p)
        ts[0] = tangential(p[0], p[1])
        for j in range(1, len(p) - 1):
            ts[j] = tangential(p[j - 1], p[j + 1])

        ts[-1] = tangential(p[-2], p[-1])

        return ts[:, 0], ts[:, 1]

    def _distance_from_start(self):
        "Initialize the distance along a profile."
        result = np.zeros_like(self.x)
        result[1::] = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2)
        return result.cumsum()


def load_profiles(filename, projection, flip):
    """Load profiles from a file filename.

    Parameters
    -----------
    filename: filename of ESRI shape file

    projection: proj4 projection object. (lon,lat) coordinates of
                points along a profile are converted to (x,y)
                coordinates in this projection. This should be the
                projection used by the dataset we're extracting
                profiles from.
    flip: boolean; set to True to flip profile directions

    Returns
    -------
    list of profiles with
    """
    profiles = []
    for (
        lat,
        lon,
        profile_id,
        name,
        clat,
        clon,
        flightline,
        glaciertype,
        flowtype,
    ) in read_shapefile(filename):
        p = Profile(
            profile_id,
            name,
            lat,
            lon,
            clat,
            clon,
            flightline,
            glaciertype,
            flowtype,
            projection,
            flip,
        )
        profiles.append(p)

    return profiles


def output_dimensions(input_dimensions, profile=True):
    """Build a list of dimension names used to define a variable in the
    output file."""

    _, _, zdim, tdim = get_dims_from_variable(input_dimensions)

    if tdim:
        result = [stationdim, tdim]
    else:
        result = [stationdim]

    if profile:
        result.append(profiledim)

    if zdim:
        result.append(zdim)

    return result


def read_shapefile(filename):
    """
    Reads lat / lon from a vector file

    Paramters
    ----------
    filename: filename of shape file.

    Returns
    -------
    lat, lon: array_like coordinates

    """

    ds = gdal.OpenEx(filename, 0)
    layer = ds.GetLayer(0)
    layer_type = ogr.GeometryTypeToName(layer.GetGeomType())
    srs = layer.GetSpatialRef()
    if not srs.IsGeographic():
        # Create spatialReference (lonlat)
        srs_geo = osr.SpatialReference()
        srs_geo.ImportFromProj4("+proj=latlon")
    profiles = []

    if layer_type == "Point":
        lon = []
        lat = []
        for pt, feature in enumerate(layer):
            feature = layer.GetFeature(pt)

            if hasattr(feature, "id"):
                profile_id = feature.id
            else:
                profile_id = str(pt)
            try:
                try:
                    name = feature.name
                except:
                    name = feature.Name
            except:
                name = str(pt)
            try:
                flightline = feature.flightline
            except:
                flightline = 2
            try:
                glaciertype = feature.gtype
            except:
                glaciertype = 4
            try:
                flowtype = feature.ftype
            except:
                flowtype = 2
            geometry = feature.GetGeometryRef()
            # Transform to latlon if needed
            if not srs.IsGeographic():
                geometry.TransformTo(srs_geo)

            point = geometry.GetPoint()
            lon.append(point[0])
            lat.append(point[1])

            try:
                clon = feature.clon
            except:
                clon = point[0]
            try:
                clat = feature.clat
            except:
                clat = point[1]

            profiles.append(
                [
                    lat,
                    lon,
                    profile_id,
                    name,
                    clat,
                    clon,
                    flightline,
                    glaciertype,
                    flowtype,
                ]
            )

    elif layer_type in ("Line String", "Multi Line String"):
        for pt, feature in enumerate(layer):
            if hasattr(feature, "id"):
                profile_id = feature.id
            else:
                profile_id = str(pt)
            if feature.name is None:
                name = "unnamed"
            else:
                try:
                    name = feature.name
                except:
                    name = "unnamed"
            try:
                clon = feature.clon
            except:
                clon = 0.0
            try:
                clat = feature.clat
            except:
                clat = 0.0
            try:
                flightline = feature.flightline
            except:
                flightline = 2
            if flightline is None:
                flightline = 2
            try:
                glaciertype = feature.gtype
            except:
                glaciertype = 4
            if glaciertype is None:
                glaciertype = 4
            try:
                flowtype = feature.ftype
            except:
                flowtype = 2
            if flowtype is None:
                flowtype = 2
            geometry = feature.GetGeometryRef()
            # Transform to latlon if needed
            if not srs.IsGeographic():
                geometry.TransformTo(srs_geo)
            lons: list = []
            lats: list = []
            for i in range(0, geometry.GetPointCount()):
                # GetPoint returns a tuple not a Geometry
                m_pt = geometry.GetPoint(i)
                lons.append(m_pt[0])
                lats.append(m_pt[1])
            # skip features with less than 2 points:
            if len(lats) > 1:
                profiles.append(
                    [
                        lats,
                        lons,
                        profile_id,
                        name,
                        clat,
                        clon,
                        flightline,
                        glaciertype,
                        flowtype,
                    ]
                )
    else:
        raise NotImplementedError(
            "Geometry type '{0}' is not supported".format(layer_type)
        )
    return profiles


def get_dims_from_variable(var_dimensions):
    """
    Gets dimensions from netcdf variable

    Parameters:
    -----------
    var: netCDF variable

    Returns:
    --------
    xdim, ydim, zdim, tdim: dimensions
    """

    def find(candidates, collection):
        """Return one of the candidates if it was found in the collection or
        None otherwise.

        """
        for name in candidates:
            if name in collection:
                return name
        return None

    # possible x-dimensions names
    xdims = ["x", "x1"]
    # possible y-dimensions names
    ydims = ["y", "y1"]
    # possible z-dimensions names
    zdims = ["z", "zb"]
    # possible time-dimensions names
    tdims = ["t", "time"]

    return [find(dim, var_dimensions) for dim in [xdims, ydims, zdims, tdims]]


def define_station_variables(nc):
    "Define variables used to store information about a station."
    # create dimensions
    nc.createDimension(stationdim)

    variables = [
        (
            "station_name",
            str,
            (stationdim,),
            {"cf_role": "timeseries_id", "long_name": "station name"},
        ),
        (
            "lon",
            "f",
            (stationdim,),
            {
                "units": "degrees_east",
                "valid_range": [-180.0, 180.0],
                "standard_name": "longitude",
            },
        ),
        (
            "lat",
            "f",
            (stationdim,),
            {
                "units": "degrees_north",
                "valid_range": [-90.0, 90.0],
                "standard_name": "latitude",
            },
        ),
    ]

    print("Defining station variables...")
    for name, datatype, dimensions, attributes in variables:
        variable = nc.createVariable(name, datatype, dimensions)
        variable.setncatts(attributes)
    print("done.")


def define_profile_variables(nc, special_vars=False):
    "Define variables used to store information about profiles."
    # create dimensions
    nc.createDimension(profiledim)
    nc.createDimension(stationdim)

    if special_vars:
        variables = [
            ("profile_id", "i", (stationdim), {"long_name": "profile id"}),
            (
                "profile_name",
                str,
                (stationdim),
                {"cf_role": "timeseries_id", "long_name": "profile name"},
            ),
            (
                "profile_axis",
                "f",
                (stationdim, profiledim),
                {"long_name": "distance along profile", "units": "m"},
            ),
            (
                "clon",
                "f",
                (stationdim),
                {
                    "long_name": "center longitude of profile",
                    "units": "degrees_east",
                    "valid_range": [-180.0, 180.0],
                },
            ),
            (
                "clat",
                "f",
                (stationdim),
                {
                    "long_name": "center latitude of profile",
                    "units": "degrees_north",
                    "valid_range": [-90.0, 90.0],
                },
            ),
            (
                "lon",
                "f",
                (stationdim, profiledim),
                {
                    "units": "degrees_east",
                    "valid_range": [-180.0, 180.0],
                    "standard_name": "longitude",
                },
            ),
            (
                "lat",
                "f",
                (stationdim, profiledim),
                {
                    "units": "degrees_north",
                    "valid_range": [-90.0, 90.0],
                    "standard_name": "latitude",
                },
            ),
            (
                "flightline",
                "b",
                (stationdim),
                {
                    "long_name": "flightline (true/false/undetermined) integer mask",
                    "flag_values": np.array([0, 1, 2], dtype=np.byte),
                    "flag_meanings": "true false undetermined",
                    "valid_range": np.array([0, 2], dtype=np.byte),
                },
            ),
            (
                "flowtype",
                "b",
                (stationdim),
                {
                    "long_name": "fast-flow type (isbrae/ice-stream) integer mask after Truffer and Echelmeyer (2003)",
                    "flag_values": np.array([0, 1, 2], dtype=np.byte),
                    "flag_meanings": "isbrae ice_stream undetermined",
                    "valid_range": np.array([0, 2], dtype=np.byte),
                },
            ),
            (
                "glaciertype",
                "b",
                (stationdim),
                {
                    "long_name": "glacier-type integer mask",
                    "comment": "glacier-type categorization after Moon et al. (2012), Science, 10.1126/science.1219985",
                    "flag_values": np.array([0, 1, 2, 3, 4], dtype=np.byte),
                    "flag_meanings": "fast_flowing_marine_terminating low_velocity_marine_terminating ice_shelf_terminating land_terminating undetermined",
                    "valid_range": np.array([0, 4], dtype=np.byte),
                },
            ),
            (
                "nx",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "x-component of the right-hand-pointing normal vector",
                    "fill_value": -2.0e9,
                },
            ),
            (
                "ny",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "y-component of the right-hand-pointing normal vector",
                    "fill_value": -2.0e9,
                },
            ),
            (
                "tx",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "x-component of the unit tangential vector",
                    "fill_value": -2.0e9,
                },
            ),
            (
                "ty",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "y-component of the tangential vector",
                    "fill_value": -2.0e9,
                },
            ),
        ]
    else:
        variables = [
            ("profile_id", "i", (stationdim), {"long_name": "profile id"}),
            (
                "profile_name",
                str,
                (stationdim),
                {"cf_role": "timeseries_id", "long_name": "profile name"},
            ),
            (
                "profile_axis",
                "f",
                (stationdim, profiledim),
                {"long_name": "distance along profile", "units": "m"},
            ),
            (
                "lon",
                "f",
                (stationdim, profiledim),
                {
                    "units": "degrees_east",
                    "valid_range": [-180.0, 180.0],
                    "standard_name": "longitude",
                },
            ),
            (
                "lat",
                "f",
                (stationdim, profiledim),
                {
                    "units": "degrees_north",
                    "valid_range": [-90.0, 90.0],
                    "standard_name": "latitude",
                },
            ),
            (
                "nx",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "x-component of the right-hand-pointing normal vector",
                    "fill_value": -2.0e9,
                },
            ),
            (
                "ny",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "y-component of the right-hand-pointing normal vector",
                    "fill_value": -2.0e9,
                },
            ),
            (
                "tx",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "x-component of the unit tangential vector",
                    "fill_value": -2.0e9,
                },
            ),
            (
                "ty",
                "f",
                (stationdim, profiledim),
                {
                    "long_name": "y-component of the tangential vector",
                    "fill_value": -2.0e9,
                },
            ),
        ]

    print("Defining profile variables...")
    for name, datatype, dimensions, attributes in variables:
        variable = nc.createVariable(name, datatype, dimensions)
        variable.setncatts(attributes)
    print("done.")


def extract_profile(
    variable,
    profile,
    xdim: str = "x",
    ydim: str = "y",
    zdim: str = "z",
    tdim: str = "time",
):
    """Extract values of a variable along a profile."""
    x = variable.coords[xdim].to_numpy()
    y = variable.coords[ydim].to_numpy()

    dim_length = dict(list(zip(variable.dims, variable.shape)))

    def init_interpolation():
        """Initialize interpolation weights. Takes care of the transpose."""
        if variable.dims.index(ydim) < variable.dims.index(xdim):
            A = InterpolationMatrix(x, y, profile.x, profile.y)
            return A, slice(A.c_min, A.c_max + 1), slice(A.r_min, A.r_max + 1)
        else:
            A = InterpolationMatrix(y, x, profile.y, profile.x)
            return A, slice(A.r_min, A.r_max + 1), slice(A.c_min, A.c_max + 1)

    # try to get the matrix we (possibly) pre-computed earlier:
    try:
        # Check if we are extracting from the grid of the same shape
        # as before. This will make sure that we re-compute weights if
        # one variable is stored as (x,y) and a different as (y,x),
        # but will not catch grids that are of the same shape, but
        # with different extents and spacings. We'll worry about this
        # case later -- if we have to.
        if profile.grid_shape == variable.shape:
            A = profile.A
            x_slice = profile.x_slice
            y_slice = profile.y_slice
        else:
            A, x_slice, y_slice = init_interpolation()
    except AttributeError:
        A, x_slice, y_slice = init_interpolation()
        profile.A = A
        profile.x_slice = x_slice
        profile.y_slice = y_slice
        profile.grid_shape = variable.shape

    def return_indexes(indexes):
        return (*indexes,)

    def read_subset(t=0, z=0):
        """Assemble the indexing tuple and get a subset from a variable."""
        index = []
        indexes = {xdim: x_slice, ydim: y_slice, zdim: z, tdim: t}
        for dim in variable.dims:
            try:
                index.append(indexes[dim])
            except KeyError:
                index.append(Ellipsis)
            starred_index = return_indexes(index)
        return variable[starred_index]

    n_points = len(profile.x)

    if tdim in variable.coords and zdim in variable.coords:
        dim_names = ["time", "profile", "z"]
        result = np.zeros((dim_length[tdim], n_points, dim_length[zdim]))
        for j in range(dim_length[tdim]):
            for k in range(dim_length[zdim]):
                result[j, :, k] = A.apply_to_subset(read_subset(t=j, z=k))
    elif tdim in variable.coords:
        dim_names = ["time", "profile"]
        result = np.zeros((dim_length[tdim], n_points))
        for j in range(dim_length[tdim]):
            result[j, :] = A.apply_to_subset(read_subset(t=j))
    elif zdim in variable.coords:
        dim_names = ["profile", "z"]
        result = np.zeros((n_points, dim_length[zdim]))
        for k in range(dim_length[zdim]):
            result[:, k] = A.apply_to_subset(read_subset(z=k))
    else:
        dim_names = ["profile"]
        result = A.apply_to_subset(read_subset())

    return result, dim_names


def copy_attributes(var_in, var_out, attributes_not_copied=None):
    """Copy attributes from var_in to var_out. Give special treatment to
    _FillValue and coordinates.

    """
    _, _, _, tdim = get_dims_from_variable(var_in.dimensions)
    for att in var_in.ncattrs():
        if att not in attributes_not_copied:
            if att == "_FillValue":
                continue
            elif att == "coordinates":
                if tdim:
                    coords = "{0} lat lon".format(tdim)
                else:
                    coords = "lat lon"
                setattr(var_out, "coordinates", coords)

            else:
                setattr(var_out, att, getattr(var_in, att))


def copy_global_attributes(in_file, out_file):
    """Copy global attributes from in_file to out_file."""
    print("Copying global attributes...")
    for attribute in in_file.ncattrs():
        setattr(out_file, attribute, getattr(in_file, attribute))
    print("done.")


def copy_dimensions(in_file, out_file, exclude_list):
    """Copy dimensions from in_file to out_file, excluding ones in
    exclude_list."""
    print("Copying dimensions...")
    for name, dim in in_file.dimensions.items():
        if name not in exclude_list and name not in out_file.dimensions:
            if dim.isunlimited():
                out_file.createDimension(name, None)
            else:
                out_file.createDimension(name, len(dim))
    print("done.")


def create_variable_like(in_file, var_name, out_file, dimensions=None, fill_value=-2e9):
    """Create a variable in an out_file that is the same var_name in
    in_file, except possibly depending on different dimensions,
    provided in dimensions.

    """
    var_in = in_file.variables[var_name]
    try:
        fill_value = var_in._FillValue
    except AttributeError:
        # fill_value was set elsewhere
        pass

    if dimensions is None:
        dimensions = var_in.dimensions

    dtype = var_in.dtype

    var_out = out_file.createVariable(
        var_name, dtype, dimensions=dimensions, fill_value=fill_value
    )
    # Fix
    attributes_not_copied: list = []
    copy_attributes(var_in, var_out, attributes_not_copied=attributes_not_copied)
    return var_out


def copy_time_dimension(in_file, out_file, name):
    """Copy time dimension, the corresponding coordinate variable, and the
    corresponding time bounds variable (if present) from an in_file to
    an out_file.

    """
    var_in = in_file.variables[name]
    var_out = create_variable_like(in_file, name, out_file)
    var_out[:] = in_file.variables[name][:]

    try:
        bounds_name = var_in.bounds
        var_out = create_variable_like(bounds_name, in_file, out_file)
        var_out[:] = in_file.variables[bounds_name][:]
    except AttributeError:
        # we get here if var_in does not have a bounds attribute
        pass


def write_station(out_file, index, profile):
    """Write information about a station (name, latitude, longitude) to an
    output file.
    """
    out_file.variables["lon"][index] = profile.lon
    out_file.variables["lat"][index] = profile.lat
    out_file.variables["station_name"][index] = profile.name


def write_profile(out_file, index, profile, special_vars=False):
    """Write information about a profile (name, latitude, longitude,
    center latitude, center longitude, normal x, normal y, distance
    along profile) to an output file.

    """
    # We have two unlimited dimensions, so we need to assign start and stop
    # start:stop where start=0 and stop is the length of the array
    # or netcdf4python will bail. See
    # https://code.google.com/p/netcdf4-python/issues/detail?id=76
    pl = len(profile.distance_from_start)
    out_file.variables["profile_axis"][index, 0:pl] = np.squeeze(
        profile.distance_from_start
    )
    out_file.variables["nx"][index, 0:pl] = np.squeeze(profile.nx)
    out_file.variables["ny"][index, 0:pl] = np.squeeze(profile.ny)
    out_file.variables["tx"][index, 0:pl] = np.squeeze(profile.tx)
    out_file.variables["ty"][index, 0:pl] = np.squeeze(profile.ty)
    out_file.variables["lon"][index, 0:pl] = np.squeeze(profile.lon)
    out_file.variables["lat"][index, 0:pl] = np.squeeze(profile.lat)
    out_file.variables["profile_id"][index] = profile.id
    out_file.variables["profile_name"][index] = profile.name
    if special_vars:
        out_file.variables["clat"][index] = profile.center_lat
        out_file.variables["clon"][index] = profile.center_lon
        out_file.variables["flightline"][index] = profile.flightline
        out_file.variables["glaciertype"][index] = profile.glaciertype
        out_file.variables["flowtype"][index] = profile.flowtype


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(("{} function took {:0.3f} s".format(f.__name__, (time2 - time1))))
        return ret

    return wrap


@timing
def extract_variable(nc_in, nc_out, profiles, var_name, stations):
    "Extract profiles from one variable."

    # fix
    vars_not_copied = []
    if var_name in vars_not_copied:
        return

    print(("  Reading variable %s" % var_name))

    var_in = nc_in.variables[var_name]
    in_dims = var_in.dimensions

    if in_dims and len(in_dims) > 1:
        # it is a non-scalar variable and it depends on more
        # than one dimension, so we probably need to extract profiles
        out_dims = output_dimensions(in_dims, stations is False)
        var_out = create_variable_like(nc_in, var_name, nc_out, dimensions=out_dims)

        for k, profile in enumerate(profiles):
            print(("    - processing profile {0}".format(profile.name)))
            values, dim_names = extract_profile(var_in, profile)

            # assemble dimension lengths
            lengths = dict(list(zip(dim_names, values.shape)))

            if stations:
                dim_names.remove(profiledim)

                # reshape extracted values to remove redundant profiledim
                values = values.reshape([lengths[d] for d in dim_names])

            try:
                index = [k] + [slice(0, k) for k in values.shape]
                var_out[index] = values
            except:
                print("extract_profiles failed while writing {}".format(var_name))
                raise
    else:
        # it is a scalar or a 1D variable; just copy it
        var_out = create_variable_like(nc_in, var_name, nc_out)
        try:
            var_out[:] = var_in[:]
        except RuntimeError as e:
            print("extract_profiles failed while writing {}".format(var_name))
            raise
        except IndexError:
            print("Failed to copy {}. Ignoring it...".format(var_name))

    # Fix
    attributes_not_copied: list = []
    copy_attributes(var_in, var_out, attributes_not_copied=attributes_not_copied)
    print(("  - done with %s" % var_name))


if __name__ == "__main__":
    # Set up the option parser
    description = """A script to extract data along (possibly multiple) profile using
    piece-wise constant or bilinear interpolation.
    The profile must be given as a ESRI shape file."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = description
    parser.add_argument(
        "SHAPEFILE", nargs=1, help="input shapefile defining profiles to extract"
    )
    parser.add_argument(
        "INPUTFILE", nargs=1, help="input NetCDF file with gridded data"
    )
    parser.add_argument(
        "OUTPUTFILE", nargs=1, help="output NetCDF file name", default="profile.nc"
    )
    parser.add_argument(
        "-f",
        "--flip",
        dest="flip",
        action="store_true",
        help="""Flip profile direction, Default=False""",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--special_vars",
        dest="special_vars",
        action="store_true",
        help="""Add special vars (glaciertype,flowtype, etc), Default=False""",
        default=False,
    )
    parser.add_argument(
        "--srs",
        dest="srs",
        help="Projection of netCDF files as a string, e.g. 'epsg:3413'",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--variable",
        dest="variables",
        help="comma-separated list with variables",
        default=None,
    )

    options = parser.parse_args()
    fill_value = -2e9
    special_vars = options.special_vars
    srs = options.srs
    if options.variables is not None:
        variables = options.variables.split(",")
    else:
        variables = None

    # print("-----------------------------------------------------------------")
    # print("Running script {} ...".format(__file__.split("/")[-1]))
    # print("-----------------------------------------------------------------")
    # print("Opening NetCDF file {} ...".format(options.INPUTFILE[0]))
    # try:
    #     # open netCDF file in 'read' mode
    #     nc_in = NC(options.INPUTFILE[0], "r")
    # except:
    #     print(
    #         "ERROR:  file '{}' not found or not NetCDF format ... ending ...".format(
    #             options.INPUTFILE[0]
    #         )
    #     )
    #     import sys

    #     sys.exit()

    # # get the dimensions
    # xdim, ydim, zdim, tdim = ppt.get_dims(nc_in)
    # # read projection information
    # if srs is not None:
    #     try:
    #         projection = Proj("{}".format(srs))
    #     except:
    #         try:
    #             projection = Proj(srs)
    #         except:
    #             print(("Could not process {}".format(srs)))
    # else:
    #     projection = ppt.get_projection_from_file(nc_in)

    # # Read in profile data
    # print("  reading profile from {}".format(options.SHAPEFILE[0]))
    # profiles = load_profiles(options.SHAPEFILE[0], projection, options.flip)

    # # switch to writing "station" information if all profiles have
    # # length 1
    # stations = np.all(np.array([len(p.x) for p in profiles]) == 1)

    # mapplane_dim_names = (xdim, ydim)

    # print("Creating dimensions")
    # nc_out = NC(options.OUTPUTFILE[0], "w", format="NETCDF4")
    # copy_global_attributes(nc_in, nc_out)

    # # define variables storing profile information
    # if stations:
    #     define_station_variables(nc_out)
    #     print("Writing stations...")
    #     for k, profile in enumerate(profiles):
    #         write_station(nc_out, k, profile)
    #     print("done.")
    # else:
    #     define_profile_variables(nc_out, special_vars=special_vars)
    #     print("Writing profiles...")
    #     for k, profile in enumerate(profiles):
    #         write_profile(nc_out, k, profile, special_vars=special_vars)
    #     print("done.")

    # # re-create dimensions from an input file in an output file, but
    # # skip x and y dimensions and dimensions that are already present
    # copy_dimensions(nc_in, nc_out, mapplane_dim_names)

    # # figure out which variables do not need to be copied to the new file.
    # # mapplane coordinate variables
    # vars_not_copied = [
    #     "lat",
    #     "lat_bnds",
    #     "lat_bounds",
    #     "lon",
    #     "lon_bnds",
    #     "lon_bounds",
    #     xdim,
    #     ydim,
    #     tdim,
    # ]
    # vars_not_copied = list(dict.fromkeys(vars_not_copied))
    # attributes_not_copied = []
    # for var_name in nc_in.variables:
    #     var = nc_in.variables[var_name]
    #     if hasattr(var, "grid_mapping"):
    #         mapping_var_name = var.grid_mapping
    #         vars_not_copied.append(mapping_var_name)
    #         attributes_not_copied.append("grid_mapping")
    #     if hasattr(var, "bounds"):
    #         bounds_var_name = var.bounds
    #         vars_not_copied.append(bounds_var_name)
    #         attributes_not_copied.append("bounds")
    # try:
    #     vars_not_copied.remove(None)
    # except:
    #     pass
    # vars_not_copied.sort()
    # last = vars_not_copied[-1]
    # for i in range(len(vars_not_copied) - 2, -1, -1):
    #     if last == vars_not_copied[i]:
    #         del vars_not_copied[i]
    #     else:
    #         last = vars_not_copied[i]

    # if tdim is not None:
    #     copy_time_dimension(nc_in, nc_out, tdim)

    # print("Copying variables...")
    # if variables is not None:
    #     vars_list = [x for x in variables if x in nc_in.variables]
    #     vars_not_found = [x for x in variables if x not in nc_in.variables]
    # else:
    #     vars_list = nc_in.variables
    #     vars_not_found = ()

    # def extract(name):
    #     extract_variable(nc_in, nc_out, profiles, name, stations)

    # for var_name in vars_list:
    #     extract(var_name)

    # if len(vars_not_found) > 0:
    #     print(
    #         (
    #             "The following variables could not be found in {}:".format(
    #                 options.INPUTFILE[0]
    #             )
    #         )
    #     )
    #     print(vars_not_found)

    # # writing global attributes
    # import sys

    # script_command = " ".join(
    #     [
    #         time.ctime(),
    #         ":",
    #         __file__.split("/")[-1],
    #         " ".join([str(s) for s in sys.argv[1:]]),
    #     ]
    # )
    # if hasattr(nc_in, "history"):
    #     history = nc_in.history
    #     nc_out.history = script_command + "\n " + history
    # else:
    #     nc_out.history = script_command

    # nc_in.close()
    # nc_out.close()
    # print("Extracted profiles to file {}".format(options.OUTPUTFILE[0]))
