import logging
import pandas as pd
import numpy as np
import scipy.constants as co
import matplotlib.pyplot as plt
import warnings

from ..utils.utils_logging import LOG_MODES, init_default_logger
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

logger = init_default_logger(__name__)


def Get_Hist(data, bins=10):
    """
    Calculates Histogram from data for a number of bins.
    Parameters
    ----------
    data : np.array
        dataset from which to calculate the histogram
    bins : int
        number of bins

    Returns
    -------
    x_data, y_data, binsize
    """
    if np.isnan(data).any():
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
    else:
        min_val = np.min(data)
        max_val = np.max(data)
    bin_size = (max_val - min_val) / bins
    x_data = np.linspace(min_val, max_val, bins)
    y_data = np.zeros_like(x_data)

    if len(data.shape) > 1:
        for i in np.arange(0, data.shape[0], 1):
            for j in np.arange(0, data.shape[1], 1):
                if np.isnan(data[i, j]):
                    # handling nan values so that they don't contribute to the histogram
                    pass
                else:
                    idx = np.abs(x_data - data[i, j]).argmin()
                    y_data[idx] += 1
    else:
        for j in np.arange(0, data.shape[0], 1):
            idx = (np.abs(x_data - data[j])).argmin()
            y_data[idx] += 1
    return x_data, y_data, bin_size


def init_header(data_object=None, method_file=None):
    """
    Initializes and returns a header dictionary for a given data object by extracting key parameters from the object's `method` attribute.
    The function aims to create or update a header with MRI scan parameters such as b-values, echo time, number of repetitions, repetition time,
    and other relevant information. If a `data_object` is provided without a header, a new header is generated. If the `data_object` already
    contains a header, it is updated with additional information from the `method` attribute.

    Parameters
    ----------
    data_object : sequence object, optional
        An object representing MRI sequence data, which contains both data and metadata, including a `method` attribute with scan parameters.
        If `None`, the function attempts to generate a header based on `method_file` if provided (default is None).

    method_file : file-like, optional
        An alternative input if `data_object` is not provided, allowing for the creation of a header from a separate method file.
        This parameter is not used in the current implementation but is reserved for future use (default is None).

    Returns
    -------
    dict
        A dictionary representing the header of the MRI scan, containing key scan parameters such as b-values (`b_values`), echo time (`echo_time`),
        number of repetitions (`nreps`), repetition time (`repetition_time`), and others. If neither `data_object` nor `method_file` is provided,
        returns `None`.

    Notes
    -----
    - The function prioritizes `data_object` over `method_file`. If `data_object` is provided, `method_file` is ignored.
    - If `data_object` does not have a header attribute, a new header dictionary is initialized and populated with default values and
      parameters extracted from `data_object.method`.
    - The function assumes that `data_object.method` is a dictionary-like object that supports the `.get()` method for key retrieval.

    Examples
    --------
    Assuming you have a `data_object` with a method attribute:

    >>> data_object = sequence_object(...)  # sequence_object is a hypothetical class instance with a `method` attribute
    >>> header = init_header(data_object)
    >>> print(header)
    {'b_values': 0.0, 'echo_time': 0.0, 'nreps': 1, 'repetition_time': 0, ...}

    To initialize a header with default values (assuming `method_file` support is implemented):

    >>> header = init_header(method_file=my_method_file)
    >>> print(header)
    {'b_values': 0.0, 'echo_time': 0.0, 'nreps': 1, 'repetition_time': 0, ...}
    """
    # cant do anything if nothing was passed:
    if data_object is None and method_file is None:
        return None
    # generate a header dictionary:
    if data_object is None and method_file is not None:
        header = {}
    # if the data_object has a header, dont generate a new one:
    if hasattr(data_object, "header"):
        header = data_object.header
    else:
        header = {}

    header["b_values"] = data_object.method.get("PVM_DwEffBval", 0.0)
    header["echo_time"] = data_object.method.get("EchoTime", 0.0)
    header["nreps"] = data_object.method.get("PVM_NRepetitions", 1)
    header["repetition_time"] = data_object.method.get("PVM_RepetitionTime", 0)
    header["rearranged"] = False  # Have the dimensions already been reconfigured?
    header["dummy_scans"] = data_object.method.get("PVM_DummyScans", 0)
    header["scan_time"] = data_object.method.get("PVM_ScanTime", 0.0)
    header["matrix"] = data_object.method.get("PVM_Matrix", [0, 0])  # matrix size.
    header["slice_thickness"] = data_object.method.get("PVM_SliceThick", 0)  # [mm].
    header["fov"] = data_object.method.get("PVM_Fov", [0, 0])  # [mm].

    header = {key: value for key, value in sorted(header.items())}

    return header


def define_imagematrix_parameters(data_obj=None):
    """
    Define the imaging matrix in voxels.

    Warning: Does not take into account the orientation and offsets of the
    object (yet).

    Parameters
    ----------
    data_obj : Sequence object
        The data object containing sequence information.

    Returns
    -------
    dim_z : int
        Number of voxels in the z dimension.
    dim_x : int
        Number of voxels in the x dimension.
    dim_y : int
        Number of voxels in the y dimension.

    Notes
    -----
    The function returns dimensions as dim_z, dim_x, dim_y, which correspond
    to z, x, y in the scanner coordinate system.
    """
    # Initialize empty return values
    dim_z = dim_x = dim_y = None

    if data_obj is None:
        return dim_z, dim_x, dim_y

    # patient position (usually Head_Prone)
    patient_pos = data_obj.acqp["ACQ_patient_pos"]
    read_orient = data_obj.method["PVM_SPackArrReadOrient"]
    slice_orient = data_obj.method["PVM_SPackArrSliceOrient"]

    # Get patient position, read orientation, and slice orientation
    patient_pos = data_obj.acqp["ACQ_patient_pos"]
    read_orient = data_obj.method["PVM_SPackArrReadOrient"]
    slice_orient = data_obj.method["PVM_SPackArrSliceOrient"]

    if patient_pos == "Head_Prone":
        # All axial slice options:
        if slice_orient == "axial":
            if read_orient == "L_R":
                # x - dimension
                dim_x = data_obj.method["PVM_Matrix"][0]
                # y - dimension
                dim_y = data_obj.method["PVM_Matrix"][1]
                # z - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_z = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_z = data_obj.method["PVM_SPackArrNSlices"]
            elif read_orient == "A_P":
                # x - dimension
                dim_x = data_obj.method["PVM_Matrix"][1]
                # y - dimension
                dim_y = data_obj.method["PVM_Matrix"][0]
                # z - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_z = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_z = data_obj.method["PVM_SPackArrNSlices"]
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        # All sagittal slice options:
        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                # y - dimension
                dim_y = data_obj.method["PVM_Matrix"][1]
                # z - dimension
                dim_z = data_obj.method["PVM_Matrix"][0]
                # x - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_x = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_x = data_obj.method["PVM_SPackArrNSlices"]
            elif read_orient == "A_P":
                # y - dimension
                dim_y = data_obj.method["PVM_Matrix"][0]
                # z - dimension
                dim_z = data_obj.method["PVM_Matrix"][1]
                # x - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_x = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_x = data_obj.method["PVM_SPackArrNSlices"]
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        # All coronal slice options:
        elif slice_orient == "coronal":
            if read_orient == "H_F":
                # z - dimension
                dim_z = data_obj.method["PVM_Matrix"][0]
                # x - dimension
                dim_x = data_obj.method["PVM_Matrix"][1]
                # y - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_y = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_y = data_obj.method["PVM_SPackArrNSlices"]
            elif read_orient == "L_R":
                # z - dimension
                dim_z = data_obj.method["PVM_Matrix"][1]
                # x - dimension
                dim_x = data_obj.method["PVM_Matrix"][0]
                # y - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_y = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_y = data_obj.method["PVM_SPackArrNSlices"]
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )

        elif slice_orient == "axial sagittal coronal":
            raise Exception(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )

        else:
            raise Exception("unknown slice orientation: " + slice_orient)

    elif patient_pos == "Head_Supine":
        Warning("Head_Supine is not implemented yet!")
        if slice_orient == "axial":
            if read_orient == "L_R":
                # scanner x - dimension
                dim_x = data_obj.method["PVM_Matrix"][0]
                # scanner y - dimension
                dim_y = data_obj.method["PVM_Matrix"][1]
                # scanner z - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_z = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_z = data_obj.method["PVM_SPackArrNSlices"]
            elif read_orient == "A_P":
                # scanner x - dimension
                dim_x = data_obj.method["PVM_Matrix"][1]
                # scanner y - dimension
                dim_y = data_obj.method["PVM_Matrix"][0]
                # scanner z - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_z = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_z = data_obj.method["PVM_SPackArrNSlices"]
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                # scanner z - dimension
                dim_z = data_obj.method["PVM_Matrix"][0]
                # scanner y - dimension
                dim_y = data_obj.method["PVM_Matrix"][1]
                # scanner x - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_x = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_x = data_obj.method["PVM_SPackArrNSlices"]
            elif read_orient == "A_P":
                # scanner z - dimension
                dim_z = data_obj.method["PVM_Matrix"][1]
                # scanner y - dimension
                dim_y = data_obj.method["PVM_Matrix"][0]
                # scanner x - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_x = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_x = data_obj.method["PVM_SPackArrNSlices"]
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        elif slice_orient == "coronal":
            if read_orient == "H_F":
                # scanner x - dimension
                dim_x = data_obj.method["PVM_Matrix"][1]
                # scanner z - dimension
                dim_z = data_obj.method["PVM_Matrix"][0]
                # scanner y - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_y = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_y = data_obj.method["PVM_SPackArrNSlices"]
            elif read_orient == "L_R":
                # scanner x - dimension
                dim_x = data_obj.method["PVM_Matrix"][0]
                # scanner z - dimension
                dim_z = data_obj.method["PVM_Matrix"][1]
                # scanner y - dimension
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    dim_y = data_obj.method["PVM_Matrix"][2]
                else:
                    dim_y = data_obj.method["PVM_SPackArrNSlices"]
            else:
                raise Exception(
                    f"define_imagematrix_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        elif slice_orient == "axial sagittal coronal":
            raise Exception(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )
        else:
            raise Exception("unknown slice orientation: " + slice_orient)
            pass

    else:
        raise Exception("unknown patient_position: " + patient_pos)

    return dim_z, dim_x, dim_y


def define_imageFOV_parameters(data_obj=None):
    """
    Warning: Does not take into account the orientation and offsets of the
    object (yet)
    Calculates the FOV in mm.
    Returns FOV in as mm_z, mm_x, mm_y, which correspond to z, x, y
    (scanner coord. system)
    Input
    -----
    data_obj: hypermri-sequence object
    """
    # init empty return
    mm_z = mm_x = mm_y = None

    if data_obj is None:
        return mm_z, mm_x, mm_y

    # patient position (usually Head_Prone)
    patient_pos = data_obj.acqp["ACQ_patient_pos"]
    read_orient = data_obj.method["PVM_SPackArrReadOrient"]
    slice_orient = data_obj.method["PVM_SPackArrSliceOrient"]
    mm_slice_gap = data_obj.method["PVM_SPackArrSliceGap"]

    if patient_pos == "Head_Prone":
        # All axial slice options:
        if slice_orient == "axial":
            # axial slices with read in x-direction, phase in y and phase/slice in z
            if read_orient == "L_R":
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][0]
                # y - dimension
                mm_y = data_obj.method["PVM_Fov"][1]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_z = data_obj.method["PVM_Fov"][2]
                else:
                    dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_z = data_obj.method["PVM_SliceThick"]
                    mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)
            # axial slices with read in y-direction, phase in x and phase/slice in z
            elif read_orient == "A_P":
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][1]
                # y - dimension
                mm_y = data_obj.method["PVM_Fov"][0]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_z = data_obj.method["PVM_Fov"][2]
                else:
                    dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_z = data_obj.method["PVM_SliceThick"]
                    mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)

            else:
                raise Exception(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        # All sagittal slice options:
        elif slice_orient == "sagittal":
            # sagittal slices with read in z-direction, phase in y and phase/slice in x
            if read_orient == "H_F":
                # x - dimension
                mm_y = data_obj.method["PVM_Fov"][1]
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][0]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_x = data_obj.method["PVM_Fov"][2]
                else:
                    _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_x = data_obj.method["PVM_SliceThick"]
                    mm_x = mm_x * dim_x + mm_slice_gap * (dim_x - 1)
            # sagittal slices with read in y-direction, phase in z and phase/slice in x
            elif read_orient == "A_P":
                # x - dimension
                mm_y = data_obj.method["PVM_Fov"][0]
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][1]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_x = data_obj.method["PVM_Fov"][2]
                else:
                    _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_x = data_obj.method["PVM_SliceThick"]
                    mm_x = mm_x * dim_x + mm_slice_gap * (mm_x - 1)
            else:
                raise Exception(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        # All coronal slice options:
        elif slice_orient == "coronal":
            # coronal slices with read in z-direction, phase in x and phase/slice in y
            if read_orient == "H_F":
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][0]
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][1]

                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    # y - dimension
                    mm_y = data_obj.method["PVM_Fov"][2]
                else:
                    _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                    mm_y = data_obj.method["PVM_SliceThick"]
                    mm_y = mm_y * dim_y + mm_slice_gap * (dim_y - 1)
            # coronal slices with read in x-direction, phase in z and phase/slice in y
            elif read_orient == "L_R":
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][0]
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][1]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    # y - dimension
                    mm_y = data_obj.method["PVM_Fov"][2]
                else:
                    _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                    mm_y = data_obj.method["PVM_SliceThick"]
                    mm_y = mm_y * dim_y + mm_slice_gap * (dim_y - 1)

            else:
                raise Exception(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        elif slice_orient == "axial sagittal coronal":
            raise Exception(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )

        # for now do the same to get values:
        else:
            raise Exception("unknown slice orientation: " + slice_orient)
            pass

    # same as for head_prone, has to be changed later!!!!
    elif patient_pos == "Head_Supine":
        if slice_orient == "axial":
            if read_orient == "L_R":
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][0]
                # y - dimension
                mm_y = data_obj.method["PVM_Fov"][1]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_z = data_obj.method["PVM_Fov"][2]
                else:
                    dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_z = data_obj.method["PVM_SliceThick"]
                    mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)
            elif read_orient == "A_P":
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][1]
                # y - dimension
                mm_y = data_obj.method["PVM_Fov"][0]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_z = data_obj.method["PVM_Fov"][2]
                else:
                    dim_z, _, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_z = data_obj.method["PVM_SliceThick"]
                    mm_z = mm_z * dim_z + mm_slice_gap * (dim_z - 1)
            else:
                raise Exception(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass
        # for now do the same to get values:
        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                # x - dimension
                mm_y = data_obj.method["PVM_Fov"][1]
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][0]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_x = data_obj.method["PVM_Fov"][2]
                else:
                    _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_x = data_obj.method["PVM_SliceThick"]
                    mm_x = mm_x * dim_x + mm_slice_gap * (mm_x - 1)
            elif read_orient == "A_P":
                # x - dimension
                mm_y = data_obj.method["PVM_Fov"][1]
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][0]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_x = data_obj.method["PVM_Fov"][2]
                else:
                    _, dim_x, _ = define_imagematrix_parameters(data_obj=data_obj)
                    mm_x = data_obj.method["PVM_SliceThick"]
                    mm_x = mm_x * dim_x + mm_slice_gap * (mm_x - 1)

            else:
                raise Exception(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        # for now do the same to get values:
        elif slice_orient == "coronal":
            if read_orient == "H_F":
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][1]
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][0]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_y = data_obj.method["PVM_Fov"][2]
                else:
                    _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                    mm_y = data_obj.method["PVM_SliceThick"]
                    mm_y = mm_y * dim_y + mm_slice_gap * (mm_y - 1)
            elif read_orient == "L_R":
                # x - dimension
                mm_x = data_obj.method["PVM_Fov"][0]
                # z - dimension
                mm_z = data_obj.method["PVM_Fov"][1]
                if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
                    mm_y = data_obj.method["PVM_Fov"][2]
                else:
                    _, _, dim_y = define_imagematrix_parameters(data_obj=data_obj)
                    mm_y = data_obj.method["PVM_SliceThick"]
                    mm_y = mm_y * dim_y + mm_slice_gap * (mm_y - 1)

            else:
                raise Exception(
                    f"define_imageFOV_parameters:\n{patient_pos}: unknown read orientation: {read_orient} for slice_orient: {slice_orient}"
                )
                pass

        elif slice_orient == "axial sagittal coronal":
            raise Exception(
                f"Slice orientation: {slice_orient} --> probably Localizer, not implemented yet!"
            )

        # for now do the same to get values:
        else:
            raise Exception("unknown slice orientation: " + slice_orient)
            pass

    else:
        raise Exception("unknown patient_position: " + patient_pos)

    return mm_z, mm_x, mm_y


def define_grid(data_obj=None, fov=None, mat=None):
    """
    Defines a grid with the points as the centers of the voxels

    --------
    Parameters:

    fov:
    Field of View in mm [fovx, fovy, fovz]

    mat:
    matrix size in [fovx, fovy, fovz]

    # Example:
    Use from within CSI.py object:
    from hypermri.utils.utils_general import (
                define_imageFOV_parameters,
                define_imagematrix_parameters,
                define_grid,
            )

    mm_read_csi, mm_phase_csi, mm_slice_csi = define_imageFOV_parameters(data_obj=self)
    dim_read_csi, dim_phase_csi, dim_slice_csi = define_imagematrix_parameters(data_obj=self)
    csi_grid = define_grid(
            mat=np.array((dim_read_csi, dim_phase_csi, dim_slice_csi)),
            fov=np.array((mm_read_csi, mm_phase_csi, mm_slice_csi)),
        )

    -----
    Returns:
    nd-arrays, defining center of voxels
    """

    # init:
    ext_1 = ext_2 = ext_3 = None

    if data_obj is not None and fov is None and mat is None:
        # get FOV and Matrix:
        fov = np.array(define_imageFOV_parameters(data_obj=data_obj))
        mat = np.array(define_imagematrix_parameters(data_obj=data_obj))

    # if no FOV was passed:
    if fov is None:
        fov = np.array(define_imageFOV_parameters(data_obj=data_obj))
    # if no matrix size was passed:
    if mat is None:
        mat = np.array(define_imagematrix_parameters(data_obj=data_obj))
    # has no point to continue if either is None
    if all(x is None for x in mat) or all(x is None for x in fov):
        return ext_1, ext_2, ext_3

    # resolution
    res = fov / mat

    # calc extent:
    if (len(fov) > 0) and (len(mat) > 0):
        ext_1 = define_grid_lines(fov=fov[0], mat=mat[0])
    if (len(fov) > 1) and (len(mat) > 1):
        ext_2 = define_grid_lines(fov=fov[1], mat=mat[1])
    if (len(fov) > 2) and (len(mat) > 2):
        ext_3 = define_grid_lines(fov=fov[2], mat=mat[2])
    return ext_1, ext_2, ext_3


def define_grid_lines(data_obj=None, fov=None, mat=None, res=None):
    if res is None and fov is not None and mat is not None:
        res = fov / mat
    if res is not None and fov is None and mat is not None:
        fov = res * mat
    if res is not None and fov is not None and mat is None:
        mat = fov // res
    grid_line = np.linspace(-fov / 2 + res / 2, fov / 2 - res / 2, mat)
    return grid_line


def get_extent(data_obj=None):
    """
    Calculate and return the extent (bounding box) of the given data object in the axial, sagittal,
    and coronal orientations based on the read orientation and offset parameters.

    Args:
        data_obj (Object, optional): Data object containing method attributes that describe
                                     the orientation and offsets of the anatomical input_data.
                                     Defaults to None.

    Returns:
        tuple: Three lists containing the extent (bounding box) for axial, sagittal,
               and coronal orientations respectively. Each list contains the minimum
               and maximum extents for two axes.
               Returns None for each orientation if data_obj is None or not provided.

    Raises:
        NotImplementedError: If the slice orientation from the data_obj's method attribute
                              is not recognized.
    """
    if data_obj is None:
        return None, None, None

    read_orient = data_obj.method["PVM_SPackArrReadOrient"]
    read_offset = data_obj.method["PVM_SPackArrReadOffset"]
    phase_offset = data_obj.method["PVM_SPackArrPhase1Offset"]
    slice_offset = data_obj.method["PVM_SPackArrSliceOffset"]
    mm_z, mm_x, mm_y = define_imageFOV_parameters(data_obj=data_obj)
    # dim_z, dim_x, dim_y = define_imagematrix_parameters(data_obj=data_obj)
    # print(f"read_orient: {read_orient}, read_offset: {read_offset}, phase_offset: {phase_offset}, slice_offset: {slice_offset}")

    patient_pos = data_obj.acqp["ACQ_patient_pos"]
    if patient_pos == "Head_Supine":
        if data_obj.method["PVM_SPackArrSliceOrient"] == "sagittal":
            if read_orient == "H_F":
                phase_offset = -phase_offset
            if read_orient == "A_P":
                pass
        elif data_obj.method["PVM_SPackArrSliceOrient"] == "coronal":
            if read_orient == "H_F":
                phase_offset = -phase_offset  # --> fixed
            elif read_orient == "L_R":
                read_offset = -read_offset  # --> fixed

        if data_obj.method["PVM_SPackArrSliceOrient"] == "axial":
            phase_offset = -phase_offset  # --> fixed
            read_offset = -read_offset  # --> fixed

            # read_offset = -read_offset # --> fixed

    ax_ext = sag_ext = cor_ext = None

    if data_obj.method["PVM_SPackArrSliceOrient"] == "coronal":
        if read_orient == "H_F":
            cor_ext = [
                -mm_z / 2 - read_offset,  # works for head_prone
                mm_z / 2 - read_offset,  # works for head_prone
                -mm_x / 2 - phase_offset,  # works for head_prone
                mm_x / 2 - phase_offset,  # works for head_prone
            ]
            ax_ext = [
                -mm_x / 2 + phase_offset,
                mm_x / 2 + phase_offset,
                -mm_y / 2 + slice_offset,
                mm_y / 2 + slice_offset,
            ]
            sag_ext = [
                -mm_z / 2 + read_offset,
                mm_z / 2 + read_offset,
                -mm_y / 2 + slice_offset,
                mm_y / 2 + slice_offset,
            ]
        elif read_orient == "L_R":
            cor_ext = [
                -mm_z / 2 - phase_offset,  # works for head_prone
                mm_z / 2 - phase_offset,  # works for head_prone
                -mm_x / 2 - read_offset,  # works for head_prone
                mm_x / 2 - read_offset,  # works for head_prone
            ]
            ax_ext = [
                -mm_x / 2 + read_offset,
                mm_x / 2 + read_offset,
                -mm_y / 2 + slice_offset,
                mm_y / 2 + slice_offset,
            ]
            sag_ext = [
                -mm_z / 2 - phase_offset,
                mm_z / 2 - phase_offset,
                -mm_y / 2 + slice_offset,
                mm_y / 2 + slice_offset,
            ]

    elif data_obj.method["PVM_SPackArrSliceOrient"] == "axial":
        if read_orient == "L_R":
            cor_ext = [
                -mm_x / 2 + read_offset,
                mm_x / 2 + read_offset,
                -mm_z / 2 + slice_offset,
                mm_z / 2 + slice_offset,
            ]
            ax_ext = [
                -mm_x / 2 - read_offset,  # -> for head_prone, works
                mm_x / 2 - read_offset,  # -> for head_prone, works
                -mm_y / 2 - phase_offset,  # -> for head_prone, works
                mm_y / 2 - phase_offset,  # -> for head_prone, works
            ]
            sag_ext = [
                -mm_z / 2 + slice_offset,
                mm_z / 2 + slice_offset,
                -mm_y / 2 + phase_offset,
                mm_y / 2 + phase_offset,
            ]
        elif read_orient == "A_P":
            cor_ext = [
                -mm_x / 2 + phase_offset,
                mm_x / 2 + phase_offset,
                -mm_z / 2 + slice_offset,
                mm_z / 2 + slice_offset,
            ]
            ax_ext = [
                -mm_x / 2 - phase_offset,  # -> for head_prone, works
                mm_x / 2 - phase_offset,  # -> for head_prone, works
                -mm_y / 2 - read_offset,  # -> for head_prone, works
                mm_y / 2 - read_offset,  # -> for head_prone, works
            ]
            sag_ext = [
                -mm_z / 2 + slice_offset,
                mm_z / 2 + slice_offset,
                -mm_y / 2 + read_offset,
                mm_y / 2 + read_offset,
            ]

    elif data_obj.method["PVM_SPackArrSliceOrient"] == "sagittal":
        # TODO needs to be verified, gives plausibly looking results
        if read_orient == "H_F":
            cor_ext = [
                -mm_x / 2 + slice_offset,
                mm_x / 2 + slice_offset,
                -mm_z / 2 + read_offset,
                mm_z / 2 + read_offset,
            ]
            ax_ext = [
                -mm_x / 2 + slice_offset,
                mm_x / 2 + slice_offset,
                -mm_y / 2 + phase_offset,
                mm_y / 2 + phase_offset,
            ]
            sag_ext = [
                -mm_y / 2 - read_offset,  # works for head_prone
                mm_y / 2 - read_offset,  # works for head_prone
                -mm_z / 2 - phase_offset,  # works for head_prone
                mm_z / 2 - phase_offset,  # works for head_prone
            ]
        elif read_orient == "A_P":
            cor_ext = [
                -mm_x / 2 + slice_offset,  # works for head_prone
                mm_x / 2 + slice_offset,  # works for head_prone
                -mm_z / 2 + phase_offset,  # works for head_prone
                mm_z / 2 + phase_offset,  # works for head_prone
            ]
            ax_ext = [
                -mm_x / 2 + slice_offset,
                mm_x / 2 + slice_offset,
                -mm_y / 2 + read_offset,
                mm_y / 2 + read_offset,
            ]
            sag_ext = [
                -mm_z / 2 - phase_offset,
                mm_z / 2 - phase_offset,
                -mm_y / 2 - read_offset,
                mm_y / 2 - read_offset,
            ]

        else:
            raise NotImplementedError(
                data_obj.method["PVM_SPackArrReadOrient"], "not implemented yet"
            )
    else:
        raise NotImplementedError(
            data_obj.method["PVM_SPackArrSliceOrient"], "not implemented yet"
        )

    data_obj.ax_ext = ax_ext
    data_obj.sag_ext = sag_ext
    data_obj.cor_ext = cor_ext

    return ax_ext, sag_ext, cor_ext


def reorient_anat(
    data_obj=None,
    input_data=None,
    overwrite_seq2d=True,
    force_reorient=False,
    mirror_first_axis=False,
    mirror_first_and_second_axis=False,
):
    """
    Reorients an anatomical input_data (FLASH or RARE) according to a specified order: echoes, z, x, y, repetitions (
    reps), and channels (chans). This reorientation is based on the patient's position, read orientation,
    slice orientation, and the presence of multiple slices or echoes. The function optionally updates the
    `seq2d_oriented` attribute of the `data_obj` with the reoriented input_data.

    Parameters
    ----------
    data_obj : hypemri.sequence object, optional
        An object containing information about the
        orientation of the anatomical input_data, including attributes for the original sequence (`seq2d`), acquisition
        parameters (`acqp`), and method parameters (`method`). If `None` is provided, the function returns `None`. This
        parameter is preferred over direct 2D sequence manipulation (Default is None).

    overwrite_seq2d : bool, optional
        Determines whether the `seq2d_oriented` attribute of the `data_obj` should be
        overwritten with the reoriented input_data. If `True`, the original `seq2d_oriented` is replaced. Otherwise,
        the reoriented input_data is returned without modifying the `data_obj` (Default is True).

    Returns
    -------
        numpy.ndarray or None The reoriented anatomical input_data as an N-Dimensional array if the operation
        is applicable; otherwise, returns the original input_data or `None` if `data_obj` is `None`. The reorientation and the
        dimensions of the returned input_data depend on the initial conditions specified by `data_obj`.

    Raises
    ------
        Warning Outputs a warning message if the function encounters untested configurations, particularly
        for certain patient positions and slice orientations.

    Notes
    -----
        - The function checks if the `data_obj` has already been reoriented by looking for a
        `reorient_counter`. If found and not equal to zero, it indicates the operation has been performed,
        and the already reoriented input_data is returned.
        - This function is specific to handling MRI data structures defined
        by `hypemri.sequence` objects, and it is tailored for images with specific orientations and configurations. The
        exact behavior depends on attributes of `data_obj`, such as patient position (`ACQ_patient_pos`),
        read orientation (`PVM_SPackArrReadOrient`), and slice orientation (`PVM_SPackArrSliceOrient`).

    Examples
    --------
    To reorient an anatomical input_data contained within a `hypemri.sequence` object:

        >>> data_obj = hypemri.sequence(...) # Assume this is a pre-defined object with necessary attributes
        >>> reoriented_image = reorient_anat(data_obj)
        >>> print(reoriented_image.shape)
        (dimensions based on the reorientation logic)

    To use the function without overwriting the original `seq2d_oriented`:

        >>> reoriented_image = reorient_anat(data_obj, overwrite_seq2d=False)
        >>> print(reoriented_image.shape)
        (dimensions based on the reorientation logic)
    """
    if data_obj is None:
        logger.warning("No data object provided. Returning None.")
        return None
    if input_data is None:
        seq2d = data_obj.seq2d
    else:
        seq2d = np.squeeze(input_data)

    # patient position (usually Head_Prone)
    patient_pos = data_obj.acqp["ACQ_patient_pos"]
    read_orient = data_obj.method["PVM_SPackArrReadOrient"]
    slice_orient = data_obj.method["PVM_SPackArrSliceOrient"]
    # mm_slice_gap = data_obj.method["PVM_SPackArrSliceGap"]
    nr = data_obj.method["PVM_NRepetitions"]
    reorient_anat_counter = get_counter(
        data_obj=data_obj, counter_name="reorient_counter"
    )
    allowed_patient_pos = ["Head_Prone", "Head_Supine"]
    if patient_pos in allowed_patient_pos:
        if (
            reorient_anat_counter == 0
            or force_reorient is True
            or input_data is not None
        ):
            dim_z, dim_x, dim_y = define_imagematrix_parameters(data_obj=data_obj)
            seq2d_per = None
            if patient_pos == "Head_Prone":
                if slice_orient == "axial":
                    if read_orient == "L_R":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            if dim_z > 1:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis
                                ]
                            else:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis, np.newaxis
                                ]
                        seq2d_per = np.transpose(seq2d_per, (4, 2, 0, 1, 3, 5))
                        seq2d_per = np.flip(seq2d_per, axis=1)

                    else:
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]
                        seq2d_per = np.transpose(seq2d_per, (4, 2, 0, 1, 3, 5))
                        seq2d_per = np.flip(seq2d_per, axis=1)

                if slice_orient == "sagittal":
                    if read_orient == "H_F":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            #
                            seq2d = np.squeeze(seq2d)
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]

                        seq2d_per = np.transpose(
                            seq2d_per, (4, 1, 2, 0, 3, 5)
                        )  # -> fixed
                    else:  # "A_P"
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]
                        seq2d_per = np.transpose(seq2d_per, (4, 1, 2, 0, 3, 5))

                elif slice_orient == "coronal":
                    if read_orient == "H_F":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d = np.squeeze(seq2d)
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]
                        seq2d_per = np.transpose(seq2d_per, (4, 1, 0, 2, 3, 5))
                    else:  # "L_R"
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d = np.squeeze(seq2d)
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]
                        seq2d_per = np.transpose(seq2d_per, (4, 1, 0, 2, 3, 5))
            elif patient_pos == "Head_Supine":
                if slice_orient == "axial":
                    if read_orient == "L_R":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            if dim_z > 1:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis
                                ]
                            else:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis, np.newaxis
                                ]
                        seq2d_per = np.transpose(seq2d_per, (4, 2, 0, 1, 3, 5))
                        seq2d_per = np.flip(seq2d_per, axis=3)
                        seq2d_per = np.flip(seq2d_per, axis=2)  # --> fixed
                        seq2d_per = np.flip(seq2d_per, axis=1)  # --> fixed

                    elif read_orient == "A_P":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]
                        seq2d_per = np.transpose(seq2d_per, (4, 2, 0, 1, 3, 5))
                        seq2d_per = np.flip(seq2d_per, axis=3)
                        seq2d_per = np.flip(seq2d_per, axis=2)  # --> fixed
                        seq2d_per = np.flip(seq2d_per, axis=1)  # --> fixed
                    else:
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d = np.squeeze(seq2d)
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]
                        seq2d_per = np.transpose(seq2d_per, (4, 2, 1, 0, 3, 5))

                elif slice_orient == "sagittal":
                    if read_orient == "H_F":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            if dim_z > 1:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis
                                ]
                            else:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis, np.newaxis
                                ]
                        seq2d_per = np.transpose(seq2d_per, (4, 1, 2, 0, 3, 5))

                        seq2d_per = np.flip(seq2d_per, axis=3)
                        seq2d_per = np.flip(seq2d_per, axis=2)  # --> fixed
                        # seq2d_per = np.flip(seq2d_per, axis=1) # --> fixed

                    elif read_orient == "A_P":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]

                        seq2d_per = np.transpose(seq2d_per, (4, 1, 2, 0, 3, 5))
                        # seq2d_per = np.flip(seq2d_per, axis=3)
                        seq2d_per = np.flip(seq2d_per, axis=2)  # --> fixed
                        # seq2d_per = np.flip(seq2d_per, axis=1) # --> fixed
                    else:
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            seq2d = np.squeeze(seq2d)
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis, np.newaxis]
                        seq2d_per = np.transpose(seq2d_per, (4, 2, 1, 0, 3, 5))

                elif slice_orient == "coronal":
                    if read_orient == "H_F":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            if dim_y > 1:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis
                                ]
                            else:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis, np.newaxis
                                ]
                        seq2d_per = np.transpose(seq2d_per, (4, 1, 0, 2, 3, 5))
                        seq2d_per = np.flip(seq2d_per, axis=3)
                        seq2d_per = np.flip(seq2d_per, axis=2)
                    if read_orient == "L_R":
                        if nr > 1:
                            seq2d_per = seq2d[..., np.newaxis, np.newaxis]
                        else:
                            if dim_y > 1:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis
                                ]
                            else:
                                seq2d = np.squeeze(seq2d)
                                seq2d_per = seq2d[
                                    ..., np.newaxis, np.newaxis, np.newaxis, np.newaxis
                                ]
                        seq2d_per = np.transpose(seq2d_per, (4, 1, 0, 2, 3, 5))
                        seq2d_per = np.flip(seq2d_per, axis=3)
                        seq2d_per = np.flip(seq2d_per, axis=2)
            if input_data is not None:
                return seq2d_per
            else:
                if overwrite_seq2d:
                    data_obj.seq2d_oriented = seq2d_per
                    add_counter(
                        data_obj=data_obj, counter_name="reorient_counter", n_counts=1
                    )
                    return seq2d_per
                else:
                    return seq2d_per
        else:
            return data_obj.seq2d_oriented
    else:
        logger.critical(
            str(patient_pos)
            + " of scan number "
            + str(data_obj.ExpNum)
            + " not implemented for reorientation. Proceeding without reorientation."
        )
        return None


def calc_mat_origin_diff(
    res_metab=None,
    fov_metab=None,
    mat_metab=None,
    res_anat=None,
    fov_anat=None,
    mat_anat=None,
):
    """
    Calculate the voxel shift offsets between metabolite and anatomical images across z, x, y dimensions.

    This function computes the differences in origin points between a metabolite input_data with a smaller matrix/resolution
    and an anatomical input_data with a higher matrix/resolution. The offsets are calculated based on the resolutions,
    field of views (FOVs), and matrix sizes of the images.

    calculations performed:
    dx[mm] = resx_metab/2 - resx_anat/2
    dx[vox] = dx[mm] * N_anat / FOV_anat


    Parameters
    ----------
    res_metab : list of float, optional
        Resolution of the metabolite input_data (small matrix), specified as [z, x, y].
    fov_metab : list of float, optional
        Field of view of the metabolite input_data, specified as [z, x, y].
    mat_metab : list of int, optional
        Matrix size of the metabolite input_data, specified as [z, x, y].
    res_anat : list of float, optional
        Resolution of the anatomical input_data (high matrix), specified as [z, x, y].
    fov_anat : list of float
        Field of view of the anatomical input_data, specified as [z, x, y].
    mat_anat : list of int
        Matrix size of the anatomical input_data, specified as [z, x, y].

    Returns
    -------
    shift_vox_list : list of float
        List of voxel shifts for each dimension (z, x, y), indicating the offset between the metabolite
        and anatomical images.

    Examples
    --------
    >>> shift_vox_list = calc_mat_origin_diff(res_metab=[1,1,1], res_anat=[2,2,2], fov_anat=[32,32,32], mat_anat=[32,32,32])
    >>> print(shift_vox_list)
    [0.5, 0.5, 0.5]

    >>> shift_vox_list = calc_mat_origin_diff(res_metab=[1,1,1], res_anat=[2,2,2], fov_anat=[1,1,1], mat_anat=[2,2,2])
    >>> print(shift_vox_list)
    [0.25, 0.25, 0.25]
    """
    if res_metab is None:
        if fov_metab is not None and mat_metab is not None:
            res_metab = [a / b for a, b in zip(fov_metab, mat_metab)]
        else:
            return None, None, None
    else:
        pass

    if res_anat is None:
        res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]
    else:
        pass

    # shift_vox_list = [
    #     d * np.sqrt((a / 2) ** 2 + (b / 2) ** 2) / c
    #     for a, b, c, d in zip(res_metab, res_anat, fov_anat, mat_anat)
    # ]

    shift_vox_list = [
        d * (a - b) / 2 / c
        for a, b, c, d in zip(res_metab, res_anat, fov_anat, mat_anat)
    ]
    return shift_vox_list


def shift_anat(
    csi_obj=None,
    anat_obj=None,
    shift_vox=None,
    use_scipy_shift=True,
    force_shift=False,
    overwrite_seq2d=True,
    input_data=None,
    mirror_first_axis=False,
    mirror_first_and_second_axis=False,
):
    """
    Adjusts the anatomical input_data to match the CSI object's orientation.

    Parameters
    ----------
    csi_obj : object, optional
        Object that contains CSI data and parameters.
    anat_obj : object, optional
        Object that contains anatomical input_data data and parameters.
    shift_vox : list of float, optional
        Voxel shifts for each dimension.
    use_scipy_shift : bool, optional
        If True, scipy's shift function will be used. Default is True.
    force_shift : bool, optional
        If True, shifting will be enforced. Default is False.
    overwrite_seq2d : bool, optional
        If True, the seq2d_oriented of the data_object will be replaced. Default is True.
    input_data : ndarray, optional
        Input anatomical data to be shifted.
    mirror_first_axis : bool, optional
        If True, the first axis will be mirrored. Default is False.
    mirror_first_and_second_axis : bool, optional
        If True, the first and second axes will be mirrored. Default is False.

    Returns
    -------
    anat_data_shifted : ndarray
        Adjusted anatomical input_data.

    """
    if anat_obj is None:
        # get anatomical FOV:
        fov_anat = None
        # get anatomical matrix size:
        mat_anat = None

        # can't overwrite seq2d:
        if overwrite_seq2d is True:
            overwrite_seq2d = False
        if input_data is None:
            raise ValueError("Either anat_obj or input_data has to be passed!")
        else:
            anat_image = input_data
    else:
        # get anatomical FOV:
        fov_anat = define_imageFOV_parameters(data_obj=anat_obj)
        # get anatomical matrix size:
        mat_anat = define_imagematrix_parameters(data_obj=anat_obj)
        # get patient position (anat)
        patient_pos_anat = anat_obj.acqp["ACQ_patient_pos"]
        # get slice orientation (anat):
        slice_orient_anat = anat_obj.method["PVM_SPackArrSliceOrient"]
        # get read orientation (anat)
        read_orient_anat = anat_obj.method["PVM_SPackArrReadOrient"]

        if input_data is None:
            # check if data was already reoriented to match bssfp orientation:
            reorient_anat_counts = get_counter(
                data_obj=anat_obj, counter_name="reorient_counter"
            )
            # if not --> reorient:
            if reorient_anat_counts == 0:
                reorient_anat(data_obj=anat_obj)
                anat_image = anat_obj.seq2d_oriented
            else:
                anat_image = anat_obj.seq2d_oriented
        else:
            anat_image = input_data
            # perform shift:
            force_shift = True
            # dont overwrite objects seq2d:
            overwrite_seq2d = False

    # if you want to perform the shift anyway:
    if force_shift is False and anat_obj is not None:
        # number of already occured shifts:
        num_shift_counts = get_counter(
            data_obj=anat_obj,
            counter_name="shift_counter",
        )
        # if no shift has been done continue
        if num_shift_counts == 0:
            pass
        # else return already shifted input_data:
        else:
            return anat_image

    # get parameters from object if passed:
    if csi_obj is None:
        fov_csi = None
        # get matrix size of CSI input_data:
        mat_csi = None
    else:
        # get FOV of CSI input_data:
        fov_csi = define_imageFOV_parameters(data_obj=csi_obj)
        # get matrix size of CSI input_data:
        mat_csi = define_imagematrix_parameters(data_obj=csi_obj)
        # get patient position of CSI input_data:
        patient_pos_csi = csi_obj.acqp["ACQ_patient_pos"]
        # get slice orientation of CSI input_data:
        slice_orient_csi = csi_obj.method["PVM_SPackArrSliceOrient"]
        # get read orientation of CSI input_data:
        read_orient_csi = csi_obj.method["PVM_SPackArrReadOrient"]
        if csi_obj.method["PVM_SpatDimEnum"] == "<3D>":
            acq_dim_csi = "3d"
        else:
            acq_dim_csi = "2d"

    if fov_csi is None and fov_anat is None:
        # can use any value, doesnt matter as
        # long as they have the same FOV:
        fov_csi = fov_anat = [1.0, 1.0]
    elif fov_csi is None and fov_anat is not None:
        # use same FOV
        fov_csi = fov_anat
    elif fov_csi is not None and fov_anat is None:
        # use same FOV
        fov_anat = fov_csi
    else:
        pass

    # if no shift was passed, calculate necessary shift depending on the patient position, the slice and the read
    # orientation:
    if shift_vox is None:
        if anat_obj is None and csi_obj is None:
            raise ValueError(
                "If no shift_vox was passed, anat_obj and csi_obj have to be passed!"
            )
            return None

        # calculate the resultions ():
        res_csi = [a / b for a, b in zip(fov_csi, mat_csi)]
        res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]

        # calc necessary shift in all directions:
        shift_vox_list = calc_mat_origin_diff(
            res_metab=res_csi, res_anat=res_anat, fov_anat=fov_anat, mat_anat=mat_anat
        )

        # init empty array:
        shift_vox = [0, 0, 0, 0, 0, 0]
        if patient_pos_anat == "Head_Prone":
            if read_orient_csi == read_orient_anat == "L_R":
                if slice_orient_csi == slice_orient_anat == "axial":
                    # both slices axial:
                    if acq_dim_csi == acq_dim_csi == "2d":
                        # no need to shift if both were acquired with multislice
                        shift_vox[1] = 0
                    else:
                        shift_vox[1] = shift_vox_list[0]
                    # left/right, x-dim of array echoes-z-x-y-rep-chans
                    shift_vox[2] = shift_vox_list[1]
                    # y-dim of array echoes-z-x-y-rep-chans
                    shift_vox[3] = shift_vox_list[2]  # --> should be positive for CSI

                else:
                    print(
                        f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                        f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                        f"slice orient = {slice_orient_anat}\n "
                        "was not tested yet!"
                    )

            elif read_orient_csi == "H_F" and read_orient_anat == "L_R":
                print(
                    f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                    f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                    f"slice orient = {slice_orient_anat}\n "
                    " was not tested yet!"
                )

                shift_vox[1] = shift_vox_list[0]
                # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[2] = shift_vox_list[1]
                # y-dim of array echoes-z-x-y-rep-chans
                shift_vox[3] = -shift_vox_list[2]  # tested --> good

            elif read_orient_csi == "H_F" and read_orient_anat == "H_F":
                ## both slices coronal:
                shift_vox[1] = -shift_vox_list[0]
                # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[2] = shift_vox_list[1]
                # y-dim of array echoes-z-x-y-rep-chans
                shift_vox[3] = shift_vox_list[2]

            else:
                print(
                    f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                    f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                    f"slice orient = {slice_orient_anat}\n "
                    "was not tested yet!"
                )

                # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[2] = shift_vox_list[1]
                # y-dim of array echoes-z-x-y-rep-chans
                shift_vox[3] = shift_vox_list[2]
                # shift_vox = [0, 0]

        elif patient_pos_csi == patient_pos_anat == "Head_Supine":
            if read_orient_csi == read_orient_anat == "L_R":
                if slice_orient_csi == slice_orient_anat == "axial":
                    print(
                        f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                        f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                        f"slice orient = {slice_orient_anat}\n "
                        "is being tested!"
                    )
                    # both slices axial:
                    if acq_dim_csi == acq_dim_csi == "2d":
                        # no need to shift if both were acquired with multislice
                        shift_vox[1] = 0
                    else:
                        shift_vox[1] = shift_vox_list[0]
                    # left/right, x-dim of array echoes-z-x-y-rep-chans
                    shift_vox[2] = shift_vox_list[1]
                    # y-dim of array echoes-z-x-y-rep-chans
                    shift_vox[3] = -shift_vox_list[2]  # --> should be positive for CSI

                else:
                    print(
                        f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                        f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                        f"slice orient = {slice_orient_anat}\n "
                        "was not tested yet!"
                    )

            elif read_orient_csi == read_orient_anat == "H_F":
                if slice_orient_csi == slice_orient_anat == "coronal":
                    print(
                        f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                        f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                        f"slice orient = {slice_orient_anat}\n "
                        "is being tested!"
                    )

                    # both slices axial:
                    if acq_dim_csi == acq_dim_csi == "2d":
                        # no need to shift if both were acquired with multislice
                        shift_vox[1] = 0
                    else:
                        shift_vox[1] = shift_vox_list[2]
                    # z-dim of array echoes-z-x-y-rep-chans
                    shift_vox[2] = shift_vox_list[0]
                    # x-dim of array echoes-z-x-y-rep-chans
                    if mirror_first_axis:
                        shift_vox[3] = shift_vox_list[
                            1
                        ]  # --> should be positive for CSI
                    else:
                        shift_vox[3] = -shift_vox_list[
                            1
                        ]  # --> should be positive for CSI

                if slice_orient_csi == slice_orient_anat == "sagittal":
                    print(
                        f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                        f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                        f"slice orient = {slice_orient_anat}\n "
                        "is being tested!"
                    )

                    # both slices axial:
                    if acq_dim_csi == acq_dim_csi == "2d":
                        # no need to shift if both were acquired with multislice
                        shift_vox[1] = 0
                    else:
                        shift_vox[1] = shift_vox_list[1]
                    # z-dim of array echoes-z-x-y-rep-chans
                    if mirror_first_and_second_axis:
                        shift_vox[2] = -shift_vox_list[0]
                    else:
                        shift_vox[2] = shift_vox_list[0]
                    # y-dim of array echoes-z-x-y-rep-chans
                    if mirror_first_and_second_axis:
                        shift_vox[3] = -shift_vox_list[
                            2
                        ]  # --> should be positive for CSI
                    else:
                        shift_vox[3] = -shift_vox_list[
                            2
                        ]  # --> should be positive for CSI

                else:
                    print(
                        f"Combination of {patient_pos_anat}, \n CSI: read orient = {read_orient_csi}, "
                        f"slice orient = {slice_orient_csi} \n Anat: read orient = {read_orient_anat}, "
                        f"slice orient = {slice_orient_anat}\n "
                        "was not tested yet!"
                    )

        else:
            if slice_orient_csi == slice_orient_anat:
                if read_orient_csi == read_orient_anat:
                    # left/right, x-dim of array echoes-z-x-y-rep-chans
                    shift_vox[2] = -shift_vox_list[1]
                    # y-dim of array echoes-z-x-y-rep-chans
                    shift_vox[3] = shift_vox_list[2]
                    # shift_vox = [0, 0]
                elif read_orient_csi == "L_R" and read_orient_anat == "A_P":
                    shift_vox[2] = shift_vox_list[
                        1
                    ]  # left/right, x-dim of array echoes-z-x-y-rep-chans
                    shift_vox[3] = -shift_vox_list[
                        2
                    ]  # y-dim of array echoes-z-x-y-rep-chans
                    # shift_vox = [0, 0]
                else:
                    print(
                        f"csi read orient:{read_orient_csi} and anat read orient:{read_orient_anat} are"
                        f"not implemented yet!"
                    )
            else:
                print(
                    f"csi slice orient:{slice_orient_csi} and anat slice orient:{slice_orient_anat} are"
                    f"not implemented yet!"
                )

    anat_image_shifted_reshaped = anat_image

    # intermezzo -------------------------------------------------------------------------------------------------------
    # acq_matrix = csi_obj.acqp["ACQ_grad_matrix"]
    # # angle in degree
    # a = 90 - np.arccos(acq_matrix) / np.pi * 180.0
    # angle = a[0, 0, 1]
    # shift_vox[2] += shift_vox[2] * (1 - np.cos(angle * np.pi / 180.0))
    # intermezzo -------------------------------------------------------------------------------------------------------

    # assume desired
    if len(shift_vox) == 3 and anat_image.ndim != 3:
        shift_vox = [0] + shift_vox + [0, 0]

    # scipy shift package:
    if use_scipy_shift:
        try:
            # import shift package:
            from scipy.ndimage import shift

            def synchronized_squeeze(a, shifts):
                """
                Remove singleton dimensions from an array and corresponding zero shifts.
                """
                assert a.ndim == len(
                    shifts
                ), f"Array and shifts list must have the same length, have {a.ndim} and {len(shifts)}"

                # Determine which dimensions have zero shift and are singleton in the array
                squeeze_dims = [
                    dim
                    for dim in range(a.ndim)
                    if shifts[dim] == 0 and a.shape[dim] == 1
                ]

                # Squeeze only the synchronized singleton dimensions
                a_squeezed = np.squeeze(a, axis=tuple(squeeze_dims))

                # Remove the corresponding shifts
                shifts_squeezed = [
                    shift for dim, shift in enumerate(shifts) if dim not in squeeze_dims
                ]

                return a_squeezed, shifts_squeezed

            # Usage example
            # Assuming im is your 6D input_data and shifts is your 6D shift vector
            im_squeezed, shifts_squeezed = synchronized_squeeze(
                a=anat_image, shifts=shift_vox
            )
            # perform shift on lower dim data (faster)
            anat_image_shifted = shift(im_squeezed, shift=shifts_squeezed, mode="wrap")

            # Reshape back to original dimensions by introducing singleton dimensions where shift was zero
            anat_image_shifted_reshaped = np.reshape(
                anat_image_shifted, anat_image.shape
            )

            if overwrite_seq2d:
                # save shfit results:
                anat_obj.seq2d_oriented = anat_image_shifted_reshaped

                # add 1 to shift counter:
                add_counter(
                    data_obj=anat_obj,
                    counter_name="shift_counter",
                    n_counts=1,
                )
            else:
                pass
        except Exception as e:
            err_msg, err_type = str(e), e.__class__.__name__

            msg = "The following error occured during "
            msg += f"utils_general.shift_anat():\n    {err_type}: {err_msg}\n"

            print(msg)
    else:
        Warning(
            "use use_scipy_shift=True, other shifting methods have not yet been implemented!"
        )

    if mirror_first_axis:
        return anat_image_shifted_reshaped[:, :, :, ::-1]
    if mirror_first_and_second_axis:
        return anat_image_shifted_reshaped[:, :, ::-1, :]
    else:
        return anat_image_shifted_reshaped


def shift_image(
    input_data=None,
    shift_vox=None,
    shift_method="phase",
    force_shift=False,
    data_obj=None,
    domain="kspace",
):
    """
    Shift the input data using either a phase ramp or a direct spatial shift.

    This function modifies the data using either a phase ramp or a direct spatial shift via the
    scipy.ndimage.shift function, based on the specified method. The input_data is expected to be
    in the k-space domain (default) and can be changed with the domain parameter.

    Parameters
    ----------
    input_data : ndarray, optional
        The input data array. If None, the k-space data from the instance variable `kspace_array` is used.
    shift_vox : array_like of int, optional
        The number of voxels to shift along each axis. If None, the shift is determined by the
        `determine_necessary_shift` method.
    shift_method : {'phase', 'scipy'}, default 'phase'
        The method used to perform the shift. 'phase' uses a linear phase ramp, while 'scipy' uses
        scipy's shift function.
    force_shift : bool, default False
        If True, forces the shift operation even if shifts have been previously applied.
    domain : {'kspace', 'image'}, default 'kspace'
        The domain of the data to be shifted.
    data_obj : hypemri.sequence object, optional
        Contains the method file containing the phase offsets.

    Returns
    -------
    kspace_shifted : ndarray
        The shifted k-space data array.

    Raises
    ------
    ValueError
        If an unsupported shift method is provided.

    Notes
    -----
    The shift is applied in the Fourier domain for the phase method and in the spatial domain for
    the scipy method. The input_data is expected to be in the k-space domain by default.
    """
    if input_data is None:
        kspace_array = data_obj.kspace_array
    else:
        kspace_array = input_data

    is_complex = np.iscomplexobj(kspace_array)

    if domain != "kspace":
        kspace_array = np.fft.ifftshift(
            np.fft.ifft(
                np.fft.ifftshift(
                    np.fft.ifftshift(
                        np.fft.ifft(np.fft.ifftshift(kspace_array, axes=2), axis=2),
                        axes=2,
                    ),
                    axes=3,
                ),
                axis=3,
            ),
            axes=3,
        )

    kspace_shift_counts = get_counter(
        data_obj=data_obj,
        counter_name="kspace_shift_counts",
    )

    if kspace_shift_counts > 0 and not force_shift:
        return kspace_array

    if shift_vox is None:
        shift_vox, _ = determine_necessary_shift(data_obj=data_obj, db=False)

    shift_vox = shift_vox[:2]

    if shift_method == "scipy":
        from scipy.ndimage import shift
        # Convert to image domain
        image_array = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(kspace_array, axes=(2, 3)), axes=(2, 3)),
            axes=(2, 3),
        )

        # Prepare shift array
        full_shift = [0] * kspace_array.ndim
        full_shift[2:4] = shift_vox

        # Apply shift to real and imaginary parts separately
        real_shifted = shift(np.real(image_array), full_shift, mode="wrap")
        imag_shifted = shift(np.imag(image_array), full_shift, mode="wrap")

        # Combine real and imaginary parts
        image_shifted = real_shifted + 1j * imag_shifted

        # Convert back to k-space
        kspace_shifted = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(image_shifted, axes=(2, 3)), axes=(2, 3)),
            axes=(2, 3),
        )

        add_counter(
            data_obj=data_obj,
            counter_name="kspace_shift_counts",
            n_counts=1,
        )
    elif shift_method == "phase":
        # Create phase ramps
        x = np.arange(kspace_array.shape[2])
        y = np.arange(kspace_array.shape[3])
        phasex = np.exp(1j * 2 * np.pi * x / kspace_array.shape[2] * shift_vox[0])
        phasey = np.exp(1j * 2 * np.pi * y / kspace_array.shape[3] * shift_vox[1])

        # Create 2D phase map
        phase_map = np.outer(phasex, phasey)

        # Create a slice object for proper broadcasting
        slice_obj = (
            (np.newaxis,) * 2
            + (slice(None), slice(None))
            + (np.newaxis,) * (kspace_array.ndim - 4)
        )

        # Apply phase map to all dimensions simultaneously
        kspace_shifted = kspace_array * phase_map[slice_obj]

        add_counter(
            data_obj=data_obj,
            counter_name="kspace_shift_counts",
            n_counts=1,
        )
    else:
        raise ValueError(
            f"Unsupported shift method: {shift_method}, use either 'phase' or 'scipy'."
        )

    if domain != "kspace":
        kspace_shifted = np.fft.fftshift(
            np.fft.fft(
                np.fft.fftshift(
                    np.fft.fftshift(
                        np.fft.fft(np.fft.fftshift(kspace_shifted, axes=2), axis=2),
                        axes=2,
                    ),
                    axes=3,
                ),
                axis=3,
            ),
            axes=3,
        )

    if not is_complex:
        kspace_shifted = np.abs(kspace_shifted)

    return kspace_shifted


def determine_necessary_shift(method=None, acqp=None, data_obj=None, db=False):
    """
    Calculate the necessary voxel and millimeter shifts required to align the MRI image
    based on the patient's position [Phase/Read Offset] and the scan orientation. The function supports
    determining shifts for axial, sagittal, and coronal slice orientations but currently
    only for the 'Head_Prone' and 'Head_Supine' patient position and specific read orientations.

    Parameters
    ----------
    method : dict, optional
        A dictionary containing various parameters related to the MRI scan method,
        including offsets and orientations. If None, uses the `method` attribute of the
        instance.
    acqp : dict, optional
        A dictionary containing acquisition parameters of the MRI scan. If None, uses
        the `acqp` attribute of the instance.
    data_obj : hypemri.sequence object, optional
    db : bool, default False
        If True, debug information (shifts in mm and voxel units) will be printed.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - The first list contains the calculated shift in voxel units for each dimension.
        - The second list contains the calculated shift in millimeters for each dimension.

    Raises
    ------
    Exception
        If the read orientation and slice orientation combination is not implemented.


    Notes
    -----
    The function currently only handles certain orientations and positions due to
    predefined settings in the conditional logic. More complex or different scenarios
    might require additional implementation efforts.
    """
    if method is None:
        method = data_obj.method
    if acqp is None:
        acqp = data_obj.acqp

    # offsets [mm]
    read_offset = method["PVM_SPackArrReadOffset"]
    phase_offset = method["PVM_SPackArrPhase1Offset"]
    slice_offset = method["PVM_SPackArrSliceOffset"]

    # patient position (usually Head_Prone)
    patient_pos = acqp["ACQ_patient_pos"]
    read_orient = method["PVM_SPackArrReadOrient"]
    slice_orient = method["PVM_SPackArrSliceOrient"]
    mm_slice_gap = method["PVM_SPackArrSliceGap"]

    mm_read_csi, mm_phase_csi, mm_slice_csi = define_imageFOV_parameters(
        data_obj=data_obj
    )
    (
        dim_read_csi,
        dim_phase_csi,
        dim_slice_csi,
    ) = define_imagematrix_parameters(data_obj=data_obj)

    shift_vox = [0, 0, 0]
    shift_mm = [0, 0, 0]

    if patient_pos == "Head_Prone":
        if slice_orient == "axial":
            if read_orient == "L_R":
                shift_mm = read_offset, phase_offset, slice_offset
                shift_vox[0] = -shift_mm[0] / mm_phase_csi * dim_phase_csi
                shift_vox[1] = -shift_mm[1] / mm_slice_csi * dim_slice_csi
                shift_vox[2] = -shift_mm[2] / mm_read_csi * dim_read_csi
            elif read_orient == "A_P":
                raise Exception(
                    "Shift not yet implemented for  read orientation: "
                    + read_orient
                    + "for slice_orient: "
                    + slice_orient
                )
                pass

            else:
                raise Exception(
                    "Shift not yet implemented for  read orientation: "
                    + read_orient
                    + "for slice_orient: "
                    + slice_orient
                )
                pass

        # for now do the same to get values:
        elif slice_orient == "sagittal":
            raise Exception(
                "Shift not yet implemented for  read orientation: "
                + read_orient
                + "for slice_orient: "
                + slice_orient
            )
            if read_orient == "H_F":
                shift_mm = read_offset, phase_offset

            elif read_orient == "A_P":
                shift_mm = read_offset, phase_offset

            else:
                pass

        # for now do the same to get values:
        elif slice_orient == "coronal":
            raise Exception(
                "Shift not yet implemented for  read orientation: "
                + read_orient
                + "for slice_orient: "
                + slice_orient
            )
            if read_orient == "H_F":
                shift_mm = read_offset, phase_offset
            elif read_orient == "L_R":
                shift_mm = read_offset, phase_offset
            else:
                pass

        # for now do the same to get values:
        else:
            raise Exception("unknown slice orientation: " + slice_orient)
            pass
    elif patient_pos == "Head_Supine":
        if slice_orient == "axial":
            if read_orient == "L_R":
                shift_mm = phase_offset, read_offset, slice_offset
                shift_vox[0] = shift_mm[1] / mm_phase_csi * dim_phase_csi
                shift_vox[1] = -shift_mm[0] / mm_slice_csi * dim_slice_csi
                shift_vox[2] = shift_mm[2] / mm_read_csi * dim_read_csi
            else:
                raise Exception(
                    "Shift not yet implemented for  read orientation: "
                    + read_orient
                    + "for slice_orient: "
                    + slice_orient
                )
                pass
    else:
        pass
    if db:
        print(f"shift_mm {shift_mm}")
        print(f"shift_vox {shift_vox}")

    return shift_vox, shift_mm


def format_y_tick_labels(ax, width=15, font_type="monospace"):
    ticks = ax.get_yticks()
    formatted_labels = []
    for tick in ticks:
        if "e" in f"{tick:.1e}":
            formatted_label = f"{tick:.1e}"
        else:
            formatted_label = f"{tick:>{width}.4g}"
        formatted_labels.append(formatted_label)
    ax.set_yticklabels(formatted_labels, family=font_type)


def get_plotting_extent(data_obj=None):
    """
    return the extent for plotting 2D data
    """
    patient_pos = data_obj.acqp["ACQ_patient_pos"]
    read_orient = data_obj.method["PVM_SPackArrReadOrient"]
    slice_orient = data_obj.method["PVM_SPackArrSliceOrient"]
    mm_slice_gap = data_obj.method["PVM_SPackArrSliceGap"]
    ax_ext, sag_ext, cor_ext = get_extent(data_obj=data_obj)

    # init default values:
    plotting_extent = [1, 1, 1, 1]
    if patient_pos == "Head_Prone":
        if slice_orient == "axial":
            if read_orient == "L_R":
                plotting_extent = ax_ext
            else:
                raise Exception(
                    "unknown read orientation: "
                    + read_orient
                    + " for slice_orient: "
                    + slice_orient
                )
                pass

        # for now do the same to get values:
        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                plotting_extent = sag_ext
            else:
                raise Exception(
                    "unknown read orientation: "
                    + read_orient
                    + " for slice_orient: "
                    + slice_orient
                )
                pass

        # for now do the same to get values:
        elif slice_orient == "coronal":
            if read_orient == "H_F":
                plotting_extent = cor_ext
            else:
                raise Exception(
                    "unknown read orientation: "
                    + read_orient
                    + " for slice_orient: "
                    + slice_orient
                )
                pass

        # for now do the same to get values:
        else:
            raise Exception("unknown slice orientation: " + slice_orient)
            pass

    # same as for head_prone, has to be changed later!!!!
    elif patient_pos == "Head_Supine":
        if slice_orient == "axial":
            if read_orient == "L_R":
                plotting_extent = ax_ext
            else:
                raise Exception(
                    "unknown read orientation: "
                    + read_orient
                    + " for slice_orient: "
                    + slice_orient
                )
                pass

        # for now do the same to get values:
        elif slice_orient == "sagittal":
            if read_orient == "H_F":
                plotting_extent = sag_ext
            else:
                raise Exception(
                    "unknown read orientation: "
                    + read_orient
                    + " for slice_orient: "
                    + slice_orient
                )
                pass

        # for now do the same to get values:
        elif slice_orient == "coronal":
            if read_orient == "H_F":
                plotting_extent = cor_ext
            else:
                raise Exception(
                    "unknown read orientation: "
                    + read_orient
                    + " for slice_orient: "
                    + slice_orient
                )
                pass

        # for now do the same to get values:
        else:
            raise Exception("unknown slice orientation: " + slice_orient)
            pass

    else:
        raise Exception("unknown patient_position: " + patient_pos)

    return plotting_extent


def load_plot_params(param_file=None, path_to_param_file=None, data_obj=None):
    """
    Load plotting parameters for the `plotting` function from a JSON file.

    This function is used to read in a set of plotting parameters that are saved
    in a JSON format. These parameters can then be utilized by the `plotting` function
    to customize the resulting plots.

    Parameters
    ----------
    param_file : str, optional
        Name of the JSON file containing the plotting parameters.
        If the filename doesn't include '.json', it will be appended.
    path_to_param_file : str, optional
        Directory path to where the `param_file` resides. If not provided,
        the function will look in the parent directory of `self.path`.

    Returns
    -------
    plot_params : dict
        Dictionary containing the loaded plotting parameters. If the loading
        process fails, the function returns None.

    Example
    -------
    obj.load_plot_params(param_file="plot_settings", path_to_param_file="path/to/directory")

    Note: An illustrative usage in the context of `plot2d` would further clarify its role.
    """

    import json
    import os

    # Ensure a filename is provided
    if param_file is None:
        Warning("param_file has to be defined")
        return {}

    # Determine the directory of the parameters file.
    # Use the specified path or the parent directory of `self.path` if not provided
    if path_to_param_file is None:
        if data_obj is not None:
            import sys

            if hasattr(data_obj, "savepath"):
                path_parentfolder = data_obj.savepath
            elif hasattr(data_obj, "path"):
                if sys.platform == "win32":
                    path_parentfolder = str(data_obj.path)[
                        0 : str(data_obj.path).rfind("\\")
                    ]
                elif sys.platform == "linux":
                    path_parentfolder = str(data_obj.path)[
                        0 : str(data_obj.path).rfind("/")
                    ]
                else:
                    path_parentfolder = str(data_obj.path)[
                        0 : str(data_obj.path).rfind("/")
                    ]

            else:
                return None
        else:
            return None
    else:
        path_parentfolder = path_to_param_file

    # Ensure the file has a '.json' extension
    if not param_file.endswith(".json"):
        param_file = param_file + ".json"

    # Attempt to load the JSON file from the constructed or provided path
    try:
        with open(os.path.join(path_parentfolder, param_file)) as f:
            plot_params = json.load(f)
    except:
        # If the above fails, try loading directly from the `param_file` as an absolute path
        try:
            with open(os.path.join(param_file)) as f:
                plot_params = json.load(f)
        except:
            # Notify the user if loading fails
            Warning("loading plot parameters did not work")
            return {}

    # Ensure the object has an attribute `plot_params` to store the loaded parameters
    if not hasattr(data_obj, "plot_params"):
        setattr(data_obj, "plot_params", "")

    # Try saving the loaded parameters to the object's attribute
    try:
        data_obj.plot_params = plot_params
    except:
        Warning("Saving plot parameters did not work")

    return plot_params


# function to perform input_data interpolation:
def img_interp(
    metab_image=None,
    interp_factor=None,
    interp_size=None,
    cmap=None,
    overlay=True,
    interp_method="bilinear",
    threshold=0,
):
    """
    Generate an interpolated version of the input input_data using matplotlib's imshow and _resample functions.

    This function scales the input input_data by a given interpolation factor or to a specific size. It can apply a colormap,
    threshold for overlay masking, and supports various interpolation methods.

    Args:
        metab_image (np.array, optional): Input 2D input_data array.
        interp_factor (float, optional): Factor by which to scale the input_data. Ignored if interp_size is provided.
        interp_size (tuple of ints, optional): Desired output size of the input_data as (height, width). Overrides interp_factor.
        cmap (str, optional): Colormap to be applied. Defaults to None which implies no colormap.
        overlay (bool, optional): If True, applies a threshold mask where values below the threshold are set to NaN. Defaults to True.
        interp_method (str, optional): Method of interpolation to be used. Options include 'nearest' (default), 'bilinear',
                                       'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                                       'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'.
                                       Each method offers a different balance of quality and performance.
        threshold (float, optional): Threshold value used for masking in overlay mode. Defaults to 0.

    Returns:
        np.array: The interpolated (and potentially masked) 2D input_data.
    Examples:
        from hypermri.utils.utils_general import img_interp
        import matplotlib.pyplot as plot
        test = np.random.rand(20,20)
        test_interp1 = img_interp(metab_image=test, interp_factor=2, cmap='gray', interp_method='bilinear')
        test_interp2 = img_interp(metab_image=test, interp_factor=4, cmap='gray', interp_method='bilinear')
        test_interp3 = img_interp(metab_image=test, interp_factor=4, cmap='gray', interp_method='lanczos')
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(test)
        plt.subplot(2,2,2)
        plt.imshow(test_interp1)
        plt.subplot(2,2,3)
        plt.imshow(test_interp2)
        plt.subplot(2,2,4)
        plt.imshow(test_interp3)
    """

    # skip interpolation if interpolation factor is 1
    if interp_factor == 1 and interp_size == None:
        if overlay is True:
            metab_image[metab_image <= threshold] = np.nan

        interpolated = metab_image
    else:
        # partially taken from:
        # https://stackoverflow.com/questions/52419509/get-interpolated-data-from-imshow
        from matplotlib.image import _resample
        from matplotlib.transforms import Affine2D
        import matplotlib.pyplot as plt

        if interp_factor:
            # define new input_data size:
            out_dimensions = (
                metab_image.shape[0] * interp_factor,
                metab_image.shape[1] * interp_factor,
            )

            metab_image_max = np.max(metab_image)
            metab_image_min = np.min(metab_image)

            # normalize (necessary for _resample)
            metab_image = (metab_image - metab_image_min) / (
                metab_image_max - metab_image_min
            )

            # generate axis object
            _, axs = plt.subplots(1)
            transform = Affine2D().scale(interp_factor, interp_factor)
            img = axs.imshow(metab_image, interpolation=interp_method, cmap=cmap)

            interpolated = _resample(
                img, metab_image, out_dimensions, transform=transform
            )

            # clear axis
            axs.clear()
            plt.close()

            # rescale
            interpolated = (
                interpolated * (metab_image_max - metab_image_min)
            ) + metab_image_min

        else:
            out_dimensions = interp_size
            interp_factor_0 = interp_size[0] / metab_image.shape[0]
            interp_factor_1 = interp_size[1] / metab_image.shape[1]
            logging.debug(interp_factor_0, interp_factor_1)
            if interp_factor_0 != interp_factor_1:
                logging.warning(
                    "Desired interp_size {} is not a multiple of metab_image size {}".format(
                        interp_size, metab_image.shape
                    )
                )
                # needs to be scale later:
                metab_image_max = np.max(metab_image)

                # normalize (necessary for _resample)
                metab_image = metab_image / metab_image_max

                # generate axis object
                _, axs = plt.subplots(1)
                transform = Affine2D().scale(interp_factor_0, interp_factor_1)
                img = axs.imshow(metab_image, interpolation=interp_method, cmap=cmap)

                interpolated = _resample(
                    img, metab_image, out_dimensions, transform=transform
                )

                # clear axis
                axs.clear()
                plt.close()

                # rescale
                interpolated = interpolated * metab_image_max

            else:
                interp_factor = interp_factor_0
                # needs to be scale later:
                metab_image_max = np.max(metab_image)

                # normalize (necessary for _resample)
                metab_image = metab_image / metab_image_max

                # generate axis object
                _, axs = plt.subplots(1)
                transform = Affine2D().scale(interp_factor, interp_factor)
                img = axs.imshow(metab_image, interpolation=interp_method, cmap=cmap)

                interpolated = _resample(
                    img, metab_image, out_dimensions, transform=transform
                )

                # clear axis
                axs.clear()
                plt.close()

                # rescale
                interpolated = interpolated * metab_image_max

        # apply mask
        if overlay is True:
            interpolated[interpolated <= threshold] = np.nan

    return interpolated


def mask_reshaper(mask=None, input_axes=None, match_standard_orient=False):
    if mask is None:
        print("Warning: no mask defined!")
        return False

    if input_axes is None:
        # assume that the input masks are from a multislice axial input_data:
        input_axes = [1, 2, 3]

    # Create a list with 6 elements initialized to 'np.newaxis'
    reshaped_axes = [np.newaxis] * 6

    # Replace the positions defined in input_axes with the value from 'range(0, len(input_axes))'
    for i, ax in enumerate(input_axes):
        reshaped_axes[ax] = slice(None)

    # Reshape the mask using the reshaped_axes
    reshaped_mask = mask[tuple(reshaped_axes)]  # Use tuple unpacking here

    if match_standard_orient:
        # reshaped_mask = np.array([np.rot90(arr, -1) for arr in reshaped_mask])
        reshaped_mask = np.rot90(reshaped_mask, k=-1, axes=(2, 3))

    return reshaped_mask


def calc_overlap(high_res_image, low_res_image):
    """
    Calculate the percentage of overlap between each voxel of a high-resolution input_data
    and the voxels of a low-resolution input_data.

    Parameters
    ----------
    high_res_image : numpy.ndarray
        A 2D high-resolution input_data.
    low_res_image : numpy.ndarray
        A 2D low-resolution input_data.

    Returns
    -------
    dict
        A dictionary where each key is a tuple representing the coordinates of a voxel
        in the high-resolution input_data, and the value is a list of tuples, each containing
        the coordinates of a low-resolution input_data voxel and the percentage of overlap with
        the high-resolution voxel.

    Raises
    ------
    AssertionError
        If either of the input images is not 2D.

    Notes
    -----
    The function assumes that both images cover the same physical extent and that
    their voxel/pixel sizes are determined by their shapes.

    Examples
    --------
    >>> high_res_image = np.random.rand(4, 4)
    >>> low_res_image = np.random.rand(3, 3)
    >>> overlap = calc_overlap(high_res_image, low_res_image)
    >>> print(overlap)
    """

    # Ensure the input images are 2D by removing any singleton dimensions
    high_res_image = np.squeeze(high_res_image)
    low_res_image = np.squeeze(low_res_image)

    # Ensure the input images are 2D
    assert (
        np.ndim(high_res_image) == 2
    ), f"high_res_image should be 2D but is {np.ndim(high_res_image)}D"
    assert (
        np.ndim(low_res_image) == 2
    ), f"low_res_image should be 2D but is {np.ndim(low_res_image)}D"

    # Calculate the shapes of the images
    high_res_shape = high_res_image.shape
    low_res_shape = low_res_image.shape

    # Dictionary to store overlap information
    accurate_overlap = {}

    # Calculate voxel dimensions for both images
    high_res_voxel_height = 1.0 / high_res_shape[0]
    high_res_voxel_width = 1.0 / high_res_shape[1]
    low_res_voxel_height = 1.0 / low_res_shape[0]
    low_res_voxel_width = 1.0 / low_res_shape[1]

    # Iterate over high-resolution input_data voxels
    for y_high in range(high_res_shape[0]):
        for x_high in range(high_res_shape[1]):
            # Define bounds of the current high-resolution voxel
            high_res_bounds = {
                "left": x_high * high_res_voxel_width,
                "right": (x_high + 1) * high_res_voxel_width,
                "top": y_high * high_res_voxel_height,
                "bottom": (y_high + 1) * high_res_voxel_height,
            }

            overlaps = []

            # Iterate over low-resolution input_data voxels to calculate overlaps
            for y_low in range(low_res_shape[0]):
                for x_low in range(low_res_shape[1]):
                    # Define bounds of the current low-resolution voxel
                    low_res_bounds = {
                        "left": x_low * low_res_voxel_width,
                        "right": (x_low + 1) * low_res_voxel_width,
                        "top": y_low * low_res_voxel_height,
                        "bottom": (y_low + 1) * low_res_voxel_height,
                    }

                    # Calculate the overlapping area dimensions
                    overlap_width = min(
                        high_res_bounds["right"], low_res_bounds["right"]
                    ) - max(high_res_bounds["left"], low_res_bounds["left"])
                    overlap_height = min(
                        high_res_bounds["bottom"], low_res_bounds["bottom"]
                    ) - max(high_res_bounds["top"], low_res_bounds["top"])

                    # If there is an overlap, calculate its percentage
                    if overlap_width > 0 and overlap_height > 0:
                        overlap_area = overlap_width * overlap_height
                        high_res_voxel_area = (
                            high_res_voxel_width * high_res_voxel_height
                        )

                        overlap_percentage = np.round(
                            (overlap_area / high_res_voxel_area) * 100, 0
                        )
                        overlaps.append(((x_low, y_low), overlap_percentage))

            # Store the calculated overlap information
            accurate_overlap[(x_high, y_high)] = overlaps

    return accurate_overlap


def calc_coverage(input_data=None, mask=None, return_nans=False, mask_drawn_on="axial"):
    """
    Computes the coverage percentage of a high-resolution 6D mask over each voxel in a low-resolution 6D volume.
    This function scales the mask to match the dimensions of the input_data, allowing for coverage calculation
    even when the resolution of the input_data and mask differ. The computation is based on the second to fourth
    dimensions (ignoring the first dimension for both input_data and mask), which typically represent spatial dimensions
    in imaging data.

    Parameters:
    - input_data (numpy.ndarray, optional): A low-resolution 6D volume of shape (echoes, z, x, y, reps, chans), where each dimension represents
      echoes, depth (z), row (x), column (y), repetitions (reps), and channels (chans), respectively.
    - mask (numpy.ndarray, optional): A high-resolution 6D mask of the same shape as `input_data`. It is used to calculate the coverage over
      the `input_data`.
    - return_nans (bool, optional): If True, positions in the coverage array where the coverage is calculated to be 0 will be replaced with NaNs. Defaults to False.

    Returns:
    - numpy.ndarray: A 6D array of the same shape as `input_data`, representing the percentage coverage of the mask for each voxel in the input_data.
      Coverage values are scaled between 0 and 1, where 1 indicates 100% coverage.

    Notes:
    - The function automatically adjusts for any dimension of the input_data or mask having a size of 1, by broadcasting the coverage calculation
      across the higher resolution dimension. This is useful for handling data with varying resolutions or when comparing different imaging modalities.
    """
    # Extend mask to 7D if input_data is 7D
    if input_data.ndim == 7:
        # Extend the 6D mask by adding a 7th dimension with size matching the 7th dimension of input_data
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, input_data.shape[6], axis=-1)

    # Adjust mask dimensions to match input_data dimensions if necessary
    mask_adjusted = mask
    for dim_index in [
        0,
        4,
        5,
    ]:  # Check and adjust for the 1st, 5th, and 6th dimensions independently
        if (
            input_data.shape[dim_index] > mask.shape[dim_index]
            and mask.shape[dim_index] == 1
        ):
            # Repeat the mask to match the input_data's resolution in this dimension
            mask_adjusted = np.repeat(
                mask_adjusted, input_data.shape[dim_index], axis=dim_index
            )

    if np.ndim(mask) < 6:
        from ..utils.utils_anatomical import make_mask_6D

        mask = make_mask_6D(
            mask_dict=None, mask_key=None, mask=mask, mask_drawn_on="axial"
        )

    # Extract the relevant dimensions
    low_res_shape = input_data.shape[1:4]
    high_res_shape = mask.shape[1:4]

    # Adjust the output shape based on the shapes of input_data and mask
    output_shape_dims = []
    for dim_a, dim_b in zip(input_data.shape, mask.shape):
        if dim_b == 1:
            output_shape_dims.append(1)
        else:
            output_shape_dims.append(dim_a)
    output_shape = tuple(output_shape_dims)

    # Calculate the scaling factors for each dimension
    depth_scale = high_res_shape[0] / (low_res_shape[0] if low_res_shape[0] != 1 else 1)
    row_scale = high_res_shape[1] / (low_res_shape[1] if low_res_shape[1] != 1 else 1)
    col_scale = high_res_shape[2] / (low_res_shape[2] if low_res_shape[2] != 1 else 1)
    if depth_scale < 1.0 or row_scale < 1.0 or col_scale < 1.0:
        logger.critical(
            f"Attention! Mask dims smaller than input data, factor: {depth_scale},{row_scale},{col_scale}!"
        )
    coverage = np.zeros(output_shape)

    # Loop over the relevant dimensions to compute the coverage
    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            for k in range(output_shape[3]):
                # Define the corresponding region in the high-res mask
                start_depth = int(i * depth_scale) if high_res_shape[0] > 1 else 0
                end_depth = (
                    min(int((i + 1) * depth_scale), mask.shape[1])
                    if high_res_shape[0] > 1
                    else 1
                )

                start_row = int(j * row_scale) if high_res_shape[1] > 1 else 0
                end_row = (
                    min(int((j + 1) * row_scale), mask.shape[2])
                    if high_res_shape[1] > 1
                    else 1
                )

                start_col = int(k * col_scale) if high_res_shape[2] > 1 else 0
                end_col = (
                    min(int((k + 1) * col_scale), mask.shape[3])
                    if high_res_shape[2] > 1
                    else 1
                )

                # number of all points:
                nelems = (
                    (end_depth - start_depth)
                    * (end_row - start_row)
                    * (end_col - start_col)
                )

                # Compute the mean value in the region
                coverage[:, i, j, k, :, :] = (
                    np.nansum(
                        mask[
                            :,
                            start_depth:end_depth,
                            start_row:end_row,
                            start_col:end_col,
                            :,
                            :,
                        ],
                        axis=(1, 2, 3),
                    )
                    / nelems
                )

    # replace 0s with NaNs
    if return_nans:
        coverage[coverage == 0] = np.nan

    return coverage


def apply_mask(
    input_data: object | None = None,
    mask_dict: dict | None = None,
    mask_key: str | None = None,
    mask: object | None = None,
    return_nans: bool = False,
    avg_all_space_dims: bool = False,
    mask_drawn_on: str = "axial",
    mask_slice_ind: int | None = None,
    bitwise: bool = False,
    **kwargs,
):
    """
    Applies a mask to input data to extract time curves from Regions of Interest (ROIs). This function supports specifying masks via a dictionary
    and can handle multi-dimensional data by averaging across spatial dimensions or applying bitwise operations based on coverage thresholds.

    Parameters:
    - input_data (numpy.ndarray, optional): The input data to be masked, expected to be a multi-dimensional array
    - mask_dict (dict): A dictionary containing masks. If provided, `mask_key` is used to select the appropriate mask from this dictionary.
    - mask_key (str): The key corresponding to the desired mask in `mask_dict`.
    - mask_drawn_on (str, optional): Specifies the orientation of the mask relative to the input data. Supported values are "axial" (default) or "coronal".
        Has to be passed if mask is 3D (old)
    - avg_all_space_dims (bool, optional): If True, averages the masked data across all spatial dimensions.
    - return_nans (bool, optional): If True, replaces voxels not in the mask with NaN; otherwise, they are set to 0.
    - mask_slice_ind (int, optional): Specifies a particular slice of the mask to use, based on the orientation.
    - bitwise (bool, optional): If True, applies a bitwise operation to the coverage data before masking, using a threshold (default 50%) to determine mask inclusion.
    - **kwargs: Additional keyword arguments for extended functionality. Supported keywords include `bitwise_lower_threshold` to customize the bitwise threshold and
        `provided_dims` to indicate which dimensionality the input_data has.

    Returns:
    - numpy.ndarray: The masked input data  or False if input data is not provided or the mask cannot be applied.

    Notes:
    - This function adjusts the orientation and dimensionality of the mask to match the input data, facilitating the extraction of meaningful time curves from specified ROIs.
    - If `mask_dict` and `mask_key` are provided but the specified mask cannot be found, a warning is issued, and False is returned.
    """

    # no need to continue if there is no data:
    if input_data is None:
        return False

    # load mask from dict if possible
    if mask is None:
        if mask_dict is not None:
            if mask_key is not None:
                try:
                    mask = mask_dict[mask_key]
                except:
                    Warning("can't extract mask!")
                    return False
        else:
            pass

    from ..utils.utils_anatomical import make_mask_6D

    mask_copy = make_mask_6D(mask=mask, mask_drawn_on=mask_drawn_on)

    # use a 6D slice index:
    if mask_slice_ind is None:
        print("Warning: No mask_slice_ind was given!")
        pass

    else:
        # assert type(mask_slice_ind) == int,
        if mask_drawn_on == "axial":
            mask_copy = mask_copy[:, [mask_slice_ind], :, :, :, :]
        elif mask_drawn_on == "coronal":
            mask_copy = mask_copy[:, :, :, [mask_slice_ind], :, :]
        else:
            pass

    if np.ndim(input_data) < 6:
        from ..utils.utils_spectroscopy import make_NDspec_6Dspec

        # if the dimensionality of the input data was passed, use that:
        if "provided_dims" in kwargs:
            provided_dims = kwargs["provided_dims"]

            # convert any strings into correspoinding integers:
            provided_dims = get_indices_for_strings(strings=provided_dims)

            # reshape the input data:
            input_data = make_NDspec_6Dspec(
                input_data=input_data, provided_dims=provided_dims
            )
        else:
            input_data = make_NDspec_6Dspec(input_data=input_data)

    # check here if input_data and mask_copy have the same shape.

    # calculate coverage of mask:
    coverage = calc_coverage(
        input_data=input_data, mask=mask_copy, return_nans=return_nans
    )

    # mask is just 0 or 1, the default threshold is 0.5 but can be edited individually
    if bitwise:
        if "bitwise_lower_threshold" in kwargs:
            bitwise_lower_threshold = kwargs["bitwise_lower_threshold"]
        else:
            bitwise_lower_threshold = 0.5
        coverage[coverage >= bitwise_lower_threshold] = 1
        # set values smaller than threshold to 0/NaN
        if return_nans:
            coverage[coverage < bitwise_lower_threshold] = np.nan
        else:
            coverage[coverage < bitwise_lower_threshold] = 0.0

    # apply mask:
    masked_data = input_data * coverage

    # do average along all spatial dimensions:
    if avg_all_space_dims is True:
        masked_data = np.squeeze(np.nanmean(masked_data, axis=(1, 2, 3), keepdims=True))

    return masked_data


def convert_mask_to_diff_fov(
    mask_a=None, fov_a=None, fov_b=None, matrix_a=None, matrix_b=None
):
    """
    Converts a mask (A) to another FOV (B).

    Parameters
    ----------
    mask_a : np.ndarray
        The high-resolution input mask.
    fov_a : list or tuple of float
        The field of view for the input mask `mask_a`. (z-x-y)
    fov_b : list or tuple of float
        The desired field of view for the output mask. (z-x-y)
    matrix_a : list or tuple of int
        The matrix size of the input mask `mask_a`. (z-x-y)
    matrix_b : list or tuple of int
        The matrix size for the desired output mask. (z-x-y)

    Returns
    -------
    mask_b : np.ndarray
        The low-resolution mask with the desired field of view `fov_b` and resolution derived from `mask_a`.

    Notes
    -----
    The function resamples `mask_a` to match the field of view `fov_b` while maintaining the same resolution in
    x and y dimensions as `mask_a`.
    """

    # Define resolution of mask A
    res_a = [f / m for f, m in zip(fov_a, matrix_a)]

    # Define resolution of mask B
    res_b = [f / m for f, m in zip(fov_b, matrix_b)]

    # Calculate new matrix size for mask B with FOV of B and resolution of A
    matrix_new = [f / r for f, r in zip(fov_b, res_a)]
    mask_b = np.zeros(
        (
            mask_a.shape[0],
            mask_a.shape[1],
            int(np.round(matrix_new[1])),
            int(np.round(matrix_new[2])),
            mask_a.shape[4],
            mask_a.shape[5],
        )
    )

    # Calculate the difference in mask size
    diff_matrix = [b - a for b, a in zip(mask_b.shape, mask_a.shape)]

    # Calculate the indices for x-dimension
    if diff_matrix[2] > 0:
        xb_start = diff_matrix[2] // 2 + 1
        xb_end = diff_matrix[2] // 2 + mask_a.shape[2]
        xa_start = 0
        xa_end = -1
    elif diff_matrix[2] < 0:
        xa_start = np.abs(diff_matrix[2] // 2 - 1)
        xa_end = diff_matrix[2] // 2 + mask_a.shape[2]
        xb_start = 0
        xb_end = -1
    else:
        xa_start = 0
        xa_end = -1
        xb_start = 0
        xb_end = -1

    # Calculate the indices for y-dimension
    if diff_matrix[3] > 0:
        yb_start = diff_matrix[3] // 2 + 1
        yb_end = diff_matrix[3] // 2 + mask_a.shape[3]
        ya_start = 0
        ya_end = -1
    elif diff_matrix[3] < 0:
        ya_start = np.abs(diff_matrix[3] // 2 - 1)
        ya_end = diff_matrix[3] // 2 + mask_a.shape[3]
        yb_start = 0
        yb_end = -1
    else:
        ya_start = 0
        ya_end = mask_a.shape[3]
        yb_start = 0
        yb_end = mask_a.shape[3]

    # Fill mask_b with values from mask_a
    mask_b[:, :, xb_start:xb_end, yb_start:yb_end, :, :] = mask_a[
        :, :, xa_start:xa_end, ya_start:ya_end, :, :
    ]

    # Return the resized mask
    return mask_b


def estimate_noise(
    input_data=None,
    mask_dict=None,
    mask_key=None,
    mask=None,
    masks_drawn_on="axial",
    slice_ind=None,
    signal_domain="spectral",
    component="complex",
    in_mask=False,
    plot=False,
):
    """
    Estimate the standard deviation of noise from MRI images.

    This function calculates the standard deviation of noise in MRI images
    using a specified mask or mask generated from a mask dictionary.
    The noise is estimated from the region OUTSIDE (default) of the supplied mask.
    It can operate on a specific slice of the input data and supports plotting the
    results for visual inspection.
    Assumes first dimension to be in spectral dimension (to temporal=FID).
    The returned standard deviation of the noise will be calculated from the
    temporal dimension to be easily usable with the fitted amplitudes from
    utils_fitting.fit_data_pseudo_inv.

    Parameters
    ----------
    input_data : ndarray, optional
        The MRI data from which to estimate noise. The data should be a 5D
        ndarray, with dimensions.
    mask_dict : dict, optional
        A dictionary containing masks. A specific mask is selected using `mask_key`.
    mask_key : str, optional
        The key corresponding to the mask to be used from `mask_dict`.
    mask : ndarray, optional
        An alternative to using `mask_dict` and `mask_key`. Directly specify the mask to use.
        All voxel will be considered if no mask is provided.
    masks_drawn_on : str, default 'axial'
        The orientation of the masks.
    slice_ind : int, optional
        The specific slice index to use from `input_data`. If not specified, defaults to 7 with a warning.
    signal_domain : str, optional
        input_data signal domain, can be spectral (default) or temporal. If spectral, an inverse FT is performed.
    in_mask: bool, optional
        whether the noise is in the mask or outside the mask. Default is False.
    plot : bool, default False
        If True, plots the mask, real and imaginary components of the noise signal, and their histograms.

    Returns
    -------
    std_noise : float
        The estimated standard deviation of the noise, calculated as the average of the standard
        deviations of the real and imaginary components of the Fourier-transformed noise signal.

    Raises
    ------
    ValueError
        If `input_data` is not provided or if a mask cannot be loaded from `mask_dict` using `mask_key`.
    """
    from ..utils.utils_fitting import fit_rayleigh, rayleigh_pdf_fit

    # Validate input_data
    if input_data is None:
        raise ValueError("input_data must be provided!")

    # Check if mask is provided directly or needs to be fetched from mask_dict
    if mask is None:
        if mask_dict is None or mask_key is None:
            mask = np.ones_like(input_data, dtype=bool)
            in_mask = True
            warnings.warn("No mask provided, using all voxels for noise estimation!")
        else:
            try:
                mask = mask_dict[mask_key]
            except KeyError:
                raise ValueError(f"Cannot load mask_key {mask_key} from mask_dict!")
    else:
        pass
        
    if in_mask:
        # use voxels in the mask
        mask_inv = mask
    else:
        # use voxels outside the mask
        mask_inv = np.ones_like(mask) - mask

    
    if not np.iscomplexobj(input_data) and component == "complex":
        warnings.warn(
            f"input_data is not complex but parameter component is {component},"
            f"consider setting component=magnitude"
        )

    # Default slice index if not provided
    if slice_ind is None:
        slice_ind = input_data.shape[1] // 2
        warnings.warn(
            f"Warning: slice_ind was not passed, defaulting to slice_ind = {slice_ind}"
        )

    match signal_domain:
        # FFT of noise, assumes noise is stored in the first echo and channel
        case "spectral":
            input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=0), axis=0)
        # already in temporal dimension:
        case "temporal":
            input_data = input_data
        # assume spectral dimension:
        case _:
            input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=0), axis=0)

    # Apply mask and calculate noise
    noise = apply_mask(
        input_data,
        mask=mask_inv,
        mask_slice_ind=slice_ind,
        return_nans=True,
        mask_drawn_on=masks_drawn_on,
        bitwise=True,
    )
    
    noise = noise.flatten()
    noise_signal = noise[~np.isnan(noise)]

    # Binned dataset:
    density, bins = np.histogram(np.abs(noise_signal), bins=30, density=1.0)
    centers = (bins[:-1] + bins[1:]) / 2
    xlin = np.linspace(bins[0], bins[-1], 100)

    popt, pcov = fit_rayleigh(input_data=np.abs(noise_signal), plot=False)

    # Calculate and return the mean and standard deviation of noise
    std_real = np.nanstd(np.real(noise_signal))
    std_imag = np.nanstd(np.imag(noise_signal))
    mean_real = np.nanmean(np.real(noise_signal))
    mean_imag = np.nanmean(np.imag(noise_signal))

    # Plotting section, if requested
    if plot:
        fig, ax = plt.subplots(1, 5, tight_layout=True, figsize=(15, 3))
        im = ax[0].imshow(
            np.rot90(
                np.squeeze(mask_inv[0:1, slice_ind : slice_ind + 1, :, :, 0:1, 0:1])
            )
        )
        plt.colorbar(im, ax=ax[0], label="Mask Coverage", shrink=0.3)
        ax[0].set_title("Inverse Signal Mask")
        ax[1].plot(np.real(noise_signal), color="blue")
        ax[1].plot(np.imag(noise_signal), color="red", alpha=0.5)
        ax[1].set_title("Signal Real/Imaginary")
        ax[2].hist(np.real(noise_signal), bins=30, color="blue", density=1.0)
        ax[2].hist(np.imag(noise_signal), bins=30, color="red", alpha=0.5, density=1)
        ax[2].set_title("Histogram Real/Imaginary")

        ax[3].plot(np.abs(noise_signal))
        ax[3].set_title("Signal Magnitude")
        ax[4].hist(np.abs(noise_signal), bins=30, density=1)
        ax[4].plot(xlin, rayleigh_pdf_fit(xlin, *popt), color="r")
        ax[4].axvline(np.mean(np.abs(noise_signal)), color="k")
        ax[4].axvline(popt[0], color="r")
        ax[4].axvline(popt[1], color="r")
        ax[4].set_title("Histogram Magnitude")
        plt.show()

    # Output the mean and standard deviation of noise
    print(f"Mean noise:                      {mean_real:.2f} + {mean_imag:.2f}*1i")
    print(f"Mean |noise|:                    {np.mean(np.abs(noise_signal))}")
    print(f"Standard deviation of noise: σ = {std_real:.2f} + {std_imag:.2f}*1i")
    print(f"Rayleigh Fit of |noise|:     σ = {popt[1]:.2f}")

    if component == "complex":
        std_noise = (std_real + std_imag) / 2
    elif component == "magnitude" or component == "mag":
        std_noise = popt[1]
    else:
        std_noise = (std_real + std_imag) / 2

    return std_noise


def interpolate_dataset(
    input_data=None,
    data_obj=None,
    interp_size=None,
    interp_method="linear",
    use_multiprocessing=True,
    number_of_cpu_cores=None,
    **kwargs,
):
    """
    Interpolates the given 6D dataset to a desired shape using the specified interpolation method.

    Parameters
    ----------
    input_data : ndarray, optional
        The input data to be interpolated. If not provided, it will be retrieved from `data_obj`.
    data_obj : object, optional
        An object containing the dataset if `input_data` is not provided.
    interp_size : tuple
        The desired shape for the interpolated data.
    interp_method : str, optional
        The interpolation method to be used. Default is "linear".
        Supported methods: "linear", "nearest", "slinear", "cubic", "quintic", "pchip".
    use_multiprocessing : bool, optional
        Whether to use parallel computing for interpolation. Default is True.
    number_of_cpu_cores : int, optional
        Number of CPU cores to be used if multiprocessing is enabled.
        If not provided, it defaults to half the available cores.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the interpolator.

    Returns
    -------
    ndarray
        The interpolated data.

    Notes
    -----
    This function performs interpolation on a 6D dataset. It can handle various interpolation methods
    and supports multiprocessing for faster computation.

    If `use_multiprocessing` is enabled, the function attempts to use the joblib library to parallelize
    the interpolation process. If `number_of_cpu_cores` is not specified, it defaults to half the available
    CPU cores, ignoring hyperthreading cores.

    The input data is converted to a 6D format using the `make_NDspec_6Dspec` function before interpolation.
    The interpolation grid is defined based on the desired `interp_size`, and the interpolation is performed
    using scipy's `RegularGridInterpolator`.

    Warnings
    --------
    If `interp_method` is not one of the supported methods, a warning is issued and "linear" interpolation
    is used as a fallback.

    Examples
    --------
    >>> input_data = np.random.rand(10, 10, 10, 10, 10, 10)
    >>> interp_size = (20, 20, 20, 20, 20, 20)
    >>> interpolated_data = interpolate_dataset(input_data=input_data, interp_size=interp_size)
    """

    from scipy.interpolate import RegularGridInterpolator
    import time
    from tqdm.auto import tqdm

    if use_multiprocessing:
        try:
            from joblib import Parallel, delayed, cpu_count

            # try to set the number of usable cpu cores to the amount of
            # available cores
            if number_of_cpu_cores is None:
                # int-divide by 2 because hyperthreading cores dont count/help:
                number_of_cpu_cores = cpu_count() // 2
            else:
                pass

        except:
            use_multiprocessing = False

    if input_data is None:
        if data_obj is None:
            logger.critical("Please specify input_data")
        else:
            input_data = data_obj.seq2d_oriented

    if interp_size is None:
        logger.critical("Don't know what to do, returning input_data")
        return input_data

    if np.array_equal(interp_size, input_data.shape):
        logger.critical(
            "interp_size can not be exact size of input_data, returning input_data"
        )
        return input_data

    # make sure input data is 6D:
    from ..utils.utils_spectroscopy import make_NDspec_6Dspec

    input_data = make_NDspec_6Dspec(input_data=input_data)

    # make sure interp_size is a integeer
    interp_size = [int(i) for i in interp_size]
    # make sure there is no 0:
    interp_size = [i if i != 0 else 1 for i in interp_size]

    # check method:
    allowed_interp_methods = [
        "linear",
        "nearest",
        "slinear",
        "cubic",
        "quintic",
        "pchip",
    ]

    if interp_method not in allowed_interp_methods:
        warnings.warn(
            f"Unknown interpolation method: {interp_method}, available options are {allowed_interp_methods},"
            f" using linear instead."
        )
        interp_method = "linear"

    ## sample positions: -----------------------------------------------------------------------------------------------
    # echo/fid positions:
    e_sample = np.linspace(1, input_data.shape[0], input_data.shape[0])
    # z, x, y positions:
    if input_data.shape[1] == 1:
        z_sample = np.array(
            [
                0,
            ]
        )
    else:
        z_sample = np.linspace(
            -input_data.shape[1] / 2, input_data.shape[1] / 2, input_data.shape[1]
        )
    x_sample = np.linspace(
        -input_data.shape[2] / 2, input_data.shape[2] / 2, input_data.shape[2]
    )
    y_sample = np.linspace(
        -input_data.shape[3] / 2, input_data.shape[3] / 2, input_data.shape[3]
    )
    # repetitions "positions"
    reps_sample = np.linspace(1, input_data.shape[4], input_data.shape[4])
    # channel "positions"
    c_sample = np.linspace(1, input_data.shape[5], input_data.shape[5])

    ## interpolation positions: ----------------------------------------------------------------------------------------
    # define resolution + FOV:
    x_res = x_sample[-1] - x_sample[-2]
    x_fov = x_res * input_data.shape[2]
    y_res = y_sample[-1] - y_sample[-2]
    y_fov = y_res * input_data.shape[3]
    if len(z_sample) == 1:
        z_res = 1
    else:
        z_res = z_sample[-1] - z_sample[-2]
    z_fov = z_res * input_data.shape[1]

    # echo/fid positions:
    e_interp = np.linspace(e_sample[0], e_sample[-1], interp_size[0])
    # z, x, y positions:
    z_interp = define_grid_lines(fov=z_fov, mat=interp_size[1])
    x_interp = define_grid_lines(fov=x_fov, mat=interp_size[2])
    y_interp = define_grid_lines(fov=y_fov, mat=interp_size[3])
    # repetitions "positions"
    reps_interp = np.linspace(reps_sample[0], reps_sample[-1], interp_size[4])
    # channel "positions"
    c_interp = np.linspace(c_sample[0], c_sample[-1], interp_size[5])
    # ------------------------------------------------------------------------------------------------------------------
    # sample points
    dims = [e_sample, z_sample, x_sample, y_sample, reps_sample, c_sample]
    # interpolation points
    interps = [e_interp, z_interp, x_interp, y_interp, reps_interp, c_interp]

    # Identify dimensions that do not need interpolation
    no_interp_dims = [
        i
        for i, (dim, interp) in enumerate(zip(dims, interps))
        if (len(dim) > 1 and len(interp) == 1)
        or np.array_equal(dim, interp)
        or (len(dim) == 1 and len(interp) > 1)
    ]

    # Identify dimensions where interpolation is necessary
    interp_dims_needed = [i for i in range(len(dims)) if i not in no_interp_dims]
    print(f"no_interp_dims {no_interp_dims}")
    print(f"interp_dims_needed {interp_dims_needed}")

    ## define sampling + interpolation grids:
    sample_grid_samplepoints = [dims[s] for s in interp_dims_needed]
    interp_grid_samplepoints = [interps[s] for s in interp_dims_needed]
    interp_grid = np.meshgrid(*interp_grid_samplepoints, indexing="ij")

    # Reorder data such that non-interpolated dimensions are at the front
    order = no_interp_dims + interp_dims_needed

    # reorder data to be able to perform fast interation
    input_data_reordered = np.moveaxis(input_data, order, range(input_data.ndim))

    # Define the interpolated data shape
    interp_shape = tuple(
        len(interp) if len(interp) > 1 else input_data.shape[i]
        for i, interp in enumerate(interps)
    )

    # init empty array
    interp_data = np.empty(interp_shape, dtype=input_data.dtype)

    # reshape
    interp_data = interp_data.transpose(order)

    # bounds_error is set to False, that means if fill_value is None (default), an extrapolation to outer grid points
    # will be performed:
    if "fill_value" in kwargs:
        fill_value = kwargs["fill_value"]
    else:
        fill_value = np.nan  # set to None to have values extrapolated

    # for tqdm:
    total_iterations = np.prod([input_data.shape[i] for i in no_interp_dims])
    # start time
    st = time.time()
    # ------------------------------------------------------------------------------------------------------------------
    if use_multiprocessing:
        # define interpolation function:
        def interpolate_slice(index_no_interp):
            slice_data = input_data_reordered[index_no_interp]

            # intialize interpolator:
            interp = RegularGridInterpolator(
                points=sample_grid_samplepoints,
                values=slice_data,
                method=interp_method,
                bounds_error=False,
                fill_value=fill_value,
            )
            # perform interpolation:
            pts = interp(tuple(interp_grid))

            return (
                pts.reshape([len(interps[i]) for i in interp_dims_needed]),
                index_no_interp,
            )

        # Updated part: wrapping the iteration with tqdm for progress tracking
        results = Parallel(n_jobs=number_of_cpu_cores)(
            delayed(interpolate_slice)(index)
            for index in tqdm(
                np.ndindex(*[input_data.shape[i] for i in no_interp_dims]),
                desc="Interpolating",
                total=total_iterations,
            )
        )

        # fill results in interp_data:
        for interp_slice, index_no_interp in results:
            full_index = index_no_interp + (Ellipsis,)
            interp_data[full_index] = interp_slice
    else:
        for index_no_interp in tqdm(
            np.ndindex(*[input_data.shape[i] for i in no_interp_dims]),
            desc="Interpolating",
            total=total_iterations,
        ):
            slice_data = input_data_reordered[index_no_interp]
            # intialize interpolator:
            interp = RegularGridInterpolator(
                points=sample_grid_samplepoints,
                values=slice_data,
                method=interp_method,
                bounds_error=False,
                fill_value=fill_value,
            )
            # perform interpolation:
            pts = interp(tuple(interp_grid))

            full_index = index_no_interp + (Ellipsis,)
            interp_data[full_index] = pts

    # Correct way to compute inverse_order
    inverse_order = [order.index(i) for i in range(len(order))]
    # Apply the corrected inverse order to transpose interp_data back
    interp_data = interp_data.transpose(inverse_order)

    et = time.time()
    print("Execution time:", et - st, "seconds")

    return interp_data


def get_sampling_dt(data_obj=None, bw_hz=0, sampling_time_axis=None):
    """
    Calculates the sampling duration per point in [s] when a spectrum/FID was acquired
    Parameters
    ----------
    data_obj: data_obj, optional
    bw_hz: float, spectral bandwidth in [Hz]
    sampling_time_axis: ndarray, sampling time axis in [s]

    Returns
    -------
    sampling duration per point
    """
    sampling_dt = 0
    if data_obj is None:
        if bw_hz != 0:
            sampling_dt = 1 / bw_hz
        elif sampling_time_axis is not None and isinstance(
            sampling_time_axis, np.ndarray
        ):
            sampling_dt = sampling_time_axis[1] - sampling_time_axis[0]
        else:
            logger.critical("Pass either data_obj or bw_hz!")
    else:
        # CSI:
        sampling_dt = data_obj.method["PVM_DigDw"] / 1000.0
    return sampling_dt


def calc_sampling_time_axis(
    data_obj=None,
    sampling_dt=0,
    npoints=0,
    bw_hz=0,
):
    """
    This function returns sampling time axis (in an FID e.g.) in [s]
    Parameters
    ----------
    data_obj: data_obj, optional
    sampling_dt: sampling time per point if no data_obj is passed
    npoints: number of points of the time axis
    bw_hz: spectral bandwidth in [Hz]

    Returns
    -------
    array from 0 to (number of sampling points -1) * sampling duration per point
    """
    if sampling_dt <= 0:
        sampling_dt = get_sampling_dt(data_obj=data_obj, bw_hz=bw_hz)
    else:
        pass
    if data_obj is None:
        pass
    else:
        if npoints == 0 or npoints is None:
            # CSI:
            npoints = data_obj.method["PVM_SpecMatrix"]

    sampling_time_axis = np.linspace(0, sampling_dt * (npoints - 1), npoints)
    return sampling_time_axis


def calc_timeaxis(data_obj=None, start_with_0=True):
    """
    calculate the time range [in s] on which the data was acquired.

    ---------
    Parameters:

    start_with_0 (bool)
        wether the time axis should start with 0 or 1 input_data repetition time

    Return:
        Time axis [in seconds]
    """
    if data_obj is None:
        return None

    # init:
    time_axis = None
    # number of repetitions
    nr = data_obj.method["PVM_NRepetitions"]
    # scan time per input_data in  seconds
    image_repetition_time = data_obj.method["PVM_ScanTime"] / nr / 1000.0

    # time range start with 0:
    if start_with_0:
        time_axis = np.linspace(0, (nr - 1) * image_repetition_time, nr)
        # time range start with 1 input_data reptition time:
    else:
        time_axis = np.linspace(image_repetition_time, nr * image_repetition_time, nr)

    return time_axis


def add_counter(data_obj=None, counter_name=None, n_counts=1):
    # add n_counts to counter_name, if counter_name is not an attribute of
    # data_obj, it will be added.

    # data_obj has to be passed:
    if data_obj is None:
        raise Exception("data_obj can not be None!")
        return None
    # counter name has to be passed:
    if counter_name is None:
        raise Exception("counter_name can not be None!")
        return None

    # get ID from object
    obj_id = str(id(data_obj))

    # check if the counter_name dictonary already exists:
    init_attr(data_obj=data_obj, attr_name=counter_name)

    # add one count to the counter counter_name:
    # get number of counts:
    if isinstance(getattr(data_obj, counter_name), dict):
        if obj_id in getattr(data_obj, counter_name):
            getattr(data_obj, counter_name)[obj_id] += n_counts
        else:
            getattr(data_obj, counter_name)[obj_id] = n_counts
    else:
        setattr(data_obj, counter_name, {obj_id: n_counts})

    num_counter_name = getattr(data_obj, counter_name)[obj_id]

    return num_counter_name


def init_attr(data_obj=None, attr_name=None, attr_type=dict):
    # adds an attribute attr_name of type attr_type to
    # the bSSFP object

    # check if bssfp object has attribute attr_name, else create:
    if hasattr(data_obj, attr_name):
        pass
    else:
        setattr(data_obj, attr_name, attr_type)
    return True


def get_counter(data_obj=None, counter_name=None):
    # get the number in the counter_name entry

    # dont continue if no obkect was passed:
    if data_obj is None:
        raise Exception("No object was passed!")
        return None

    # dont continue if no name was passed:
    if counter_name is None:
        raise Exception("No Name was passed!")
        return None

    init_attr(data_obj=data_obj, attr_name=counter_name, attr_type=dict)

    # get object ID:
    obj_id = str(id(data_obj))

    if isinstance(getattr(data_obj, counter_name), dict):
        if obj_id in getattr(data_obj, counter_name):
            num_counter_name = getattr(data_obj, counter_name)[obj_id]
        else:
            getattr(data_obj, counter_name)[obj_id] = 0
            num_counter_name = 0
    else:
        setattr(data_obj, counter_name, {obj_id: 0})
        num_counter_name = 0

    return num_counter_name


def flipangle_corr(T1_obs, flipangle, TR):
    """
    Corrects the observed flipangle for low flipangle experiments with a certain
    repetition time.

    Parameters
    ----------
    T1_obs : float
        Observed T1 decay constant in seconds, usually around 50 for
        [1-13C]pyruvate.
    flipangle : float
        Flipangle in degree.
    TR : float
        Repetition time in seconds.

    Returns
    -------
    T1 : float
        corrected T1 decay constant
    """
    T1 = 1 / ((1 / T1_obs) + (np.log(np.cos(flipangle * np.pi / 180)) / TR))
    return T1


def calc_pol_level(
    thermal_data,
    hyper_data,
    time_to_diss,
    bg_region_hyper=[90, 100],
    bg_region_thermal=[90, 100],
    molarity_hyper=0.08,
    molarity_thermal=0.08,
    T1_for_backcalculation=False,
    linebroadening=0,
    integration_width_hyper=3,
    integration_width_thermal=3,
    select_peak_ppm_thermal=False,
    select_peak_ppm_hyper=False,
    first_spec=0,
    take_one_thermal_spec=False,
    thermal_phase_input=False,
    hyper_phase_input=False,
    B_field=7,
    Temp=18,
    print_output=True,
    plot=True,
):
    """
    Calculates the polarization level by comparing a thermal dataset with a hyperpolarized one.

    Parameters
    ----------
    thermal_data : BrukerExp instance
        Contains thermal spectra (.specs) and meta data (.method)
    hyper_data : BrukerExp instance
        Contains hyperpolarized spectra with a given TR
    time_to_diss : float
        Time from experiment to dissolution
    bg_region_hyper : list, optional
        ppm values where background region is taken from, by default [0,10]
    bg_region_thermal : list, optional
        ppm values where background region is taken from, by default [0,10]
    molarity_hyper : float, optional
        Molarity of hyperpolarized sample in mols / l, for pyruvate this is
        80mM, by default 0.08
    molarity_thermal : float, optional
        Molarity of thermal sample, by default 0.08
    T1_for_backcalculation : bool/float, optional
        Gives us the option to use an externally known T1 for the decay outside the bore in seconds.
        If False, it uses the calculated/flipangle corrected T1 inside the bore.
    linebroadening : float, optional
        Linebroadening applied to both spectra before integration, default is 0 Hz.
    integration_width_hyper : float, optional
        Integration width around peak of hyper spectra in ppm, default is 3.
    integration_width_thermal : float, optional
        Integration width around peak of thermal spectrum in ppm, default is 3.
    first_spec : int, optional
        First repetition that is used, by default 0.
    B_field : int, optional
        Magnetic field in spectrometer in Tesla, by default 1
    Temp : int, optional
        Temperature in the bore in degree Celsius, by default 28.5
    print_output : bool,optional
        Prints results into notebook, by default True

    Returns
    -------
    Polarization_level : float
        Polarization level in hyperpolarized state at time_to_diss in percent.
    T1_hyper_corr : float
        T1 decay constant in seconds, corrected for flipangle and TR.
    SNR_thermal : float
        Thermal SNR from measurement, corrected for flipangle and averages.
    SNR_hyper : float
        Hyperpolarized SNR backcalculated to time_to_diss, corrected for flipangle.
    Pol_lvl_thermal : float
        Thermal polarization level (NOT IN PERCENT)
    """
    from scipy.optimize import curve_fit

    # Step 1
    # calculate thermal polarization level according to Boltzmann
    Pol_lvl_thermal = np.tanh(
        co.hbar * 67.2828 * 1e6 * B_field / (2 * co.k * (273.15 + Temp))
    )

    # Step 2: calculate hyperpolarized SNR dependent on time
    # Norm all spectra to background region
    # Integrate the peak to obtain SNR array
    # fit it to exponential decay function
    # correct the T1 decay constant through the flipangle and TR of the sequence used to monitor the
    # hyperpolarized decay
    # convert input flip angle to radians
    FA_hyper = float(hyper_data.method["ExcPulse1"][2]) * np.pi / 180.0
    # get the number of spectra for hyper measurement
    Nspec_hyper = int(hyper_data.method["PVM_NRepetitions"])
    # Repetition time
    TR_hyper = float(hyper_data.method["PVM_RepetitionTime"]) / 1000  # into s

    ppm_axis_hyper = hyper_data.Get_ppm_axis(70)
    lower_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[0])
    lower_bound_bg_index_hyper = np.argmin(
        lower_bound_bg_hyper - np.min(lower_bound_bg_hyper)
    )
    upper_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[1])
    upper_bound_bg_index_hyper = np.argmin(
        upper_bound_bg_hyper - np.min(upper_bound_bg_hyper)
    )
    bg_region_hyper_indices = [lower_bound_bg_index_hyper, upper_bound_bg_index_hyper]
    bg_region_hyper_indices.sort()

    hyper_spectra = hyper_data.get(linebroadening, 70)[1]
    hyper_normed = np.zeros_like(hyper_spectra)
    for spectrum in range(Nspec_hyper):
        hyper_normed[spectrum, :] = (
            hyper_spectra[spectrum, :]
            - np.mean(hyper_spectra[spectrum, bg_region_hyper[0] : bg_region_hyper[1]])
        ) / np.std(hyper_spectra[spectrum, bg_region_hyper[0] : bg_region_hyper[1]])

    # integrate a selected peak
    if select_peak_ppm_hyper:
        center_ppm_hyper = select_peak_ppm_hyper
    else:
        # otherwise find largest peak
        center_hyper = np.squeeze(np.where(hyper_normed - np.max(hyper_normed) == 0))[1]
        center_ppm_hyper = ppm_axis_hyper[center_hyper]

    lower_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper - integration_width_hyper)
    )
    lower_bound_integration_index_hyper = np.argmin(
        lower_bound_integration_ppm_hyper - np.min(lower_bound_integration_ppm_hyper)
    )
    upper_bound_integration_ppm_hyper = np.abs(
        ppm_axis_hyper - (center_ppm_hyper + integration_width_hyper)
    )
    upper_bound_integration_index_hyper = np.argmin(
        upper_bound_integration_ppm_hyper - np.min(upper_bound_integration_ppm_hyper)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_hyper = [
        lower_bound_integration_index_hyper,
        upper_bound_integration_index_hyper,
    ]
    integrated_peak_roi_hyper.sort()

    time_of_first_spec = first_spec * TR_hyper
    SNR_hyper = np.sum(
        hyper_normed[
            first_spec, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]
        ]
    )

    # fit exponential to hyper SNR to backcalculate
    def exponential(x, M, T1, offset):
        return M * np.exp(-x / T1) + offset

    # integrate the spectra, not the normalized spectra for the T1 fit, as otherwhise there could be fit issues due to high noise
    Hyper_Signal_for_T1_fit = np.array(
        [
            np.sum(
                hyper_spectra[
                    spectrum,
                    integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1],
                ]
            )
            for spectrum in range(first_spec, Nspec_hyper)
        ]
    )
    # define the time axis during which the scans took place
    hyp_time_axis = np.arange(first_spec * TR_hyper, TR_hyper * Nspec_hyper, TR_hyper)
    # Fit
    coeff, err = curve_fit(
        exponential,
        hyp_time_axis,
        Hyper_Signal_for_T1_fit,
        p0=(np.max(Hyper_Signal_for_T1_fit), 50, np.mean(Hyper_Signal_for_T1_fit)),
    )
    # flipangle correct for time outside bore
    flipangle_corr_T1 = flipangle_corr(coeff[1], FA_hyper, TR_hyper)
    error_t1 = np.sqrt(np.diag(err))[1]
    # backcalculate to time of dissolution
    if T1_for_backcalculation:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, T1_for_backcalculation, 0
        )
    else:
        SNR_hyper_backcalculated = exponential(
            -time_to_diss, SNR_hyper, flipangle_corr_T1, 0
        )
    # now norm thermal
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 3 Calculate thermal SNR
    FA_thermal = float(thermal_data.method["ExcPulse1"][2]) * np.pi / 180.0
    ppm_axis_thermal = thermal_data.ppm_axis
    lower_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[0])
    lower_bound_bg_index_thermal = np.argmin(
        lower_bound_bg_thermal - np.min(lower_bound_bg_thermal)
    )
    upper_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[1])
    upper_bound_bg_index_thermal = np.argmin(
        upper_bound_bg_thermal - np.min(upper_bound_bg_thermal)
    )
    bg_region_thermal_indices = [
        lower_bound_bg_index_thermal,
        upper_bound_bg_index_thermal,
    ]
    bg_region_thermal_indices.sort()

    Nspec_thermal = float(thermal_data.method["PVM_NRepetitions"]) * float(
        thermal_data.method["PVM_NAverages"]
    )
    # option to only use one thermal spectrum
    if take_one_thermal_spec:
        Nspec_thermal = 1
        therm_spectra = thermal_data.Get_spec_singlepulse_reps(linebroadening, 70)[1][0]
        thermal_normed = (
            therm_spectra
            - np.mean(
                therm_spectra[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
        ) / np.std(
            therm_spectra[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]]
        )
    else:
        # mean all thermal spectra we have
        if thermal_data.method["PVM_NRepetitions"] > 1:
            # mean thermal spectra if we have multiple that need to be averaged by us
            # i.e. Repetitions instead of Averages
            # number of thermal spectra
            Nspec_thermal = thermal_data.method["PVM_NRepetitions"]
            therm_spectra = thermal_data.Get_meaned_spec_singlepulse_reps(
                linebroadening, 70
            )[1]
            thermal_normed = (
                therm_spectra
                - np.mean(
                    therm_spectra[
                        bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                    ]
                )
            ) / np.std(
                therm_spectra[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )

        elif thermal_data.method["PVM_NAverages"] > 1:
            # in case we have averages
            therm_spectra = thermal_data.Get_spec_singlepulse_reps(linebroadening, 70)[
                1
            ]
            thermal_normed = (
                therm_spectra
                - np.mean(
                    therm_spectra[
                        bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                    ]
                )
            ) / np.std(
                therm_spectra[
                    bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
                ]
            )
        else:
            pass

    # in case we want to integrate a specific peak
    if select_peak_ppm_thermal:
        center_ppm_thermal = select_peak_ppm_thermal
    else:
        # find largest peak
        center_thermal = np.squeeze(
            np.where(thermal_normed - np.max(thermal_normed) == 0)
        )
        center_ppm_thermal = ppm_axis_thermal[center_thermal]

    #  integrate around peak
    lower_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal - integration_width_thermal)
    )
    lower_bound_integration_index_thermal = np.argmin(
        lower_bound_integration_ppm_thermal
        - np.min(lower_bound_integration_ppm_thermal)
    )
    upper_bound_integration_ppm_thermal = np.abs(
        ppm_axis_thermal - (center_ppm_thermal + integration_width_thermal)
    )
    upper_bound_integration_index_thermal = np.argmin(
        upper_bound_integration_ppm_thermal
        - np.min(upper_bound_integration_ppm_thermal)
    )
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_thermal = [
        lower_bound_integration_index_thermal,
        upper_bound_integration_index_thermal,
    ]
    integrated_peak_roi_thermal.sort()

    # print(integrated_peak_roi_thermal)
    SNR_thermal = np.sum(
        thermal_normed[integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]]
    )
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 4: Correction factors
    Receiver_Gain_thermal = thermal_data.acqp["RG"]
    Receicer_Gain_hyper = hyper_data.acqp["RG"]

    correction_factor = (
        np.sqrt(Nspec_thermal)
        * (np.sin(FA_thermal) / np.sin(FA_hyper))
        * (molarity_thermal / molarity_hyper)
        * (Receiver_Gain_thermal / Receicer_Gain_hyper)
    )
    enhancement_factor = (SNR_hyper_backcalculated / SNR_thermal) * correction_factor
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    "------------------------------------------------------------------------------------------------------------"
    # Step 5: compare and plot results

    Polarization_level = Pol_lvl_thermal * enhancement_factor
    Polarization_level_at_first_spec = (
        Pol_lvl_thermal * (SNR_hyper / SNR_thermal) * correction_factor
    )
    Polarization_level = np.round(Polarization_level * 100, 1)
    Polarization_level_at_first_spec = np.round(
        Polarization_level_at_first_spec * 100, 1
    )

    if print_output is True:
        print("--------------------------------------------------------------")
        print(
            "Corrected observed T1=",
            np.round(coeff[1]),
            " s, for a flipangle of ",
            FA_hyper * 180 / np.pi,
            " ° and a TR of ",
            TR_hyper,
            " s ",
        )
        print("Resulting in T1_corr = ", np.round(flipangle_corr_T1, 1), " s")

        print(
            "Receiver Gain difference - Hyper RX Gain = ",
            Receicer_Gain_hyper,
            " vs Thermal RX Gain = ",
            Receiver_Gain_thermal,
        )
        print(
            "Molarity  difference - Hyper Sample 13C Molarity = ",
            molarity_hyper,
            " vs Thermal Sample 13C Molarity  = ",
            molarity_thermal,
        )
        print(
            "Number of spectra  difference - Hyper Scan 1 sample vs Thermal scan ",
            Nspec_thermal,
            " sample",
        )
        print(
            "Flipangle difference correction - Hyper flip angle ",
            FA_hyper * 180 / np.pi,
            " ° - vs Thermal flip angle ",
            FA_thermal * 180 / np.pi,
            " °",
        )
        print(
            "Enhancement factor from thermal to hyper",
            "{:.1e}".format(enhancement_factor),
        )
        if T1_for_backcalculation:
            print(
                "Externally used T1 from other fit function = ",
                np.round(T1_for_backcalculation, 1),
            )
        else:
            print(
                "T1_hyper_corr = ",
                np.round(flipangle_corr_T1, 1),
                "pm",
                np.round(error_t1, 1),
                " s",
            )

        print(
            "SNR_thermal normed to Molarity and Number of spectra",
            np.round(
                SNR_thermal
                * (molarity_hyper / molarity_thermal)
                / np.sqrt(Nspec_thermal),
                1,
            ),
        )
        print(
            "SNR_thermal / correction factor = ",
            np.round(SNR_thermal / correction_factor, 3),
        )
        print("--------------------------------------------------------------")
        print("THERMAL Polarization = ", Pol_lvl_thermal)
        print("SNR_thermal = ", np.round(SNR_thermal, 1))
        print("SNR_hyper_backcalculated = ", np.round(SNR_hyper_backcalculated, 1))
        print("SNR_hyper_at_first_spec = ", np.round(SNR_hyper, 1))

        time_of_first_spec = TR_hyper * first_spec
        print(
            "HYPER - Polarization level of first spec at T = ",
            time_of_first_spec,
            " s, is ",
            Polarization_level_at_first_spec,
            " %",
        )
        print(
            "HYPER - Polarization level at T = ",
            time_of_first_spec - time_to_diss,
            " s, is ",
            Polarization_level,
            " %",
        )
        print("--------------------------------------------------------------")
    else:
        pass

    if plot is True:
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(6, 4))
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        if T1_for_backcalculation:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
            )
            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    T1_for_backcalculation,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        else:
            ax.plot(
                backcalc_axis,
                exponential(
                    backcalc_axis,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
            )
            ax.scatter(
                time_of_first_spec,
                exponential(
                    time_of_first_spec,
                    Polarization_level,
                    flipangle_corr_T1,
                    Pol_lvl_thermal,
                ),
                label="First measurement point",
                color="C1",
            )
        ax.set_xlabel("Time since start of experiment [s]")
        ax.set_ylabel(r"Polarization level [$\%$]")
        ax.set_title("Polarization level")
        ax.legend()

        fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
        ax[0].plot(thermal_data.ppm_axis, thermal_normed)
        ax[0].set_xlabel("ppm")
        ax[0].set_title("Thermal spectrum - " + str(Nspec_thermal) + " averages")
        ax[0].set_ylabel("SNR")
        ax[0].fill_between(
            [
                thermal_data.ppm_axis[bg_region_thermal_indices[0]],
                thermal_data.ppm_axis[bg_region_thermal_indices[1]],
            ],
            np.min(thermal_normed),
            np.max(thermal_normed),
            alpha=0.3,
            color="C2",
            label="Background",
        )
        ax[0].fill_between(
            [
                thermal_data.ppm_axis[integrated_peak_roi_thermal[0]],
                thermal_data.ppm_axis[integrated_peak_roi_thermal[1]],
            ],
            np.min(thermal_normed),
            np.max(thermal_normed),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )

        ax[1].plot(ppm_axis_hyper, hyper_normed[first_spec])
        ax[1].set_xlabel("ppm")
        ax[1].set_title("First hyper spectrum")
        ax[1].set_ylabel("SNR")
        ax[1].fill_between(
            [
                ppm_axis_hyper[bg_region_hyper_indices[0]],
                ppm_axis_hyper[bg_region_hyper_indices[1]],
            ],
            np.min(hyper_normed[first_spec]),
            np.max(hyper_normed[first_spec]),
            alpha=0.3,
            color="C2",
            label="Background",
        )
        ax[1].fill_between(
            [
                ppm_axis_hyper[integrated_peak_roi_hyper[0]],
                ppm_axis_hyper[integrated_peak_roi_hyper[1]],
            ],
            np.min(hyper_normed[first_spec]),
            np.max(hyper_normed[first_spec]),
            alpha=0.3,
            color="C1",
            label="Peak integration",
        )
        ax[1].legend()

        ax[2].set_title("Hyper Signal for T1 fit")
        ax[2].scatter(hyp_time_axis, Hyper_Signal_for_T1_fit, label="Data points")
        backcalc_axis = np.arange(
            time_of_first_spec - time_to_diss, TR_hyper * Nspec_hyper, TR_hyper
        )
        ax[2].plot(
            hyp_time_axis,
            exponential(hyp_time_axis, coeff[0], coeff[1], coeff[2]),
            label="Fit - T1=" + str(np.round(coeff[1], 1)) + "s",
        )
        if T1_for_backcalculation:
            ax[2].plot(
                backcalc_axis,
                exponential(backcalc_axis, coeff[0], T1_for_backcalculation, coeff[2]),
                label="T1 manual input ="
                + str(np.round(T1_for_backcalculation, 1))
                + "s",
            )
        else:
            ax[2].plot(
                backcalc_axis,
                exponential(backcalc_axis, coeff[0], flipangle_corr_T1, coeff[2]),
                label="T1 corrected =" + str(np.round(flipangle_corr_T1, 1)) + "s",
            )

        ax[2].legend()
        ax[2].set_ylabel("Hyper Signal [a.u.]")
        ax[2].set_xlabel("Time since start of experiment [s]")
        ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # second plot showing background levels

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(
            ppm_axis_thermal[
                bg_region_thermal_indices[0] : bg_region_thermal_indices[1]
            ],
            thermal_normed[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]],
        )
        ax2[0, 1].plot(
            ppm_axis_hyper[bg_region_hyper_indices[0] : bg_region_hyper_indices[1]],
            hyper_normed[
                first_spec, bg_region_hyper_indices[0] : bg_region_hyper_indices[1]
            ],
        )

        ax2[0, 0].set_title("Thermal BG region")
        ax2[0, 1].set_title("Hyper BG region")
        x_data, y_data, bin_size = Get_Hist(
            thermal_normed[bg_region_thermal_indices[0] : bg_region_thermal_indices[1]],
            25,
        )
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = Get_Hist(
            hyper_normed[
                first_spec, bg_region_hyper_indices[0] : bg_region_hyper_indices[1]
            ],
            25,
        )
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title("Thermal BG - Histogram")
        ax2[1, 1].set_title("Hyper BG - Histogram")
        ax2[1, 0].set_xlabel("SNR val")
        ax2[1, 1].set_xlabel("SNR val")
        ax2[0, 0].set_xlabel("ppm")
        ax2[0, 1].set_xlabel("ppm")
        ax2[1, 0].set_ylabel("Nr Points")
        ax2[1, 1].set_ylabel("Nr Points")

        # third plot showing peak integration regions

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(
            ppm_axis_thermal[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
        )
        ax2[0, 1].plot(
            ppm_axis_hyper[integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]],
            hyper_normed[
                first_spec, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]
            ],
        )

        ax2[0, 0].set_title("Thermal Signal region")
        ax2[0, 1].set_title("Hyper Signal region")
        x_data, y_data, bin_size = Get_Hist(
            thermal_normed[
                integrated_peak_roi_thermal[0] : integrated_peak_roi_thermal[1]
            ],
            100,
        )
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = Get_Hist(
            hyper_normed[
                first_spec, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]
            ],
            100,
        )
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title("Thermal Signal - Histogram")
        ax2[1, 1].set_title("Hyper Signal - Histogram")
        ax2[1, 0].set_xlabel("SNR val")
        ax2[1, 1].set_xlabel("SNR val")
        ax2[0, 0].set_xlabel("ppm")
        ax2[0, 1].set_xlabel("ppm")
        ax2[1, 0].set_ylabel("Nr Points")
        ax2[1, 1].set_ylabel("Nr Points")

    else:
        pass

    return (
        Polarization_level,
        SNR_thermal,
        SNR_hyper_backcalculated,
        Pol_lvl_thermal,
        correction_factor,
    )


def mri_fft(input_data=None, axes=(0, 1, 2), ift=False):
    """
    Quickly computes the fftshift(fft(fftshift(input_data))) along the input axis.
    If input_data is not set, returns the fft along axis of the fid file

    Parameters:
    ----------
    input_data : (complex) ndarray (None)
        data that should be fourier-transformed.
        If blank, self.fid is used

    axes : tuple (0,1,2)
        axes along the fourier transform should be performed

    ift : use inverse Fourier transform and inverese fftshifts
    """

    # if no input dat was given, use fid
    if input_data is not None:
        fid = input_data
    else:
        logger.warning("No input_data was passed!")

    # init empty array:
    fid_ft = fid
    # perform ft along given axes
    try:
        if ift:
            for d in range(len(axes)):
                fid_ft = np.fft.ifftshift(
                    np.fft.ifft(
                        np.fft.ifftshift(
                            fid_ft,
                            axes=(axes[d],),
                        ),
                        axis=axes[d],
                    ),
                    axes=(axes[d],),
                )
        else:
            for d in range(len(axes)):
                fid_ft = np.fft.fftshift(
                    np.fft.fft(
                        np.fft.fftshift(
                            fid_ft,
                            axes=(axes[d],),
                        ),
                        axis=axes[d],
                    ),
                    axes=(axes[d],),
                )
    except:
        logger.warning("Could not perform ffts!")

    return fid_ft


def gauss_to_kHz(B1_G=0.0, nucleus="13c"):
    """
    Calculates B1_G (Gauss) in kHz
    Parameters
    ----------
    B1_G: float
        The B1 amplitude in Gauss
    nucleus: str
        can be 1h, 2h, 13c, ...

    Returns
    -------
    The B1 amplitude in kHz

    """
    gmr = get_gmr(nucleus=nucleus, unit="MHz_T")
    return B1_G * gmr


def kHz_to_gauss(B1_kHz=0.0, nucleus="13c"):
    """
    Calculates B1_kHz in Gauss
    Parameters
    ----------
    B1_kHz: float
        The B1 amplitude in kHz
    nucleus: str
        can be 1h, 2h, 13c, ...

    Returns
    -------
    The B1 amplitude in kHz

    """
    gmr = get_gmr(nucleus=nucleus, unit="MHz_T")
    return B1_kHz / gmr


def rad_to_deg(angle_deg=0):
    """
    Convert an angle from degrees to radians.

    Parameters
    ----------
    angle_deg : float, optional
        Angle in degrees. Default is 0.

    Returns
    -------
    float
        The angle in radians.
    """
    return angle_deg / 180 * np.pi


def calc_B1_amp_G(fa_deg=90.0, trf_ms=1.0, nucleus="13c", sint=1.0):
    """
    Calculate the B1 amplitude in Gauss.

    Parameters
    ----------
    fa_deg : float, optional
        The flip angle in degrees. Default is 90.
    trf_ms : float, optional
        The pulse duration in milliseconds. Default is 1.
    nucleus : str, optional
        The nucleus type, e.g., '1h', '2h', '13c'. Default is '13c'.
    sint : float, optional
        Integration factor (0-1). Represents the area the RF pulse takes up compared to a block pulse of the same length.
        Default is 1 for a block pulse. For example, sint = 0.762 for a Gaussian of 1ms and fwhm=1/e.

    Returns
    -------
    float
        The B1 amplitude in Gauss [G]. Use `Gauss_to_kHz` to convert the amplitude to kHz.

    """
    # check if integration factor is 0:
    if sint == 0:
        logger.critical("Can't divide by 0, please change sint!")
        return None
    # check if pulse duration in [ms] is 0:
    if trf_ms == 0:
        logger.warning("Can't divide by 0, please change trf_ms!")
        return None

    # gyromagnetic ratio in radian / Tesla / second:
    gmr_rad_Ts = get_gmr(nucleus=nucleus, unit="rad_Ts", db=False)

    # flip-angle  in radian:
    fa_rad = rad_to_deg(angle_deg=fa_deg)

    return fa_rad / (gmr_rad_Ts * trf_ms * 1e-3 * sint) * 1e4


def calc_fa_deg(rfpow_W=1.0, reference_pow_W=1.0, trf_ms=1.0, sint=1.0, fa_deg=90.0):
    """
    Calculate the flip angle in degrees based on the RF pulse amplitude, reference power, pulse duration, and integration factor.

    Parameters
    ----------
    rfpow_W : float
        RF pulse amplitude in watts.
    reference_pow_W : float
        Reference power for a 1ms pulse with sint=1 and block pulse, resulting in a 90-degree flip angle.
    trf_ms : float
        Pulse duration in milliseconds.
    sint : float
        Integration factor.
    fa_deg: float
        set flipangle in degrees.

    Returns
    -------
    float
        The flip angle in degrees.
    """
    assert trf_ms > 0, "Pulse duration trf_ms must greater 0"
    assert sint > 0, "Integration factor must greater 0"

    # calculate flipangle from input parameters:
    return fa_deg * trf_ms * sint * np.sqrt(rfpow_W / reference_pow_W)


def get_gmr(nucleus="13c", unit="MHz_T", db=False):
    """
    Get the gyromagnetic ratio of any nucleus
    Parameters
    ----------
    nucleus: str, can be 1h, 2h, 13c, ...
    unit: str, Either MHz/T (MHz_T, "gamma_bar") or 10e6 rad/T/s (rad_Ts)

    Returns
    -------
    The gyromagnetic ratio in standard units
    """

    # valid nuclei:
    nuclei = [
        "1H",
        "1H (in H2O)",
        "2H",
        "3H",
        "3He",
        "7Li",
        "13C",
        "14N",
        "15N",
        "17O",
        "19F",
        "23Na",
        "27Al",
        "29Si",
        "31P",
        "57Fe",
        "63Cu",
        "67Zn",
        "129Xe",
    ]

    nucleus = nucleus.lower()

    # Checking for valid nucleus and unit
    valid_nuclei = [nuc.lower() for nuc in nuclei]
    valid_units = ["rad_Ts", "MHz_T"]

    # Dictionary with gmr values (taken from wikipedia)
    # https://en.wikipedia.org/wiki/Gyromagnetic_ratio
    gmr_values = {
        "1h_rad_Ts": 267.52218744 * 1e6,
        "1h_MHz_T": 42.577478518,
        "1h_in_h2o_rad_Ts": 267.5153151 * 1e6,
        "1h_in_h2o_MHz_T": 42.57638474,
        "2h_rad_Ts": 41.065 * 1e6,
        "2h_MHz_T": 6.536,
        "3h_rad_Ts": 285.3508 * 1e6,
        "3h_MHz_T": 45.415,
        "3he_rad_Ts": -203.7894569 * 1e6,
        "3he_MHz_T": -32.43409942,
        "7li_rad_Ts": 103.962 * 1e6,
        "7li_MHz_T": 16.546,
        "13c_rad_Ts": 67.2828 * 1e6,
        "13c_MHz_T": 10.7084,
        "14n_rad_Ts": 19.331 * 1e6,
        "14n_MHz_T": 3.077,
        "15n_rad_Ts": -27.116 * 1e6,
        "15n_MHz_T": -4.316,
        "17o_rad_Ts": -36.264 * 1e6,
        "17o_MHz_T": -5.772,
        "19f_rad_Ts": 251.815 * 1e6,
        "19f_MHz_T": 40.078,
        "23na_rad_Ts": 70.761 * 1e6,
        "23na_MHz_T": 11.262,
        "27al_rad_Ts": 69.763 * 1e6,
        "27al_MHz_T": 11.103,
        "29si_rad_Ts": -53.19 * 1e6,
        "29si_MHz_T": -8.465,
        "31p_rad_Ts": 108.291 * 1e6,
        "31p_MHz_T": 17.235,
        "57fe_rad_Ts": 8.681 * 1e6,
        "57fe_MHz_T": 1.382,
        "63cu_rad_Ts": 71.118 * 1e6,
        "63cu_MHz_T": 11.319,
        "67zn_rad_Ts": 16.767 * 1e6,
        "67zn_MHz_T": 2.669,
        "129xe_rad_Ts": -73.997 * 1e6,
        "129xe_MHz_T": -11.777,
    }

    # Constructing the key for the dictionary lookup
    key = f"{nucleus.lower()}_{unit}"

    # check if passed input is valid:
    if nucleus.lower() not in valid_nuclei:
        return f"Unknown nucleus: {nucleus}. Please provide a valid nucleus."
    elif unit not in valid_units:
        return f"Unknown unit: {unit}. Please provide a valid unit (either 'rad_Ts' or 'MHz_T')."
    else:
        pass

    # print which unit is passed:
    if db:
        if unit == "MHz_T":
            print("in [MHz/T]")
        elif unit == "rad_Ts":
            print("in [10e6 rad/T/s]")
        else:
            print("unknown unit")

    # Returning the value if the key exists, otherwise return None
    return gmr_values.get(key, None)


def rfpower_from_refpower(refpower_W=1, fa_deg=90, trf_ms=1, sint=1):
    """
    Calculate the RF power necessary for a pulse.

    Parameters
    ----------
    refpower_W : float, optional
        Reference power for a 90-degree flip angle, 1ms pulse duration, and block pulse (sint=1). Default is 1.
    fa_deg : float, optional
        Flip angle in degrees. Default is 90.
    trf_ms : float, optional
        Pulse duration in milliseconds. Default is 1.
    sint : float, optional
        Integration factor (0-1, with 1 representing a block pulse). Default is 1.

    Returns
    -------
    float
        The RF power in watts.
    """
    assert trf_ms > 0, "Pulse duration trf_ms must greater 0"
    assert sint > 0, "Integration factor must greater 0"
    return refpower_W * (fa_deg / (90 * trf_ms * sint)) ** 2


def translate_coordinates(original_coords, fov, matrix_size):
    """
    Translates coordinates based on a given Field of View (FOV) to indices suitable for input_data processing.

    Parameters:
    - original_coords (list of float): The original x or y coordinates based on the FOV.
    - fov (list of float, [start, end]): The Field of View range. E.g., [-4, 4].
    - matrix_size (int): The size of the input_data matrix in the respective dimension. E.g., 100 for a 100x100 input_data.

    Returns:
    - list of int: Translated coordinates as pixel indices.
    """

    fov_start, fov_end = fov
    fov_size = fov_end - fov_start
    translated_coords = []

    for coord in original_coords:
        translated_coord = (coord - fov_start) * (matrix_size / fov_size)
        translated_coords.append(int(np.round(translated_coord)))

    return translated_coords


def celsius_to_kelvin(celsius=0):
    """
    Convert a temperature from Celsius to Kelvin.

    Parameters
    ----------
    celsius : float, optional
        Temperature in Celsius. Default is 0.

    Returns
    -------
    float
        Temperature in Kelvin.
    """
    assert np.all(celsius >= -273.15), "temperature must be greater -273.15C"
    kelvin = celsius + 273.15
    return kelvin


def kelvin_to_celsius(kelvin=273.15):
    """
    Convert a temperature from Kelvin to Celsius.

    Parameters
    ----------
    kelvin : float, optional
        Temperature in Kelvin. Default is 273.15.

    Returns
    -------
    float
        Temperature in Celsius.
    """
    # assert np.all(kelvin >= 0), "temperature must be greater/equal 0K"
    celsius = kelvin - 273.15
    return celsius


def get_zeeman_energy_split(nucleus="13c", fieldstrength=7, gyromagnetic_ratio=None):
    """
    Calculate the energy difference between E_alpha and E_beta of a spin of a nucleus at a given field strength.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    fieldstrength : float, optional
        Field strength in Tesla. Default is 7.
    gyromagnetic_ratio : float, optional
        Gyromagnetic ratio (in case nucleus was not passed). Default is None.

    Returns
    -------
    float
        The energy difference in Joules.
    """
    assert fieldstrength > 0, "fieldstrength must be greater than 0"

    # define constants:
    planck_constant = 6.62607015e-34  # Planck constant in J*s
    h_bar = planck_constant / (2 * np.pi)  # Reduced Planck constant

    # get gyromagnetic ratio
    gmr = get_gmr(nucleus=nucleus, unit="MHz_T") * 1e6

    return h_bar * gmr * fieldstrength


def get_thermal_polarisation_ratio(
    nucleus="13c", temperature=230, fieldstrength=7, temperature_unit="K"
):
    """
    Calculate the polarisation (in ppm) of spins of a given nucleus at a specified temperature and field strength.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    temperature : float, optional
        Temperature in Kelvin. Default is 230.
    fieldstrength : float, optional
        Field strength in Tesla. Default is 7.
    temperature_unit : str, optional
        Temperature unit, either "K" (Kelvin) or "C" (Celsius). Default is "K".

    Returns
    -------
    float
        The polarisation ratio.
    """
    assert fieldstrength > 0, "fieldstrength must be greater than 0"

    # define constants:
    planck_constant = 6.62607015e-34  # Planck constant in J*s
    boltzmann_constant = 1.380649e-23  # Boltzmann constant in J/K
    h_bar = planck_constant / (2 * np.pi)  # Reduced Planck constant

    if temperature_unit == "K":
        pass
    elif temperature_unit == "C":
        temperature = celsius_to_kelvin(celsius=temperature)
    else:
        raise ValueError("temperature unit must be either K or C!")

    # calculate energy of zeeman splitting at fieldstrength:
    dE = get_zeeman_energy_split(nucleus=nucleus, fieldstrength=fieldstrength)

    population_ratio = np.exp(-dE / (boltzmann_constant * temperature))

    return population_ratio


def get_thermal_polarisation_ppm(
    nucleus="13c", temperature=230, fieldstrength=7, temperature_unit="K"
):
    """
    Get the number of spins (in ppm) that contribute to the NMR signal.

    Parameters
    ----------
    nucleus : str, optional
        The type of nucleus (e.g., "13c"). Default is "13c".
    temperature : float, optional
        Temperature in Kelvin or Celsius. Default is 230.
    fieldstrength : float, optional
        Field strength in Tesla. Default is 7.
    temperature_unit : str, optional
        Temperature unit, either "K" (Kelvin) or "C" (Celsius). Default is "K".

    Returns
    -------
    float
        The polarisation in ppm.
    """
    # ratio of spin up / spin down:
    population_ratio = get_thermal_polarisation_ratio(
        nucleus=nucleus,
        temperature=temperature,
        fieldstrength=fieldstrength,
        temperature_unit=temperature_unit,
    )

    # polarisation in ppm:
    polarisation_ppm = (1 - population_ratio) * 1e6

    return polarisation_ppm


def plot_b1_overlay(
    b1_data=None,
    anat_obj=None,
    axlist=None,
    fig=None,
    path_to_b1=r"D:\OneDrive - TUM\Data\2022\ISMRM 2023\data\lin cryo\FLASH for SNR\lin_cryo_b1.pkl",
    axial_coronal="axial",
    output_nuc="1h",
):
    """
    Plot B1 overlay on anatomical images.

    Parameters
    ----------
    b1_data : dict, optional
        Preloaded B1 data. If None, data will be loaded from `path_to_b1`.
    anat_obj : object, optional
        Anatomical object containing image data.
    axlist : list, optional
        List of axes for plotting.
    fig : matplotlib.figure.Figure, optional
        Figure object for plotting.
    path_to_b1 : str, optional
        Path to the B1 data file. Default is a specific path.
    axial_coronal : str, optional
        Orientation of the plot, either 'axial' or 'coronal'. Default is 'axial'.
    output_nuc : str, optional
        Output nucleus type. Default is '1h'.

    Returns
    -------
    None
    """
    from hypermri.utils.utils_general import format_y_tick_labels

    if b1_data is None:
        # unpack the b1 map (assumes it was generated with self.b1_fitting (see above)
        import pickle

        # Load the data from the file
        with open(path_to_b1, "rb") as file:
            loaded_data = pickle.load(file)

        # Accessing the items from the loaded data
        b1_dirpath = loaded_data["dirpath"]
        try:
            # don't know if the B1 data was generated in one or two iterations:
            B1_map_first_round = loaded_data["B1_map_first_round"]
            refpow_map_first_round = loaded_data["refpow_map_first_round"]
            data_fit_first_round = loaded_data["data_fit_first_round"]
        except:
            pass
        B1_map = np.rot90(loaded_data["B1_map"])
        refpow_map = np.rot90(loaded_data["refpow_map"])
        data_fit = loaded_data["data_fit"]
        # niter = loaded_data["niter"]
        # perlen = loaded_data["perlen"]
        b1_method = loaded_data["method"]
        b1_acqp = loaded_data["acqp"]
    else:
        pass

    # Now you can work with the loaded variables as you wish

    # get extents of anatomical and b1 map:
    anat_extent = get_extent(data_obj=anat_obj)[0]

    class DummyObject:
        pass

    b1_dummy_obj = DummyObject()
    b1_dummy_obj.method = b1_method
    b1_dummy_obj.acqp = b1_acqp

    # define the extent of the generated B1 map
    b1_extent = get_extent(data_obj=b1_dummy_obj)[0]
    # define grid of fieldmap:
    b1_grid = define_grid(data_obj=b1_dummy_obj)

    if anat_obj is None:
        anat_img = None

    else:
        if hasattr(anat_obj, "seq2d_oriented"):
            anat_img = anat_obj.seq2d_oriented

        else:
            anat_img = anat_obj.seq2d

            if anat_obj.method["PVM_NRepetitions"] > 1:
                anat_img = np.mean(anat_img, axis=0)
            # anat_img = np.rot90(anat_img)

            if anat_img.ndim > 2:
                anat_img = anat_img[:, :, int(anat_img.shape[-1] / 2)]
            anat_img = anat_img[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]
            anat_img = np.transpose(anat_img, [2, 3, 0, 1, 4, 5])

    import ipywidgets as widgets

    #
    #  plot B1 map and reference powermap
    # if axial_coronal == "axial":
    slider_rf_power_W = widgets.FloatSlider(
        value=1,
        min=0,
        max=20,
        step=0.001,
        description="RF power [W]: ",
        layout=widgets.Layout(width="50%"),
    )

    slider_rf_duration_ms = widgets.FloatSlider(
        value=1,
        min=0.000001,
        max=25,
        step=0.01,
        description="RF duration [ms]: ",
        layout=widgets.Layout(width="50%"),
    )

    slider_rf_integration_factor = widgets.FloatSlider(
        value=1,
        min=0.000001,
        max=1,
        step=0.01,
        description="Integration factor: ",
        layout=widgets.Layout(width="50%"),
    )

    slider_rf_flipangle_degree = widgets.FloatSlider(
        value=90,
        description="Flipangle [degree]: ",
        layout=widgets.Layout(width="50%"),
    )

    slider_anat_overlay_alpha = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        description="Overlay Anat: ",
        layout=widgets.Layout(width="50%"),
    )

    rangeslider_b1_rx_crange = widgets.FloatRangeSlider(
        value=[0, 1],
        min=-1,
        max=2,
        description="Receive B1 [a.u.]: ",
        layout=widgets.Layout(width="50%"),
    )

    rangeslider_refpow_crange = widgets.FloatRangeSlider(
        value=[0, 1],
        min=-1,
        max=10,
        description="Refpow [W]: ",
        layout=widgets.Layout(width="50%"),
    )

    checkbox_exc_inv = widgets.Checkbox(value=False, description="0-180°: ")

    checkbox_anat_overlay = widgets.Checkbox(value=False, description="anat overlay:")

    checkbox_b1_g_kHz = widgets.Dropdown(
        value="Gauss", description="B1 Map units: ", options=["Gauss", "kHz"]
    )

    dropdown_nuc = widgets.Dropdown(
        options=[
            "1H",
            "1H (in H2O)",
            "2H",
            "3H",
            "3He",
            "7Li",
            "13C",
            "14N",
            "15N",
            "17O",
            "19F",
            "23Na",
            "27Al",
            "29Si",
            "31P",
            "57Fe",
            "63Cu",
            "67Zn",
            "129Xe",
        ],
        value="13C",
    )

    def calc_fa_map(rf_power_W=1, reference_power_W=1, trf_ms=1, sint=1, fa_deg=90):
        # Vectorize your function
        v_calc_fa_deg = np.vectorize(calc_fa_deg)
        # Calculate the flip angle map
        flip_angle_map_deg = v_calc_fa_deg(
            rfpow_W=rf_power_W,
            reference_pow_W=reference_power_W,
            trf_ms=trf_ms,
            sint=sint,
            fa_deg=fa_deg,
        )
        return flip_angle_map_deg

    def calc_b1_map_g(rf_power_W, reference_power_W):
        b1_amp_g_map = calc_B1_amp_G(
            fa_deg=90, trf_ms=1, nucleus=dropdown_nuc.value, sint=1
        ) * np.sqrt(rf_power_W / reference_power_W)
        return b1_amp_g_map

    def b1_g_to_kHz(b1_map_g, nuc):
        v_gauss_to_kHz = np.vectorize(gauss_to_kHz)
        b1_map_kHz = v_gauss_to_kHz(B1_G=b1_map_g, nucleus=nuc)
        return b1_map_kHz

    def plot_img_b1map(
        rf_power_W,
        rf_duration_ms,
        rf_integration_factor,
        rf_flipangle_degree,
        b1_crange,
        refpow_crange,
        exc_inv,
        nuc,
        b1_unit_g_kHz,
        anat_overlay_alpha,
        anat_overlay,
    ):
        # anatomical

        # underlay anatomical input_data:
        if anat_overlay:
            axlist[0].imshow(
                np.rot90(
                    np.squeeze(
                        anat_img[
                            0,
                            int(np.floor(anat_img.shape[1] / 2)),
                            :,
                            :,
                            int(np.floor(anat_img.shape[4] / 2)),
                            0,
                        ]
                    )
                ),
                extent=anat_extent,
                cmap="bone",
            )
            im_b1_rx = axlist[0].imshow(
                np.squeeze(np.abs(B1_map)) / np.max(np.max(np.abs(np.squeeze(B1_map)))),
                extent=b1_extent,
                cmap="jet",
                vmin=b1_crange[0],
                vmax=b1_crange[1],
                alpha=anat_overlay_alpha,
            )
        else:
            im_b1_rx = axlist[0].imshow(
                np.squeeze(np.abs(B1_map)) / np.max(np.max(np.abs(np.squeeze(B1_map)))),
                extent=b1_extent,
                cmap="jet",
                vmin=b1_crange[0],
                vmax=b1_crange[1],
            )
        axlist[0].set_title("B1(-) receive")

        if anat_overlay:
            axlist[1].imshow(
                np.rot90(
                    np.squeeze(
                        anat_img[
                            0,
                            int(np.floor(anat_img.shape[1] / 2)),
                            :,
                            :,
                            int(np.floor(anat_img.shape[4] / 2)),
                            0,
                        ]
                    )
                ),
                extent=anat_extent,
                cmap="bone",
            )
            im_refpow = axlist[1].imshow(
                np.squeeze(refpow_map),
                extent=b1_extent,
                cmap="jet",
                vmin=refpow_crange[0],
                vmax=refpow_crange[1],
                alpha=anat_overlay_alpha,
            )
        else:
            im_refpow = axlist[1].imshow(
                np.squeeze(refpow_map),
                extent=b1_extent,
                cmap="jet",
                vmin=refpow_crange[0],
                vmax=refpow_crange[1],
            )
        axlist[1].set_title("Reference power map")

        fa_map = calc_fa_map(
            rf_power_W=rf_power_W,
            reference_power_W=np.squeeze(refpow_map),
            trf_ms=rf_duration_ms,
            sint=rf_integration_factor,
            fa_deg=rf_flipangle_degree,
        )
        if anat_overlay:
            axlist[2].imshow(
                np.rot90(
                    np.squeeze(
                        anat_img[
                            0,
                            int(np.floor(anat_img.shape[1] / 2)),
                            :,
                            :,
                            int(np.floor(anat_img.shape[4] / 2)),
                            0,
                        ]
                    )
                ),
                extent=anat_extent,
                cmap="bone",
            )
            if exc_inv:
                fa_map = np.abs((fa_map + 180) % 360 - 180)
                im_fa = axlist[2].imshow(
                    np.squeeze(fa_map),
                    extent=b1_extent,
                    cmap="seismic",
                    alpha=anat_overlay_alpha,
                )
                axlist[2].set_title("Flipangle map")
            else:
                im_fa = axlist[2].imshow(
                    np.squeeze(fa_map),
                    extent=b1_extent,
                    cmap="jet",
                    alpha=anat_overlay_alpha,
                )
                axlist[2].set_title("Flipangle map")
        else:
            if exc_inv:
                fa_map = np.abs((fa_map + 180) % 360 - 180)
                im_fa = axlist[2].imshow(
                    np.squeeze(fa_map),
                    extent=b1_extent,
                    cmap="seismic",
                )
                axlist[2].set_title("Flipangle map")
            else:
                im_fa = axlist[2].imshow(
                    np.squeeze(fa_map),
                    extent=b1_extent,
                    cmap="jet",
                )
                axlist[2].set_title("Flipangle map")

        # get b1_that is necessary for 90 degree (depending on nucleus and so on)"
        b1_map_g = calc_b1_map_g(rf_power_W=rf_power_W, reference_power_W=refpow_map)
        if b1_unit_g_kHz == "Gauss":
            b1_tx_map = b1_map_g
        else:
            b1_tx_map = b1_g_to_kHz(b1_map_g=b1_map_g, nuc=nuc)

        if anat_overlay:
            axlist[3].imshow(
                np.rot90(
                    np.squeeze(
                        anat_img[
                            0,
                            int(np.floor(anat_img.shape[1] / 2)),
                            :,
                            :,
                            int(np.floor(anat_img.shape[4] / 2)),
                            0,
                        ]
                    )
                ),
                extent=anat_extent,
                cmap="bone",
            )
            im_b1_tx = axlist[3].imshow(
                np.squeeze(b1_tx_map),
                extent=b1_extent,
                cmap="jet",
                alpha=anat_overlay_alpha,
            )
        else:
            im_b1_tx = axlist[3].imshow(
                np.squeeze(b1_tx_map),
                extent=b1_extent,
                cmap="jet",
            )
        axlist[3].set_title("B1(+) transmit")

        global cbar_b1_rx, cbar_refpow, cbar_fa, cbar_b1_tx
        try:
            cbar_b1_rx.remove()
        except:
            pass
        try:
            cbar_refpow.remove()
        except:
            pass
        try:
            cbar_fa.remove()
        except:
            pass

        try:
            cbar_b1_tx.remove()
        except:
            pass

        cbar_b1_rx = fig.colorbar(im_b1_rx, ax=axlist[0], fraction=0.033, pad=0.04)
        cbar_b1_rx.remove()
        cbar_b1_rx = fig.colorbar(
            im_b1_rx, ax=axlist[0], fraction=0.033, pad=0.04, label="B1(-) Rx [a.u.]"
        )

        cbar_refpow = fig.colorbar(im_refpow, ax=axlist[1], fraction=0.033, pad=0.04)
        cbar_refpow.remove()
        cbar_refpow = fig.colorbar(
            im_refpow, ax=axlist[1], fraction=0.033, pad=0.04, label="Ref. Power [W]"
        )

        cbar_fa = fig.colorbar(im_fa, ax=axlist[2], fraction=0.033, pad=0.04)
        cbar_fa.remove()

        if exc_inv:
            cbar_fa = fig.colorbar(
                im_fa,
                ax=axlist[2],
                fraction=0.033,
                pad=0.04,
                label="Eff. Flipangle [deg]",
            )

        else:
            cbar_fa = fig.colorbar(
                im_fa, ax=axlist[2], fraction=0.033, pad=0.04, label="Flipangle [deg]"
            )

        cbar_b1_tx = fig.colorbar(im_b1_tx, ax=axlist[3], fraction=0.033, pad=0.04)
        cbar_b1_tx.remove()
        if b1_unit_g_kHz == "Gauss":
            cbar_b1_tx = fig.colorbar(
                im_b1_tx, ax=axlist[3], fraction=0.033, pad=0.04, label="B1 [Gauss]"
            )
        else:
            cbar_b1_tx = fig.colorbar(
                im_b1_tx, ax=axlist[3], fraction=0.033, pad=0.04, label="B1 [kHz]"
            )

    out = widgets.interactive_output(
        plot_img_b1map,
        {
            "rf_power_W": slider_rf_power_W,
            "rf_duration_ms": slider_rf_duration_ms,
            "rf_integration_factor": slider_rf_integration_factor,
            "rf_flipangle_degree": slider_rf_flipangle_degree,
            "b1_crange": rangeslider_b1_rx_crange,
            "refpow_crange": rangeslider_refpow_crange,
            "exc_inv": checkbox_exc_inv,
            "nuc": dropdown_nuc,
            "b1_unit_g_kHz": checkbox_b1_g_kHz,
            "anat_overlay_alpha": slider_anat_overlay_alpha,
            "anat_overlay": checkbox_anat_overlay,
        },
    )

    display_opts = widgets.VBox(
        [
            rangeslider_b1_rx_crange,
            rangeslider_refpow_crange,
            checkbox_exc_inv,
            checkbox_b1_g_kHz,
            slider_anat_overlay_alpha,
            checkbox_anat_overlay,
        ],
        layout=widgets.Layout(width="50%"),
    )
    pulse_opts = widgets.VBox(
        [
            slider_rf_power_W,
            slider_rf_duration_ms,
            slider_rf_flipangle_degree,
            slider_rf_integration_factor,
            dropdown_nuc,
        ],
        layout=widgets.Layout(width="50%"),
    )
    main_ui = widgets.HBox([pulse_opts, display_opts])

    display(main_ui, out)


def load_pc_sam_temp_file(filepath, day, time_offset, plot=False):
    """
    Loads .txt file saved from PC-SAM and returns a temperature Dataframe
    Parameters
    ----------
    filepath:str
        path to .txt file
    day: str
        day of the measurement in format: 'YYYY-MM-DD'
    time_offset: float
        time offset from PC-SAM computer to MR computer. Usually this is 3600s for a measurement during daylight savings time
        can be 2*3600s during non daylight savings time, or plus minus a few other seconds
    Returns
    -------
    temperature_dataframe: pandas Dataframe
    """
    import datetime

    try:
        temperature_dataframe = pd.read_csv(
            filepath, decimal=",", sep=", ", engine="python"
        )
    except UnicodeDecodeError:
        logger.error(
            "UTF8 codec error when reading %s, try removing the degree sign in the txt file."
            % filepath
        )
        return None
    day = day + " "
    temperature_dataframe.columns = temperature_dataframe.iloc[1]
    temperature_dataframe = temperature_dataframe.drop([0, 1, 2])
    temperature_dataframe = temperature_dataframe.reset_index()
    # removing all columns that are not of interest here (just need time and temperature)
    remove_columns = []
    for col in temperature_dataframe.columns:
        if col not in ['"Time"', '"Temp 1"']:
            remove_columns.append(col)
        else:
            pass
    temperature_dataframe = temperature_dataframe.drop(remove_columns, axis=1)
    # rename column names
    temperature_dataframe = temperature_dataframe.rename(
        columns={'"Time"': "Time", '"Temp 1"': "Temperature"}, index=None
    )
    # convert entries to str and float
    conversion_dict = {"Time": str, "Temperature": float}
    temperature_dataframe = temperature_dataframe.astype(conversion_dict)
    # convert to datetime objects
    temperature_dataframe["Time"] = day + temperature_dataframe["Time"]
    temperature_dataframe["Time"] = pd.to_datetime(
        temperature_dataframe["Time"], format="%Y-%m-%d %H:%M:%S"
    )
    # convert to epoch time
    # depending if there is daylight savings time at this time, we need to subtract 2 or 1 hour (in winter 1 hour, in summer 2 hours)

    temperature_dataframe["EpochTime"] = (
        temperature_dataframe["Time"] - datetime.datetime(1970, 1, 1)
    ).dt.total_seconds() - time_offset
    temperature_dataframe = temperature_dataframe.astype({"EpochTime": float})

    if plot:

        def avg_temp(temp_array, window_size=5):
            """
            averages temperature in a given window size for easier accessibility
            Parameters
            ----------
            temp_array
            window_size

            Returns
            -------

            """
            i = 0
            multi_sec_avg = []
            while i < len(temp_array) - window_size + 1:
                window_average = round(
                    np.sum(temp_array[i : i + window_size]) / window_size, 2
                )
                multi_sec_avg.append(window_average)
                i += 1
            return multi_sec_avg

        import matplotlib.dates as mdates

        fig, ax = plt.subplots(1, tight_layout=True, figsize=(15, 5))
        window = 20
        dtFmt = mdates.DateFormatter("%H:%M:%S")
        temp_change = np.pad(
            np.gradient(avg_temp(temperature_dataframe["Temperature"], window)),
            (0, window - 1),
            "constant",
        )
        temperature_dataframe["temp change"] = temp_change
        temperature_dataframe.plot.scatter(
            ax=ax,
            x="Time",
            y="Temperature",
            s=3,
            c="temp change",
            cmap="jet",
            vmin=-0.025,
            vmax=0.025,
        )
        ax.xaxis.set_major_formatter(dtFmt)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.tick_params(
            axis="x",
            rotation=45,
            size=2,
        )
        ax.set_title(day)

    return temperature_dataframe


def plot_anat_overlayed_with_masks(
    anatomical=None,
    mask_dict={},
    slice_ind=0,
    savepath=None,
    mask_color=None,
    colormap=None,
    overlay_alpha=0.5,
    slice_orient="axial",
):
    """
    Plots masks over an anatomical input_data with the option to use a single color or a colormap for different masks
    Parameters
    ----------
    - axial: The anatomical input_data data.
    - masks: Dictionary or list of masks to plot.
    - slice_ind: The index of the slice to be plotted.
    - basepath: The base path for saving the output images.
    - single_color: A color name or tuple defining the RGBA for a single color for all masks. If None, colormap is used.
    - colormap: The name of the colormap to use for different masks. Ignored if single_color is not None.
    Returns
    ----------
    figure with subplots, each containing the axial input_data (slice slice_ind) and an overlayed mask on top of it
    Examples
    ----------
    plot_anat_overlayed_with_masks(axial=axial, mask_dict=masks, slice_ind=6, savepath=basepath, colormap="jet")

    """
    import os
    from matplotlib.colors import to_rgba, ListedColormap
    from matplotlib.cm import get_cmap

    # Calculate the subplot grid size
    num_masks = len(mask_dict)
    grid_size = int(np.ceil(np.sqrt(num_masks)))
    grid_cols = grid_size - 1 if grid_size * (grid_size - 1) >= num_masks else grid_size
    grid_rows = grid_size

    if slice_orient == "axial":
        ax_ext, _, _ = get_extent(data_obj=anatomical)
    elif slice_orient == "coronal":
        ax_ext, _, cor_ext = get_extent(data_obj=anatomical)
    else:
        raise Exception(f"slice_orient={slice_orient} not yet implemented!")

    if colormap is None and mask_color is None:
        mask_color = "r"

    # Get a list of colors from the colormap
    if colormap is not None and mask_color is None:
        cmap = get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, num_masks))

    # Get a list of colors from the colormap
    if colormap is not None and mask_color is not None:
        logger.warning("Choose either color or colormap, continuing with colormap")
        cmap = get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, num_masks))

    if slice_orient == "axial":
        plt.figure(figsize=(5, 5))
        for k, m in enumerate(mask_dict):
            plt.subplot(grid_rows, grid_cols, k + 1)
            plt.imshow(
                np.rot90(
                    np.squeeze(anatomical.seq2d_oriented[0, slice_ind, :, :, 0, 0])
                ),
                extent=ax_ext,
                cmap="bone",
            )

            if np.ndim(mask_dict[m]) == 3:
                mask_overlay = np.squeeze(
                    np.transpose(mask_dict[m], (2, 0, 1))[slice_ind, :, :]
                )
            # mask was already translated to be 6:
            elif np.ndim(mask_dict[m]) == 6:
                mask_overlay = np.rot90(
                    np.squeeze(mask_dict[m][0, slice_ind, :, :, 0, 0])
                )
            else:
                raise Exception(
                    f"mask {m} has dim={np.ndim(mask_dict[m])}, should be 3 or 6!"
                )

            if mask_color:
                if isinstance(mask_color, str):
                    mask_color = to_rgba(mask_color)
                color = mask_color  # Use the RGBA color directly
            elif colormap is not None:
                color = colors[k]  # Use the RGBA color directly from the colormap

            # Ensure color is a 4-element tuple
            if len(color) != 4:
                raise ValueError("Color must be a tuple/list with 4 elements (RGBA).")

            mask_overlay_colored = np.zeros(
                (*mask_overlay.shape, 4)
            )  # Create an RGBA input_data based on the mask
            mask_overlay_colored[
                mask_overlay == 1
            ] = color  # Apply color only where the mask is present

            plt.imshow(mask_overlay_colored, extent=ax_ext, alpha=overlay_alpha)
            plt.title(m)
    else:
        plt.figure(figsize=(5, 5))
        for k, m in enumerate(mask_dict):
            plt.subplot(grid_rows, grid_cols, k + 1)
            plt.imshow(
                np.squeeze(anatomical.seq2d_oriented[0, :, :, slice_ind, 0, 0]),
                extent=cor_ext,
                cmap="bone",
            )
            if np.ndim(mask_dict[m]) == 3:
                mask_overlay = np.squeeze(mask_dict[m][:, :, slice_ind])

            # mask was already translated to be 6:
            elif np.ndim(mask_dict[m]) == 6:
                mask_overlay = np.squeeze(mask_dict[m][0, :, :, slice_ind, 0, 0])

            if mask_color:
                if isinstance(mask_color, str):
                    mask_color = to_rgba(mask_color)
                color = mask_color  # Use the RGBA color directly
            elif colormap is not None:
                color = colors[k]  # Use the RGBA color directly from the colormap

            # Ensure color is a 4-element tuple
            if len(color) != 4:
                raise ValueError("Color must be a tuple/list with 4 elements (RGBA).")

            mask_overlay_colored = np.zeros(
                (*mask_overlay.shape, 4)
            )  # Create an RGBA input_data based on the mask
            mask_overlay_colored[
                mask_overlay == 1
            ] = color  # Apply color only where the mask is present

            plt.imshow(mask_overlay_colored, extent=cor_ext, alpha=overlay_alpha)
            plt.title(m)

    plt.tight_layout()

    if savepath is None:
        pass
    else:
        plt.savefig(
            os.path.join(savepath, "masks_on_anat.svg"), dpi=600, transparent=True
        )
        plt.savefig(
            os.path.join(savepath, "masks_on_anat.png"), dpi=600, transparent=True
        )
    plt.show()


def get_indices_for_strings(strings):
    """
    This functions lets you find out which dimension name corresponds to which dimension index
    The default used order of dimensions is [repetitions - z-axis - x-axis - y-axis - repetitions - channels]
    Parameters
    ----------
    strings: string, list of strings

    Returns
    index, list of indices
    -------

    Example:
    -------
    # Example usage with an array of strings, including shorthands
    >>>test_strings = ["x", "chan"]
    >>>indices = get_indices_for_strings(test_strings)
    """
    fixed_order = ["t", "z", "x", "y", "r", "c"]

    # Define a mapping for shorthand versions to the full versions
    shorthand_mapping = {
        "time": "t",
        "fid": "t",
        "spectral": "t",
        "echoes": "t",
        "sampling": "t",
        "spec": "t",
        "repetitions": "r",
        "reps": "r",
        "x-axis": "x",
        "z-axis": "z",
        "y-axis": "y",
        "chan": "c",
        "ch": "c",
        "chans": "c",
        "channels": "c",
        "channel": "c",
    }

    indices = []
    for s in strings:
        if isinstance(s, str):
            # Map the shorthand to its full version if applicable
            full_version = shorthand_mapping.get(s, s)
            try:
                # Return the index of the full version in the fixed order
                index = fixed_order.index(full_version)
                indices.append(index)
            except ValueError:
                # If the string is not found in the fixed order, handle the error
                # Here, you can either append None or skip the string, depending on your preference
                indices.append(None)
        elif isinstance(s, int):
            indices.append(s)
        else:
            indices.append(None)

    return indices


def save_data(
    dir_path: str | None = None,
    filename: str | None = None,
    file: object | None = None,
    file_keys: list[str] | None = None,
    use_timestamp: bool = True,
    print_file_path: bool = True,
    save_as_mat: bool = False,
) -> None:
    """
    Saves a given file, which can be a Python object or a NumPy array, to the specified directory with an optional
    timestamp.

    Parameters
    ----------
    dir_path : str | None, optional
        The directory path where the file will be saved. If the directory does not exist, it will be created.
    filename : str | None
        The base name of the file to which the data will be saved. This function will add a timestamp and appropriate
        file extension.
    file : object | np.ndarray | None
        The data to be saved. Can be any serializable Python object or a NumPy array.
    use_timestamp : bool, optional
        If True, appends a timestamp to the filename to ensure uniqueness. Default is True.
    print_file_path : bool, optional
        If True, prints the full path of the saved file after saving. Default is True.
    save_as_mat: bool, optional
        If True, data is saved as a .mat-file, readable by MATLAB (experimental)
    Returns
    -------
    None
        This function does not return any value, but will print the path of the saved file if `print_file_path` is set
        to True.

    Raises
    ------
    ImportError
        If required modules are not found (implicit exception due to module import).

    RuntimeError
        If both `.npz` and `.pkl` files are found with the same filename.

    Examples
    --------
    >>> data_array = np.array([1, 2, 3])
    >>> save_data('my_data', 'array_data', data_array)
    Data saved as array_data.npz at my_data/array_data_2023-04-29_14-30-00

    >>> data_dict = {'key': 'value'}
    >>> save_data('my_data', 'dict_data', data_dict, use_timestamp=False)
    Data saved as dict_data.pkl at my_data/dict_data
    """
    import os
    import datetime
    import pickle
    import scipy.io as sio  # Add this import

    os.makedirs(dir_path, exist_ok=True)

    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if save_as_mat:
            filepath = os.path.join(dir_path, filename + "_" + timestamp + ".mat")
        else:
            filepath = os.path.join(dir_path, filename + "_" + timestamp)
    else:
        if save_as_mat:
            filepath = os.path.join(dir_path, filename + ".mat")
        else:
            filepath = os.path.join(dir_path, filename)

    # Helper function to ensure keys are a list
    def ensure_keys_list(keys):
        # Check if keys is already a list
        if keys is None:
            return keys
        if isinstance(keys, list):
            return keys
        else:
            # If keys is not a list, attempt to convert it to a list
            try:
                return list(keys)
            except TypeError:
                # If keys cannot be converted to a list, raise an error or handle it as needed
                raise ValueError("keys cannot be converted to a list")

    file_keys = ensure_keys_list(keys=file_keys)

    if save_as_mat:
        if isinstance(file, dict):
            # Convert dictionary values to 2D NumPy arrays, replace spaces in keys with underscores
            data_to_save = {
                k.replace(" ", "_"): np.atleast_2d(np.asarray(v))
                for k, v in file.items()
            }
        elif isinstance(file, np.ndarray) and file_keys is None:
            data_to_save = {"data": np.atleast_2d(file)}
        elif isinstance(file, list) and file_keys is not None:
            # Map list elements to dictionary keys, replace spaces in keys with underscores
            data_to_save = {
                key.replace(" ", "_"): np.atleast_2d(np.asarray(file[k]))
                for k, key in enumerate(file_keys)
            }
        else:
            raise ValueError("Unsupported file type or missing file_keys for lists")

        sio.savemat(filepath, data_to_save)
        if print_file_path:
            print(f"Data saved as {filename}.mat at {filepath}")

        sio.savemat(filepath, data_to_save, format="5")
        if print_file_path:
            print(f"Data saved as {filename}.mat at {filepath}")
    elif isinstance(file, np.ndarray) and file_keys is None:
        np.savez_compressed(filepath, file)
        if print_file_path:
            print(f"Data saved as {filename}.npz at {filepath}")
    else:
        if isinstance(file, list) and file_keys is not None:
            # If file is a dictionary and custom keys are provided, map the data to these keys
            dict_to_save = {key: file[k] for k, key in enumerate(file_keys)}
            file = dict_to_save
        if isinstance(file, np.ndarray) and file_keys is not None:
            # If file is a dictionary and custom keys are provided, map the data to these keys
            dict_to_save = {key: file[k] for k, key in enumerate(file_keys)}
            file = dict_to_save

        with open(filepath + ".pkl", "wb") as save_file:
            pickle.dump(file, save_file, protocol=pickle.HIGHEST_PROTOCOL)
        if print_file_path:
            print(f"Data saved as {filepath}.pkl")


def load_data(
    dir_path: str | None = None,
    filename: str | None = None,
    global_vars: dict = None,
) -> object:
    """
    Loads data from a file in the specified directory. Can handle both `.npz` and `.pkl` files.
    Optionally updates the global variable dictionary with contents from a loaded dictionary.

    Parameters
    ----------
    dir_path : str
        The directory path from where the file will be loaded.
    filename : str
        The base name of the file to be loaded, without extension.
    global_vars : dict, optional
        If provided, and the loaded file is a dictionary, this function will update the globals with the contents of the
        dictionary.

    Returns
    -------
    object
        The data loaded from the file, which could be a NumPy array or any other Python object stored in the file.

    Raises
    ------
    FileNotFoundError
        If no file is found matching the criteria.

    Examples
    --------
    >>> data = load_data('my_data', 'array_data')
    >>> print(data)
    [1 2 3]

    >>> data = load_data('my_data', 'dict_data', use_timestamp=True, global_vars=globals())
    Select a file to load:
    1: dict_data_2023-04-29_14-30-00.pkl
    2: dict_data_2023-04-29_15-00-00.pkl
    Enter selection: 1
    """
    import os
    import pickle

    # Ensure the directory exists before attempting to list or load files
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # List all files that start with the filename and have the proper extensions (.pkl, .npz)
    files = [
        f
        for f in os.listdir(dir_path)
        if f.startswith(filename) and f.endswith((".pkl", ".npz"))
    ]
    if not files:
        raise FileNotFoundError(
            f"No files found starting with {filename} in {dir_path}"
        )

    if len(files) > 1:
        # Print available files for user selection
        print("Select a file to load:")
        for index, file in enumerate(files, start=1):
            print(f"{index}: {file}")
        selected_index = int(input("Enter selection: ")) - 1
        filepath = os.path.join(dir_path, files[selected_index])
    else:
        filepath = os.path.join(dir_path, files[0])

    # Load the file based on its extension
    if filepath.endswith(".npz"):
        data = np.load(filepath, allow_pickle=True)
        key = next(iter(data))  # Assume file contains only one array
        loaded_data = data[key]
    elif filepath.endswith(".pkl"):
        with open(filepath, "rb") as file:
            loaded_data = pickle.load(file)

    # If global_vars was provided and the loaded data is a dictionary, update globals
    if global_vars is not None and isinstance(loaded_data, dict):
        global_vars.update(
            loaded_data
        )  # Update the calling environment's global variables
    elif global_vars is not None and not isinstance(loaded_data, dict):
        global_vars[filename] = loaded_data
    else:
        pass

    return loaded_data


def save_as_pkl(
    dir_path: str | None = None,
    filename: str | None = None,
    file: object | None = None,
    file_keys: list[str] | None = None,
    use_timestamp: bool = True,
    print_file_path: bool = True,
) -> None:
    """
    Saves provided data as a pickle file, optionally using a timestamp in the filename for uniqueness.

    This function allows saving one or multiple data objects into a single pickle file. If multiple objects are
    provided, they are saved in a dictionary format within the pickle file. The keys for this dictionary can be
    specified; otherwise, they are automatically generated. A timestamp can be appended to the filename to avoid
    overwriting existing files.

    Parameters
    ----------
    dir_path : str
        The directory path where the pickle file will be saved.
    filename : str
        The base name of the pickle file. A timestamp may be appended if `use_timestamp` is True.
    file : object or list of objects
        The data object(s) to be saved. Can be any Python object(s) that pickle can handle.
    file_keys : list of str, optional
        The keys corresponding to each data object. These are used as keys in the dictionary saved within the pickle
        file. If not provided, keys are generated as sequential numbers in string format.
    use_timestamp : bool, optional
        If True (default), appends a timestamp to the filename to ensure uniqueness.
    print_file_path : bool, optional
        If True (default), prints the full path of the saved pickle file.

    Returns
    -------
    None
    Examples
    -------
    # Example data objects
    >>>my_array = [1, 2, 3, 4, 5]  # An example list
    >>>my_dict = {'a': 1, 'b': 2, 'c': 3}  # An example dictionary
    >>>my_int = 42  # An example integer
    >>>dir_path = 'path/to/your/directory'
    >>>filename = 'my_saved_data'
    >>>objects_to_save = [my_array, my_dict, my_int]

    # Optional: Corresponding keys for the objects in the pickle file
    # These keys will be used to access the objects when the pickle file is loaded
    >>>file_keys = ['test_array', 'test_dict', 'test_int']

    # Call the function to save the objects into a single pickle file
    >>>save_as_pkl(dir_path=dir_path, filename=filename, file=objects_to_save, file_keys=file_keys, use_timestamp=True)

    """
    import os
    import pickle
    import datetime

    # Ensure required parameters are provided
    assert dir_path is not None, "dir_path has to be passed"
    assert filename is not None, "filename has to be passed"
    assert file is not None, "file to save has to be passed"

    # Generate default keys if not provided
    if file_keys is None:
        if isinstance(file, dict):
            file_keys = file.keys()
        else:
            file_keys = [str(i) for i in range(len(file))]

    def ensure_keys_list(keys):
        # Check if keys is already a list
        if isinstance(keys, list):
            return keys
        else:
            # If keys is not a list, attempt to convert it to a list
            try:
                return list(keys)
            except TypeError:
                # If keys cannot be converted to a list, raise an error or handle it as needed
                raise ValueError("keys cannot be converted to a list")

    file_keys = ensure_keys_list(keys=file_keys)

    # Ensure file_keys is a list
    if not isinstance(file_keys, list):
        file_keys = [file_keys]

    # Append additional keys if not enough are provided
    if len(file_keys) < len(file):
        for i in range(len(file) - len(file_keys)):
            file_keys.append(str(len(file_keys) + i))

        # Prepare the data to be saved
    if isinstance(file, dict) and file_keys is None:
        # If file is a dictionary and no custom keys are provided, save it directly
        dict_to_save = file
    elif isinstance(file, dict) and file_keys is not None:
        # If file is a dictionary and custom keys are provided, map the data to these keys
        dict_to_save = {key: file[key] for key in file_keys if key in file}
    else:
        # Handle other types (e.g., list of objects) with custom keys or default to index-based keys
        if file_keys is None or len(file_keys) != len(file):
            file_keys = [str(i) for i in range(len(file))]
        dict_to_save = dict(zip(file_keys, file))

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Append timestamp to filename if required
    filename_with_timestamp = (
        f"{filename}_{timestamp}.pkl" if use_timestamp else f"{filename}.pkl"
    )

    # Full path for the file to be saved
    full_path = os.path.join(dir_path, filename_with_timestamp)

    # Save the dictionary to a pickle file
    with open(full_path, "wb") as file:
        pickle.dump(dict_to_save, file)

    # Optionally print the full path of the saved file
    if print_file_path:
        print(f"Data saved to {full_path}")


def load_as_pkl(dir_path=None, filename=None, global_vars=None):
    """
    Load a pickle file and optionally create variables named after the keys in the loaded dictionary.

    Parameters
    ----------
    dir_path : str
        Directory where the file will be loaded from.
    filename : str
        Name of the pickle file to load.
    load_individual : bool, optional
        If True, the dict entries of the loaded file will generate individual global variables named after the dict keys.

    Returns
    -------
    loaded_file : object or dict
        The object loaded from the pickle file. If the file contains a dictionary and load_individual is True,
        also creates global variables for each entry.

    Examples
    --------
    #### Example data objects
    >>>my_array = [1, 2, 3, 4, 5]  # An example list
    >>>my_dict = {'a': 1, 'b': 2, 'c': 3}  # An example dictionary
    >>>my_int = 42  # An example integer
    >>>dir_path = 'path/to/your/directory'
    >>>filename = 'my_saved_data'
    >>>objects_to_save = [my_array, my_dict, my_int]

    # Optional: Corresponding keys for the objects in the pickle file
    #### These keys will be used to access the objects when the pickle file is loaded
    >>>file_keys = ['test_array', 'test_dict', 'test_int']

    ##### Call the function to save the objects into a single pickle file
    >>>save_as_pkl(dir_path=dir_path, filename=filename, file=objects_to_save, file_keys=file_keys, use_timestamp=True)

    #### to directly generate variables from the loaded file (dict) pass globals() (this will generate the variables
    'test_array', 'test_dict', test_int')
    >>>load_as_pkl(dir_path="your_dir_path", filename="your_filename.pkl", global_vars=globals())

    """
    import os
    import pickle

    assert dir_path is not None, "dir_path has to be passed"
    assert filename is not None, "filename has to be passed"

    if filename[-4:] == ".pkl":
        pass
    else:
        filename += ".pkl"

    # Combine directory path and filename
    full_path = os.path.join(dir_path, filename)

    # Load the data from the file
    with open(full_path, "rb") as file:
        loaded_file = pickle.load(file)

    if global_vars is not None and isinstance(loaded_file, dict):
        global_vars.update(loaded_file)

    return loaded_file


def underline(text):
    """
    Underline a string, and print.
    """
    l = ""
    for i in range(len(text)):
        l += "-"
    text_new = text + "\n" + l
    print(text_new)


def underoverline(text):
    """
    Overline, underline a string, and print.
    """
    l = ""
    for i in range(len(text)):
        l += "-"
    text_new = l + "\n" + text + "\n" + l
    print(text_new)


def make_consistent(a, b):
    """
    Function takes x2 2D arrays e.g. parameter maps, and sets all the same pixels to NaN in both  maps.
    """
    coords = np.argwhere(np.isnan(a))  # Returns coordnates of NaN in a.
    for item in coords:
        b[item[0], item[1]] = np.nan

    return a, b


def analyze_freq_maps_generic(
    mask,
    frequency_map,
    csi_snr_map=None,
    anat_ref_img=None,
    compute_temperature=False,
    temperature_function="5mM",
    is_ppm_boolean=True,
    colormap="plasma",
    bins=30,
    plot_csi_anat_ref=False,
    savepath=None,
    colorbarticks=None,
):
    """
    This functions analyzes frequency maps retrieved from fitting peaks for the temperature dependency study of pyruvate and lactate.
    It has been adapted for generic vendor CSI data
    Parameters
    ----------
    mask: 2D np.array containing np.nan where the pixel is not to be included and 1 everywhere else
    frequency_map: 2D np.array of same shape as mask containing frequency information
    anat_ref_img: BrukerExp instance of anatomical reference scan
    anat_ref_slice: int, axial slice number to be displayed
    temperature_function: function that defines how frequency is translated into temperature
    colormap: str, default is 'plasma'
    bins: int, default is 30. Number of bins for histogram and gaussian fitting

    Returns
    -------
    output_dict: dictionary containing frequency and temperature averages for all pixels and mean+std from gaussian fitting
    """
    t_vmin = 30.0
    t_vmax = 42.0
    t_alpha = 1
    from ..utils.utils_fitting import temperature_from_frequency

    def gaussian(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    if is_ppm_boolean:
        frequency_str = "ppm"
    else:
        frequency_str = "Hz"
    masked_frq_map = frequency_map * mask
    meaned_frq_all = np.nanmean(masked_frq_map)
    std_frq_all = np.nanstd(masked_frq_map)
    if compute_temperature:
        masked_temp_map, masked_temp_map_error = temperature_from_frequency(
            masked_frq_map, temperature_function, is_ppm_boolean
        )
        meaned_temp_all = np.nanmean(masked_temp_map)
        std_temp_all = np.nanstd(masked_temp_map)
    else:
        logger.warning("No temperature calibration function passed.")

    if plot_csi_anat_ref is True:
        fig, ax = plt.subplots(2, 2, figsize=(13, 5), tight_layout=True)
        im0 = ax[0, 0].imshow(csi_snr_map, cmap=colormap)
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im0, cax=cax, orientation="vertical", label="SNR [a.u.]")
        ax[0, 1].imshow(
            anat_ref_img,
            cmap="bone",
        )
        im2 = ax[1, 0].imshow(
            masked_frq_map,
            cmap=colormap,
            vmin=np.nanmin(masked_frq_map),
            vmax=np.nanmax(masked_frq_map),
        )
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(
            im2, cax=cax, orientation="vertical", label="f [" + frequency_str + "]"
        )
        im3 = ax[1, 1].imshow(
            masked_temp_map,
            cmap=colormap,
            interpolation="None",
            vmin=np.nanmin(masked_temp_map),
            vmax=np.nanmax(masked_temp_map),
        )
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax, orientation="vertical", label="T [°C]")
        ax[1, 0].set_title("Frequency map")
        ax[1, 1].set_title("Temperature map")
        ax[0, 1].set_title("Anatomical")
        ax[0, 0].set_title("CSI SNR")
        [ax[0, n].set_xticks([]) for n in range(2)]
        [ax[0, n].set_yticks([]) for n in range(2)]
        [ax[1, n].set_xticks([]) for n in range(2)]
        [ax[1, n].set_yticks([]) for n in range(2)]

    if compute_temperature:
        fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3))
        im1 = ax[0].imshow(
            masked_temp_map,
            interpolation="None",
            cmap=colormap,
            vmin=t_vmin,
            vmax=t_vmax,
            alpha=t_alpha,
        )
        ax[0].set_title(
            "T= "
            + str(np.round(meaned_temp_all, 2))
            + " °C - f="
            + str(np.round(meaned_frq_all, 2))
            + frequency_str
        )
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(
            im1,
            cax=cax,
            orientation="vertical",
            label="T [°C]",
            ticks=colorbarticks,
        )

        # fit a gaussian to that data
        # gaussian fitting adapted from https://www.wasyresearch.com/tutorial-python-for-fitting-gaussian-distribution-on-data/
        x_data = np.ravel(masked_temp_map)[~np.isnan(np.ravel(masked_temp_map))]

        hist, bin_edges = np.histogram(x_data, bins=bins)
        hist = hist / sum(hist)
        n = len(hist)
        x_hist = np.zeros((n), dtype=float)
        for ii in range(n):
            x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2

        y_hist = hist
        mean = sum(x_hist * y_hist) / sum(y_hist)
        sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)
        # try:

        param_optimised, param_covariance_matrix = curve_fit(
            gaussian, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=5000
        )  # ,
        # bounds=([0.0, 0.0, 0.0], [2 * np.max(y_hist), np.max(x_hist), 100]),
        # )
        # print('Max',2*np.max(y_hist),np.max(x_hist),100)
        # print(param_optimised)
        # except:
        # logger.critical("Gaussian fitting not successful")
        #    param_optimised = [np.nan, np.nan, np.nan]

        x_hist_2 = np.linspace(np.min(x_hist), np.max(x_hist), 500)
        ax[1].plot(
            x_hist_2,
            gaussian(x_hist_2, *param_optimised),
            label="Gaussian fit",
            color="C1",
        )

        # Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        ax[1].hist(
            x_data,
            weights=weights,
            alpha=0.7,
            color="C0",
            edgecolor="black",
            bins=bins,
        )

        # bins=np.linspace(np.min(masked_temp_map_high_combined),np.max(masked_temp_map_high_combined),len(high_temps_masked_combined))

        # sns.histplot(ax=ax[1],x=high_temps_masked_combined,y=bins,kde=True)
        ax[1].vlines(
            param_optimised[1],
            0,
            param_optimised[0],
            color="k",
            linestyle="dashed",
            label="Mean",
        )
        ax[1].set_title(
            r"$\mu=$"
            + str(np.round(param_optimised[1], 1))
            + r"$ / \sigma =$"
            + str(np.round(param_optimised[2], 2))
            + r"$^{\circ} C$"
        )
        ax[1].set_xlabel("T [$^{\circ}$C]")
        ax[1].legend()
        ax[1].set_ylabel("Probability")
        print("bin size Temperature", x_hist[1] - x_hist[0])

        meaned_temp = param_optimised[1]
        meaned_temp_std = param_optimised[2]
        # fit a gaussian to that data
        # gaussian fitting adapted from https://www.wasyresearch.com/tutorial-python-for-fitting-gaussian-distribution-on-data/
        x_data = np.ravel(masked_frq_map)[~np.isnan(np.ravel(masked_frq_map))]

        hist, bin_edges = np.histogram(x_data, bins=bins)
        hist = hist / sum(hist)
        n = len(hist)
        x_hist = np.zeros((n), dtype=float)
        for ii in range(n):
            x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2

        y_hist = hist
        mean = sum(x_hist * y_hist) / sum(y_hist)
        sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)
        try:
            param_optimised, param_covariance_matrix = curve_fit(
                gaussian, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=5000
            )
        except:
            logger.critical("Gaussian fitting not successful")
            param_optimised = [np.nan, np.nan, np.nan]
        x_hist_2 = np.linspace(np.min(x_hist), np.max(x_hist), 500)
        ax[2].plot(
            x_hist_2,
            gaussian(x_hist_2, *param_optimised),
            label="Gaussian fit",
            color="C1",
        )

        # Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        ax[2].hist(
            x_data,
            weights=weights,
            alpha=0.7,
            color="C0",
            edgecolor="black",
            bins=bins,
        )
        ax[2].vlines(
            param_optimised[1],
            0,
            param_optimised[0],
            color="k",
            linestyle="dashed",
            label="Mean",
        )
        ax[2].set_title(
            r"$\mu=$"
            + str(np.round(param_optimised[1], 2))
            + r"$ / \sigma =$"
            + str(np.round(param_optimised[2], 2))
            + frequency_str
        )
        ax[2].set_xlabel("f " + frequency_str)
        ax[2].legend()
        ax[2].set_ylabel("Probability")
        # optinal setting of x limits:
        ax[1].set_xlim([28, 50])
        print("careful we set x limits in the statistical plots manually")

        print("bin size Frequency:", x_hist[1] - x_hist[0])

        meaned_frq = param_optimised[1]
        meaned_frq_std = param_optimised[2]

        output_dict = {
            "meaned_frq_all_pixels": meaned_frq_all,
            "std_frq_all_pixels": std_frq_all,
            "meaned_frq_from_gauss": meaned_frq,
            "std_frq_from_gauss": meaned_frq_std,
            "meaned_temp_all_pixels": meaned_temp_all,
            "std_temp_all_pixels": std_temp_all,
            "meaned_temp_from_gauss": meaned_temp,
            "std_temp_from_gauss": meaned_temp_std,
            "masked_frq_map": masked_frq_map,
            "masked_temp_map": masked_temp_map,
        }
    else:
        fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(8, 3))

        im1 = ax[0].imshow(
            masked_frq_map,
            alpha=0.7,
            interpolation="None",
            cmap=colormap,
        )
        ax[0].set_title(" °C - f=" + str(np.round(meaned_frq_all, 2)) + frequency_str)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(
            im1,
            cax=cax,
            orientation="vertical",
            label="f" + frequency_str,
            ticks=colorbarticks,
        )

        x_data = np.ravel(masked_frq_map)[~np.isnan(np.ravel(masked_frq_map))]

        hist, bin_edges = np.histogram(x_data, bins=bins)
        hist = hist / sum(hist)
        n = len(hist)
        x_hist = np.zeros((n), dtype=float)
        for ii in range(n):
            x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2

        y_hist = hist
        mean = sum(x_hist * y_hist) / sum(y_hist)
        sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)
        try:
            param_optimised, param_covariance_matrix = curve_fit(
                gaussian, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=5000
            )
        except:
            logger.critical("Gaussian fitting not successful")
            param_optimised = [np.nan, np.nan, np.nan]
        x_hist_2 = np.linspace(np.min(x_hist), np.max(x_hist), 500)
        ax[1].plot(
            x_hist_2,
            gaussian(x_hist_2, *param_optimised),
            label="Gaussian fit",
            color="C1",
        )

        # Normalise the histogram values
        weights = np.ones_like(x_data) / len(x_data)
        ax[1].hist(
            x_data,
            weights=weights,
            alpha=0.7,
            color="C0",
            edgecolor="black",
            bins=bins,
        )
        ax[1].vlines(
            param_optimised[1],
            0,
            param_optimised[0],
            color="k",
            linestyle="dashed",
            label="Mean",
        )
        ax[1].set_title(
            r"$\mu=$"
            + str(np.round(param_optimised[1], 2))
            + r"$ / \sigma =$"
            + str(np.round(param_optimised[2], 2))
            + frequency_str
        )
        ax[1].set_xlabel("f" + frequency_str)
        ax[1].legend()
        ax[1].set_ylabel("Probability")

        print("bin size Frequency:", x_hist[1] - x_hist[0])

        meaned_frq = param_optimised[1]
        meaned_frq_std = param_optimised[2]

        output_dict = {
            "meaned_frq_all_pixels": meaned_frq_all,
            "std_frq_all_pixels": std_frq_all,
            "meaned_frq_from_gauss": meaned_frq,
            "std_frq_from_gauss": meaned_frq_std,
            "masked_frq_map": masked_frq_map,
        }
    if savepath:
        plt.savefig(savepath)
    else:
        pass
        return output_dict


def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian function.

    Parameters:
    - x: array_like, the independent variable where the gaussian is evaluated.
    - amplitude: float, the peak amplitude of the Gaussian curve.
    - mean: float, the mean of the Gaussian curve, also its center.
    - stddev: float, the standard deviation of the Gaussian curve, determining its width.

    Returns:
    - Gaussian function evaluated at x.
    """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev**2))


def compute_weighted_mean_and_std(data, weights):
    """
    Computes the weighted mean for a dataset.
    According to Case I from http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    which took the implementation from: Bevington, P. R., Data Reduction and Error Analysis for the Physical Sciences, 336 pp., McGraw-Hill, 1969.
    We assume here that some of our datapoints are more important than others, e.g.
    because they are pixels in a mask and some pixels are only contributing 80%.
    Parameters
    ----------
    data: np.array, contains data over which the mean and std should be computed
    weights: np.array, weighting of data, has same shape as data

    Returns
    -------
    mean: float, weighed mean of the dataset
    std: float, unbiased standard error of the dataset
    """
    mean = sum(data * weights) / sum(weights)
    a = ((sum(weights * (data**2))) / sum(weights)) - mean**2
    b = sum(weights**2) / (sum(weights) ** 2 - sum(weights**2))
    std = np.sqrt(a * b)
    return mean, std


def simplify_element(input=None):
    """
    Convert a single-element NumPy array to a scalar, or return the original value if not applicable.

    Parameters
    ----------
    input : numpy.ndarray or scalar
        The value to be simplified.

    Returns
    -------
    scalar or numpy.ndarray
        The simplified scalar value if input was a single-element array, otherwise the original input.

    Examples
    --------
    >>> simplify_element(np.array([100]))
    100
    >>> simplify_element(np.array([1, 2, 3]))
    array([1, 2, 3])
    >>> simplify_element(100)
    100
    """
    if isinstance(input, np.ndarray) and input.size == 1:
        return input.item()
    return input

def hyperpolarized_Mz_flipangle(M0_au, n_excitations, alpha_deg):
    """
    Calculate the hyperpolarized signal considering flip angle.

    Parameters
    ----------
    M0_au : float
        Initial magnetization in arbitrary units.
    n_excitations : int
        Number of excitations (pulses).
    alpha_deg : float
        Flip angle in degrees.

    Returns
    -------
    polarization : ndarray
        Calculated polarization values.
    """
    return M0_au * (np.cos(np.deg2rad(alpha_deg)) ** np.arange(n_excitations))


def hyperpolarized_Mz_T1(M0_au, t_s, T1_s):
    """
    Calculate the hyperpolarized signal considering T1 relaxation.

    Parameters
    ----------
    M0_au : float
        Initial magnetization in arbitrary units.
    t_s : array_like
        Time points in seconds.
    T1_s : float
        T1 relaxation time in seconds.

    Returns
    -------
    polarization : ndarray
        Calculated polarization values.
    """
    return M0_au * np.exp(-t_s / T1_s)


def hyperpolarized_Mz_flipangle_T1(
    M0_au=1e5,
    n_excitations=100,
    TR_s=0.2,
    T1_s=30,
    alpha_deg=5,
    plot=True,
    interactive=False,
    data_obj=None,
):
    """
    Calculate and optionally plot the hyperpolarized signal considering both flip angle and T1 relaxation.

    Parameters
    ----------
    M0_au : float, optional
        Initial magnetization in arbitrary units. Default is 1e5.
    n_excitations : int, optional
        Number of excitations (pulses). Default is 100.
    TR_s : float, optional
        Repetition time in seconds. Default is 0.2.
    T1_s : float, optional
        T1 relaxation time in seconds. Default is 30.
    alpha_deg : float, optional
        Flip angle in degrees. Default is 5.
    plot : bool, optional
        If True, create a plot. Default is True.
    interactive : bool, optional
        If True, create an interactive plot. Default is False.
    data_obj: hypermri object, optional
        If passed, inital Flipangle, Repetition time and number of repetitions will be taken from
        parameters of the object.

    Returns
    -------
    polarization : ndarray
        Calculated polarization values.
    t_s : ndarray
        Time points in seconds.

    Examples
    --------
    >>> polarization, t_s = hyperpolarized_Mz_flipangle_T1(M0_au=1e5, n_excitations=100, TR_s=0.2, T1_s=30, alpha_deg=5, plot=True, interactive=True)
    """
    from ipywidgets import interact, FloatSlider, IntSlider, Checkbox

    plt.ion()  # Turn on interactive mode
    if data_obj is None:
        pass
    else:
        TR_s = data_obj.repetition_time_ms / 1000.0
        n_excitations = np.prod(data_obj.matrix) * data_obj.n_repetitions
        alpha_deg = data_obj.flipangle_deg

    t_s = np.arange(n_excitations) * TR_s

    polarization = (
        M0_au
        * (np.cos(np.deg2rad(alpha_deg)) ** np.arange(n_excitations))
        * np.exp(-t_s / T1_s)
    )

    if plot:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twiny()  # Create a twin axis for pulse numbers

        def update_plot(
            M0_au=M0_au,
            n_excitations=n_excitations,
            TR_s=TR_s,
            T1_s=T1_s,
            alpha_deg=alpha_deg,
            signal_t1=False,
        ):
            t_s = np.arange(n_excitations) * TR_s
            if signal_t1:
                signal = (
                    M0_au
                    * (np.cos(np.deg2rad(alpha_deg)) ** np.arange(n_excitations))
                    * np.exp(-t_s / T1_s)
                    * np.sin(np.deg2rad(alpha_deg))
                )
            else:
                pass
            polarization = (
                M0_au
                * (np.cos(np.deg2rad(alpha_deg)) ** np.arange(n_excitations))
                * np.exp(-t_s / T1_s)
            )
            # To avoid clipping when the total duration is too short, we generate this dummy time array for the calculation of the effective T1
            t_eff_T1_s = np.arange(0, 10 * T1_s, TR_s)
            polarization_eff_T1_s = (
                M0_au
                * (np.cos(np.deg2rad(alpha_deg)) ** np.arange(len(t_eff_T1_s)))
                * np.exp(-t_eff_T1_s / T1_s)
            )
            eff_T1_s = t_eff_T1_s[
                np.argmin(np.abs(polarization_eff_T1_s - M0_au / np.exp(1)))
            ]

            ax1.clear()
            ax2.clear()
            if signal_t1:
                ax1.plot(
                    t_s,
                    signal,
                    label=f"Signal Sum={np.round(np.sum(np.abs(signal))/1e5, 2)}e5",
                    color="green",
                )
                ax1.plot(t_s, polarization, label="Mz combined effects", color="blue")

                ax1.plot(
                    t_s,
                    hyperpolarized_Mz_flipangle(M0_au, n_excitations, alpha_deg),
                    label=f"Flip Angle Relaxation ={np.round(alpha_deg,1)}°",
                    color="gray",
                    alpha=0.5,
                )
                ax1.plot(
                    t_s,
                    hyperpolarized_Mz_T1(M0_au, t_s, T1_s),
                    label=f"T1 Relaxation (T1={np.round(T1_s,1)} s)",
                    color="lightgray",
                    alpha=0.5,
                )
                ax1.vlines(
                    eff_T1_s,
                    0,
                    M0_au / np.exp(1),
                    color="r",
                    linestyle="--",
                    label=f"M0/e (effective T1={np.round(eff_T1_s, 1)} s)",
                )

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("M (a.u.)")
                ax1.set_title("Hyperpolarized Mz - Combined Effects")
                ax1.legend()
                ax1.set_ylim(0, M0_au * 1.1)
                ax1.set_xlim(0, t_s[-1])
            else:
                ax1.plot(t_s, polarization, label="Mz combined effects", color="blue")

                ax1.plot(
                    t_s,
                    hyperpolarized_Mz_flipangle(M0_au, n_excitations, alpha_deg),
                    label=f"Flip Angle Relaxation ={np.round(alpha_deg,1)}°",
                    color="gray",
                    alpha=0.5,
                )
                ax1.plot(
                    t_s,
                    hyperpolarized_Mz_T1(M0_au, t_s, T1_s),
                    label=f"T1 Relaxation (T1={np.round(T1_s,1)} s)",
                    color="lightgray",
                    alpha=0.5,
                )
                ax1.vlines(
                    eff_T1_s,
                    0,
                    M0_au / np.exp(1),
                    color="r",
                    linestyle="--",
                    label=f"M0/e (effective T1={np.round(eff_T1_s, 1)} s)",
                )

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Mz (a.u.)")
                ax1.set_title("Hyperpolarized Mz - Combined Effects")
                ax1.legend()
                ax1.set_ylim(0, M0_au * 1.1)
                ax1.set_xlim(0, t_s[-1])

            # Set up the secondary x-axis for pulse numbers
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(
                np.arange(0, n_excitations, max(1, n_excitations // 10)) * TR_s
            )
            ax2.set_xticklabels(
                np.arange(0, n_excitations, max(1, n_excitations // 10))
            )
            # ax2.set_xlabel('Excitation Number')

            fig.canvas.draw_idle()

        if interactive:
            interact(
                update_plot,
                M0_au=FloatSlider(
                    min=1e4, max=1e6, step=1e4, value=M0_au, description="M0 (a.u.)"
                ),
                n_excitations=IntSlider(
                    min=10,
                    max=500,
                    step=10,
                    value=n_excitations,
                    description="# Excitations",
                ),
                TR_s=FloatSlider(
                    min=0.01, max=1, step=0.01, value=TR_s, description="TR (s)"
                ),
                T1_s=FloatSlider(
                    min=1, max=60, step=1, value=T1_s, description="T1 (s)"
                ),
                alpha_deg=FloatSlider(
                    min=0, max=90, step=1, value=alpha_deg, description="Flip Angle (°)"
                ),
                signal_t1=Checkbox(value=False, description="Show signal"),
            )
        else:
            update_plot()

    return polarization, t_s

