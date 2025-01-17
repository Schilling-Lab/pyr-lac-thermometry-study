import numpy as np
from skimage.measure import block_reduce
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom


def interpolate_binary_mask(mask, new_shape, method="nearest", **kwargs):
    """Interpolate a binary mask to a new shape.

    Parameters
    ----------
    mask : ndarray
        The binary mask to be interpolated.
    new_shape : tuple
        The new shape (dimensions) for the interpolated mask.
    method : str, optional
        The interpolation method. Default is 'nearest'.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to RegularGridInterpolator.

    Returns
    -------
    ndarray
        The interpolated binary mask.

    Notes
    -----
    This function interpolates a binary mask to a new shape using RegularGridInterpolator
    from scipy.interpolate. The mask is assumed to contain binary values (0 or 1), and
    the interpolation is performed based on the specified method.
    """
    assert (
        np.min(mask) >= 0 and np.max(mask) <= 1
    ), "Not a binary mask. Values outside [0,1]."

    n, m, k = mask.shape
    n_new, m_new, k_new = new_shape

    # Create a meshgrid of voxel coordinates
    x, y, z = (
        np.arange(n),
        np.arange(m),
        np.arange(k),
    )

    x_new, y_new, z_new = np.meshgrid(
        np.linspace(0, (n - 1), n_new),
        np.linspace(0, (m - 1), m_new),
        np.linspace(0, (k - 1), k_new),
        indexing="ij",
    )

    # Create a regular grid interpolator
    interpolator = RegularGridInterpolator(
        (x, y, z), mask.astype(float), method=method, **kwargs
    )

    # Interpolate values on the new grid
    interpolated_mask = interpolator((x_new, y_new, z_new))

    # Round the values to 0 or 1
    interpolated_mask = np.round(interpolated_mask).astype(int)

    return interpolated_mask


def interpolate_3d_array(
    arr, new_dimensions, voxel_size=None, return_grid=False, **kwargs
):
    """Interpolate a(n unmasked!) 3D array onto a new grid using regular grid interpolation.

    Parameters
    ----------
    arr : numpy.ndarray
        The input 3D array
    new_dimensions : tuple
        A tuple specifying the new dimensions (n_new, m_new, k_new) of the
        interpolated array.
    voxel_size : numpy.ndarray, optional
        The voxel size of the original array. If not provided, it defaults to
        [1.0, 1.0, 1.0].
    return_grid : bool, optional
        If True, the function also returns the coordinates of the new grid.
        Defaults to False.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the RegularGridInterpolator.
        You can e.g. specify the interpolation type with method='cubic'.

    Returns
    -------
    numpy.ndarray or tuple
        - If return_grid is False (default), returns the interpolated masked
        array on the new grid.
        - If return_grid is True, returns a tuple containing the interpolated
        masked array and the coordinates of the new grid.

    Raises
    ------
    AssertionError
        If the input array is not a np.ndarray
    """
    assert isinstance(arr, np.ndarray), "We need a default numpy.ndarray mate."

    if voxel_size is None:
        voxel_size = np.array((1.0, 1.0, 1.0))

    n, m, k = arr.shape

    n_new, m_new, k_new = new_dimensions

    # Create a meshgrid of voxel coordinates
    x, y, z = (
        np.arange(n) * voxel_size[0],
        np.arange(m) * voxel_size[1],
        np.arange(k) * voxel_size[2],
    )

    x_new, y_new, z_new = np.meshgrid(
        np.linspace(0, (n - 1) * voxel_size[0], n_new),
        np.linspace(0, (m - 1) * voxel_size[1], m_new),
        np.linspace(0, (k - 1) * voxel_size[2], k_new),
        indexing="ij",
    )

    # Create a regular grid interpolator
    interpolator = RegularGridInterpolator((x, y, z), arr, **kwargs)

    # Interpolate the pH values on the new grid
    array_interpolated = interpolator((x_new, y_new, z_new))

    return array_interpolated


def interpolate_masked_3d_array(
    masked_array, new_dimensions, voxel_size=None, return_grid=False, **kwargs
):
    """
    Interpolate a masked 3D array onto a new grid using regular grid interpolation.

    Parameters
    ----------
    masked_array : numpy.ma.MaskedArray
        The input 3D array with missing values masked.
    new_dimensions : tuple
        A tuple specifying the new dimensions (n_new, m_new, k_new) of the
        interpolated array.
    voxel_size : numpy.ndarray, optional
        The voxel size of the original array. If not provided, it defaults to
        [1.0, 1.0, 1.0].
    return_grid : bool, optional
        If True, the function also returns the coordinates of the new grid.
        Defaults to False.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the RegularGridInterpolator.
        You can e.g. specify the interpolation type with method='cubic'.

    Returns
    -------
    numpy.ma.MaskedArray or tuple
        - If return_grid is False (default), returns the interpolated masked
        array on the new grid.
        - If return_grid is True, returns a tuple containing the interpolated
        masked array and the coordinates of the new grid.

    Raises
    ------
    AssertionError
        If the input array is not a masked array.
    """
    assert np.ma.is_masked(masked_array), "We need a masked array mate."

    if voxel_size is None:
        voxel_size = np.array((1.0, 1.0, 1.0))

    n, m, k = masked_array.shape

    n_new, m_new, k_new = new_dimensions

    # Create a meshgrid of voxel coordinates
    x, y, z = (
        np.arange(n) * voxel_size[0],
        np.arange(m) * voxel_size[1],
        np.arange(k) * voxel_size[2],
    )

    x_new, y_new, z_new = np.meshgrid(
        np.linspace(0, (n - 1) * voxel_size[0], n_new),
        np.linspace(0, (m - 1) * voxel_size[1], m_new),
        np.linspace(0, (k - 1) * voxel_size[2], k_new),
        indexing="ij",
    )

    # Create a regular grid interpolator
    interpolator = RegularGridInterpolator((x, y, z), masked_array, **kwargs)

    # Interpolate the pH values on the new grid
    array_interpolated = interpolator((x_new, y_new, z_new))

    # # Interpolate mask:
    # mask_interpolated = interpolate_binary_mask(masked_array.mask, new_dimensions, method='linear')

    # # Apply the interpolated mask to the interpolated data
    # masked_array_interpolated = np.ma.masked_array(
    #     array_interpolated, mask_interpolated
    # )
    # Create a separate interpolator for the mask
    mask_interpolator = RegularGridInterpolator(
        (x, y, z), masked_array.mask, bounds_error=False, fill_value=False, **kwargs
    )

    # Interpolate the mask values on the new grid
    mask_interpolated = mask_interpolator((x_new, y_new, z_new))

    # Apply the interpolated mask to the interpolated data
    masked_array_interpolated = np.ma.masked_array(
        array_interpolated, mask_interpolated
    )

    if return_grid:
        return masked_array_interpolated, (x_new, y_new, z_new)
    return masked_array_interpolated


def downsample_masked_array(masked_array, downsampling_factor):
    """
    Downsamples a masked array using scikit-image's block_reduce function while preserving the mask information.

    Parameters:
        masked_array (numpy.ma.masked_array): The original 3D masked array.
        downsampling_factor (tuple): The downsampling factor for each dimension.

    Returns:
        numpy.ma.masked_array: The downsampled masked array.

    Notes:
        This function uses `skimage.measure.block_reduce` to downsample the array
        by applying a block-wise reduction operation. It avoids interpolation
        artifacts and is well-suited for downsampling with fixed downsampling
        factors. (And more robust to noise!) Unlike `scipy.ndimage.zoom`,
        `block_reduce` only supports downsampling and does not perform
        interpolation or upsampling.
    """
    # Downsample the array
    downsampled_array = block_reduce(masked_array.data, downsampling_factor, np.mean)
    downsampled_mask = block_reduce(masked_array.mask, downsampling_factor, np.any)

    # Create a new masked array with the downsampled data and mask
    downsampled_masked_array = np.ma.masked_array(
        downsampled_array, mask=downsampled_mask
    )

    return downsampled_masked_array
