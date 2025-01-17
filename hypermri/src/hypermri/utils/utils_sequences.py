###########################################################################################################################################################

# Statistical analysis of datasets

###########################################################################################################################################################


# =========================================================================================================================================================
# Import necessary packages
# =========================================================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os


def generate_gradient_file(grad_array=None,
                           savepath=None,
                           savename=None,
                           sampling_dt=10e-6,
                           gradient_duration=10 * 1e-3,
                           db=False):
    """
    Generates a gradient file that can be read by ParaVision
    copy the generated file to:
    /opt/PV7.0.0/prog/curdir/nmrsu/ParaVision/exp/lists/gp/
    Parameters
    ----------
    grad_array: array
        1D Gradient shape array. will be normalized to be [-1, 1]
    savepath: str
        Path where to store the generate gradient file
    savename: str
        Name of the generated gradient shape file. file-type will be "number-of-points"
    sampling_dt: float
        Scanner gradient sampling rate. Min @ Bruker 7T = 8.6us
    gradient_duration: float
        Total duration of the generated gradient file. This will determine the number of points

    Returns
    -------

    """
    from scipy.interpolate import interp1d
    if grad_array is None:
        warnings.warn("Please provide grad_array")
    if np.ndim(grad_array) != 1:
        warnings.warn(f"grad_array has to be 1D but is {np.ndim(grad_array)}")

    if sampling_dt < 8.6 * 1e-6:
        warnings.warn(f"Gradient sampling time of {sampling_dt}, is too small, setting it to 8.6 us")
        sampling_dt = 8.6

    # get number of points:
    npoints = len(grad_array)
    # calculate the total duration
    total_duration = npoints * sampling_dt
    # reduce number of points if resulting sampling rate would be too high:
    if total_duration >= gradient_duration:
        # define number of points that the gradient file should have to match the desired sampling rate:
        desired_npoints = int(np.floor(gradient_duration / sampling_dt))

        # Generate 2 time axis, one that was provided, one that the gradient shape will be interpolated onto:
        desired_time_axis = np.linspace(0, 1, desired_npoints)
        current_time_axis = np.linspace(0, 1, npoints)

        # Create interpolation function:
        interpolation_func = interp1d(current_time_axis, grad_array, kind='linear')

        # Save of gradient array:
        grad_array_old = grad_array.copy()

        # Interpolate data
        grad_array = interpolation_func(desired_time_axis)
        print(f"npoints before:={len(current_time_axis)}, npoints after={len(desired_time_axis)}")

        # plot before and after interpolation:
        if db:
            fig, ax = plt.subplots(1, 3)
            ax[0].plot(current_time_axis, grad_array_old)
            ax[1].plot(desired_time_axis, grad_array)
            ax[2].plot(current_time_axis, grad_array_old)
            ax[2].plot(desired_time_axis, grad_array)
    else:
        desired_npoints = npoints

    def normalize_grad(input_grad_array):
        input_grad_array = input_grad_array / np.max(input_grad_array)
        return input_grad_array

    def format_number_E_minus_01(num):
        # Format the number to move the decimal point one digit to the left
        adjusted_number = num * 10
        return f"{adjusted_number:.6f}E-01"

    normalized_grad_array = normalize_grad(input_grad_array=grad_array)
    formatted_numbers = [format_number_E_minus_01(num) for num in normalized_grad_array]

    # Importatnt to not put an offset here and keep this part
    header = f"""##TITLE=
##JCAMP-DX= 5.00 Bruker JCAMP library
##DATA TYPE= Shape Data
##ORIGIN= Bruker BioSpin GmbH
##OWNER= <demo>
##DATE= 2008/09/08
##TIME= 13:20:52
##$SHAPE_PARAMETERS= Type: Sinus ; Number of Cycles 1.0 ; Phase Angle [deg] 0.0
##MINX= 0.000000E00
##MAXX= 1.000000E02
##MINY= 0.000000E00
##MAXY= 1.000000E00
##$SHAPE_EXMODE= Gradient
##$SHAPE_TOTROT= 9.000000E01
##$SHAPE_TYPE= None
##$SHAPE_USER_DEF=
##$SHAPE_REPHFAC=
##$SHAPE_BWFAC= 0.000000E00
##$SHAPE_BWFAC50=
##$SHAPE_INTEGFAC= 6.3656674E-01
##$SHAPE_MODE= 0
##NPOINTS= {desired_npoints}
##XYDATA= (X++(Y..Y))
"""
    footer = "##END="
    filename = f"{savename}.{desired_npoints}"
    savename = os.path.join(savepath, filename)
    # Write to file:
    with open(savename, 'w') as file:
        file.write(header)
        for number in formatted_numbers:
            file.write(number + '\n')
        file.write(footer)


def generate_csi_sampling_pattern(size_x, size_y, pattern_type="spiral", **kwargs):
    """
    Generate a sampling pattern for chemical shift imaging (CSI) experiments similar to PV sequence lucaCSI4.

    Parameters
    ----------
    size_x : int
        The width of the output array.
    size_y : int
        The height of the output array.
    pattern_type : str
        The type of sampling pattern to generate. Can be either 'spiral' or 'centric'.
    **kwargs
        Additional keyword arguments to pass to the sampling pattern generation function (spatial_resolution_mm in case of pattern_type = "centric").

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (size_y, size_x) containing the sampling pattern.

    Examples
    --------
    >>> import hypermri.utils.utils_sequences as utseq
    >>> spiral_pattern = utseq.generate_csi_sampling_pattern(size_x=16, size_y=16, pattern_type="spiral")
    >>> centric_pattern = utseq.generate_csi_sampling_pattern(size_x=16, size_y=16, pattern_type="centric")
    >>> fig, ax = plt.subplots(1,2)
    >>> ax=ax.flatten()
    >>> ax[0].imshow(spiral_pattern)
    >>> ax[1].imshow(centric_pattern)
    """

    def generate_centric_sampling(size_x, size_y, spatial_resolution_mm):
        """
        Generate an equidistant sampling pattern based on distance to k-space center.

        This function creates a 2D numpy array representing a sampling pattern
        where points are ordered based on their distance to the k-space center,
        taking into account the spatial resolution.

        Parameters
        ----------
        size_x : int
            The width of the output array.
        size_y : int
            The height of the output array.
        spatial_resolution_mm : tuple
            A tuple of two floats representing the spatial resolution in mm
            for x and y dimensions.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of shape (size_y, size_x) containing the sampling
            pattern. The array is filled with integers from 0 to
            (size_x * size_y - 1), representing the order of sampling.

        Notes
        -----
        This function implements a sampling pattern similar to a spiral but
        orders points based on their distance to the k-space center, calculated
        in fractions of k_bwx and k_bwy.

        Examples
        --------
        >>> pattern = generate_equidistant_sampling(64, 64, (1.0, 1.0))
        >>> plt.imshow(pattern, cmap='viridis')
        >>> plt.colorbar(label='Sampling Order')
        >>> plt.title('Equidistant Sampling Pattern')
        >>> plt.show()
        """
        if spatial_resolution_mm is None:
            spatial_resolution_mm = np.zeros(2)
            spatial_resolution_mm[0] = 1.0
            spatial_resolution_mm[1] = 1.0

        # Calculate k-space "bandwidth"
        k_bwx = 1 / spatial_resolution_mm[0]
        k_bwy = 1 / spatial_resolution_mm[1]

        # Create k-space coordinate matrices
        kx = np.linspace(-1 + 1 / size_x, 1 - 1 / size_x, size_x)
        ky = np.linspace(-1 + 1 / size_y, 1 - 1 / size_y, size_y)
        kx_mat, ky_mat = np.meshgrid(kx, ky)

        # Calculate actual k-space values
        kgrad_matx = kx_mat * k_bwx / 2
        kgrad_maty = ky_mat * k_bwy / 2

        # Calculate distance to k-space center
        k_mat_dist = np.sqrt(kgrad_matx**2 + kgrad_maty**2)

        # Flatten the arrays
        k_mat_dist_1d = k_mat_dist.flatten()
        kx_1d = kx_mat.flatten()
        ky_1d = ky_mat.flatten()
        kgrad_matx_1d = kgrad_matx.flatten()
        kgrad_maty_1d = kgrad_maty.flatten()

        # Sort based on distance to k-space center
        sorted_indices = np.argsort(k_mat_dist_1d)
        k_mat_dist_1d = k_mat_dist_1d[sorted_indices]
        kx_1d = kx_1d[sorted_indices]
        ky_1d = ky_1d[sorted_indices]
        kgrad_matx_1d = kgrad_matx_1d[sorted_indices]
        kgrad_maty_1d = kgrad_maty_1d[sorted_indices]

        # Create the sampling pattern
        pattern = np.zeros((size_y, size_x), dtype=int)
        for i, (x, y) in enumerate(zip(kx_1d, ky_1d)):
            x_idx = np.argmin(np.abs(kx - x))
            y_idx = np.argmin(np.abs(ky - y))
            pattern[y_idx, x_idx] = i

        return pattern

    def generate_spiral_sampling(size_x, size_y):
        """
        Generate a center-out spiral sampling pattern.

        This function creates a 2D numpy array representing a center-out spiral
        sampling pattern. The spiral starts from the center of the array and
        moves outward. It handles square arrays.

        Parameters
        ----------
        size_x : int
            The width of the output array.
        size_y : int
            The height of the output array.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of shape (size_y, size_x) containing the spiral
            sampling pattern. The array is filled with integers from 0 to
            (size_x * size_y - 1), representing the order of sampling.

        Notes
        -----
        This function adapts the logic from a provided C function to create
        a spiral sampling pattern. It handles square arrays.

        Examples
        --------
        >>> pattern = generate_spiral_sampling_adapted(5, 5)
        >>> print(pattern)
        [[20 19 18 17 16]
        [21  6  5  4 15]
        [22  7  0  3 14]
        [23  8  1  2 13]
        [24  9 10 11 12]]
        """
        if size_x != size_y:
            raise ValueError(
                f"Size must be rectangular but is size_x={size_x} x size_y={size_y}"
            )
        total_points = size_x * size_y

        # Create arrays to store the spiral order
        order_spiral_x = np.zeros(total_points, dtype=int)
        order_spiral_y = np.zeros(total_points, dtype=int)

        if size_x == size_y:
            # Square matrix case
            arr_x = np.arange(size_x).reshape(size_x, 1).repeat(size_y, axis=1)
            arr_y = np.arange(size_y).reshape(1, size_y).repeat(size_x, axis=0)
            arr_y = np.flipud(arr_y)

            offset = 0
            cols = size_y - 1
            rows = size_x - 1
            counter = 0

            while offset < (size_x - 1):
                # Top row
                for col in range(offset, cols + 1):
                    order_spiral_x[counter] = arr_x[offset, col]
                    order_spiral_y[counter] = arr_y[offset, col]
                    counter += 1

                # Right column
                for row in range(offset + 1, rows + 1):
                    order_spiral_x[counter] = arr_x[row, cols]
                    order_spiral_y[counter] = arr_y[row, cols]
                    counter += 1

                # Bottom row
                for col in range(cols - 1, offset - 1, -1):
                    order_spiral_x[counter] = arr_x[rows, col]
                    order_spiral_y[counter] = arr_y[rows, col]
                    counter += 1

                # Left column
                for row in range(rows - 1, offset, -1):
                    order_spiral_x[counter] = arr_x[row, offset]
                    order_spiral_y[counter] = arr_y[row, offset]
                    counter += 1

                offset += 1
                rows -= 1
                cols -= 1

        else:
            # Rectangular matrix case
            y_bigger_than_x = size_y > size_x
            leftover = abs(size_y - size_x)
            half_leftover_ceil = math.ceil(leftover / 2)
            half_leftover_floor = math.floor(leftover / 2)

            if y_bigger_than_x:
                arr_x = np.arange(size_x).reshape(size_x, 1).repeat(size_y, axis=1)
                arr_y = np.arange(size_y).reshape(1, size_y).repeat(size_x, axis=0)
                arr_y = np.flipud(arr_y)

                for i in range(size_x):
                    for j in range(half_leftover_floor, size_x + half_leftover_floor):
                        arr_x[i, j] = i
                        arr_y[i, j] = j
            else:
                arr_x = np.arange(size_x).reshape(size_x, 1).repeat(size_y, axis=1)
                arr_y = np.arange(size_y).reshape(1, size_y).repeat(size_x, axis=0)
                arr_y = np.flipud(arr_y)

                for i in range(half_leftover_floor, size_y + half_leftover_floor):
                    for j in range(size_y):
                        arr_x[i, j] = i
                        arr_y[i, j] = j

            # Now implement the spiral ordering for the rectangular case
            # (This part would need to be implemented similar to the square case,
            # but accounting for the rectangular shape)

        # Create the final pattern array
        pattern = np.full((size_y, size_x), -1, dtype=int)
        for i in range(total_points):
            x, y = order_spiral_x[i], order_spiral_y[i]
            if pattern[y, x] == -1:
                pattern[y, x] = i

        # Fill any remaining -1 values
        remaining = np.where(pattern == -1)
        pattern[remaining] = range(np.sum(pattern >= 0), total_points)
        pattern = np.abs(pattern - np.max(pattern))

        return pattern


    def generate_bruker_centric_centric():
        """TODO Wolfgang"""
        pass
    if pattern_type == "spiral":
        return generate_spiral_sampling(size_x, size_y)
    elif pattern_type == "centric":
        if kwargs is None:
            kwargs = {}
            kwargs["spatial_resolution_mm"] = (1.0, 1.0)
        elif "spatial_resolution_mm" not in kwargs:
            kwargs["spatial_resolution_mm"] = (1.0, 1.0)
        return generate_centric_sampling(size_x, size_y, **kwargs)


def create_sample_image(nx, ny, shape, size):
    """
    Create a sample image with specified shape and size.

    Parameters
    ----------
    nx, ny : int
        Dimensions of the image.
    shape : str
        Shape of the sample ('square' or 'circle').
    size : float
        Size of the sample as a fraction of the image size (0 to 1).

    Returns
    -------
    sample_image : ndarray
        2D array representing the sample image.
    """
    sample_image = np.zeros((nx, ny), dtype="complex")
    center_x, center_y = nx // 2, ny // 2
    radius = int(min(nx, ny) * size / 2)

    if shape == "square":
        x_start, x_end = center_x - radius, center_x + radius
        y_start, y_end = center_y - radius, center_y + radius
        sample_image[x_start:x_end, y_start:y_end] = 1 + 0j

    if shape == "2 squares":
        x_start_1, x_end_1 = center_x + radius // 2, center_x + 3 * radius // 2
        y_start_1, y_end_1 = center_y + radius // 2, center_y + 3 * radius // 2
        sample_image[x_start_1:x_end_1, y_start_1:y_end_1] = 1 + 0j

        x_start_2, x_end_2 = center_x - 3 * radius // 2, center_x - radius // 2
        y_start_2, y_end_2 = center_y - 3 * radius // 2, center_y - radius // 2
        sample_image[x_start_2:x_end_2, y_start_2:y_end_2] = 1 + 0j
        print(x_start_1, x_end_1, y_start_1, y_end_1)
        print(x_start_2, x_end_2, y_start_2, y_end_2)

    elif shape == "circle":
        y, x = np.ogrid[-center_x : nx - center_x, -center_y : ny - center_y]
        mask = x * x + y * y <= radius * radius
        sample_image[mask] = 1 + 0j

    return sample_image

