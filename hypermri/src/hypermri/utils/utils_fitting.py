import numpy as np
import matplotlib.pyplot as plt

from ..utils.utils_logging import LOG_MODES, init_default_logger
from IPython.display import clear_output
from IPython.display import display


logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")
from ..utils.utils_spectroscopy import get_freq_axis, freq_to_index, make_NDspec_6Dspec
from ..utils.utils_general import simplify_element as se
from ..utils.utils_general import (
    get_gmr,
    calc_sampling_time_axis,
)  # get gyromagnetic ratio


def subtract_peak_from_spec(
    input_data=None, fit_result=None, peak_to_subtract=None, fit_params=None
):
    """
    Subtract the main peak from the spectrum and return the resulting difference spectrum.

    This function is intended to subtract the dominant main peak from the spectrum to redo
    the fitting with the main peak removed. If no specific peak is provided to subtract,
    it subtracts the peak with the maximum amplitude.

    Parameters
    ----------
    input_data : array-like, optional
        The spectrum data to process.
    fit_result : object, optional
        The fit results from a previous operation, contains peak properties.
    peak_to_subtract : int or None, optional
        Index of the peak to be subtracted. If not provided, the peak with the
        maximum amplitude will be subtracted.
    fit_params : dict, optional
        Parameters from the fitting operation. Should include 'metabs' which
        is a list of metabolites or peaks fitted, and 'freq_range' which
        specifies the frequency range of input_data.

    Returns
    -------
    diff_spectrum : array-like
        Spectrum data after the subtraction of the main peak.

    Notes
    -----
    The function modifies a copy of the fit results to zero out all amplitudes
    except for the one to be subtracted, then generates a spectrum with only
    that peak and subtracts it from the input data.

    See Also
    --------
    find_max_amp : Function used to find the peak with maximum amplitude if not specified.
    """
    # Check if input data is provided; if not, return with a warning
    if input_data is None:
        Warning("No input_data passed")
        return None

    # Check if fitting results are provided; if not, return with a warning
    if fit_result is None:
        Warning("No fitting results fit_result passed")
        return None

    # Copy fit results to ensure original data remains unaltered
    from copy import deepcopy

    fit_result_copy = deepcopy(fit_result)

    # If peak_to_subtract is not provided, determine which peak to subtract based on max amplitude
    if peak_to_subtract is None:
        peak_to_subtract = (
            find_max_amp(fit_result=fit_result_copy, fit_params=fit_params) + 1
        )

    # Zero out all the amplitudes except for the one we want to subtract
    for k, metab in enumerate(fit_params["metabs"]):
        if (k + 1) != peak_to_subtract:
            setattr(fit_result_copy, f"amplitude_{k + 1}", 0.0)

    # Update fit_params to indicate which peak is being subtracted
    fit_params["peak_to_subtract"] = peak_to_subtract

    # Generate a spectrum with only the peak to subtract
    fit_spectrum = fit_result_copy(fit_params["freq_range"])
    # Calculate the difference between the input spectrum and the generated fit spectrum
    diff_spectrum = input_data - fit_spectrum

    return diff_spectrum


def find_max_amp(fit_result=None, fit_params=None):
    """
    Find the index of the peak with the highest amplitude in fit_results.

    Parameters
    ----------
    fit_result : object, optional
        The fit results from a previous operation, contains peak properties.
    fit_params : dict, optional
        Parameters from the fitting operation. Should include 'metabs' which
        is a list of metabolites or peaks fitted.

    Returns
    -------
    int or None
        The index of the peak with the maximum amplitude. Returns None if
        fit_result or fit_params is not provided.

    Notes
    -----
    This function iterates through all peaks in the fit results, extracts their
    amplitudes, and returns the index of the peak with the maximum amplitude.

    See Also
    --------
    subtract_peak_from_spec : Function that uses this to find the main peak to subtract.
    """
    # Check if fitting results are provided; if not, return with a warning
    if fit_result is None:
        Warning("No fitting results fit_result passed")
        return None

    # Check if fitting parameters are provided; if not, return with a warning
    if fit_params is None:
        Warning("No fitting parameters fit_params passed")
        return None

    # Initialize an array to store amplitudes of all peaks
    amps = np.zeros((len(fit_params["metabs"]), 1))

    # Copy fit results to ensure original data remains unaltered
    from copy import deepcopy

    fit_result_copy = deepcopy(fit_result)

    # Loop through each metabolite or peak to extract its amplitude
    for k, metab in enumerate(fit_params["metabs"]):
        # Fetch the amplitude attribute for the k-th peak from the fit results
        amp = getattr(fit_result_copy, f"amplitude_{k + 1}")
        amps[k] = amp.value

    # Return the index of the peak with the highest amplitude
    return np.argmax(amps)


def ppm_from_temp_and_concentration(c="5mM", t=0):
    """
    Calculate the ppm value based on the concentration and temperature.

    Parameters
    ----------
    c : str, optional
        The concentration of the solution. Supported values are '5mM', '10mM', '20mM', '50mM', '100mM', '200mM', '400mM', '600mM'.
    t : float, optional
        The temperature in degrees Celsius.

    Returns
    -------
    float
        The calculated ppm value.

    Raises
    ------
    ValueError
        If an unsupported concentration is provided.

    Notes
    -----
    This function uses a predefined mapping of concentrations to their respective
    parameters for calculating the ppm value.

    Examples
    --------
    >>> ppm_from_temp_and_concentration('10mM', 25)
    15.9155
    """
    # Mapping concentrations to their respective parameters
    concentration_params = {
        "5mM": (-0.01297, 16.23),
        "10mM": (-0.01298, 16.24),
        "20mM": (-0.01311, 16.29),
        "50mM": (-0.01339, 16.41),
        "100mM": (-0.01373, 16.57),
        "200mM": (-0.01441, 16.86),
        "400mM": (-0.01544, 17.34),
        "600mM": (-0.01623, 17.72),
    }

    if c in concentration_params:
        slope, intercept = concentration_params[c]
        return slope * t + intercept
    else:
        raise ValueError(
            "Unsupported concentration. Please choose from '5mM', '10mM', '20mM', '50mM', '100mM', '200mM', '400mM', '600mM'."
        )


def temp_from_ppm_and_concentration(c="5mM", ppm=0, return_kelvin=True):
    """
    Calculate the temperature based on the concentration and ppm value.

    Parameters
    ----------
    c : str, optional
        The concentration of the solution. Supported values are '5mM', '10mM', '20mM', '50mM', '100mM', '200mM', '400mM', '600mM'.
    ppm : float, optional
        The ppm value.
    return_kelvin : bool, optional
        If True, return temperature in Kelvin. If False, return in Celsius.

    Returns
    -------
    float
        The calculated temperature in Kelvin or Celsius.

    Raises
    ------
    ValueError
        If an unsupported concentration is provided.

    Notes
    -----
    This function uses a predefined mapping of concentrations to their respective
    parameters for calculating the temperature.

    See Also
    --------
    ppm_from_temp_and_concentration : Inverse function to calculate ppm from temperature and concentration.

    Examples
    --------
    >>> temp_from_ppm_and_concentration('10mM', 16.0, return_kelvin=False)
    18.49
    """
    # Mapping concentrations to their respective parameters
    from ..utils.utils_general import kelvin_to_celsius

    concentration_params = {
        "5mM": (-0.01297, 16.23),
        "10mM": (-0.01298, 16.24),
        "20mM": (-0.01311, 16.29),
        "50mM": (-0.01339, 16.41),
        "100mM": (-0.01373, 16.57),
        "200mM": (-0.01441, 16.86),
        "400mM": (-0.01544, 17.34),
        "600mM": (-0.01623, 17.72),
    }
    if c in concentration_params:
        slope, intercept = concentration_params[c]
        # Rearranging the equation ppm = slope * t + intercept to solve for t
        t = (ppm - intercept) / slope
        if return_kelvin:
            return t
        else:
            return kelvin_to_celsius(kelvin=t)
    else:
        raise ValueError(
            "Unsupported concentration. Please choose from '5mM', '10mM', '20mM', '50mM', '100mM', '200mM', '400mM', '600mM'."
        )


def temperature_from_frequency(
    frequency, calibration_type="blood", frequency_is_ppm=True
):
    """
    Returns the temperature of a pyruvate-lactate solution for a desired calibration function.

    This function is mostly used for the 2024 Publication on Thermometry.

    Parameters
    ----------
    frequency: float
        Pyruvate-lactate relative chemical shift in Hz or ppm.
    calibration_type: str, optional
        Type of calibration to use. Default is 'Blood'.
        Other options are: 'LDH', 'Blood_meaned','LDH_meaned','cryo_thermal' or
        concentrations in the form: '5mM' - from the following options: 5,10,20,50,100,200,400,600.
    frequency_is_ppm: bool, optional
        If input in Hz at 7T is desired then set this to False. You can only use calibration functions from the following list
        ['cryo_thermal', 'LDH', 'Blood', 'Blood_meaned','LDH_meaned'] since other measurements were not at 7T.

    Returns
    -------
    temperature, temperature_error: float
        Calculated temperature and its error.

    Raises
    ------
    ValueError
        If an invalid calibration type is provided or if frequency is not in ppm for certain calibrations.

    Notes
    -----
    The function uses different calibration parameters based on the calibration type and whether the frequency is in ppm or Hz.

    Examples
    --------
    >>> temperature_from_frequency(16.0, calibration_type='Blood', frequency_is_ppm=True)
    (310.15, 0.5)
    """

    def linear(f, pitch, crossing):
        return (f - crossing) / pitch

    def err_linear(f, pitch, pitch_error, crossing, crossing_err):
        return np.sqrt(
            (crossing_error / pitch) ** 2
            + (pitch_error * ((crossing - f) / pitch)) ** 2
        )

    if (
        calibration_type
        not in ["cryo_thermal", "LDH", "Blood", "Blood_meaned", "LDH_meaned"]
        and frequency_is_ppm == False
    ):
        raise ValueError(
            "You must input the frequency in ppm for calibrations at 11.9T"
        )
    if frequency_is_ppm == True:
        if calibration_type == "cryo_thermal":
            pitch = -0.016321184066004
            pitch_error = 0.001151351715244
            crossing = 12.843448632366941
            crossing_error = 4.4252367604e-05
        elif calibration_type == "LDH":
            pitch = -0.01472863465268
            pitch_error = 0.009713604636182
            crossing = 12.780660470812476
            crossing_error = 0.000357109774727
        elif calibration_type == "LDH_meaned":
            pitch = -0.014386495546791
            pitch_error = 0.056925861243265
            crossing = 12.763982363066608
            crossing_error = 0.00202276880719
        elif calibration_type == "Blood":
            pitch = -0.013011837418234
            pitch_error = 0.03850878645128
            crossing = 12.737055542464372
            crossing_error = 0.001205824445164
        elif calibration_type == "Blood_meaned":
            pitch = -0.014365470412533
            pitch_error = 0.067207631500598
            crossing = 12.765264460856539
            crossing_error = 0.002331100420903
        elif calibration_type == "5mM":
            pitch = -0.012967023642931
            pitch_error = 0.001611918011057
            crossing = 12.687991248319072
            crossing_error = 0.000499970341949
        elif calibration_type == "10mM":
            pitch = -0.012976898165725
            pitch_error = 0.003458085864921
            crossing = 12.69612502856522
            crossing_error = 0.001072727320076
        elif calibration_type == "20mM":
            pitch = -0.013109496965684
            pitch_error = 0.00236649843376
            crossing = 12.712149898413177
            crossing_error = 0.000740674010679
        elif calibration_type == "50mM":
            pitch = -0.013385022660291
            pitch_error = 0.003073372872851
            crossing = 12.755002271910573
            crossing_error = 0.000978830935923
        elif calibration_type == "100mM":
            pitch = -0.013731512844978
            pitch_error = 0.005107782357626
            crossing = 12.814997258304782
            crossing_error = 0.001661063045042
        elif calibration_type == "200mM":
            pitch = -0.014408217740115
            pitch_error = 0.007887358921608
            crossing = 12.926556654978315
            crossing_error = 0.00266816618565
        elif calibration_type == "400mM":
            pitch = -0.01544148517688
            pitch_error = 0.009940461235411
            crossing = 13.118609741470681
            crossing_error = 0.003551089800098
        elif calibration_type == "600mM":
            pitch = -0.016231456018516
            pitch_error = 0.012464416792304
            crossing = 13.288926529656862
            crossing_error = 0.00462054712916
        else:
            print("calibration type not recognized, defaulting to blood at 7T")
            pitch = -0.013011837418234
            pitch_error = 0.03850878645128
            crossing = 12.737055542464372
            crossing_error = 0.001205824445164

    else:
        # input in Hz at 7T
        if calibration_type == "cryo_thermal":
            pitch = -1.232111910557836
            pitch_error = 0.001151351782786
            crossing = 969.5721817181965
            crossing_error = 4.4252370713e-05
        elif calibration_type == "LDH":
            pitch = -1.111887813410422
            pitch_error = 0.009713604908939
            crossing = 964.8322034375896
            crossing_error = 0.000357109790553
        elif calibration_type == "LDH_meaned":
            pitch = -1.086059180561179
            pitch_error = 0.056925856223268
            crossing = 963.57314641691
            crossing_error = 0.00202276880719
        elif calibration_type == "Blood":
            pitch = -0.982284162038326
            pitch_error = 0.038508790923977
            crossing = 961.5404010527442
            crossing_error = 0.001205824603659
        elif calibration_type == "Blood_meaned":
            pitch = -1.084472026996133
            pitch_error = 0.067207630777528
            crossing = 963.6699360643906
            crossing_error = 0.002331100486608
        else:
            print("calibration type not recognized, defaulting to blood at 7T")
            pitch = -0.982284162038326
            pitch_error = 0.038508790923977
            crossing = 961.5404010527442
            crossing_error = 0.001205824603659

    temperature, temperature_error = linear(frequency, pitch, crossing), err_linear(
        frequency, pitch, pitch_error, crossing, crossing_error
    )

    return temperature, temperature_error


def remove_peak_from_spec(
    input_data=None, fit_result=None, peak_to_avoid=None, fit_params=None, n_fwhm=6
):
    """
    Remove the area around a specified peak in a spectrum.

    This function removes a range around a peak defined by n_fwhm * FWHM. If no peak is specified,
    the function removes around the peak that has the maximum amplitude.

    Parameters
    ----------
    input_data : array-like, optional
        The spectrum data to process.
    fit_result : object, optional
        The fit results from a previous operation, contains peak properties.
    peak_to_avoid : int or None, optional
        Index of the peak to be removed. If not provided, the peak subtracted
        earlier or the biggest peak will be selected.
    fit_params : dict, optional
        Parameters from the fitting operation. Should include 'freq_range' which
        specifies the frequency range of input_data.
    n_fwhm : int, default=6
        Number of times the FWHM is used to define the range around the peak to remove.

    Returns
    -------
    freq_range_nan : array-like
        Frequency range after removal.
    input_data_nan : array-like
        Spectrum data after removal.

    Notes
    -----
    This function is useful for removing dominant peaks to allow better analysis of smaller peaks.
    """
    # Check if input data is provided; if not, return with a warning
    if input_data is None:
        Warning("No input_data passed")
        return None

    # Check if fitting results are provided; if not, return with a warning
    if fit_result is None:
        Warning("No fitting results fit_result passed")
        return None

    # Copy fit results to ensure original data remains unaltered
    from copy import deepcopy

    fit_result_copy = deepcopy(fit_result)

    # Determine the peak to subtract if not explicitly specified by the user
    if peak_to_avoid is None:
        # Try to get the previously subtracted peak from fitting parameters
        peak_to_avoid = fit_params.get("peak_to_subtract", None)
        if peak_to_avoid is None:
            # If no previous peak was subtracted, choose the peak with the max amplitude
            peak_to_avoid = (
                find_max_amp(fit_result=fit_result_copy, fit_params=fit_params) + 1
            )

    # Convert peak center frequency to its index in the data array for easier manipulation
    x_0 = freq_to_index(
        getattr(getattr(fit_result_copy, f"x_0_{peak_to_avoid}"), "value"),
        freq_range=fit_params["freq_range"],
    )

    # Compute the resolution of the frequency data
    freq_res = np.abs(fit_params["freq_range"][1] - fit_params["freq_range"][0])
    # Calculate the number of data points that span the FWHM of the peak
    nan_range = int(
        np.ceil(
            getattr(getattr(fit_result_copy, f"fwhm_{peak_to_avoid}"), "value")
            / freq_res
        )
    )

    # Determine the range to be set to NaN based on the FWHM and n_fwhm parameter
    start = x_0 - n_fwhm * nan_range
    end = x_0 + n_fwhm * nan_range

    # Extract and concatenate the frequency range, excluding the NaN range
    freq_range_nan = deepcopy(fit_params["freq_range"])
    freq_range_nan = np.concatenate([freq_range_nan[:start], freq_range_nan[end:]])

    # Extract and concatenate the spectrum data, excluding the NaN range
    input_data_nan = deepcopy(input_data)
    input_data_nan = np.concatenate([input_data_nan[:start], input_data_nan[end:]])

    return freq_range_nan, input_data_nan


def rayleigh_pdf_fit(x, mu, sigma):
    """
    Custom function for fitting the Rayleigh PDF, adjusted for the scipy curve_fit function.

    Parameters
    ----------
    x : array-like
        Input data points.
    mu : float
        Location parameter of the Rayleigh distribution.
    sigma : float
        Scale parameter of the Rayleigh distribution.

    Returns
    -------
    array-like
        Rayleigh probability density function values for the input data points.

    Notes
    -----
    This function is designed to be used with scipy's curve_fit for fitting Rayleigh distributions.
    """
    from scipy import stats

    return stats.rayleigh.pdf(x, loc=mu, scale=sigma)


def fit_rayleigh(input_data=None, plot=True, **kwargs):
    """
    Fit a Rayleigh distribution to the magnitude of the input data.

    Parameters
    ----------
    input_data : array_like, optional
        The input data for which the Rayleigh distribution is to be fitted.
        This data is expected to be complex and its magnitude is used.
    plot : bool, optional
        If True, plots the signal magnitude and the histogram with the fitted
        Rayleigh PDF. Defaults to True.
    **kwargs : dict, optional
        Additional keyword arguments. May include:
        - bins : int, number of bins for the histogram. Defaults to 30.
        - density : bool, if True, the histogram is normalized. Defaults to True.
        - p0 : tuple, initial guess for the fitting coefficients (mu, sigma).

    Returns
    -------
    popt, pcov : ndarray
        Optimal values for the parameters so that the sum of the squared
        residuals of `f(xdata, *popt) - ydata` is minimized, where `f` is the
        Rayleigh PDF.

    Raises
    ------
    ValueError
        If `input_data` is not provided.

    Notes
    -----
    The function fits a Rayleigh distribution to the absolute values of the input data
    using scipy's curve_fit function. If plotting is enabled, two subplots are created:
    one showing the magnitude of the signal and the other showing the histogram of the
    signal's magnitude with the fitted Rayleigh PDF overlaid.
    """

    from scipy import optimize

    if input_data is None:
        raise ValueError("input_data has to be passed!")

    input_data = np.abs(
        input_data
    ).flatten()  # Ensure input is flattened and magnitude is taken

    # Extract or set default keyword arguments
    bins = kwargs.get("bins", 30)
    plot_density = kwargs.get(
        "density", True
    )  # Renamed to avoid confusion with histogram output
    p0 = kwargs.get("p0", [np.mean(input_data), np.std(input_data)])

    # Compute histogram density and bins
    hist_density, bins = np.histogram(input_data, bins=bins, density=plot_density)
    centers = (bins[:-1] + bins[1:]) / 2
    xlin = np.linspace(bins[0], bins[-1], 100)

    # Fit Rayleigh distribution
    popt, pcov = optimize.curve_fit(rayleigh_pdf_fit, centers, hist_density, p0=p0)

    if plot:
        fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 6))
        ax[0].plot(input_data)
        ax[0].set_title("Signal Magnitude")
        ax[1].hist(input_data, bins=bins, density=plot_density)
        ax[1].plot(xlin, rayleigh_pdf_fit(xlin, *popt), color="r")
        ax[1].axvline(np.mean(input_data), color="k", label="Mean")
        ax[1].axvline(popt[0], color="r", label="Fitted Mu")
        ax[1].axvline(popt[1], color="g", label="Fitted Sigma")
        ax[1].set_title("Histogram Magnitude")
        ax[1].legend()

    return popt, pcov


# ----pseudo-inv-fitting----pseudo-inv-fitting----pseudo-inv-fitting----pseudo-inv-fitting----pseudo-inv-fitting----pseud
def basefunc(freq, T2_star, t):
    """
    Calculate the Free Induction Decay (FID) signal for given frequencies and T2* relaxation times.

    Parameters
    ----------
    freq : array_like
        Frequencies (in Hz) for which to calculate the FID signal.
    T2_star : array_like
        T2* relaxation times (in seconds) corresponding to each frequency in `freq`.
    t : array_like or scalar
        Time points (in seconds) at which the FID signal is calculated.

    Returns
    -------
    numpy.ndarray
        An array of complex values representing the FID signal at each time point for each
        frequency and T2* pair.

    Notes
    -----
    The FID signal is calculated using the formula:
    s(t) = exp(1i * 2 * pi * freq * t) * exp(-t / T2_star)

    Examples
    --------
    >>> freq = [500, 600]
    >>> T2_star = [0.05, 0.03]
    >>> t = np.linspace(0, 0.1, 100)
    >>> signal = basefunc(freq, T2_star, t)
    """
    if not isinstance(freq, list):
        freq = [freq]
    if not isinstance(T2_star, list):
        T2_star = [T2_star]

    # avoid divide by 0 issues:
    if isinstance(T2_star, list) and 0 in T2_star:
        sig = np.zeros_like(
            [
                np.exp(1j * 2.0 * np.pi * f * t) * np.exp(-t / 1.0)
                for f, T2 in zip(freq, T2_star)
            ]
        )
    else:
        sig = np.array(
            [
                np.exp(1j * 2.0 * np.pi * f * t) * np.exp(-t / T2)
                for f, T2 in zip(freq, T2_star)
            ]
        )
    return sig


def basefunc_jacobian(freq, T2_star):
    """
    Calculate the Jacobian of the basefunc with respect to its parameters.

    This function is a placeholder and currently returns True. It should be implemented
    to return the actual Jacobian matrix of the basefunc with respect to frequency and T2*.

    Parameters
    ----------
    freq : array_like
        Frequencies for which to calculate the Jacobian.
    T2_star : array_like
        T2* relaxation times for which to calculate the Jacobian.

    Returns
    -------
    bool
        Currently returns True. Should be implemented to return the Jacobian matrix.

    Notes
    -----
    This function needs to be properly implemented to return the Jacobian matrix
    for optimization algorithms that require it.
    """


def calc_X(x, intercept=1):
    """
    Define a parameter Matrix X for linear regression models.

    This function prepares the input x into a format suitable for linear regression,
    optionally including an intercept term.

    Parameters
    ----------
    x : array_like
        Input data to be transformed into the parameter matrix.
    intercept : int, optional
        If 1 (default), includes a column of ones for the intercept term.
        If 0, no intercept term is included.

    Returns
    -------
    numpy.ndarray
        The parameter matrix X, where each row corresponds to an observation
        and each column to a parameter (including the intercept if specified).

    Notes
    -----
    This function handles various input shapes and dimensions, making it versatile
    for different types of regression problems.
    """
    if x.shape[0] == 1:
        x = x.T
    # X contains the sampling points [[1, t0], [1, t1], ..., [1, tN]]
    if isinstance(x[0], np.ndarray):
        if len(x[0]) > 1:
            x_arr = []
            if intercept != 0:
                for k in range(x.shape[0]):
                    xi = x[k, :]
                    x_arr.append([intercept] + [x for x in xi])
                return np.array(x_arr)
            else:
                # default path for fitting spectra:
                for k in range(x.shape[0]):
                    xi = x[k, :]
                    x_arr.append([x for x in xi])
                return np.array(x_arr)

        if intercept != 0:
            return np.array([[intercept] + [xi[0]] for xi in x])
        else:
            return np.array([[xi[0]] for xi in x])
    else:
        if intercept != 0:
            return np.array([[intercept, xi] for xi in x])
        else:
            return np.array([[xi] for xi in x])


def calc_beta(X, y, x=None, intercept=1):
    """
    Calculate the beta coefficients for a linear regression model.

    This function solves the equation Y = X * beta using the pseudoinverse method.

    Parameters
    ----------
    X : array_like
        The design matrix of shape (n_samples, n_features).
    y : array_like
        The target values of shape (n_samples,).
    x : array_like, optional
        Raw input data. If provided and X is None, calc_X will be used to create X.
    intercept : int, optional
        Passed to calc_X if x is provided. Default is 1.

    Returns
    -------
    numpy.ndarray
        The calculated beta coefficients.

    Notes
    -----
    This function uses numpy's pseudoinverse (pinv) to solve for beta,
    which is suitable for both overdetermined and underdetermined systems.
    """
    if X is None and x is not None:
        X = calc_X(x, intercept=intercept)
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


def eta(X=None, beta=[0, 1], x=None):
    """
    Calculate the predicted values for a linear model.

    Parameters
    ----------
    X : array_like, optional
        The design matrix. If None, it will be calculated from x using calc_X.
    beta : array_like, optional
        The coefficients of the linear model. Default is [0, 1].
    x : array_like, optional
        Raw input data to be used if X is None.

    Returns
    -------
    numpy.ndarray
        The predicted values based on X and beta.

    Notes
    -----
    This function calculates X.dot(beta) to produce the predicted values.
    It ensures X is a 2D array before performing the dot product.
    """
    if X is None:
        X = calc_X(x)
    if X.ndim == 1:
        X = X[:, np.newaxis]  # Ensure X is a column vector if 1D
    return X.dot(beta)


def S(y, X, b):
    """
    Calculate the residual sum of squares for a linear model.

    Parameters
    ----------
    y : array_like
        The true target values.
    X : array_like
        The design matrix.
    b : array_like
        The coefficients of the linear model.

    Returns
    -------
    float
        The residual sum of squares.

    Notes
    -----
    This function computes ||y - X.dot(b)||^2, which is the squared
    Euclidean 2-norm of the residual vector.
    """
    # define residual sum of squares:
    return np.linalg.norm(y - X.dot(b)) ** 2


def s_square(y, X, b, N, P):
    # residual mean square / variance estimate:
    # y = X[:, 1]
    assert N > P
    return S(y, X, b) / (N - P)


def fishers_f(N, P, alpha):
    """Fishers'F distribution for the upper alpha quantile with P and N-P
    # degrees of freedom"""

    # example from Watts: fishers_f(N=28, P=2, alpha=0.05) = 3.37
    assert N > P
    from scipy import stats

    return stats.f.ppf(1 - alpha, P, N - P)


def students_t(N, P, alpha):
    """ "Students T distribution for alpha/2 and N-P degrees of freedom:"""
    assert N > P
    from scipy import stats

    return stats.t.ppf(1 - alpha / 2, N - P)


def standard_error(X, p, s=None, y=None, b=None, N=None, P=None):
    # se = s * sqrt((X.T * X) ^ -1 [p]
    if s is None:
        s = np.sqrt(s_square(y=y, X=X, b=b, N=N, P=P))

    # sqrt[(X.T * X) ^ -1] (for debugging)
    term2 = np.sqrt(np.linalg.inv(X.T.dot(X)).diagonal()[p])

    # the pth diagonal term of the matrix (X.T X)^-1
    return s * term2


def mci(b, p, N, P, X, y, alpha):
    """
    marginal confidence intervall
    Parameters
    ----------
    b
    p
    N
    P
    X
    y
    alpha

    Returns
    -------

    """
    return standard_error(X=X, p=p, y=y, b=b, N=N, P=P) * students_t(
        N=N, P=P, alpha=alpha
    )


def cb(x, y, X, b, N, P, alpha):
    """
    Calculates the confidence bands

    Parameters
    ----------
    x: derivative matrix (N x P)
    y: measured signal (N)
    X: ...
    b: estimated values
    N: Number of observations (measured points)
    P: Number of parameters:
    alpha: confidence interval (100-alpha) %

    Returns
    -------

    """
    if x.shape[0] > 2:
        band = []
        for k in range(x.shape[0]):
            band.append(
                np.sqrt(s_square(y=y, X=X, b=b, N=N, P=P))
                * np.sqrt(x[k, :].T.dot(np.linalg.inv(X.T.dot(X))).dot(x[k, :]))
                * np.sqrt(P * fishers_f(N, P, alpha))
            )

        return np.array(band)
    else:
        # b1 + b2 +/ s * sqrt(
        return (
            np.sqrt(s_square(y=y, X=X, b=b, N=N, P=P))
            * np.sqrt(x.T.dot(np.linalg.inv(X.T.dot(X))).dot(x))
            * np.sqrt(P * fishers_f(N, P, alpha))
        )


def fid_func_variable_peaks(t, *params):
    """
    Calculates the free induction decay (FID) signal for a variable number of peaks.

    Parameters
    ----------
    t : numpy.ndarray
        An array of time points at which the FID signal is calculated.
    *params : float
        A sequence of parameters for each peak in the FID signal. The parameters for each peak
        are given in the order: frequency (w0_1, w0_2, ..., w0_N), decay time (T2_1, T2_2, ..., T2_N),
        and amplitude (amp_1, amp_2, ..., amp_N), where N is the number of peaks.

    Returns
    -------
    numpy.ndarray
        The calculated FID signal at each time point in `t`.

    Notes
    -----
    The function assumes a complex signal representation, where each peak in the FID signal
    is modeled as a decaying exponential modulated by a sinusoidal function of its frequency.
    """
    # Calculate number of peaks from parameters
    num_peaks = len(params) // 3
    amp = params[:num_peaks]
    w0 = params[num_peaks : 2 * num_peaks]
    t2 = params[2 * num_peaks :]

    # Initialize signal
    sig = 0.0 + 1j * 0.0
    # Accumulate signal from each peak
    for a, w, t22 in zip(amp, w0, t2):
        sig += a * np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
    return sig


def jac_fid_func_variable_peaks_real(t, *params):
    """
    Calculates the Jacobian matrix of the real part of the FID signal
    for a variable number of peaks with respect to its parameters.

    Parameters
    ----------
    t : numpy.ndarray
        An array of time points at which the Jacobian is calculated.
    *params : float
        Parameters for the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The Jacobian matrix of the real part of the FID signal with respect
        to the parameters.

    Notes
    -----
    The Jacobian matrix is structured with rows corresponding to parameters
    (in the order of amplitudes, frequencies, and T2 values) and columns
    corresponding to time points in `t`.
    """
    num_peaks = len(params) // 3
    amp = params[:num_peaks]
    w0 = params[num_peaks : 2 * num_peaks]
    t2 = params[2 * num_peaks :]

    jac = np.empty((3 * num_peaks, t.size), dtype=complex)

    c = 0
    # Compute derivatives with respect to amplitudes
    for w, t22 in zip(w0, t2):
        jac[c, :] = np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        c += 1

    # Compute derivatives with respect to frequencies
    for a, w, t22 in zip(amp, w0, t2):
        jac[c, :] = (
            a
            * (1j * 2.0 * np.pi * t)
            * np.exp(1j * 2.0 * np.pi * w * t)
            * np.exp(-t / t22)
        )
        c += 1

    # Compute derivatives with respect to T2 values
    for a, w, t22 in zip(amp, w0, t2):
        jac[c, :] = (
            a * (-t / (t22**2)) * np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        )
        c += 1

    return np.real(jac)


def jac_fid_func_variable_peaks_imag(t, *params):
    """
    Computes the Jacobian of the imaginary part of the FID signal with respect to its parameters,
    for a variable number of peaks.

    Similar to `jac_fid_func_variable_peaks_real`, but focuses on the imaginary part of the signal.

    Parameters
    ----------
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as detailed for `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The Jacobian matrix of the imaginary part of the FID signal, with respect to the parameters,
        evaluated at each time point in `t`.
    """

    num_peaks = len(params) // 3
    amp = params[:num_peaks]
    w0 = params[num_peaks : 2 * num_peaks]
    t2 = params[2 * num_peaks :]

    jac = np.empty((3 * num_peaks, t.size), dtype=complex)

    c = 0

    # amplitudes:
    for w, t22 in zip(w0, t2):
        jac[c, :] = np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        c += 1

    # fequencies:
    for a, w, t22 in zip(amp, w0, t2):
        # print(f"a={a}, w={w}, t2={t22}")
        jac[c, :] = (
            a
            * (1j * 2.0 * np.pi * t)
            * np.exp(1j * 2.0 * np.pi * w * t)
            * np.exp(-t / t22)
        )
        c += 1
    # T2s:
    for a, w, t22 in zip(amp, w0, t2):
        jac[c, :] = (
            a * (-t / (t22**2)) * np.exp(1j * 2.0 * np.pi * w * t) * np.exp(-t / t22)
        )
        c += 1
    return np.imag(jac)


def std_squared_real(input_data, t, *params):
    """
    Calculates the squared standard deviation of the real part of the FID signal from the observed signal.

    This function is useful for estimating the noise level in the real component of the signal,
    based on the difference between the observed signal and the model predicted by `fid_func_variable_peaks`.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal as a numpy array.
    t : numpy.ndarray
        A 1D numpy array of time points at which the signal and model are evaluated.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the squared standard deviation of the real part of the FID signal.
    """

    def fid_func_variable_peaks_real_part(t, *params):
        return np.real(fid_func_variable_peaks(t, *params))

    n = len(params)
    m = len(t)
    if n + 1 >= m:
        raise ValueError(f"m={m} has to be bigger than n={n}+1")
    std = (
        1
        / (m - n + 1)
        * (np.real(input_data) - fid_func_variable_peaks_real_part(t, *params)).T
        @ (np.real(input_data) - fid_func_variable_peaks_real_part(t, *params))
    )
    std = std * np.eye(m)
    return std


def std_squared_imag(input_data, t, *params):
    """
    Calculates the squared standard deviation of the imaginary part of the FID signal from the observed signal.

    This function is useful for estimating the noise level in the imaginary component of the signal,
    based on the difference between the observed signal and the model predicted by `fid_func_variable_peaks`.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal as a numpy array.
    t : numpy.ndarray
        A 1D numpy array of time points at which the signal and model are evaluated.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the squared standard deviation of the imaginary part of the FID signal.
    """

    def fid_func_variable_peaks_imag_part(t, *params):
        return np.imag(fid_func_variable_peaks(t, *params))

    n = len(params)
    m = len(t)
    if n + 1 >= m:
        raise ValueError(f"m={m} has to be bigger than n={n}+1")
    std = (
        1
        / (m - n + 1)
        * (np.imag(input_data) - fid_func_variable_peaks_imag_part(t, *params)).T
        @ (np.imag(input_data) - fid_func_variable_peaks_imag_part(t, *params))
    )
    std = std * np.eye(m)
    return std


def weights_real(input_data, t, *params):
    """
    Computes weighting coefficients for the real part of the FID signal based on its noise characteristics.

    These weights are inversely proportional to the variance of the noise in the real part of the signal,
    useful for weighted least squares fitting or analysis.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A diagonal matrix of weights for the real part of the FID signal.
    """

    std = std_squared_real(input_data, t, *params)
    m = len(t)
    weights = np.zeros_like(std)
    # Take the inverse of the diagonal of std, which represents variance
    np.fill_diagonal(weights, 1 / np.diag(std))
    return weights


def weights_imag(input_data, t, *params):
    """
    Computes weighting coefficients for the imagninary part of the FID signal based on its noise characteristics.

    These weights are inversely proportional to the variance of the noise in the imagninary part of the signal,
    useful for weighted least squares fitting or analysis.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        A diagonal matrix of weights for the imagninary part of the FID input_data.
    """

    std = std_squared_imag(input_data, t, *params)

    weights = np.zeros_like(std)
    # Take the inverse of the diagonal of std, which represents variance
    np.fill_diagonal(weights, 1 / np.diag(std))
    return weights


def parameter_covariance_real(input_data, t, *params):
    """
    Estimates the covariance matrix of the parameters for the real part of the FID signal.

    This covariance matrix is useful for understanding the confidence intervals and correlations
    between parameters estimated from the real part of the FID signal.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The covariance matrix of the parameters estimated from the real part of the FID signal.
    """

    jac = jac_fid_func_variable_peaks_real(t, *params)
    weights = weights_real(input_data, t, *params)
    # print(f"jac.shape={jac.shape}")
    # print(f"weights.shape={weights.shape}")

    # The multiplication should ensure that the weights are applied correctly
    temp = jac @ weights @ jac.T
    param_cov = np.linalg.inv(temp)
    return param_cov


def parameter_covariance_imag(input_data, t, *params):
    """
    Estimates the covariance matrix of the parameters for the imaginary part of the FID signal.

    This covariance matrix is useful for understanding the confidence intervals and correlations
    between parameters estimated from the imaginary part of the FID signal.

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed FID signal.
    t : numpy.ndarray
        A 1D numpy array of time points.
    *params : tuple
        Parameters of the FID signal as expected by `fid_func_variable_peaks`.

    Returns
    -------
    numpy.ndarray
        The covariance matrix of the parameters estimated from the imaginary part of the FID signal.
    """

    jac = jac_fid_func_variable_peaks_imag(t, *params)
    weights = weights_imag(input_data, t, *params)
    # print(f"jac.shape={jac.shape}")
    # print(f"weights.shape={weights.shape}")

    # The multiplication should ensure that the weights are applied correctly
    temp = jac @ weights @ jac.T
    param_cov = np.linalg.inv(temp)
    return param_cov


def estimate_std_fit_results(
    popt, fit_params, input_data, verbose=False, full_output=False
):
    """
    Estimates the standard deviation of fit results for both real and imaginary parts of the FID signal.

    This function calculates the standard deviation of the fitted parameters, providing an insight into
    the precision of the fit.

    Parameters
    ----------
    popt : numpy.ndarray
        Optimized parameters from fitting the FID signal.
    fit_params : dict
        Dictionary containing fit parameters including the time axis (`time_axis`).
    input_data : numpy.ndarray
        The observed FID signal.
    verbose : bool, optional
        If True, print detailed information about the amplitude, frequency, and T2 estimates.
    full_output : bool, optional
        If True, return the detailed standard deviations of the fit results.

    Returns
    -------
    numpy.ndarray
        Standard deviation of the fitted parameters. If `full_output` is True, returns detailed standard deviations.
    """
    # define measured signal as FID:
    if fit_params["signal_domain"] == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data))
    else:
        pass

    param_cov = parameter_covariance_real(input_data, fit_params["time_axis"], *popt)
    cov = np.diag(param_cov)
    std_real = np.sqrt(cov)

    param_cov = parameter_covariance_imag(input_data, fit_params["time_axis"], *popt)
    cov = np.diag(param_cov)
    std_imag = np.sqrt(cov)

    num_peaks = len(popt) // 3

    # Caclulate the mean of real and imaginary part
    std = np.sqrt(std_real**2 + std_imag**2) / np.sqrt(2)

    # 4 = (complex) amplitude, freq, T2, (magnitude) amplitude
    std_res = np.zeros((4, num_peaks), dtype=complex)

    std = np.reshape(std, [-1, num_peaks])
    std_real = np.reshape(std_real, [-1, num_peaks])
    std_imag = np.reshape(std_imag, [-1, num_peaks])

    # complex amplitude:
    std_res[0, :] = std_real[0, :] + 1j * std_imag[0, :]
    # frequency
    std_res[1, :] = std[1, :]
    # T2
    std_res[2, :] = std[2, :]
    # Magnitude Amplitude:
    std_res[3, :] = std[0, :]

    popt_res = np.reshape(popt, [-1, num_peaks])

    std_res = np.transpose(std_res, (1, 0))
    popt_res = np.transpose(popt_res, (1, 0))

    if verbose:
        print("Amplitude Estimates:")
        for i in range(num_peaks):
            print(
                f"Amp({i + 1}): {np.abs(np.round(popt_res[i, 0], 2)):>5} +/- "
                f"{np.round(std_res[i, 3], 2)} a.u."
            )
            print(
                f"Amp({i + 1}): {np.round(popt_res[i, 0], 2):>5} +/- "
                f"{np.round(std_res[i, 0], 2)} a.u."
            )

        print("\nFrequency Estimates:")
        for i in range(num_peaks):
            print(
                f"freq({i + 1}): {np.real(np.round(popt_res[i, 1], 2)):>5} +/- "
                f"{np.round(np.real(std_res[i, 1]), 2)} Hz"
            )

        print("\nT2 Estimates:")
        for i in range(num_peaks):
            print(
                f"T2({i + 1}): {np.real(np.round(1000 * popt_res[i, 2], 2)):>5} +/- "
                f"{np.round(1000 * np.real(std_res[i, 2]), 2)} ms"
            )

    if full_output:
        return std_res
    else:
        return std_res


def fit_func_pseudo_inv_error(X=None, y=None, beta=None, alpha=0.05):
    """
    Calculate the marginal confidence interval (mci) [(100-alpha) %]

    Parameters
    ----------
    X (np.ndarray):
        parameter matrix
    y (np.ndarray):
        measured input_data
    beta:
        estimates fit coefficients
    alpha:
        confidence interval

    Returns
    -------
    marginal confidence interval

    """
    # error estimation of beta:
    intercept = 0

    # number of observations:
    N = X.shape[0]

    # number of regressor variables:
    P = X.shape[1]

    # residual sum of squares:
    rSS = S(y=y, X=X, b=beta)

    # residual mean squares:
    rmS = s_square(y=y, X=X, b=beta, N=N, P=P)

    # Fishers'F
    FF = fishers_f(N=N, P=P, alpha=alpha)

    # standard error:
    sb1 = standard_error(X=X, p=0, s=np.sqrt(rmS))

    if intercept == 0:
        sb2 = 0
    else:
        sb2 = standard_error(X=X, p=1, s=np.sqrt(rmS))

    calc_mci = True
    if calc_mci:
        # marginal confidence interval:
        mcis = [mci(b=beta, p=p, N=N, P=P, X=X, y=y, alpha=alpha) for p in range(P)]

        if intercept == 0:
            mci2 = 0
        else:
            mci2 = mci(b=beta, p=1, N=N, P=P, X=X, y=y, alpha=alpha)

    calc_cb = False
    if calc_cb:
        if intercept == 0:
            cb1 = np.ones_like(X)
        else:
            x_temp = calc_X(x=np.linspace(np.min(X), np.max(X), X.shape[0]))
            cb1 = cb(x=x_temp, y=y, X=X, b=beta, N=N, P=P, alpha=alpha)
    return mcis


def fit_func_pseudo_inv(
    input_data,
    freqs,
    metabs_t2s,
    time_axis,
    signal_domain="spectral",
    dbmode=False,
    pseudo_inv=True,
):
    """
    Fits the input data to a model based on given frequencies and T2* relaxation times.

    This function generates a simulated signal using the base function defined for Free Induction Decay (FID),
    and then fits the input data to this simulated signal. The fitting is done in the spectral or time domain,
    based on the `signal_domain` parameter. The function computes the coefficients that best match the simulated
    signal to the input data using a pseudo-inverse approach.

    Parameters
    ----------
    input_data : array_like
        The input data to fit. This can be either in the time domain or the spectral domain,
        depending on the value of `signal_domain`.

    freqs : array_like
        A sequence of frequencies (in Hz) corresponding to different components or metabolites
        in the system.

    metabs_t2s : array_like
        A sequence of T2* relaxation times (in seconds) corresponding to each frequency in `freqs`.

    time_axis : array_like
        The time points (in seconds) at which the FID signal is calculated.

    signal_domain : str, optional
        The domain of the `input_data`. Can be 'spectral' or 'time'. Default is 'spectral'.

    Returns
    -------
    final_signal : numpy.ndarray
        The fitted signal in the same domain as `input_data`. It's the product of the simulated
        signal and the calculated coefficients.

    a : numpy.ndarray
        The coefficients calculated by fitting the simulated signal to the `input_data`.

    Notes
    -----
    The fitting process involves generating a simulated signal using the `basefunc` function and
    then calculating the pseudo-inverse of this simulated signal. The coefficients `a` are obtained
    by applying the pseudo-inverse to the `input_data`. The final fitted signal is then the product
    of the simulated signal and these coefficients.


    .. math::
        \\text{measured signal}   =  \\text{Signal time dependency factors (freq, T2, time)} \\cdot \\text{(complex) Amplitude}

    .. math::
        \\begin{align*}
            \\text{measured signal:} \\quad &S\\left(t\\right) \\\\
            \\text{Complex amplitude:} \\quad &A  \\\\
        \\end{align*}

    Signal equation of a Free Induction Decay FID:

    .. math::
        S(t) = A \\cdot \\left[ \\exp(i \\omega t) \\exp\\left(-\\frac{t}{T_2^*}\\right) \\right] = \\text{basefunc} \\cdot A

    Or written out:

    .. math::
        \\begin{align*}
            S(t=t_1) &= A \\cdot \\left[\\exp(i \\omega_1 t_1) \\cdot \\exp\\left(-\\frac{t_1}{T_{2_1}^*}\\right)\\right], \\\\
            S(t=t_2) &= A \\cdot \\left[\\exp(i \\omega_2 t_2) \\cdot \\exp\\left(-\\frac{t_2}{T_{2_2}^*}\\right)\\right], \\\\
                       &\\vdots \\\\
            S(t=t_N) &= A \\cdot \\left[\\exp(i \\omega_N t_N) \\cdot \\exp\\left(-\\frac{t_N}{T_{2_N}^*}\\right)\\right] \\\\
        \\end{align*}


    So the amplitude :math:`A` can be determined (if frequency :math:`\\omega` and decaying constant :math:`T_2^*` are
    known) by solving this system of linear equations:

    .. math::
        A = \\left(\\text{basefunc} ^{-1}\\right) * S(t)

    Examples
    --------

    >>> freqs = [500, 600]
    >>> metabs_t2s = [0.05, 0.03]
    >>> time_axis = np.linspace(0, 0.1, 100)
    >>> input_data = np.random.random(100)
    >>> final_signal, coefficients, coefficients_mci = fit_func_pseudo_inv(input_data, freqs, metabs_t2s, time_axis)

    """
    # generate signal:
    sim_signal = basefunc(freq=freqs, T2_star=metabs_t2s, t=time_axis)

    # define measured signal as FID:
    if signal_domain == "spectral":
        meas_signal = np.fft.ifft(np.fft.ifftshift(input_data))
    else:
        meas_signal = input_data

    # "faster" version
    if pseudo_inv:
        # define a parameter/derivate Matrix X (for error estimation):
        # Compute the pseudo-inverse of m
        m_pseudo_inv = np.linalg.pinv(sim_signal.T)

        # Compute amplitudes 'beta' using the pseudo-inverse
        beta = m_pseudo_inv.dot(meas_signal)
    else:
        # See 1988, Bates, Watts

        # define a parameter/derivate Matrix X:
        X = calc_X(x=sim_signal.T, intercept=0)

        # Solve the Equation Y = X * beta --> beta  = (X.T X)^-1 X.T y
        beta = calc_beta(X=X, y=meas_signal)

    # Compute 'a' using the pseudo-inverse
    final_signal = sim_signal.T * beta

    # return intermediate arrays for debugging:
    if dbmode:
        return final_signal, beta, sim_signal, m_pseudo_inv
    else:
        return final_signal, beta


def fit_params_setup_tool(
    input_data=None, fit_params={}, data_obj=None, plot_params={}
):
    """
    Interactive viewer for 6D MRI spectroscopy input_data + helper to set up the
    parameters metabs_freqs_Hz, metabs_freqs_ppm, metabs_freqs_Hz by clicking and
    metabs_freq_ranges by dragging, using the right mouse button.

    This function creates an interactive plot for visualizing and analyzing
    6D MRI spectroscopy input_data. It allows navigation through different dimensions
    of the input_data, switching between frequency units (Hz and ppm), and selecting
    different input_data types (real, imaginary, phase, magnitude).

    Parameters
    ----------
    input_data : ndarray
        6D array containing MRI spectroscopy input_data.
        Dimensions should be in the order:
        (spectroscopic, z, x, y, repetitions, channels)
    fit_params : dict
        Dictionary containing fitting parameters and metabolite information.
        Required keys:
        - 'metabs': list of str, metabolite names
        - 'metabs_freqs_Hz': list of float, metabolite frequencies in Hz
        - 'metabs_freqs_ppm': list of float, metabolite frequencies in ppm
        - 'freq_range_Hz': array-like, frequency range in Hz
        - 'freq_range_ppm': array-like, frequency range in ppm
        - 'metabs_freq_ranges': list of float, frequency ranges in Hz for each metabolite

    Returns
    -------
    None
        This function doesn't return any value but creates an interactive plot.

    Notes
    -----
    The viewer includes the following interactive features:
    - Navigation through z-slices, repetitions, and channels
    - Switching between Hz and ppm for frequency axis
    - Selecting different metabolites (highlighted with original color, others grayed out)
    - Clicking on the x-y image to update the spectrum
    - Clicking on the spectrum to set metabolite frequencies (automatically updates both Hz and ppm)
    - Right-clicking and dragging to select a spectral range for each metabolite
    - Choosing between real, imaginary, phase, and magnitude input_data display
    - Adaptive scaling for different input_data types
    - Table display of all metabolite frequencies and ranges

    The plot consists of three main elements:
    1. X-Y View: Shows a 2D slice of the input_data
    2. Spectrum: Shows the spectrum at the selected x-y position
    3. Frequency Table: Displays frequencies and ranges for all metabolites
    """
    from matplotlib.widgets import RectangleSelector
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    from hypermri.utils.utils_spectroscopy import freq_ppm_to_Hz, freq_Hz_to_ppm
    from matplotlib.patches import Rectangle

    if np.ndim(input_data) != 6:
        input_data = make_NDspec_6Dspec(
            input_data, provided_dims=fit_params["provided_dims"]
        )
    if fit_params["signal_domain"] == "spectral":
        pass
    else:
        input_data = np.fft.fftshift(np.fft.fft(input_data, axis=0), axes=0)
    input_data = np.rot90(
        input_data,
        k=1,
        axes=(2, 3),
    )

    current_z, current_rep, current_channel = (
        input_data.shape[1] // 2,
        input_data.shape[4] // 2,
        0,
    )
    current_x, current_y = input_data.shape[2] // 2, input_data.shape[3] // 2
    freq_axis_mode = "Hz"
    data_type = "Magnitude"
    figsize = plot_params.get("figsize", (15, 10))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    plt.subplots_adjust(bottom=0.25)

    # Prepare frequency axis
    freq_hz = fit_params["freq_range_Hz"]
    freq_ppm = fit_params["freq_range_ppm"]

    # Generate a color map for metabolite lines
    color_map = plt.cm.get_cmap("tab10")
    metab_colors = [color_map(i) for i in np.linspace(0, 1, len(fit_params["metabs"]))]

    # Add vertical lines for metabolites with different colors and labels
    metab_lines = []
    metab_labels = []
    metab_ranges = []
    for i, (metab, freq, color, range_hz) in enumerate(
        zip(
            fit_params["metabs"],
            fit_params["metabs_freqs_Hz"],
            metab_colors,
            fit_params["metabs_freq_ranges"],
        )
    ):
        
        if freq is None:
            fit_params["metabs_freqs_Hz"][i] = 0.0
        
        line = ax2.axvline(
            freq, color="gray", linestyle="--", linewidth=0.5, visible=False
        )
        label = ax2.text(
            freq,
            0,
            metab,
            color="gray",
            rotation=90,
            va="bottom",
            ha="right",
            visible=False,
        )
        range_rect = ax2.add_patch(
            Rectangle(
                (freq - range_hz / 2, 0),
                range_hz,
                1,
                facecolor=color,
                alpha=0.2,
                visible=False,
            )
        )
        metab_lines.append(line)
        metab_labels.append(label)
        metab_ranges.append(range_rect)

    def get_display_data(d):
        if data_type == "Real":
            return np.real(d)
        elif data_type == "Imaginary":
            return np.imag(d)
        elif data_type == "Phase":
            return np.angle(d)
        else:  # Magnitude
            return np.abs(d)

    def update_plots(z, rep, channel):
        nonlocal current_z, current_rep, current_channel
        current_z, current_rep, current_channel = z, rep, channel

        display_data = get_display_data(input_data[:, z, :, :, rep, channel])

        # Update image
        im.set_data(np.mean(display_data, axis=0))
        vmin, vmax = np.percentile(np.mean(display_data, axis=0), [2, 98])
        im.set_clim(vmin, vmax)

        # Update spectrum
        spectrum_data = display_data[:, current_x, current_y]
        spectrum_line.set_ydata(spectrum_data)

        # Update y-axis limits for spectrum
        if data_type == "Phase":
            ax2.set_ylim(-np.pi, np.pi)
        else:
            ax2.set_ylim(np.min(spectrum_data), 1.1 * np.max(spectrum_data))

        update_metab_lines()
        update_freq_table()

    def on_click_xy(event):
        nonlocal current_x, current_y
        if event.inaxes == ax1:
            current_x, current_y = int(event.ydata), int(event.xdata)  # Swap x and y
            spectrum_data = get_display_data(
                input_data[
                    :, current_z, current_x, current_y, current_rep, current_channel
                ]
            )
            spectrum_line.set_ydata(spectrum_data)
            ax2.set_title(
                f"Spectrum at X:{current_y}, Y:{current_x}"
            )  # Swap x and y in the title
            point_highlight.set_data(current_y, current_x)  # Swap x and y

            # Update y-axis limits for spectrum
            if data_type == "Phase":
                ax2.set_ylim(-np.pi, np.pi)
            else:
                ax2.set_ylim(np.min(spectrum_data), 1.1 * np.max(spectrum_data))

            fig.canvas.draw_idle()

    def on_click_spectrum(event):
        if event.inaxes == ax2 and event.button == 1:  # Left click
            freq = event.xdata
            metab_str = metab_selector.value
            metab_idx = fit_params["metabs"].index(metab_str)
            if freq_axis_mode == "Hz":
                fit_params["metabs_freqs_Hz"][metab_idx] = freq
                fit_params["metabs_freqs_ppm"][metab_idx] = freq_Hz_to_ppm(
                    freq_Hz=[freq], hz_axis=freq_hz, ppm_axis=freq_ppm
                )[0]
            else:
                fit_params["metabs_freqs_ppm"][metab_idx] = freq
                fit_params["metabs_freqs_Hz"][metab_idx] = freq_ppm_to_Hz(
                    freq_ppm=[freq], ppm_axis=freq_ppm, hz_axis=freq_hz
                )[0]

            update_metab_lines()
            update_freq_table()

    def on_select_range(eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        metab_str = metab_selector.value
        metab_idx = fit_params["metabs"].index(metab_str)
        new_range = abs(x2 - x1)
        if freq_axis_mode == "ppm":
            # Convert range from ppm to Hz
            new_range = (
                new_range
                * (fit_params["freq_range_Hz"][-1] - fit_params["freq_range_Hz"][0])
                / (fit_params["freq_range_ppm"][-1] - fit_params["freq_range_ppm"][0])
            )
        fit_params["metabs_freq_ranges"][metab_idx] = new_range
        update_metab_lines()
        update_freq_table()

        # Clear the RectangleSelector
        rs.set_active(False)
        rs.set_visible(False)
        fig.canvas.draw_idle()

        # Reactivate the RectangleSelector for future use
        rs.set_active(True)

    def update_metab_lines():
        y_positions = np.linspace(
            0.1, np.max(spectrum_line.get_data()[1]), len(metab_lines)
        )  # Distribute labels vertically

        sorted_indices = sorted(
            range(len(metab_lines)),
            key=lambda k: fit_params["metabs_freqs_Hz"][k]
            if freq_axis_mode == "Hz"
            else fit_params["metabs_freqs_ppm"][k],
            reverse=(freq_axis_mode == "ppm"),
        )

        selected_metab = metab_selector.value
        for i, idx in enumerate(sorted_indices):
            freq_hz = fit_params["metabs_freqs_Hz"][idx]
            freq_ppm = fit_params["metabs_freqs_ppm"][idx]
            range_hz = fit_params["metabs_freq_ranges"][idx]

            if freq_axis_mode == "Hz":
                freq = freq_hz
                range_start = freq - range_hz / 2
                range_end = freq + range_hz / 2
            else:
                freq = freq_ppm
                range_ppm = (
                    range_hz
                    * (
                        fit_params["freq_range_ppm"][-1]
                        - fit_params["freq_range_ppm"][0]
                    )
                    / (fit_params["freq_range_Hz"][-1] - fit_params["freq_range_Hz"][0])
                )
                range_start = freq - range_ppm / 2
                range_end = freq + range_ppm / 2

            metab_lines[idx].set_xdata(freq)
            metab_lines[idx].set_visible(True)

            if fit_params["metabs"][idx] == selected_metab:
                metab_lines[idx].set_color(metab_colors[idx])
                metab_labels[idx].set_color(metab_colors[idx])
            else:
                metab_lines[idx].set_color("gray")
                metab_labels[idx].set_color("gray")

            label = metab_labels[idx]
            label.set_position((freq, y_positions[i]))
            label.set_visible(True)

            metab_ranges[idx].set_xy((range_start, 0))
            metab_ranges[idx].set_width(range_end - range_start)
            metab_ranges[idx].set_height(ax2.get_ylim()[1])
            metab_ranges[idx].set_visible(True)

        fig.canvas.draw_idle()

    def update_freq_table():
        ax3.clear()
        ax3.axis("off")
        table_data = []
        for i, metab in enumerate(fit_params["metabs"]):
            freq_hz = fit_params["metabs_freqs_Hz"][i]
            freq_ppm = fit_params["metabs_freqs_ppm"][i]
            range_hz = fit_params["metabs_freq_ranges"][i]
            table_data.append(
                [metab, f"{freq_hz:.2f}", f"{freq_ppm:.2f}", f"±{range_hz / 2:.2f}"]
            )

        table = ax3.table(
            cellText=table_data,
            colLabels=["Metabolite", "Freq (Hz)", "Freq (ppm)", "Range (±Hz)"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Highlight the selected metabolite
        selected_metab = metab_selector.value
        selected_idx = fit_params["metabs"].index(selected_metab)
        for i in range(len(table_data[0])):
            table[(selected_idx + 1, i)].set_facecolor(metab_colors[selected_idx])
            table[(selected_idx + 1, i)].set_text_props(weight="bold")

        ax3.set_title("Metabolite Frequencies and Ranges")
        fig.canvas.draw_idle()

    def switch_freq_axis(change):
        nonlocal freq_axis_mode
        freq_axis_mode = change.new
        if freq_axis_mode == "Hz":
            spectrum_line.set_xdata(freq_hz)
            ax2.set_xlim(freq_hz[0], freq_hz[-1])
            ax2.set_xlabel("Frequency (Hz)")
        else:
            spectrum_line.set_xdata(freq_ppm)
            ax2.set_xlim(freq_ppm[-1], freq_ppm[0])  # Flip x-axis for ppm
            ax2.set_xlabel("Frequency (ppm)")
        update_metab_lines()
        update_freq_table()

    def switch_data_type(change):
        nonlocal data_type
        data_type = change.new
        update_plots(current_z, current_rep, current_channel)

    def on_metab_select(change):
        update_metab_lines()
        update_freq_table()

    # Initial plot setup
    display_data = get_display_data(
        input_data[:, current_z, :, :, current_rep, current_channel]
    )
    im = ax1.imshow(np.mean(display_data, axis=0), cmap="viridis")
    ax1.set_title("X-Y View")
    (point_highlight,) = ax1.plot(
        current_x, current_y, "ro", markersize=10, fillstyle="none"
    )

    spectrum_data = display_data[:, current_x, current_y]
    (spectrum_line,) = ax2.plot(freq_hz, spectrum_data)
    ax2.set_title("Spectrum")
    ax2.set_xlabel("Frequency (Hz)")

    # Set initial color limits and y-axis limits
    vmin, vmax = np.percentile(display_data, [2, 98])
    im.set_clim(vmin, vmax)
    ymin, ymax = np.percentile(spectrum_data, [1, 99])
    y_range = ymax - ymin
    ax2.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    fig.canvas.mpl_connect("button_press_event", on_click_xy)
    fig.canvas.mpl_connect("button_press_event", on_click_spectrum)

    # Set up RectangleSelector for range selection
    rs = RectangleSelector(
        ax2,
        on_select_range,
        useblit=True,
        button=[3],  # right click
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )  # Changed to False

    # Ensure the RectangleSelector stays active
    ax2.set_navigate(True)
    ax2.set_navigate_mode(None)

    # Create widgets
    z_slider = widgets.IntSlider(
        min=0, max=input_data.shape[1] - 1, step=1, value=current_z, description="Z:"
    )
    rep_slider = widgets.IntSlider(
        min=0,
        max=input_data.shape[4] - 1,
        step=1,
        value=current_rep,
        description="Repetition:",
    )
    channel_slider = widgets.IntSlider(
        min=0,
        max=input_data.shape[5] - 1,
        step=1,
        value=current_channel,
        description="Channel:",
    )
    freq_axis_toggle = widgets.ToggleButtons(
        options=["Hz", "ppm"], value="Hz", description="Frequency Axis:"
    )
    metab_selector = widgets.Dropdown(
        options=fit_params["metabs"],
        value=fit_params["metabs"][0],
        description="Metabolite:",
    )
    data_type_selector = widgets.Dropdown(
        options=["Magnitude", "Real", "Imaginary", "Phase"],
        value="Magnitude",
        description="Data Type:",
    )

    freq_axis_toggle.observe(switch_freq_axis, names="value")
    data_type_selector.observe(switch_data_type, names="value")
    metab_selector.observe(on_metab_select, names="value")

    interact(update_plots, z=z_slider, rep=rep_slider, channel=channel_slider)

    display(widgets.HBox([freq_axis_toggle, metab_selector, data_type_selector]))

    # Initial update of metabolite lines and frequency table
    update_metab_lines()
    update_freq_table()

    # Keep a reference to rs to prevent garbage collection
    plt.gcf().rs = rs

    plt.tight_layout()


def def_fit_params_calc_sampling(fit_params=None, data_obj=None, **kwargs):
    """
    Helper function that defines time/frequency parameters for fitting, checks for consistency"

    Parameters
    ----------
    fit_params: dict
        dictonary containing the fit parameters
    data_obj: hypermri-sequence object
        Object that contains the data parameters
    kwargs

    Returns
    -------
    fit_params: dict
        Updated fit parameter dictonary

    """
    from ..utils.utils_spectroscopy import get_freq_axis
    from ..utils.utils_general import get_sampling_dt, calc_sampling_time_axis

    if fit_params is None and data_obj is None:
        return {}
    if fit_params is None:
        fit_params = {}

    if data_obj is None:
        cut_off = fit_params.get("cut_off", 0)
        if cut_off != 0:
            raise ValueError(
                f"cut_off is {cut_off}, has to be 0 if no data_obj is provided!"
            )
        sampling_dt = fit_params.get("sampling_dt", None)

        # Extracting the values from fit_params
        time_axis = fit_params.get("time_axis")
        freq_range_Hz = fit_params.get("freq_range_Hz")
        freq_range_ppm = fit_params.get("freq_range_ppm")
        nsamplepoints = fit_params.get("nsamplepoints")

        # Initialize variables for checking

        # Check if freq_range_Hz is not None
        is_freq_range_Hz_valid = False
        if freq_range_Hz is not None:
            if (
                isinstance(freq_range_Hz, (list, tuple, range))
                and len(freq_range_Hz) > 0
            ):
                # freq_range_Hz is a non-empty list or tuple
                is_freq_range_Hz_valid = True
            elif isinstance(freq_range_Hz, np.ndarray):
                is_freq_range_Hz_valid = True
            else:
                is_freq_range_Hz_valid = False

        is_freq_range_ppm_valid = False
        if freq_range_ppm is not None:
            if (
                isinstance(freq_range_ppm, (list, tuple, range))
                and len(freq_range_ppm) > 0
            ):
                # freq_range_ppm is a non-empty list or tuple
                is_freq_range_ppm_valid = True
            elif isinstance(freq_range_ppm, np.ndarray):
                # freq_range_ppm is a non-empty list or tuple
                is_freq_range_ppm_valid = True
            else:
                is_freq_range_ppm_valid = False

        is_time_axis_valid = False
        if time_axis is not None:
            if isinstance(time_axis, (list, tuple, range)) and len(time_axis) > 0:
                # time_axis is a non-empty list or tuple
                is_time_axis_valid = True
            elif isinstance(time_axis, np.ndarray):
                is_time_axis_valid = True
            else:
                print(is_time_axis_valid)
                is_time_axis_valid = False

        is_nsamplepoints_valid = False
        if nsamplepoints is not None:
            if isinstance(nsamplepoints, int):
                # time_axis is a non-empty list or tuple
                is_nsamplepoints_valid = True
            else:
                is_nsamplepoints_valid = False

        sampling_dt_freq_range_Hz = None
        sampling_dt_time_axis = None

        # Check if freq_range_Hz and nsamplepoints match
        if is_freq_range_Hz_valid and is_nsamplepoints_valid:
            if len(freq_range_Hz) != nsamplepoints:
                raise ValueError(
                    "The length of 'freq_range_Hz' does not match 'nsamplepoints'."
                )
                return False

        # Check if freq_range_Hz and freq_range_ppm have the same length
        if is_freq_range_Hz_valid and is_freq_range_ppm_valid:
            if len(freq_range_Hz) != len(freq_range_ppm):
                raise ValueError(
                    "The lengths of 'freq_range_Hz' and 'freq_range_ppm' do not match."
                )
                return False

        # Check if freq_range_Hz and freq_range_ppm have the same length
        if is_time_axis_valid and is_nsamplepoints_valid:
            if len(time_axis) != nsamplepoints:
                raise ValueError(
                    "The length of 'time_axis' does not match 'nsamplepoints'."
                )
                return False

        # Define number of points:
        if is_nsamplepoints_valid is False:
            if is_freq_range_Hz_valid:
                nsamplepoints = len(freq_range_Hz)
                is_nsamplepoints_valid = True
            elif is_freq_range_ppm_valid:
                nsamplepoints = len(freq_range_ppm)
                is_nsamplepoints_valid = True
            elif is_time_axis_valid:
                nsamplepoints = len(time_axis)
                is_nsamplepoints_valid = True
            else:
                nsamplepoints = 200
                print(
                    f"nsamplepoints is not defined, defaulting to nsamplepoints={nsamplepoints}!"
                )
                is_nsamplepoints_valid = True
        if is_time_axis_valid:
            sampling_dt_time_axis = np.round(time_axis[1] - time_axis[0], 7)
        if is_freq_range_Hz_valid:
            from ..utils.utils_spectroscopy import get_bw_from_freq_axis

            bw_Hz = get_bw_from_freq_axis(freq_axis=freq_range_Hz)
            sampling_dt_freq_range_Hz = np.round(
                get_sampling_dt(data_obj=data_obj, bw_hz=bw_Hz), 7
            )

        if (
            is_time_axis_valid is False
            and sampling_dt is None
            and is_freq_range_Hz_valid is False
        ):
            print("-----------------------")
            print(f"Y/N: freq_range_Hz: {is_freq_range_Hz_valid}")
            print(f"Y/N: freq_range_ppm: {is_freq_range_ppm_valid}")
            print(f"Y/N: time_axis: {is_time_axis_valid}")
            print(f"Y/N: nsamplepoints: {is_nsamplepoints_valid}")
            print(f"-----------------------------------------------")
            return fit_params
        try:
            sampling_dt = np.round(sampling_dt, 7)
        except:
            pass
        try:
            sampling_dt_time_axis = np.round(sampling_dt_time_axis, 7)
        except:
            pass
        try:
            sampling_dt_freq_range_Hz = np.round(sampling_dt_freq_range_Hz, 7)
        except:
            pass
        # check for consistency between sampling time, time axis and freq_range_Hz
        if is_time_axis_valid and is_freq_range_Hz_valid and sampling_dt is not None:
            if sampling_dt_freq_range_Hz == sampling_dt_time_axis == sampling_dt:
                pass
            else:
                raise ValueError(
                    f"Sampling time; sampling_dt={sampling_dt}, freq_range_Hz={sampling_dt_freq_range_Hz},"
                    f"time_axis={sampling_dt_time_axis}"
                )
        elif (
            is_time_axis_valid is False
            and is_freq_range_Hz_valid
            and sampling_dt is not None
        ):
            if sampling_dt_freq_range_Hz == sampling_dt:
                time_axis = calc_sampling_time_axis(
                    data_obj=data_obj, sampling_dt=sampling_dt, npoints=nsamplepoints
                )
            else:
                raise ValueError(
                    f"Sampling time; sampling_dt={sampling_dt}, freq_range_Hz={sampling_dt_freq_range_Hz}"
                )
        # if only samplig time was passed, calculate freq axis and time axis
        elif (
            is_time_axis_valid is False
            and is_freq_range_Hz_valid is False
            and sampling_dt is not None
        ):
            time_axis = calc_sampling_time_axis(
                data_obj=data_obj, sampling_dt=sampling_dt, npoints=nsamplepoints
            )

            freq_range_Hz = get_freq_axis(
                scan=data_obj,
                unit="Hz",
                centered_at_0=False,
                npoints=nsamplepoints,
                sampling_dt=sampling_dt,
            )

        elif (
            is_time_axis_valid is False
            and is_freq_range_Hz_valid
            and sampling_dt is not None
        ):
            if sampling_dt_freq_range_Hz == sampling_dt:
                time_axis = calc_sampling_time_axis(
                    data_obj=data_obj, sampling_dt=sampling_dt, npoints=nsamplepoints
                )

            else:
                raise ValueError(
                    f"Sampling time; sampling_dt={sampling_dt}, freq_range_Hz={sampling_dt_freq_range_Hz}"
                )

        elif (
            is_time_axis_valid is False
            and is_freq_range_Hz_valid
            and sampling_dt is None
        ):
            bw_Hz = get_bw_from_freq_axis(freq_axis=freq_range_Hz)
            sampling_dt = get_sampling_dt(
                data_obj=data_obj,
                bw_hz=bw_Hz,
            )
            time_axis = calc_sampling_time_axis(
                data_obj=data_obj, sampling_dt=sampling_dt, npoints=nsamplepoints
            )

        elif is_time_axis_valid and is_freq_range_Hz_valid and sampling_dt is None:
            if sampling_dt_freq_range_Hz == sampling_dt_time_axis:
                sampling_dt = sampling_dt_time_axis

            else:
                raise ValueError(
                    f"Sampling time; time_axis={sampling_dt_time_axis}, freq_range_Hz={sampling_dt_freq_range_Hz}"
                )
        elif (
            is_time_axis_valid
            and is_freq_range_Hz_valid is False
            and sampling_dt is None
        ):
            sampling_dt = sampling_dt_time_axis

            freq_range_Hz = get_freq_axis(
                scan=data_obj,
                unit="Hz",
                centered_at_0=False,
                npoints=nsamplepoints,
                sampling_dt=sampling_dt,
            )

        else:
            print("-----------------------")
            print(f"Y/N: freq_range_Hz: {is_freq_range_Hz_valid}")
            print(f"Y/N: freq_range_ppm: {is_freq_range_ppm_valid}")
            print(f"Y/N: time_axis: {is_time_axis_valid}")
            print(f"Y/N: nsamplepoints: {is_nsamplepoints_valid}")
            print(f"-----------------------------------------------")

            print(f"dt calculated from freq_range_Hz: {sampling_dt_freq_range_Hz}")
            print(f"dt calculated from time_axis: {sampling_dt_time_axis}")
            print(f"dt passed: {sampling_dt}")

        # fill all values:
        fit_params["freq_range_Hz"] = freq_range_Hz
        fit_params["time_axis"] = time_axis
        fit_params["sampling_dt"] = sampling_dt
        fit_params["nsamplepoints"] = nsamplepoints

        # calculate ppm axis if it was not passed:
        if is_freq_range_ppm_valid:
            # should have already been tested above
            if len(freq_range_Hz) == len(freq_range_ppm):
                pass
        else:
            if freq_range_ppm is None:
                # calculate ppm range from b0 and gyromagnetic ratio:
                b0 = fit_params.get("b0", 7)  # assume 7 Tesla
                nucleus = fit_params.get("nucleus", "13c")
                freq_range_ppm_center = fit_params.get("freq_range_ppm_center", 0)
                gmr = get_gmr(nucleus=nucleus, unit="MHz_T")
                Hz_to_ppm = gmr * b0
                freq_range_ppm = freq_range_Hz / Hz_to_ppm + freq_range_ppm_center
                fit_params["freq_range_ppm"] = freq_range_ppm
            else:
                raise ValueError("freq_range_ppm is not Valid!")
                return fit_params
        return fit_params
    else:
        from ..utils.utils_spectroscopy import get_bw_from_freq_axis
        from ..utils.utils_general import calc_sampling_time_axis

        cut_off = fit_params.get("cut_off", 0)
        # check if number of points was defined:
        # Extracting the values from fit_params
        nsamplepoints = fit_params.get("nsamplepoints")

        if nsamplepoints is None:
            pass
        elif isinstance(nsamplepoints, int):
            if nsamplepoints == 0:
                raise ValueError(f"nsamplepoints has to be >0")
        else:
            nsamplepoints = None

        # get frequencies axis in Hz
        freq_range_Hz = get_freq_axis(
            scan=data_obj, unit="Hz", cut_off=cut_off, npoints=nsamplepoints
        )
        # get frequencies axis in ppm
        freq_range_ppm = get_freq_axis(
            scan=data_obj, unit="ppm", cut_off=cut_off, npoints=nsamplepoints
        )
        # get bandwidth in Hz
        bw_Hz = get_bw_from_freq_axis(freq_axis=freq_range_Hz)
        # get sampling duration in s
        sampling_dt = np.round(get_sampling_dt(data_obj=data_obj, bw_hz=bw_Hz), 7)
        # get sampling axis in s
        sampling_time_axis = calc_sampling_time_axis(
            data_obj=data_obj, npoints=nsamplepoints
        )

        # get number of sampling points if it was None:
        if nsamplepoints is None:
            nsamplepoints = len(freq_range_Hz)

        # check consistency:
        sampling_dt_time_axis = np.round(
            sampling_time_axis[1] - sampling_time_axis[0], 7
        )
        sampling_dt_freq_range_Hz = np.round(
            get_sampling_dt(data_obj=None, bw_hz=bw_Hz), 7
        )
        if sampling_dt_freq_range_Hz == sampling_dt_time_axis:
            if sampling_dt_freq_range_Hz == sampling_dt:
                pass
            else:
                raise ValueError(
                    f"Inconsistency in sampling duration between sampling duration from "
                    f"freq_range_Hz{sampling_dt_freq_range_Hz}, "
                    f"time_axis {sampling_dt_time_axis} and sampling_dt{sampling_dt}"
                )
        nsamplepoints = len(freq_range_Hz)
        # fill fit_params dictonary with calculated points:
        fit_params["nsamplepoints"] = nsamplepoints
        fit_params["freq_range_Hz"] = freq_range_Hz
        fit_params["freq_range_ppm"] = freq_range_ppm
        fit_params["sampling_dt"] = sampling_dt
        fit_params["time_axis"] = sampling_time_axis
        fit_params["bw_Hz"] = bw_Hz

        return fit_params


def def_fit_params_get_metabs(fit_params=None, data_obj=None):
    """
    Helper function to define the metabolite frequencies in fit_params

    Parameters
    ----------
    fit_params: dict
        dictonary containing the fit parameters
    data_obj: hypermri-sequence object
        Object that contains the data parameters

    Returns
    -------
    fit_params: dict
        Updated fit parameter dictonary

    """
    from .utils_spectroscopy import get_metab_cs_ppm

    # use function to get frequency axis etc.
    fit_params = def_fit_params_calc_sampling(fit_params=fit_params, data_obj=data_obj)
    freq_range_Hz = fit_params.get("freq_range_Hz")
    freq_range_ppm = fit_params.get("freq_range_ppm")

    # Check passed metabolite names and initial metabolite frequencies:
    metabs = fit_params.get("metabs", None)
    if metabs is not None and isinstance(metabs, str):
        metabs = [metabs]
    metabs_freqs_ppm = fit_params.get("metabs_freqs_ppm", None)
    metabs_freqs_Hz = fit_params.get("metabs_freqs_Hz", None)

    # make provided metabolite frequencies a list
    if not isinstance(metabs_freqs_Hz, (list, np.ndarray)):
        metabs_freqs_Hz = [metabs_freqs_Hz]

    # make provided metabolite frequencies a list
    if not isinstance(metabs_freqs_ppm, (list, np.ndarray)):
        metabs_freqs_ppm = [metabs_freqs_ppm]

    if (
        metabs is not None
        and metabs_freqs_Hz[0] is None
        and metabs_freqs_ppm[0] is None
    ):
        from ..utils.utils_spectroscopy import freq_ppm_to_Hz

        metabs_freqs_ppm = fit_params.get(
            "metabs_freqs_ppm", [get_metab_cs_ppm(m) for m in metabs]
        )
        metabs_freqs_Hz = freq_ppm_to_Hz(
            freq_ppm=metabs_freqs_ppm, ppm_axis=freq_range_ppm, hz_axis=freq_range_Hz
        )

    elif (
        metabs is not None
        and metabs_freqs_Hz[0] is None
        and metabs_freqs_ppm[0] is not None
    ):
        from ..utils.utils_spectroscopy import freq_ppm_to_Hz

        metabs_freqs_ppm = fit_params.get(
            "metabs_freqs_ppm", [get_metab_cs_ppm(m) for m in metabs]
        )
        metabs_freqs_Hz = freq_ppm_to_Hz(
            freq_ppm=metabs_freqs_ppm, ppm_axis=freq_range_ppm, hz_axis=freq_range_Hz
        )

    elif (
        metabs is not None
        and metabs_freqs_Hz[0] is not None
        and metabs_freqs_ppm[0] is None
    ):
        from ..utils.utils_spectroscopy import freq_Hz_to_ppm

        metabs_freqs_Hz = fit_params.get("metabs_freqs_Hz")

        metabs_freqs_ppm = freq_Hz_to_ppm(
            freq_Hz=metabs_freqs_Hz, hz_axis=freq_range_Hz, ppm_axis=freq_range_ppm
        )

    elif (
        metabs is None
        and metabs_freqs_ppm[0] is None
        and metabs_freqs_Hz[0] is not None
    ):
        from ..utils.utils_spectroscopy import freq_Hz_to_ppm

        metabs = []
        for k, _ in enumerate(metabs_freqs_Hz):
            metabs.append("metab_" + str(k))
        metabs_freqs_ppm = freq_Hz_to_ppm(
            freq_Hz=metabs_freqs_Hz, ppm_axis=freq_range_ppm, hz_axis=freq_range_Hz
        )
    elif (
        metabs is None
        and metabs_freqs_Hz[0] is None
        and metabs_freqs_ppm[0] is not None
    ):
        metabs = []
        from ..utils.utils_spectroscopy import freq_ppm_to_Hz

        for k, _ in enumerate(metabs_freqs_ppm):
            metabs.append("metab_" + str(k))
        metabs_freqs_Hz = freq_ppm_to_Hz(
            freq_ppm=metabs_freqs_ppm, ppm_axis=freq_range_ppm, hz_axis=freq_range_Hz
        )

    elif metabs is None and metabs_freqs_Hz[0] is None and metabs_freqs_ppm[0] is None:
        # st some default values
        from ..utils.utils_spectroscopy import freq_ppm_to_Hz

        metabs = ["pyruvate", "lactate", "alanine", "pyruvatehydrate", "urea"]
        metabs_freqs_ppm = fit_params.get(
            "metabs_freqs_ppm", [get_metab_cs_ppm(m) for m in metabs]
        )
        metabs_freqs_Hz = freq_ppm_to_Hz(
            freq_ppm=metabs_freqs_ppm, ppm_axis=freq_range_ppm, hz_axis=freq_range_Hz
        )

    # Ensure metabs_freqs_Hz is a list
    if not isinstance(metabs_freqs_Hz, (list, np.ndarray)):
        metabs_freqs_Hz = [metabs_freqs_Hz]

    if not isinstance(metabs_freqs_ppm, (list, np.ndarray)):
        metabs_freqs_ppm = [metabs_freqs_ppm]

    if metabs is None and metabs_freqs_Hz is not None and metabs_freqs_ppm is not None:
        metabs = []
        for k, _ in enumerate(metabs_freqs_Hz):
            metabs.append("metab_" + str(k))

    if metabs_freqs_ppm is None:
        nmetabs_freqs_ppm = 0
    else:
        nmetabs_freqs_ppm = len(metabs_freqs_ppm)
    if metabs_freqs_Hz is None:
        nmetabs_freqs_Hz = 0
    else:
        nmetabs_freqs_Hz = len(metabs_freqs_Hz)
    if len(metabs) > nmetabs_freqs_Hz and len(metabs) > nmetabs_freqs_ppm:
        raise ValueError(
            f"Please provide as many metab_freqs_Hz (N={nmetabs_freqs_Hz}) or \..."
            f"metab_freqs_ppm (N={nmetabs_freqs_ppm}) as metabs (N={len(metabs)})!"
        )

    # Ensure metabs is a list
    if isinstance(metabs, str):
        metabs = [metabs]

    fit_params["metabs_freqs_ppm"] = metabs_freqs_ppm
    fit_params["metabs_freqs_Hz"] = metabs_freqs_Hz
    fit_params["metabs"] = metabs

    return fit_params


def def_fit_params(fit_params=None, data_obj=None):
    """
    Define and update fitting parameters for spectroscopy analysis.

    This function initializes or updates a comprehensive set of fitting parameters for spectroscopy
    data fitting. It sets default values for parameters not provided and ensures consistency
    across all parameters.

    Parameters
    ----------
    fit_params : dict, optional
        A dictionary of fitting parameters. If a parameter is not included in this dictionary,
        its default value will be set. Default is None, which creates a new dictionary.
    data_obj : object, optional
        A scan object containing information about the spectroscopy scan, used to derive
        default parameter values when not explicitly provided.

    Returns
    -------
    dict
        A dictionary containing the complete set of fitting parameters, including both
        user-provided and default values.

    Notes
    -----
    This function is the main entry point for setting up fitting parameters. It calls several
    helper functions to handle specific groups of parameters (e.g., sampling, metabolites).
    The function sets various parameters such as number of iterations, T2* range, frequency range,
    metabolites to consider, and many others.

    See Also
    --------
    def_fit_params_calc_sampling : Helper function for time/frequency sampling parameters.
    def_fit_params_get_metabs : Helper function for metabolite-related parameters.

    Examples
    ________
    >>> fit_params = def_fit_params(data_obj=data_obj)

    2. Set specific parameters and fill the rest with defaults:

    >>> fit_params = {}
    >>> fit_params["niter"] = 3
    >>> fit_params = def_fit_params(fit_params=fit_params, data_obj=data_obj)

    3. Get an overview of available parameters and set specific ones:

    >>> fit_params = def_fit_params()
    >>> fit_params["niter"] = 3

    Notes
    -----
    The function sets various parameters such as number of points per iteration (`npoints`),
    number of iterations (`niter`), range of T2* times (`range_t2s_s`), frequency range in Hz
    (`range_freqs_Hz`), sample points (`nsamplepoints`), metabolites to consider (`metabs`),
    and others. These parameters are crucial in defining how the fitting process behaves.
    """

    if fit_params is None:
        fit_params = {}

    # define time_axis, frequency axis:
    fit_params = def_fit_params_calc_sampling(fit_params=fit_params, data_obj=data_obj)
    # define metab freqs in Hz and ppm:
    fit_params = def_fit_params_get_metabs(fit_params=fit_params, data_obj=data_obj)

    # number of points the range_t2_s and range_freqs_Hz are split into (resoltion per fitting repetition is (e.g. freq):
    # df = range_freqs_Hz / npoints / (zoomfactor ** (fitting repetition))
    npoints = fit_params.get("npoints", 31)
    # number of iterations per fitting repetition
    niter = fit_params.get("niter", 1)
    # range of T2s that should be fit [s]
    default_range_t2s_s = fit_params.get("range_t2s_s", 0.03)
    # Default range of freqs that should be fit [Hz]
    default_range_freqs_Hz = fit_params.get("range_freqs_Hz", 70.0)
    # Magnetic field strength:
    b0 = fit_params.get("b0", 7)
    # which nucleus
    nucleus = fit_params.get("nucleus", "13c")
    # if an intial fit without zooming should be performed (not necessary):
    init_fit = fit_params.get("init_fit", False)
    # in which signal domain the input data is provided (very important!)
    signal_domain = fit_params.get("signal_domain", "spectral")
    # how much to zoom in on frequency and T2 per fitting repetition
    zoomfactor = fit_params.get("zoomfactor", 1.5)
    # UP to which point the FID should be cut off:
    cut_off = fit_params.get("cut_off", 0)
    # number of fitting freqs and T2s, using the zoomfactor to reduce the search range each iteration
    rep_fitting = fit_params.get("rep_fitting", 11)
    # Whether to save the fit results (Defaults to False):
    fit_params["save_fit_results"] = fit_params.get("save_fit_results", False)
    sampling_dt = fit_params.get("sampling_dt", None)
    time_axis = fit_params.get("time_axis", None)
    # init savepath if not passed:
    fit_params["savepath"] = fit_params.get("savepath", None)
    # define provided dimensions of data array: (assume time(FID), z-axis, x-axis. y-axis, repetitions, channels:
    fit_params["provided_dims"] = fit_params.get(
        "provided_dims", ["t", "z", "x", "y", "r", "c"]
    )
    # use function to get frequency axis etc.
    freq_range_Hz = fit_params.get("freq_range_Hz")
    freq_range_ppm = fit_params.get("freq_range_ppm")

    # Check passed metabolite names and initial metabolite frequencies:
    metabs = fit_params.get("metabs", None)
    if metabs is not None and isinstance(metabs, str):
        metabs = [metabs]
    metabs_freqs_ppm = fit_params.get("metabs_freqs_ppm", None)
    metabs_freqs_Hz = fit_params.get("metabs_freqs_Hz", None)

    # New: Get individual frequency ranges for each metabolite
    # Default range of freqs that should be fit [Hz]
    default_range_freqs_Hz = fit_params.get("range_freqs_Hz", 70.0)

    # Get metabolites
    metabs = fit_params.get(
        "metabs", ["pyruvate", "lactate", "alanine", "pyruvatehydrate", "urea"]
    )

    # Handle metabs_freq_ranges
    metabs_freq_ranges = fit_params.get("metabs_freq_ranges", None)
    metabs_t2_ranges = fit_params.get("metabs_t2_ranges", None)

    if metabs_freq_ranges is None:
        # If not provided, create an array of default values
        metabs_freq_ranges = [default_range_freqs_Hz] * len(metabs)
    elif isinstance(metabs_freq_ranges, (int, float)):
        # If a single value is provided, create an array with that value
        metabs_freq_ranges = [metabs_freq_ranges] * len(metabs)
    elif isinstance(metabs_freq_ranges, list):
        # If it's already a list, ensure it has the correct length
        if len(metabs_freq_ranges) < len(metabs):
            metabs_freq_ranges.extend(
                [default_range_freqs_Hz] * (len(metabs) - len(metabs_freq_ranges))
            )
        elif len(metabs_freq_ranges) > len(metabs):
            metabs_freq_ranges = metabs_freq_ranges[: len(metabs)]
    else:
        raise ValueError("metabs_freq_ranges must be None, a number, or a list")
    metabs_freq_ranges = [se(m) for m in metabs_freq_ranges]
    fit_params["metabs_freq_ranges"] = metabs_freq_ranges

    if metabs_t2_ranges is None:
        # If not provided, create an array of default values
        metabs_t2_ranges = [default_range_t2s_s] * len(metabs)
    elif isinstance(metabs_t2_ranges, (int, float)):
        # If a single value is provided, create an array with that value
        metabs_t2_ranges = [metabs_t2_ranges] * len(metabs)
    elif isinstance(metabs_t2_ranges, list):
        # If it's already a list, ensure it has the correct length
        if len(metabs_t2_ranges) < len(metabs):
            metabs_t2_ranges.extend(
                [default_range_t2s_s] * (len(metabs) - len(metabs_t2_ranges))
            )
        elif len(metabs_t2_ranges) > len(metabs):
            metabs_t2_ranges = metabs_t2_ranges[: len(metabs)]
    else:
        raise ValueError("metabs_t2_ranges must be None, a number, or a list")
    metabs_t2_ranges = [se(m) for m in metabs_t2_ranges]
    fit_params["metabs_t2_ranges"] = metabs_t2_ranges

    from ..utils.utils_general import get_indices_for_strings

    # convert to indices:
    fit_params["provided_dims_ind"] = fit_params.get(
        "provided_dims_ind", get_indices_for_strings(fit_params["provided_dims"])
    )

    # if data_obj was passed, try to get data_obj's savepath:
    if data_obj is not None and fit_params["savepath"] is None:
        try:
            fit_params["savepath"] = data_obj.savepath
        except:
            pass

    # cant save fit results if no path was passed:
    if fit_params["savepath"] is None:
        fit_params["save_fit_results"] = False

    # frequency range (ppm)
    if data_obj is None:
        fit_range_repetitions = fit_params.get("fit_range_repetitions", range(1))
    else:
        fit_range_repetitions = fit_params.get(
            "fit_range_repetitions", range(data_obj.method["PVM_NRepetitions"])
        )

    # get initial metabolute T2* value:
    metabs_t2s = fit_params.get("metabs_t2s", 0.03)

    # make provided metabolite T2*s a list
    if not isinstance(metabs_t2s, (list, np.ndarray)):
        metabs_t2s = [metabs_t2s for _ in metabs]
    # if not enough metabolute t2s were provided, extent the list of provided T2s
    elif len(metabs_t2s) < len(metabs):
        # Extend the metabs_t2s list with 0.03 for missing values
        metabs_t2s.extend([0.03] * (len(metabs) - len(metabs_t2s)))
    elif len(metabs_t2s) > len(metabs):
        # Reduce the metabs_t2s list
        metabs_t2s = metabs_t2s[:: len(metabs)]

    metabs_t2s = [se(m) for m in metabs_t2s]
    # Replace np.nan with 0.03 in the list
    fit_params["metabs_t2s"] = [
        0.03 if np.isnan(value) else value for value in metabs_t2s
    ]

    # define if all CPU cores should be used (default set to max/2 due to hyperthreading):
    fit_params["use_all_cores"] = fit_params.get("use_all_cores", False)
    # if fitting progress bar should be shown
    fit_params["show_tqdm"] = fit_params.get("show_tqdm", True)

    # Choose fitting method (True --> uses np.linalg.pinv):
    fit_params["pseudo_inv"] = fit_params.get("pseudo_inv", True)

    # Try to correct the inconsistencies in MCI calculations (they are frequency dependend)
    fit_params["correct_freqs_MCI"] = fit_params.get("correct_freqs_MCI", False)

    # Try to correct the inconsistencies in MCI calculations (they are frequency dependend)
    fit_params["return_MCI"] = fit_params.get("return_MCI", False)

    # fill in parameters:
    metabs_freqs_Hz = [se(m) for m in metabs_freqs_Hz]
    fit_params["metabs_freqs_Hz"] = metabs_freqs_Hz
    # Add a new parameter for initial frequencies --> should help to keep the fitted peaks in the inital range:
    fit_params["initial_metabs_freqs_Hz"] = fit_params.get("metabs_freqs_Hz").copy()
    metabs_freqs_ppm = [se(m) for m in metabs_freqs_ppm]
    fit_params["metabs_freqs_ppm"] = metabs_freqs_ppm
    fit_params["freq_range_Hz"] = freq_range_Hz
    fit_params["freq_range_ppm"] = freq_range_ppm
    fit_params["zoomfactor"] = zoomfactor
    fit_params["signal_domain"] = signal_domain
    fit_params["npoints"] = npoints
    fit_params["niter"] = niter
    fit_params["metabs_t2s"] = metabs_t2s
    # fit_params["range_t2s_s"] = range_t2s_s
    # fit_params["range_freqs_Hz"] = range_freqs_Hz
    fit_params["init_fit"] = init_fit
    # make parameter iterable:
    if isinstance(fit_range_repetitions, int):
        fit_range_repetitions = range(fit_range_repetitions)
    fit_params["fit_range_repetitions"] = fit_range_repetitions
    fit_params["cut_off"] = cut_off
    fit_params["nucleus"] = nucleus
    fit_params["b0"] = b0
    # calculate the fit resolultion:
    fit_params["res_freqs_Hz"] = [
        (
            metabs_freq_ranges[m]
            / (fit_params["zoomfactor"] ** fit_params["niter"])
            / fit_params["npoints"]
        )
        for m, _ in enumerate(metabs)
    ]
    fit_params["res_t2s_s"] = [
        (
            metabs_t2_ranges[m]
            / (fit_params["zoomfactor"] ** fit_params["niter"])
            / fit_params["npoints"]
        )
        for m, _ in enumerate(metabs)
    ]
    fit_params["rep_fitting"] = rep_fitting
    fit_params["metabs"] = metabs

    # minimum T2:
    min_t2_s = fit_params.get("min_t2_s", 0.003)
    # If min_t2_s is a float, create a list with the float repeated
    if not isinstance(min_t2_s, (list, np.ndarray)):
        min_t2_s = [min_t2_s for _ in metabs]
    elif len(min_t2_s) < len(metabs):
        # Extend the min_t2_s list with 0.008 for missing values
        min_t2_s.extend([0.008] * (len(metabs) - len(min_t2_s)))
    # Replace np.nan with 0.0008 in the list
    fit_params["min_t2_s"] = [0.008 if np.isnan(value) else value for value in min_t2_s]

    # max T2:
    max_t2_s = fit_params.get("max_t2_s", 0.1)
    # If max_t2_s is a float, create a list with the float repeated
    if not isinstance(max_t2_s, (list, np.ndarray)):
        max_t2_s = [max_t2_s for _ in metabs]
    elif len(max_t2_s) < len(metabs):
        # Extend the max_t2_s list with 0.3 for missing values
        max_t2_s.extend([0.1] * (len(metabs) - len(max_t2_s)))
    # Replace np.nan with 0.1 in the list
    fit_params["max_t2_s"] = [0.1 if np.isnan(value) else value for value in max_t2_s]

    for m, min_t2 in enumerate(fit_params["min_t2_s"]):
        if min_t2 > fit_params["max_t2_s"][m]:
            fit_params["max_t2_s"][m] = fit_params["min_t2_s"][m]
    fit_params = def_fit_params_check_consistency(fit_params=fit_params)
    # Sorting keys alphabetically
    sorted_keys = sorted(fit_params)

    # Creating a new dictionary with sorted keys
    fit_params = {key: fit_params[key] for key in sorted_keys}

    # Generate description for each metabolite
    metabs_description = "# Metabolite Fit Parameters Description\n\n"
    metabs_description += "This description provides an overview of the fitting parameters for each metabolite and general spectral information.\n\n"
    try:
        for i, metab in enumerate(fit_params["metabs"]):
            metabs_description += f"## {metab}\n\n"
            metabs_description += f"- Initial frequency: {fit_params['metabs_freqs_Hz'][i]:.2f} Hz ({fit_params['metabs_freqs_ppm'][i]:.2f} ppm)\n"
            metabs_description += (
                f"- Initial T2*: {fit_params['metabs_t2s'][i]:.3f} s\n"
            )
            metabs_description += f"- Frequency search range: {fit_params['metabs_freq_ranges'][i]:.2f} Hz\n"
            metabs_description += (
                f"- T2* search range: {fit_params['metabs_t2_ranges'][i]:.3f} s\n"
            )
            metabs_description += f"- Minimum T2*: {fit_params['min_t2_s'][i]:.3f} s\n"
            metabs_description += (
                f"- Maximum T2*: {fit_params['max_t2_s'][i]:.3f} s\n\n"
            )
    except:
        pass
    metabs_description += "## General fitting parameters:\n"
    metabs_description += f"- Number of fitting points: {fit_params['npoints']}\n"
    metabs_description += f"- Number of iterations: {fit_params['niter']}\n"
    metabs_description += f"- Zoom factor: {fit_params['zoomfactor']}\n"
    metabs_description += f"- Signal domain: {fit_params['signal_domain']}\n"
    metabs_description += (
        f"- Number of sample points: {fit_params['nsamplepoints']}\n\n"
    )

    metabs_description += "## Spectral axis information:\n"
    metabs_description += f"- Frequency range: {fit_params['freq_range_Hz'][0]:.2f} to {fit_params['freq_range_Hz'][-1]:.2f} Hz\n"
    metabs_description += f"- PPM range: {fit_params['freq_range_ppm'][0]:.2f} to {fit_params['freq_range_ppm'][-1]:.2f} ppm\n"
    metabs_description += f"- Spectral resolution: {abs(fit_params['freq_range_Hz'][1] - fit_params['freq_range_Hz'][0]):.2f} Hz "
    metabs_description += f"({abs(fit_params['freq_range_ppm'][1] - fit_params['freq_range_ppm'][0]):.4f} ppm)\n"
    metabs_description += (
        f"- Time point resolution: {fit_params['sampling_dt']:.6f} s\n"
    )

    # Add the description to fit_params
    fit_params["metabs_description"] = metabs_description

    # Add the description to fit_params

    return fit_params


def def_fit_params_check_consistency(fit_params=None, data_obj=None):
    """
    Ensures that the entries in the `fit_params` dictionary are not lists, to maintain consistency in parameter formats.
    This function iterates through the `fit_params` dictionary and checks each value. If a value is found to be a list,
    an appropriate action is taken to ensure consistency, such as converting lists to single values or handling them
    as needed based on the context of the parameter. This function is crucial for preparing the `fit_params` for
    further processing where list values might cause errors or unexpected behavior.

    Parameters
    ----------
    fit_params : dict, optional
        A dictionary containing the fitting parameters. Each key-value pair represents a parameter and its value.
        If a parameter's value is a list, this function aims to correct it to maintain consistency.
    data_obj : hypermri-sequence object, optional
        An object representing the spectroscopy scan. This object can provide additional context or default values
        for the fitting parameters, aiding in the consistency check and correction process.

    Returns
    -------
    dict
        A dictionary containing the fitting parameters after ensuring that no parameter values are lists,
        thus maintaining consistency across all parameters.

    Notes
    -----
    - The specific actions taken when a list is encountered depend on the nature of the parameter and the context
      provided by `data_obj` if available.
    - This function is part of a larger preprocessing step to prepare fitting parameters for spectroscopy data analysis.
    """

    def flatten_array(arr):
        # Check if the array is nested by looking at the first element
        if arr and isinstance(arr[0], list):
            # Flatten the array using a list comprehension
            return [item for sublist in arr for item in sublist]
        else:
            # Return the array as is if it's not nested
            return arr

    fit_params["metabs_freqs_Hz"] = flatten_array(fit_params["metabs_freqs_Hz"])
    fit_params["metabs_freqs_ppm"] = flatten_array(fit_params["metabs_freqs_ppm"])
    fit_params["metabs_t2s"] = flatten_array(fit_params["metabs_t2s"])
    fit_params["max_t2_s"] = flatten_array(fit_params["max_t2_s"])
    fit_params["min_t2_s"] = flatten_array(fit_params["min_t2_s"])
    return fit_params


def fit_freq_pseudo_inv(
    input_data=None,
    metabs=None,
    fit_params={},
    data_obj=None,
    plot=False,
    dbplot=False,
    dbmode=False,
):
    """
    Fits frequency data for complex_amps set of metabolites using iterative optimization.

    This function fits frequency data for given metabolites by iteratively adjusting the frequencies
    to minimize the residual sum of squares (RSS) error. It supports fitting in both the spectral
    and temporal domains and offers optional plotting for analysis and debugging.

    Parameters
    ----------
    input_data : array_like
        The input data to fit. This can be in the spectral or temporal domain.

    metabs : list, optional
        A list of metabolites for which the frequencies are to be fitted.

    fit_params : dict, optional
        A dictionary of parameters for fitting. If not provided, default parameters are used.

    data_obj: hypermri-sequence object
        Scan object containing additional information used in the fitting process.

    plot : bool, optional
        If True, plot the final fitted data.

    dbplot : bool, optional
        If True, enable debugging plots to visualize the fitting process.

    dbmode : bool, optional
        If True, returns additional debugging information.

    Returns
    -------
    tuple
        A tuple containing the final fitted signal in the frequency domain, the final frequencies for
        each metabolite, and the coefficients from the fitting process. If `dbmode` is True, it also
        includes debugging information.

    Notes
    -----
    The function works by generating complex_amps simulated signal based on initial guesses for frequencies
    and then iteratively adjusts these frequencies. The adjustment is based on the minimization
    of the RSS error between the simulated signal and the input data.
    """
    import time

    fit_params = def_fit_params(fit_params=fit_params, data_obj=data_obj)

    # Get individual frequency ranges for each metabolite
    metabs_freq_ranges = fit_params.get("metabs_freq_ranges")

    # number of fitting per iteration:
    npoints = fit_params.get("npoints")
    niter = fit_params.get("niter")
    init_fit = fit_params.get("init_fit")
    if init_fit:
        pass
        # niter += 1
    zoomfactor = fit_params.get("zoomfactor")
    signal_domain = fit_params.get("signal_domain")

    # frequency range (ppm)
    # freq_range_ppm = fit_params.get("freq_range_ppm")
    freq_range_Hz = fit_params.get("freq_range_Hz")
    # metabs_freqs_ppm = fit_params.get("metabs_freqs_ppm")
    metabs_freqs_Hz = fit_params.get("metabs_freqs_Hz")
    # Use the initial frequencies and ranges
    initial_freqs_Hz = fit_params.get("initial_metabs_freqs_Hz")
    metabs_freq_ranges = fit_params.get("metabs_freq_ranges")
    current_freqs_Hz = fit_params.get("metabs_freqs_Hz")
    nsamplepoints = fit_params.get("nsamplepoints")
    metabs = fit_params.get("metabs", metabs)
    use_pseudo_inv = fit_params.get("pseudo_inv")

    # initial T2 guesses:
    metabs_t2s = fit_params.get("metabs_t2s")
    time_axis = fit_params.get("time_axis", None)

    if dbmode:
        all_r2 = np.zeros((len(metabs), niter, npoints, 2))

    if time_axis is None:
        Warning("fit_parameter['time_axis'] has to be passed if no data_obj is passed!")

    # enforce the signal to be in temporal domain to be able to reduce number of points:
    if nsamplepoints != input_data.shape[0]:
        if signal_domain == "spectral":
            temp_signal = np.fft.ifft(np.fft.ifftshift(input_data, axes=(0,)), axis=0)
            temp_signal = temp_signal[:nsamplepoints]
            input_data = np.fft.fftshift(np.fft.fft(temp_signal, axis=0), axes=(0,))
        else:
            pass

    # init empty array:
    r2 = np.zeros((len(metabs), npoints))
    freq_min = np.zeros((len(metabs), niter))
    r2_mins = np.zeros((len(metabs), niter))

    if dbplot:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(freq_range_Hz, np.abs(input_data))
        st = time.time()

    # define initial metabolite frequencies:
    # freqs_Hz = metabs_freqs_Hz.copy()

    for i in range(niter):
        for k, m in enumerate(metabs):
            freqs = current_freqs_Hz.copy()
            # Use the current frequency to set the range, but ensure it's within the initial bounds
            lower_bound = max(
                current_freqs_Hz[k] - metabs_freq_ranges[k] / 2,
                initial_freqs_Hz[k] - metabs_freq_ranges[k] / 2,
            )
            upper_bound = min(
                current_freqs_Hz[k] + metabs_freq_ranges[k] / 2,
                initial_freqs_Hz[k] + metabs_freq_ranges[k] / 2,
            )
            search_range_Hz = np.linspace(lower_bound, upper_bound, npoints)
            for l, f in enumerate(search_range_Hz):
                # set the frequency of the metabolite
                freqs[k] = f

                # fit complex amplitude, using the set frequencies
                # fitted signal, complex amplitudes, 95% confidence of amplitudes:
                fitted_signal, complex_amps = fit_func_pseudo_inv(
                    input_data=input_data,
                    freqs=freqs,
                    metabs_t2s=metabs_t2s,
                    time_axis=time_axis,
                    signal_domain=signal_domain,
                    pseudo_inv=use_pseudo_inv,
                )

                if signal_domain == "spectral":
                    # save difference between fitted signal and measured signal:
                    # r2[k, l] = np.sum(np.abs(np.sum(np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0), axis=1) - input_data) ** 2 / np.abs(input_data**2))
                    r2[k, l] = np.sum(
                        np.abs(
                            np.sum(
                                np.fft.fftshift(
                                    np.fft.fft(fitted_signal, axis=0), axes=0
                                ),
                                axis=1,
                            )
                            - input_data
                        )
                        ** 2
                    )
                else:
                    # save difference between fitted signal and measured signal:
                    # r2[k, l] = np.sum(np.abs(np.sum(np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0), axis=1) - input_data) ** 2 / np.abs(input_data**2))
                    r2[k, l] = np.sum(
                        np.abs(
                            np.sum(
                                np.fft.fftshift(
                                    np.fft.fft(fitted_signal, axis=0), axes=0
                                ),
                                axis=1,
                            )
                            - np.fft.fftshift(np.fft.fft(input_data, axis=0), axes=0)
                        )
                        ** 2
                    )

                # debug mode, save the r2:
                if dbmode:
                    all_r2[k, i, l, 0] = r2[k, l]
                    all_r2[k, i, l, 1] = freqs[k]

            # find smallest difference between fit and measured signal
            r2_min = np.argmin(r2, axis=1)
            # find corresponding frequency
            freq_min[k, i] = search_range_Hz[r2_min[k]]

            # Update the current frequency, ensuring it stays within the initial range
            current_freqs_Hz[k] = np.clip(
                freq_min[k, i],
                initial_freqs_Hz[k] - metabs_freq_ranges[k] / 2,
                initial_freqs_Hz[k] + metabs_freq_ranges[k] / 2,
            )

            r2_mins[k, i] = np.min(r2[k, :])

        if dbplot:
            # fitted signal, complex amplitudes, 95% confidence of amplitudes:
            tempsignal, tempa = fit_func_pseudo_inv(
                input_data=input_data,
                freqs=current_freqs_Hz,
                metabs_t2s=metabs_t2s,
                time_axis=time_axis,
                signal_domain=signal_domain,
                pseudo_inv=use_pseudo_inv,
            )
            ax1.plot(
                freq_range_Hz,
                np.abs(
                    (
                        np.sum(
                            np.fft.fftshift(np.fft.fft(tempsignal, axis=0), axes=0),
                            axis=1,
                        )
                    )
                ),
                label=i,
            )
            ax1.legend()

        # if initial fit should be performed, skip zooming for first round
        if init_fit and i == 0:
            pass
        else:
            # Zoom in for each metabolite individually
            for k, m in enumerate(metabs):
                metabs_freq_ranges[k] *= 1 / zoomfactor

    if dbplot:
        # print(f"Fitting took: {np.round(time.time() - st, 2)}s")
        ax2.plot(freq_min.T, label=metabs)
        ax2.legend()

        ax3.plot(r2_mins.T, label=metabs)
        ax3.legend()
        # print(f"r2_min = {r2_mins[:, -1]}")
        display(plt.gcf())  # Display the current figure
        time.sleep(0.5)
        plt.close()  # Close the figure to prevent re-displaying it later
        clear_output(wait=True)  # Clear the output and wait for the next plot

    # fitted signal, complex amplitudes, 95% confidence of amplitudes:
    fitted_signal, complex_amps = fit_func_pseudo_inv(
        input_data=input_data,
        freqs=current_freqs_Hz,
        metabs_t2s=metabs_t2s,
        time_axis=time_axis,
        signal_domain=signal_domain,
        pseudo_inv=use_pseudo_inv,
    )

    if plot is True:
        plt.figure()
        for k, m in enumerate(metabs):
            plt.subplot(2, 3, k + 1)
            plt.plot(freq_range_Hz, np.abs(input_data))
            plt.plot(
                freq_range_Hz,
                np.abs(
                    np.fft.fftshift(np.fft.fft(fitted_signal[:, k], axis=0), axes=0)
                ),
            )
        plt.title(m)
        plt.subplot(2, 3, 6)
        plt.plot(freq_range_Hz, np.abs(input_data))
        plt.plot(
            freq_range_Hz,
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(np.sum(fitted_signal, axis=1), axis=0), axes=0
                )
            ),
            label="sum",
        )
        plt.title("sum")

    if dbmode:
        return (
            np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0),
            current_freqs_Hz,
            complex_amps,
            all_r2,
        )
    else:
        return (
            np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0),
            current_freqs_Hz,
            complex_amps,
        )


def fit_t2_pseudo_inv(
    input_data=None,
    metabs=None,
    fit_params={},
    data_obj=None,
    plot=False,
    dbplot=False,
    dbmode=False,
):
    """
    Fits a multi-exponential decay model to input data using a pseudo-inverse approach.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input signal data (time or frequency domain).
    metabs : list
        List of metabolite names.
    fit_params : dict
        Dictionary of fitting parameters.  Includes keys like:
           * 'range_t2s_s' (float): Range of T2 values to search over.
           * 'npoints' (int): Number of points in the T2 search grid.
           * 'niter' (int): Number of fitting iterations.
           * ... (other relevant fitting parameters)
    data_obj: hypermri-sequence object
        Data object (if parameters are extracted from it).
    plot : bool, optional
        If True, generate plots of the fitted signal. Default is False.
    dbplot : bool, optional
        If True, generate intermediate fit debugging plots. Default is False.
    dbmode : bool, optional
        If True, return additional debugging information. Default is False.

    Returns
    -------
    fitted_signal : numpy.ndarray
        The fitted signal in the frequency domain.
    t2s_s : numpy.ndarray
        Final estimated T2 values for each metabolite.
    complex_amps : numpy.ndarray
        Complex amplitudes of the fitted components.
    all_r2 : numpy.ndarray, optional
        Debugging array (only returned if dbmode is True).
    """
    import time

    fit_params = def_fit_params(fit_params=fit_params, data_obj=data_obj)

    # range of frequencies to fit over:
    metabs_t2_ranges = fit_params.get("metabs_t2_ranges")

    # number of fitting per iteration:
    npoints = fit_params.get("npoints")
    niter = fit_params.get("niter")
    init_fit = fit_params.get("init_fit")
    if init_fit:
        pass
        # niter += 1
    zoomfactor = fit_params.get("zoomfactor")
    signal_domain = fit_params.get("signal_domain")

    # frequency range (ppm)
    freq_range_Hz = fit_params.get("freq_range_Hz")
    metabs_freqs_Hz = fit_params.get("metabs_freqs_Hz")
    nsamplepoints = fit_params.get("nsamplepoints")
    min_t2_s = fit_params.get("min_t2_s")
    max_t2_s = fit_params.get("max_t2_s")
    use_pseudo_inv = fit_params.get("pseudo_inv")

    # initial T2 guesses:
    metabs_t2s_s = fit_params.get("metabs_t2s")
    time_axis = fit_params.get("time_axis", None)

    if time_axis is None:
        Warning("fit_parameter['time_axis'] has to be passed if no data_obj is passed!")

    # enfore the signal to be in temporal domain to be able to reduce number of points:
    if nsamplepoints != input_data.shape[0]:
        if signal_domain == "spectral":
            temp_signal = np.fft.ifft(np.fft.ifftshift(input_data, axes=(0,)), axis=0)
            temp_signal = temp_signal[:nsamplepoints]
            input_data = np.fft.fftshift(np.fft.fft(temp_signal, axis=0), axes=(0,))
        else:
            pass

    # init empty array:
    r2 = np.zeros((len(metabs), npoints))
    r2_mins = np.zeros((len(metabs), niter))
    t2_min = np.zeros((len(metabs), niter))

    # generate copy (this will be overwritten)

    if dbplot:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(freq_range_Hz, np.abs(input_data))
        st = time.time()
        print(f"metabs_t2s_s -- {metabs_t2s_s} ")

    if dbmode:
        all_r2 = np.zeros((len(metabs), niter, npoints, 2))

    t2s_s = metabs_t2s_s.copy()

    # loop over number of iterations:
    for i in range(niter):
        # loop over metabolites:
        for k, m in enumerate(metabs):
            t2s = t2s_s.copy()
            # set up search range for T2, depending on the metabolite:
            search_range_s = np.linspace(
                -metabs_t2_ranges[k] / 2, metabs_t2_ranges[k] / 2, npoints
            )

            for l, t in enumerate(search_range_s):
                # set the frequency of the metabolite

                t2s[k] = t2s_s[k] + t
                if t2s[k] > 0:
                    # fitted signal, complex amplitudes, 95% confidence of amplitudes:
                    fitted_signal, complex_amps = fit_func_pseudo_inv(
                        input_data=input_data,
                        freqs=metabs_freqs_Hz,
                        metabs_t2s=t2s,
                        time_axis=time_axis,
                        signal_domain=signal_domain,
                        pseudo_inv=use_pseudo_inv,
                    )

                    if signal_domain == "spectral":
                        # save difference between fitted signal and measured signal:
                        # r2[k, l] = np.sum(np.abs(np.sum(np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0), axis=1) - input_data) ** 2 / np.abs(input_data**2))
                        r2[k, l] = np.sum(
                            np.abs(
                                np.sum(
                                    np.fft.fftshift(
                                        np.fft.fft(fitted_signal, axis=0), axes=0
                                    ),
                                    axis=1,
                                )
                                - input_data
                            )
                            ** 2
                        )
                    else:
                        # save difference between fitted signal and measured signal:
                        # r2[k, l] = np.sum(np.abs(np.sum(np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0), axis=1) - input_data) ** 2 / np.abs(input_data**2))
                        r2[k, l] = np.sum(
                            np.abs(
                                np.sum(
                                    np.fft.fftshift(
                                        np.fft.fft(fitted_signal, axis=0), axes=0
                                    ),
                                    axis=1,
                                )
                                - np.fft.fftshift(np.fft.fft(input_data))
                            )
                            ** 2
                        )
                else:
                    r2[k, l] = np.inf
                # print(f"k={k}, l={l}, r2={r2}")
                assert r2[k, l] > 0.0
                # debug mode, save the r2:
                if dbmode:
                    all_r2[k, i, l, 0] = r2[k, l]
                    all_r2[k, i, l, 1] = t2s[k]

            # find smallest number along axis of different T2 for each metabolite (0th axis)
            # print(f"r2={r2}")
            r2_min = np.argmin(r2, axis=1)
            # print(r2_min)

            # print(f"i={i}-k={k}-l={l}- r2_min={r2_min}")
            t2_min[k, i] = t2s_s[k] + search_range_s[r2_min[k]]
            # print(t2_min)

            # Update t2s_s with the new optimal value
            t2s_s[k] = t2_min[k, i]

            # Ensure T2 stays within bounds
            t2s_s[k] = max(min_t2_s[k], min(t2s_s[k], max_t2_s[k]))

            r2_mins[k, i] = np.min(r2[k, :])

        # if inital fit should be performed, skip zooming for first round
        if init_fit and i == 0:
            pass
        else:
            # Zoom in for each metabolite individually
            for k, m in enumerate(metabs):
                metabs_t2_ranges[k] *= 1 / zoomfactor

        if dbplot:
            # fitted signal, complex amplitudes, 95% confidence of amplitudes:
            tempsignal, tempa = fit_func_pseudo_inv(
                input_data=input_data,
                freqs=metabs_freqs_Hz,
                metabs_t2s=t2s_s,
                time_axis=time_axis,
                signal_domain=signal_domain,
                pseudo_inv=use_pseudo_inv,
            )

            ax1.plot(
                freq_range_Hz,
                np.abs(
                    (
                        np.sum(
                            np.fft.fftshift(np.fft.fft(tempsignal, axis=0), axes=0),
                            axis=1,
                        )
                    )
                ),
                label=i,
            )
            ax1.legend()

            ax2.plot(t2_min.T, label=metabs)
            ax2.legend()
            ax3.plot(r2_mins.T, label=metabs)
            ax3.legend()
            # print(f"Fitting took: {np.round(time.time() - st, 2)}s")
            # print(f"r2_min = {r2_mins[:, -1]}")
            display(plt.gcf())  # Display the current figure
            time.sleep(0.5)
            plt.close()  # Close the figure to prevent re-displaying it later
            clear_output(wait=True)  # Clear the output and wait for the next plot

    # fitted signal, complex amplitudes, 95% confidence of amplitudes:
    fitted_signal, complex_amps = fit_func_pseudo_inv(
        input_data=input_data,
        freqs=metabs_freqs_Hz,
        metabs_t2s=t2s_s,
        time_axis=time_axis,
        signal_domain=signal_domain,
        pseudo_inv=use_pseudo_inv,
    )
    # print(f"fit_t2s: a= {complex_amps}")

    if plot is True:
        plt.figure()
        for k, m in enumerate(metabs):
            plt.subplot(2, 3, k + 1)
            plt.plot(freq_range_Hz, np.abs(input_data))
            plt.plot(
                freq_range_Hz,
                np.abs(
                    np.fft.fftshift(np.fft.fft(fitted_signal[:, k], axis=0), axes=0)
                ),
            )
        plt.title(m)
        plt.subplot(2, 3, 6)
        plt.plot(freq_range_Hz, np.abs(input_data))
        plt.plot(
            freq_range_Hz,
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(np.sum(fitted_signal, axis=1), axis=0), axes=0
                )
            ),
            label="sum",
        )
        plt.title("sum")

        # debug mode, save the r2:
    if dbmode:
        return (
            np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0),
            t2s_s,
            complex_amps,
            all_r2,
        )
    else:
        return (
            np.fft.fftshift(np.fft.fft(fitted_signal, axis=0), axes=0),
            t2s_s,
            complex_amps,
        )


def fit_data_pseudo_inv(
    input_data=None,
    data_obj=None,
    fit_params={},
    use_multiprocessing=True,
    dbplot=False,
    dbmode=False,
    dict_output=False,
):
    """
    Perform spectral data fitting using the pseudoinverse method. This function
    handles complex spectral data in the time domain and allows for multiprocessing
    to improve performance.

    Parameters
    ----------
    input_data : ndarray, optional
        Input data array containing spectral data to be fitted. If None, data will be taken from `data_obj`.
    data_obj: hypermri-sequence object
        Object containing spectral data and other properties. If `input_data` is None, the spectral data
        will be extracted from this object.
    fit_params : dict, optional
        Dictionary containing parameters for fitting. Keys should include 'provided_dims_ind' and other
        fitting-specific settings.
    use_multiprocessing : bool, optional
        If True, use parallel processing to speed up the fitting process (default is True).
    dbplot : bool, optional
        If True, debugging plots will be displayed (default is False).
    dbmode : bool, optional
        If True, additional debugging information will be provided during processing (default is False).
        The `r2_freq` and `r2_t2` arrays (R2 values for frequency and T2 fitting) are only returned if this is True.
    use_all_cores : bool, optional
        If True, all CPU cores are used for multiprocessing. If False, only half are used. This parameter is
        set to be deprecated and moved to `fit_params`.
    dict_output: bool
        temporary parameter that toogles an output as a dict. Helps to implement this change

    Returns
    -------
    tuple of ndarrays
        Returns multiple items in a tuple:
        - fit_spectrums : ndarray
            Fitted spectra array sampled with the same resolution as the input.
        - fit_amps : ndarray
            Amplitudes of the fitted spectra.
        - fit_freqs : ndarray
            Frequencies of the fitted spectra.
        - fit_t2s : ndarray
            T2 values of the fitted spectra.
        - fit_r2_freq : ndarray, optional
            R2 values for frequency fitting, returned only if `dbmode=True`.
        - fit_r2_t2 : ndarray, optional
            R2 values for T2 fitting, returned only if `dbmode=True`.
        - fit_stds : ndarray
            Standard deviations of fitted parameters, in the order of [complex amplitude, frequency, T2,
            magnitude amplitude].
    OR dict containing the above

    Raises
    ------
    ValueError
        If both `input_data` and `data_obj` are None, a ValueError is raised indicating that at least one data source
        must be provided.

    Notes
    -----
    The function is designed to be flexible and efficient, utilizing multiprocessing when possible to handle
    large datasets and complex fitting operations. Debugging modes are available to trace through the fitting
    process and visualize the fitting results.

    Examples
    --------
    Perform fitting:
    >>> fit_results = utf.fit_data_pseudo_inv(input_data=geoff_csi.csi_image,
    >>>                                       fit_params=fit_params,
    >>>                                       use_multiprocessing=True,
    >>>                                       dict_output=True)


    Extract results from dict:
    >>> fit_spectrum = fit_results["fit_spectrum"]
    >>> fit_amps = fit_results["fit_amps"]
    >>> fit_freqs = fit_results["fit_freqs"]
    >>> fit_t2s = fit_results["fit_t2s"]
    >>> r2_freq_array = fit_results["r2_freq_array"]
    >>> r2_t2_array = fit_results["r2_t2_array"]
    >>> fit_stds = fit_results["fit_stds"]

    """
    from tqdm.auto import tqdm
    import time

    if input_data is None:
        if data_obj is not None:
            try:
                # assume nspect:
                if data_obj.spec.ndim == 2:
                    input_data = make_NDspec_6Dspec(
                        input_data=data_obj.spec,
                        provided_dims=[4, 0],
                    )
                else:
                    input_data = data_obj.spec
            except:
                try:
                    # assume CSI:
                    input_data = data_obj.csi_image
                except AttributeError as e:
                    print("Cant load data!")
                    return None
        else:
            raise ValueError("passe either input_data or data_obj!")

    # get a list of the provided dimensions indices:
    provided_dims_ind = fit_params.get("provided_dims_ind")

    # if mismatch between the number of the provided dimensions (index list) and the actual dataset, replace by list
    if len(provided_dims_ind) != np.ndim(input_data):
        provided_dims_ind = [i for i in range(np.ndim(input_data))]

    # makes the spectrum 6D for further processing
    input_data = make_NDspec_6Dspec(
        input_data=input_data, provided_dims=provided_dims_ind
    )

    # fill fit_params:
    fit_params = def_fit_params(fit_params=fit_params, data_obj=data_obj)

    # use multiple CPU cores should be used (speeds up processing)
    if use_multiprocessing:
        from joblib import Parallel, delayed, cpu_count

        use_all_cores = fit_params.get("use_all_cores", False)

        if use_all_cores:
            # use all cpu cores for fast performance (doesnt help if hyperthreading is activated)
            number_of_cpu_cores = cpu_count()
        else:
            number_of_cpu_cores = cpu_count() // 2

    # toggles progess bar:
    show_tqdm = fit_params.get("show_tqdm", True)

    # check number of points in FID to fit:
    if fit_params["nsamplepoints"] is None:
        fit_params["nsamplepoints"] = input_data.shape[0]
    else:
        if fit_params["nsamplepoints"] > input_data.shape[0]:
            fit_params["nsamplepoints"] = input_data.shape[0]
        elif fit_params["nsamplepoints"] < input_data.shape[0]:
            if fit_params["cut_off"] == 0:
                if fit_params["signal_domain"] == "spectral":
                    # need to put it to the temporal domain to remove first points in the FID:
                    temp_signal = np.fft.ifft(
                        np.fft.ifftshift(input_data, axes=(0,)), axis=0
                    )
                    temp_signal = temp_signal[
                        : fit_params["nsamplepoints"], :, :, :, :, :
                    ]
                    input_data = np.fft.fftshift(
                        np.fft.fft(temp_signal, axis=0), axes=(0,)
                    )
                else:
                    input_data = input_data[
                        fit_params["nsamplepoints"] : :, :, :, :, :, :
                    ]
            else:
                if fit_params["signal_domain"] == "spectral":
                    # need to put it to the temporal domain to remove first points in the FID:
                    temp_signal = np.fft.ifft(
                        np.fft.ifftshift(input_data, axes=(0,)), axis=0
                    )
                    temp_signal = temp_signal[fit_params["cut_off"] : :, :, :, :, :, :]
                    input_data = np.fft.fftshift(
                        np.fft.fft(temp_signal, axis=0), axes=(0,)
                    )
                else:
                    input_data = input_data[fit_params["cut_off"] : :, :, :, :, :, :]
        else:
            pass

    # get metaboites:
    metabs = fit_params.get(
        "metabs", ["pyruvate", "lactate", "alanine", "pyruvatehydrate", "urea"]
    )

    # how often the fitting of T2 and freq (interleaved) should be repeated while reducing the range of fitted valuves:
    rep_fitting = fit_params.get("rep_fitting", 4)

    #
    try:
        fit_range_repetitions = fit_params.get(
            "fit_range_repetitions", range(input_data.shape[4])
        )
    except:
        fit_range_repetitions = range(1)

    # if number of repetitions to fit is bigger than the number of repetitions of the input data, use number of
    # repetitions of the input data
    if len(fit_range_repetitions) > input_data.shape[4]:
        fit_range_repetitions = range(input_data.shape[4])

    fit_params["fit_range_repetitions"] = fit_range_repetitions

    # define empty arrays:
    fit_spectrums = np.zeros(input_data.shape + (len(metabs),), dtype="complex")
    fit_amps = np.zeros((1,) + input_data.shape[1:] + (len(metabs),), dtype="complex")
    fit_stds = np.zeros((1,) + input_data.shape[1:] + (len(metabs), 4), dtype="complex")
    fit_freqs = np.zeros((1,) + input_data.shape[1:] + (len(metabs),))
    fit_t2s = np.zeros((1,) + input_data.shape[1:] + (len(metabs),))

    # it's a stupid solution, should be done more elegantly in the future:
    if (
        dbmode
    ):  # freq/t2/freq2/t2       - iterations     -    metabolites     -       number of points per iteration
        fit_r2_freq = np.zeros(
            (1,)
            + input_data.shape[1:]
            + (len(fit_params["metabs"]),)
            + (rep_fitting,)
            + (fit_params["niter"],)
            + (fit_params["npoints"],)
            + (2,)
        )
        fit_r2_t2 = np.zeros(
            (1,)
            + input_data.shape[1:]
            + (len(fit_params["metabs"]),)
            + (rep_fitting,)
            + (fit_params["niter"],)
            + (fit_params["npoints"],)
            + (2,)
        )

        # Define the modified fit_loop function
        def fit_loop(
            dim1,
            dim2,
            dim3,
            dim4,
            dim5,
            input_data,
            data_obj,
            fit_params,
            dbplot=dbplot,
            dbmode=dbmode,
        ):
            # define data to be fit:
            data_slice = np.squeeze(input_data[:, dim1, dim2, dim3, dim4, dim5])

            # init modified: fit_parameters:
            fit_params_mod = fit_params.copy()
            fit_params_mod = def_fit_params(
                fit_params=fit_params_mod, data_obj=data_obj
            )

            # init empty array for the R2s
            r2_freq_array = np.zeros(
                (len(fit_params["metabs"]),)
                + (rep_fitting,)
                + (fit_params["niter"],)
                + (fit_params["npoints"],)
                + (2,)  # r2 + actual vaules
            )
            r2_t2_array = np.zeros(
                (len(fit_params["metabs"]),)
                + (rep_fitting,)
                + (fit_params["niter"],)
                + (fit_params["npoints"],)
                + (2,)
            )

            for r in range(rep_fitting):
                # fit frequencies:
                _, freq_Hz, _, r2_freq = fit_freq_pseudo_inv(
                    input_data=data_slice,
                    metabs=metabs,
                    fit_params=fit_params_mod,
                    data_obj=data_obj,
                    dbplot=dbplot,
                    dbmode=dbmode,
                )

                fit_params_mod[
                    "metabs_freqs_Hz"
                ] = freq_Hz  # Update current best estimate

                # fit T2*:
                _, t2_s, _, r2_t2 = fit_t2_pseudo_inv(
                    input_data=data_slice,
                    metabs=metabs,
                    fit_params=fit_params_mod,
                    data_obj=data_obj,
                    dbplot=dbplot,
                    dbmode=dbmode,
                )
                fit_params_mod["metabs_t2s"] = t2_s

                # save R2s:
                r2_freq_array[:, r, :, :] = r2_freq
                r2_t2_array[:, r, :, :] = r2_t2

                # reduce fitting range:
                # fit_params_mod["range_t2s_s"] *= 1 / fit_params_mod["zoomfactor"]
                # fit_params_mod["range_freqs_Hz"] *= 1 / fit_params_mod["zoomfactor"]

            # final fit with optimized parameters:
            fit_spectrum, t2_s, fit_amps = fit_t2_pseudo_inv(
                input_data=data_slice,
                metabs=metabs,
                fit_params=fit_params_mod,
                data_obj=data_obj,
                dbplot=dbplot,
                dbmode=False,
            )

            popt = np.concatenate(
                [
                    fit_amps,  # Convert numpy scalar to Python scalar
                    freq_Hz,
                    t2_s,
                ]
            ).tolist()

            fit_stds = estimate_std_fit_results(
                popt,
                fit_params,
                input_data=data_slice,
                verbose=False,
                full_output=False,
            )

            return (
                fit_spectrum,
                fit_amps,
                freq_Hz,
                t2_s,
                r2_freq_array,
                r2_t2_array,
                fit_stds,
            )

        # Generate tasks for parallel execution
        tasks = [
            (
                dim1,
                dim2,
                dim3,
                dim4,
                dim5,
                input_data,
                data_obj,
                fit_params,
                dbplot,
                dbmode,
            )
            for dim4 in fit_range_repetitions
            for dim1 in range(input_data.shape[1])
            for dim2 in range(input_data.shape[2])
            for dim3 in range(input_data.shape[3])
            for dim5 in range(input_data.shape[5])
        ]

        results = []
        st = time.time()
        if use_multiprocessing:
            # Parallel execution
            results = Parallel(n_jobs=number_of_cpu_cores)(
                delayed(fit_loop)(*task) for task in tqdm(tasks)
            )
        else:
            for task in tqdm(tasks):
                results.append(fit_loop(*task))
        if dbmode:
            print(f"Fitting took: {np.round(time.time() - st,3)}s")
            pass
        # Aggregate results
        for idx, (dim1, dim2, dim3, dim4, dim5, _, _, _, _, _) in enumerate(tasks):
            (
                fit_spectrum,
                fit_amp,
                freq_Hz,
                t2_s,
                r2_freqs,
                r2_t2s,
                fit_std,
            ) = results[idx]
            fit_amps[0, dim1, dim2, dim3, dim4, dim5, :] = fit_amp
            fit_stds[0, dim1, dim2, dim3, dim4, dim5, :] = fit_std
            fit_freqs[0, dim1, dim2, dim3, dim4, dim5, :] = freq_Hz
            fit_t2s[0, dim1, dim2, dim3, dim4, dim5, :] = t2_s
            fit_spectrums[:, dim1, dim2, dim3, dim4, dim5, :] = fit_spectrum
            fit_r2_freq[:, dim1, dim2, dim3, dim4, dim5, :, :, :] = r2_freqs
            fit_r2_t2[:, dim1, dim2, dim3, dim4, dim5, :, :, :] = r2_t2s

        if fit_params["save_fit_results"]:
            from hypermri.utils.utils_general import save_as_pkl

            # Your data to be saved
            fit_results = [
                fit_amps,
                fit_spectrums,
                fit_t2s,
                fit_freqs,
                fit_r2_freq,
                fit_r2_t2,
                fit_stds,
                fit_params,
            ]
            # corresponding keys:
            file_keys = [
                "fit_amps",
                "fit_spectrums",
                "fit_t2s",
                "fit_freqs",
                "fit_r2_freq",
                "fit_r2_t2",
                "fit_stds",
                "fit_params",
            ]

            # save fit results (using timestamp to avoid overwritting old data)
            save_as_pkl(
                dir_path=fit_params["savepath"],  # directory
                filename="fit_results",  # name of file to be saved:
                file_keys=file_keys,  # keys in dict to be saved
                file=fit_results,  # dict to be saved
                use_timestamp=True,  # using timestamp in name
                print_file_path=True,
            )  # output the filepath and name afte saveing
        if dict_output:
            fit_results = {}
            fit_results["fit_spectrum"] = fit_spectrums
            fit_results["fit_amps"] = fit_amps
            fit_results["fit_freqs"] = fit_freqs
            fit_results["fit_t2s"] = fit_t2s
            fit_results["fit_r2_t2"] = fit_r2_t2
            fit_results["fit_r2_freq"] = fit_r2_freq
            fit_results["fit_stds"] = fit_stds
            return fit_results
        else:
            return (
                fit_spectrums,
                fit_amps,
                fit_freqs,
                fit_t2s,
                fit_r2_freq,
                fit_r2_t2,
                fit_stds,
            )
    else:
        # Define the modified fit_loop function
        def fit_loop(
            dim1,
            dim2,
            dim3,
            dim4,
            dim5,
            input_data,
            data_obj,
            fit_params,
            dbplot=dbplot,
        ):
            # prefer input data over data_obj data:
            data_slice = np.squeeze(input_data[:, dim1, dim2, dim3, dim4, dim5])

            # quick check if data_slice contains actual data:
            if np.sum(data_slice) == 0:
                raise Exception(
                    f"input_data at index [:,{dim1},{dim2},{dim3},{dim4},{dim5}] contains no data!"
                )

            # init modified: fit_parameters:
            fit_params_mod = fit_params.copy()

            fit_params_mod = def_fit_params(
                fit_params=fit_params_mod, data_obj=data_obj
            )
            data_slice = data_slice[: fit_params_mod["nsamplepoints"]]

            for r in range(rep_fitting):
                # fit frequencies:
                _, freq_Hz, _ = fit_freq_pseudo_inv(
                    input_data=data_slice,
                    metabs=metabs,
                    fit_params=fit_params_mod,
                    data_obj=data_obj,
                    dbplot=dbplot,
                )

                fit_params_mod["metabs_freqs_Hz"] = freq_Hz

                # fit T2*:
                _, t2_s, _ = fit_t2_pseudo_inv(
                    input_data=data_slice,
                    metabs=metabs,
                    fit_params=fit_params_mod,
                    data_obj=data_obj,
                    dbplot=dbplot,
                )
                fit_params_mod["metabs_t2s"] = t2_s

                # reduce fitting range:
                # fit_params_mod["range_t2s_s"] *= 1 / fit_params_mod["zoomfactor"]
                # fit_params_mod["range_freqs_Hz"] *= 1 / fit_params_mod["zoomfactor"]

            # (final) fit T2*:
            fit_spectrum, t2_s, fit_amps = fit_t2_pseudo_inv(
                input_data=data_slice,
                metabs=metabs,
                fit_params=fit_params_mod,
                data_obj=data_obj,
                dbplot=dbplot,
                dbmode=False,
            )

            popt = np.concatenate(
                [
                    fit_amps,  # Convert numpy scalar to Python scalar
                    freq_Hz,
                    t2_s,
                ]
            ).tolist()

            fit_stds = estimate_std_fit_results(
                popt,
                fit_params,
                input_data=data_slice,
                verbose=False,
                full_output=False,
            )

            return fit_spectrum, fit_amps, freq_Hz, t2_s, fit_stds

        # Generate tasks for parallel execution
        tasks = [
            (dim1, dim2, dim3, dim4, dim5, input_data, data_obj, fit_params, dbplot)
            for dim4 in fit_range_repetitions
            for dim1 in range(input_data.shape[1])
            for dim2 in range(input_data.shape[2])
            for dim3 in range(input_data.shape[3])
            for dim5 in range(input_data.shape[5])
        ]

        results = []
        if use_multiprocessing:
            # Parallel execution
            results = Parallel(n_jobs=number_of_cpu_cores)(
                delayed(fit_loop)(*task)
                for task in tqdm(
                    tasks,
                    leave=False,
                    desc="Fitting spectra on " + str(number_of_cpu_cores) + " cores",
                    disable=not show_tqdm,
                )
            )
        else:
            for task in tqdm(
                tasks, leave=False, desc="Fitting spectra", disable=not show_tqdm
            ):
                results.append(fit_loop(*task))

        # Aggregate results
        for idx, (dim1, dim2, dim3, dim4, dim5, _, _, _, _) in enumerate(tasks):
            fit_spectrum, fit_amp, freq_Hz, t2_s, fit_std = results[idx]
            fit_amps[0, dim1, dim2, dim3, dim4, dim5, :] = fit_amp
            fit_stds[0, dim1, dim2, dim3, dim4, dim5, :] = fit_std
            fit_freqs[0, dim1, dim2, dim3, dim4, dim5, :] = freq_Hz
            fit_t2s[0, dim1, dim2, dim3, dim4, dim5, :] = t2_s
            fit_spectrums[:, dim1, dim2, dim3, dim4, dim5, :] = fit_spectrum

        # save the fit results:
        if fit_params["save_fit_results"]:
            from ..utils.utils_general import save_as_pkl

            # Your data to be saved
            fit_results = [
                fit_amps,
                fit_spectrums,
                fit_t2s,
                fit_freqs,
                fit_stds,
                fit_params,
            ]
            # corresponding keys:
            file_keys = [
                "fit_amps",
                "fit_spectrums",
                "fit_t2s",
                "fit_freqs",
                "fit_stds",
                "fit_params",
            ]

            # save fit results (using timestamp to avoid overwritting old data)
            save_as_pkl(
                dir_path=fit_params["savepath"],  # directory
                filename="fit_results",  # name of file to be saved:
                file_keys=file_keys,  # keys in dict to be saved
                file=fit_results,  # dict to be saved
                use_timestamp=True,  # using timestamp in name
                print_file_path=True,
            )  # output the filepath and name afte saveing
        if dict_output:
            fit_results = {}
            fit_results["fit_spectrum"] = fit_spectrums
            fit_results["fit_amps"] = fit_amps
            fit_results["fit_freqs"] = fit_freqs
            fit_results["fit_t2s"] = fit_t2s
            fit_results["fit_r2_freq"] = np.zeros_like(fit_amps)  # for consistency
            fit_results["fit_r2_t2"] = np.zeros_like(fit_amps)  # for consistency
            fit_results["fit_stds"] = fit_stds
            return fit_results
        else:
            return fit_spectrums, fit_amps, fit_freqs, fit_t2s, fit_stds


def plot_fitted_spectra(
    measured_data=None,
    fitted_data=None,
    fit_r2_t2=None,
    fit_r2_freq=None,
    plot_params={},
    fit_params={},
    data_obj=None,
):
    """
    Combines functionality of plot_fitted_spectra and analyse_fit_r2 with clickable image for voxel selection.
    Adapts layout based on available data.

    Parameters
    ----------
    measured_data: array, optional
        measured spectra
    fitted_data: array, optional
        fitted spectra, should be 7D (echoes - z - x - y - reps - chans - metabolites)
    fit_r2_t2: array, optional
        R2 fitting results for T2*
    fit_r2_freq: array, optional
        R2 fitting results for frequency
    plot_params: dict
        plotting parameters
    fit_params: dict
        fitting parameters
    data_obj: hypermri-sequence object, optional

    Returns
    -------
    An interactive plot to visualize fitted spectra and/or R2 analysis
    """
    import ipywidgets as widgets
    from IPython.display import display

    # Set up parameters
    metabs = fit_params.get(
        "metabs", ["urea", "pyruvate", "alanine", "pyruvatehydrate", "lactate"]
    )
    figsize = plot_params.get("figsize", (15, 5))  # default

    # Determine which data is available
    has_measured = measured_data is not None
    has_fitted = fitted_data is not None
    has_r2 = fit_r2_t2 is not None or fit_r2_freq is not None

    if not has_fitted and not has_r2:
        raise ValueError("Either fitted data or R2 data must be provided.")

    if has_measured:
        if np.ndim(measured_data) != 6:
            # ("Measured data must be 6D (echoes - z - x - y - reps - chans)"
            measured_data = make_NDspec_6Dspec(
                input_data=measured_data,
                provided_dims=fit_params.get(
                    "provided_dims",
                    [
                        "spectral",
                    ],
                ),
            )

    # Determine the number of rows based on the data
    num_rows = 1  # Start with one row for the image and spectra
    if has_fitted and fitted_data.shape[4] > 1:
        num_rows += 1  # Add row for amplitude plots if there are repetitions
    if has_r2:
        num_rows += 1  # Add row for R2 plots

    # Set up the main figure
    fig = plt.figure(figsize=(figsize[0], figsize[1] * num_rows))
    gs = fig.add_gridspec(num_rows, 3)

    # Initialize axes
    ax_image = fig.add_subplot(gs[0, 0])
    ax_spec1 = fig.add_subplot(gs[0, 1])
    ax_spec2 = fig.add_subplot(gs[0, 2])

    if has_fitted and fitted_data.shape[4] > 1:
        ax_amp = fig.add_subplot(gs[1, 0:2])
        ax_cumamp = fig.add_subplot(gs[1, 2])
    else:
        ax_amp = ax_cumamp = None

    if has_r2:
        if num_rows == 3:
            ax_r2_t2 = fig.add_subplot(gs[2, 0:2])
            ax_r2_freq = fig.add_subplot(gs[2, 2])
        else:
            ax_r2_t2 = fig.add_subplot(gs[1, 0:2])
            ax_r2_freq = fig.add_subplot(gs[1, 2])
    else:
        ax_r2_t2 = ax_r2_freq = None

    # Data rotation (if applicable)
    if measured_data is not None:
        measured_data = np.rot90(measured_data, k=1, axes=(2, 3))
    if fitted_data is not None:
        fitted_data = np.rot90(fitted_data, k=1, axes=(2, 3))
    if fit_r2_t2 is not None:
        fit_r2_t2 = np.rot90(fit_r2_t2, k=1, axes=(2, 3))
    if fit_r2_freq is not None:
        fit_r2_freq = np.rot90(fit_r2_freq, k=1, axes=(2, 3))

    # Prepare the image for clicking
    if has_measured:
        sum_image = np.sum(
            np.abs(measured_data), axis=(0, 4, 5)
        )  # Sum over echoes, reps, and channels
    elif has_fitted:
        sum_image = np.sum(
            np.abs(fitted_data), axis=(0, 4, 5, 6)
        )  # Sum over echoes, reps, channels, and metabolites
    else:
        # Create a dummy image from R2 data
        sum_image = np.sum(
            fit_r2_t2 if fit_r2_t2 is not None else fit_r2_freq,
            axis=(0, 4, 5, 6, 7, 8, 9),
        )
    img = ax_image.imshow(sum_image[0], cmap="viridis")
    ax_image.set_title("Click to select voxel")

    # Set up the interactive widgets
    slice_slider = widgets.IntSlider(
        min=0, max=sum_image.shape[0] - 1, step=1, description="Slice:"
    )
    widgets_list = [slice_slider]

    if has_fitted:
        rep_slider = widgets.IntSlider(
            min=0, max=fitted_data.shape[4] - 1, step=1, description="Rep:"
        )
        chan_slider = widgets.IntSlider(
            min=0, max=fitted_data.shape[5] - 1, step=1, description="Chan:"
        )
        widgets_list.extend([rep_slider, chan_slider])

    metab_slider = widgets.IntSlider(
        min=0, max=len(metabs) - 1, step=1, description="Metab:"
    )
    widgets_list.append(metab_slider)

    if has_r2:
        fit_iter_slider = widgets.IntRangeSlider(
            min=0,
            max=fit_r2_t2.shape[8] - 1
            if fit_r2_t2 is not None
            else fit_r2_freq.shape[8] - 1,
            step=1,
            value=[
                0,
                fit_r2_t2.shape[8] - 1
                if fit_r2_t2 is not None
                else fit_r2_freq.shape[8] - 1,
            ],
            description="Iter:",
        )
        fit_rep_slider = widgets.IntRangeSlider(
            min=0,
            max=fit_r2_t2.shape[7] - 1
            if fit_r2_t2 is not None
            else fit_r2_freq.shape[7] - 1,
            step=1,
            value=[
                0,
                fit_r2_t2.shape[7] - 1
                if fit_r2_t2 is not None
                else fit_r2_freq.shape[7] - 1,
            ],
            description="Fit Reps:",
        )
        show_range_checkbox = widgets.Checkbox(value=True, description="Show Range:")
        widgets_list.extend([fit_iter_slider, fit_rep_slider, show_range_checkbox])

    # Define color map and markers for R2 plots
    cmap_name = plot_params.get("cmap", "jet")
    marker_size = plot_params.get("markersize", 20)
    markers = plot_params.get("markers", ["o", ">", "x", "^", "*", "."])
    colormap = getattr(plt.cm, cmap_name, plt.cm.jet)
    if has_r2:
        colors = [
            colormap(i)
            for i in np.linspace(
                0,
                1,
                fit_r2_t2.shape[7] if fit_r2_t2 is not None else fit_r2_freq.shape[7],
            )
        ]
    else:
        colors = []

    # Variable to store the clicked position
    clicked_position = [None, None]

    # Function to update the plot
    def update_plot(
        slice=0,
        x=0,
        y=0,
        rep=0,
        chan=0,
        metab=0,
        fit_iters=[0, -1],
        fit_reps=[0, -1],
        show_range=True,
    ):
        ax_image.clear()
        ax_image.imshow(sum_image[slice], cmap="viridis")
        ax_image.set_title("Click to select voxel")
        if clicked_position[0] is not None and clicked_position[1] is not None:
            ax_image.plot(
                clicked_position[1],
                clicked_position[0],
                "ro",
                markersize=10,
                markerfacecolor="none",
            )

        if has_fitted:
            ax_spec1.clear()
            ax_spec2.clear()

            # Plot spectra
            if has_measured:
                ax_spec1.plot(
                    freq_range,
                    np.abs(measured_data[:, slice, x, y, rep, chan]),
                    label="Measured",
                )
            ax_spec1.plot(
                freq_range,
                np.abs(np.sum(fitted_data[:, slice, x, y, rep, chan, :], axis=1)),
                label="Fitted",
            )
            ax_spec1.set_xlim(xlim)
            ax_spec1.set_title("Absolute Spectra")
            ax_spec1.legend()

            # Plot fitted metabolites with selected metabolite highlighted
            for k, m in enumerate(metabs):
                if k == metab:
                    ax_spec2.plot(
                        freq_range,
                        np.abs(fitted_data[:, slice, x, y, rep, chan, k]),
                        label=m,
                        linewidth=3,
                    )
                else:
                    ax_spec2.plot(
                        freq_range,
                        np.abs(fitted_data[:, slice, x, y, rep, chan, k]),
                        label=m,
                        alpha=0.5,
                    )
            ax_spec2.set_xlim(xlim)
            ax_spec2.set_title("Fitted Metabolites")
            ax_spec2.legend()

            # Plot amplitudes if applicable
            if ax_amp and ax_cumamp:
                ax_amp.clear()
                ax_cumamp.clear()
                for k, m in enumerate(metabs):
                    ax_amp.plot(
                        np.sum(np.abs(fitted_data[:, slice, x, y, :, chan, k]), axis=0),
                        label=m,
                    )
                ax_amp.axvline(x=rep, color="k", linestyle="--")
                ax_amp.set_title("Metabolite Amplitudes")
                ax_amp.legend()

                for k, m in enumerate(metabs):
                    ax_cumamp.plot(
                        np.cumsum(
                            np.sum(
                                np.abs(fitted_data[:, slice, x, y, :, chan, k]), axis=0
                            )
                        ),
                        label=m,
                    )
                ax_cumamp.axvline(x=rep, color="k", linestyle="--")
                ax_cumamp.set_title("Cumulative Metabolite Amplitudes")
                ax_cumamp.legend()

        if has_r2:
            ax_r2_t2.clear()
            ax_r2_freq.clear()

            # Plot R2(T2*) analysis
            if fit_r2_t2 is not None:
                for fit_rep in range(fit_reps[0], fit_reps[1] + 1):
                    for fit_iter in range(fit_iters[0], fit_iters[1] + 1):
                        ax_r2_t2.scatter(
                            np.squeeze(
                                fit_r2_t2[
                                    0,
                                    slice,
                                    x,
                                    y,
                                    rep,
                                    chan,
                                    metab,
                                    fit_rep,
                                    fit_iter,
                                    :,
                                    1,
                                ]
                            ),
                            np.squeeze(
                                fit_r2_t2[
                                    0,
                                    slice,
                                    x,
                                    y,
                                    rep,
                                    chan,
                                    metab,
                                    fit_rep,
                                    fit_iter,
                                    :,
                                    0,
                                ]
                            ),
                            marker=markers[fit_iter],
                            s=marker_size,
                            color=colors[fit_rep],
                            label=f"Rep {fit_rep}, Iter {fit_iter}",
                        )
                if show_range:
                    ax_r2_t2.axvline(
                        fit_params["min_t2_s"][metab], color="k", alpha=0.5
                    )
                    ax_r2_t2.axvline(
                        fit_params["max_t2_s"][metab], color="k", alpha=0.5
                    )
                ax_r2_t2.set_title(f"R2(T2*) {metabs[metab]}")
                ax_r2_t2.set_xlabel("T2* [s]")
                ax_r2_t2.set_ylabel("R^2 [a.u.]")
                ax_r2_t2.legend()

            # Plot R2(freq) analysis
            if fit_r2_freq is not None:
                for fit_rep in range(fit_reps[0], fit_reps[1] + 1):
                    for fit_iter in range(fit_iters[0], fit_iters[1] + 1):
                        ax_r2_freq.scatter(
                            np.squeeze(
                                fit_r2_freq[
                                    0,
                                    slice,
                                    x,
                                    y,
                                    rep,
                                    chan,
                                    metab,
                                    fit_rep,
                                    fit_iter,
                                    :,
                                    1,
                                ]
                            ),
                            np.squeeze(
                                fit_r2_freq[
                                    0,
                                    slice,
                                    x,
                                    y,
                                    rep,
                                    chan,
                                    metab,
                                    fit_rep,
                                    fit_iter,
                                    :,
                                    0,
                                ]
                            ),
                            marker=markers[fit_iter],
                            s=marker_size,
                            color=colors[fit_rep],
                            label=f"Rep {fit_rep}, Iter {fit_iter}",
                        )
                ax_r2_freq.set_title(f"R2(freq) {metabs[metab]}")
                ax_r2_freq.set_xlabel("Frequency [Hz]")
                ax_r2_freq.set_ylabel("R^2 [a.u.]")
                ax_r2_freq.legend()

        plt.tight_layout()
        fig.canvas.draw_idle()

    # Function to handle click events
    def on_click(event):
        if event.inaxes == ax_image:
            x, y = int(event.ydata), int(event.xdata)
            clicked_position[0], clicked_position[1] = x, y
            update_plot(
                slice_slider.value,
                x,
                y,
                rep_slider.value if "rep_slider" in locals() else 0,
                chan_slider.value if "chan_slider" in locals() else 0,
                metab_slider.value,
                fit_iter_slider.value if "fit_iter_slider" in locals() else [0, -1],
                fit_rep_slider.value if "fit_rep_slider" in locals() else [0, -1],
                show_range_checkbox.value
                if "show_range_checkbox" in locals()
                else True,
            )

    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_change(change):
        if clicked_position[0] is not None and clicked_position[1] is not None:
            x, y = clicked_position
        else:
            x, y = 0, 0
        update_plot(
            slice_slider.value,
            x,
            y,
            rep_slider.value if "rep_slider" in locals() else 0,
            chan_slider.value if "chan_slider" in locals() else 0,
            metab_slider.value,
            fit_iter_slider.value if "fit_iter_slider" in locals() else [0, -1],
            fit_rep_slider.value if "fit_rep_slider" in locals() else [0, -1],
            show_range_checkbox.value if "show_range_checkbox" in locals() else True,
        )

    # Connect the sliders
    for widget in widgets_list:
        widget.observe(on_change, names="value")

    # Create horizontal layout for sliders
    sliders_box = widgets.HBox(widgets_list[:4])  # First 4 widgets in one row
    if len(widgets_list) > 4:
        r2_sliders_box = widgets.HBox(
            widgets_list[4:]
        )  # Remaining widgets in another row
        all_widgets = widgets.VBox([sliders_box, r2_sliders_box])
    else:
        all_widgets = sliders_box

    # Display the widgets
    display(all_widgets)

    # Set up frequency range and xlim if spectral data is available
    if has_fitted:
        freq_range = fit_params.get(
            "freq_range_Hz", np.linspace(-1 / 2, 1 / 2, fitted_data.shape[0])
        )
        xlim = plot_params.get("xlim", (freq_range[0], freq_range[-1]))
    else:
        freq_range = None
        xlim = None

    # Show the initial plot
    update_plot()
    plt.tight_layout()
    plt.show()


def b1_fitting(
    input_data,
    refpower,
    data_obj=None,
    fa_deg=90,
    trf_ms=1,
    sint=1,
    niter=10,
    perlen=500,
    high_prec=True,
    red_fact=1 / 1.75,
    use_multiprocessing=True,
):
    """
    Generates a B1 Map of the input_data. input_data is expected to be a list of images, acquired with different
    RF powers refpower. Can take some time as it's quite calculation intense.

    Parameters
    ----------
    input_data : list of complex values 2D images, optional
        The input data to be used for B1 mapping. Default is None.
    refpower : list, optional
        List of used reference powers (same number of elements as input_data). Default is None.
    data_obj : object, optional
        BrukerXP object for passing parameters. Default is None.
    fa_deg : float, optional
        Defined flip angle (only necessary if no data_obj was passed). Default is 90.
    trf_ms : float, optional
        Pulse duration in milliseconds (only necessary if no data_obj was passed). Default is 1.
    sint : float, optional
        Integration factor (only necessary if no data_obj was passed). Default is 1.
    niter : int, optional
        Number of iterations (recommended 10). Default is 10.
    perlen : int, optional
        Number of steps per iteration (don't use less than 500). Default is 500.
    high_prec : bool, optional
        If True, a second round will be performed for higher precision. Default is True.
    red_fact : float, optional
        Reduction factor by which the search range should be reduced each iteration. Small number = fast reduction. Default is 1/1.75.
    use_multiprocessing: bool, optional
      If True, multiprocessing will be used. Default is True.

    Returns
    -------
    tuple
        B1 map, reference power map, fitted data, and optionally higher precision results.
    Notes
    -----
    So far only works for axial (x-y) images!
    """
    from ..utils.utils_spectroscopy import make_NDspec_6Dspec
    from joblib import Parallel, delayed, cpu_count
    from tqdm import tqdm
    import time

    # check if input_data is a list and convert to numpy array
    if isinstance(input_data, list):
        input_data = np.array(input_data)
    # check if input_data is 7D and reduce to 6D
    if np.ndim(input_data) == 7:
        input_data = np.squeeze(np.mean(input_data, axis=5))

    # int-divide by 2 because hyperthreading cores don't count/help:
    number_of_cpu_cores = cpu_count() // 2

    # number of acquisitions
    naq = input_data.shape[0]
    # number of pixels in x and y direction
    nx, ny = input_data.shape[1:3]

    if data_obj is not None:
        fa_deg = data_obj.method["ExcPulse1"][2]
        trf_ms = data_obj.method["ExcPulse1"][0]
        sint = data_obj.method["ExcPulse1"][6]

    print("Pulse Parameters:")
    print("-----------------")
    print(f"Flipangle: {fa_deg}°")
    print(f"Pulse duration: {trf_ms} ms")
    print(f"Integration factor: {sint}")
    print("-----------------")
    print(
        f"Fitting B1 in {niter} steps, each time fitting {perlen} different RF powers"
    )

    def perform_b1_fit(stop_ind=None):
        # start time
        start_time = time.time()

        # calculate sqrt of reference power (B1 ~ sin(sqrt(RF Power)))
        sqrt_refpow = np.sqrt(refpower)
        if stop_ind is None:
            # Generates an array of siz (nx, ny), filled with naq (all stop indices are naq)
            stop_ind = np.full((nx, ny), naq)

        # define center period
        percent = 3.1
        # define period range
        perextent = 3
        # create period array
        per = np.linspace(percent - perextent, percent + perextent, perlen)
        # create a 3D array of the period array
        per = np.tile(per, (nx, ny, 1))

        # init empty arrays:
        fit_results = np.zeros((1, nx, ny, perlen), dtype=complex)
        data_fit = np.zeros((naq, nx, ny, perlen), dtype=complex)
        r2 = np.zeros((nx, ny, perlen))

        # define the inner loop function
        def compute_inner_loop(ix, iy):
            # initialze empty array for R^2
            local_r2 = np.zeros(perlen)
            # initialize empty array for fit results
            local_fit_results = np.zeros((1, perlen), dtype=complex)
            # initialize empty array for data fit
            local_data_fit = np.zeros((naq, perlen), dtype=complex)
            # get the stop index for the current pixel
            local_stop = int(stop_ind[ix, iy])

            # create the A matrix (modelled signal)
            A = np.sin(np.pi / per[ix, iy] * sqrt_refpow[:local_stop, np.newaxis])
            # create the b vector (measured signal)
            b = input_data[:local_stop, ix, iy]

            for ip in range(perlen):
                # solve the least squares problem
                x = np.linalg.lstsq(A[:, ip : ip + 1], b, rcond=None)[0]
                # store the results:
                local_fit_results[0, ip] = x
                local_data_fit[:local_stop, ip] = A[:, ip] * x
                local_r2[ip] = np.sum(np.abs(b - A[:, ip] * x) ** 2)

            return ix, iy, local_fit_results, local_data_fit, local_r2

        for iter in tqdm(range(niter), desc="B1 Fitting Progress"):
            if use_multiprocessing:
                results = Parallel(n_jobs=number_of_cpu_cores)(
                    delayed(compute_inner_loop)(ix, iy)
                    for ix in range(nx)
                    for iy in range(ny)
                )
            else:
                results = [
                    compute_inner_loop(ix, iy) for ix in range(nx) for iy in range(ny)
                ]

            for ix, iy, local_fit_results, local_data_fit, local_r2 in results:
                fit_results[0, ix, iy, :] = local_fit_results
                data_fit[:, ix, iy, :] = local_data_fit
                r2[ix, iy, :] = local_r2

            r2_min_ind = np.argmin(r2, axis=2)
            perextent *= red_fact

            for ixx in range(nx):
                for iyy in range(ny):
                    percent = per[ixx, iyy, r2_min_ind[ixx, iyy]]
                    temp = np.linspace(
                        max(0.05, percent - perextent), percent + perextent, perlen
                    )
                    per[ixx, iyy, :] = temp

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} seconds")

        r2_min_ind = np.argmin(r2, axis=2)
        a = fit_results[0, np.arange(nx)[:, None], np.arange(ny), r2_min_ind]
        data_fit_final = data_fit[
            np.arange(naq)[:, None, None],
            np.arange(nx)[:, None],
            np.arange(ny),
            r2_min_ind,
        ]
        periods = per[np.arange(nx)[:, None], np.arange(ny), r2_min_ind]
        refpow_map = (0.5 * periods) ** 2
        # refpow_map = refpow_map * (90 * trf_ms * sint / fa_deg) ** 2

        a = make_NDspec_6Dspec(input_data=a, provided_dims=(2, 3))
        refpow_map = make_NDspec_6Dspec(input_data=refpow_map, provided_dims=(2, 3))
        periods = make_NDspec_6Dspec(input_data=periods, provided_dims=(2, 3))
        data_fit_final = make_NDspec_6Dspec(
            input_data=data_fit_final, provided_dims=(0, 2, 3)
        )

        return a, periods, data_fit_final, refpow_map, r2_min_ind

    # perform fitting, return Amplitude, oscillation periods depending on reference power, fitted data and reference
    # power map
    a, periods, data_fit_final, refpow_map, r2_min_ind = perform_b1_fit()

    if high_prec:
        print("-----------------")
        print("Performing second search for higher precision")
        stop_ind = np.zeros((nx, ny))
        for iy in range(ny):
            for ix in range(nx):
                stop_ind[ix, iy] = (
                    int(
                        np.argmin(
                            np.abs(
                                np.sqrt(refpower)
                                - 4 * np.sqrt(refpow_map[0, 0, ix, iy, 0, 0])
                            )
                        )
                    )
                    + 1
                )
        a_2, periods_2, data_fit_final_2, refpow_map_2, r2_min_ind_2 = perform_b1_fit(
            stop_ind=stop_ind
        )
        # # scale the reference power map by integration factor, flip angle and pulse duration
        # refpow_map = refpow_map * (90 * trf_ms * sint / fa_deg) ** 2
        # refpow_map_2 = refpow_map_2 * (90 * trf_ms * sint / fa_deg) ** 2

        return (
            a,
            a_2,
            refpow_map,
            refpow_map_2,
            data_fit_final,
            data_fit_final_2,
            periods,
            periods_2,
        )
    else:
        # scale the reference power map by integration factor, flip angle and pulse duration
        # refpow_map = refpow_map * (90 * trf_ms * sint / fa_deg) ** 2
        return a, refpow_map, data_fit_final, periods





def fit_single_point_apy(
    data_obj=None,
    data_point=None,
    f_init=None,
    freq_range=None,
    verblevel=0,
    fit_params=None,
):
    """
    Fit a model (in f_init) to the data data_point (1D) along freq_range.

    Parameters
    ----------
    data_obj: hypermri-sequence object

    data_point : ndarray
        1D spectral data point.

    f_init : object
        Initial model for fitting.

    freq_range : ndarray
        Frequency values corresponding to data_point.

    verblevel : int, optional
        Verbose level of fit, 0 = off.

    fit_params : dict, optional
        Additional fit parameters.

    Returns
    -------
    fit_model_result : object
        Fitted model for the provided data point.

    """
    from astropy.modeling import fitting

    if data_point is None:
        Warning("data_point is None!")
        return None

    if fit_params is None:
        fit_params = generate_default_fit_params_apy(
            data_obj=data_obj, input_data=data_point
        )

    # FIXME move these (the ones that have to be set to avoid crashing) to a standard value defining function that is always called:
    fit_method = fit_params.get("fit_method", "SLSQPLSQFitter")

    # define frequency range:
    # print(freq_range)
    if freq_range is not None:
        pass
    else:
        freq_range = fit_params.get(
            "freq_range",
            get_freq_axis(
                scan=data_obj, unit=fit_params["freq_unit"], npoints=data_point.shape[0]
            ),
        )

    # data_point = data_point[fit_params["fit_range"][0]:fit_params["fit_range"][1]]
    # freq_range = freq_range[fit_params["fit_range"][0]:fit_params["fit_range"][1]]

    # normalize data:
    if fit_params["fit_area_opt"]:
        data_point = data_point / fit_params["fit_area_opt_factor"]

    # match the initial fit parameters to the data (input_data)
    f_init = match_fit_init_apy(
        data_obj=data_obj,
        fit_init=f_init,
        input_data=data_point,
        fit_params=fit_params,
    )

    # choose fitting method:
    if fit_method == "LevMarLSQFitter":
        fit_sol = fitting.LevMarLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
        )
    elif fit_method == "SLSQPLSQFitter":
        fit_sol = fitting.SLSQPLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
            verblevel=verblevel,
            iter=fit_params["fit_maxiter"],
            epsilon=fit_params["fit_eps"],
        )
    elif fit_method == "LinearLSQFitter":
        fit_sol = fitting.LinearLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
        )
    else:
        # set fitting method:
        fit_params["fit_method"] = "SLSQPLSQFitter"
        fit_sol = fitting.SLSQPLSQFitter()
        fit_model_result = fit_sol(
            f_init,
            freq_range,
            data_point,
            verblevel=verblevel,
            iter=fit_params["fit_maxiter"],
            epsilon=fit_params["fit_eps"],
        )

    # save fit parameters:
    fit_params["fit_info"] = fit_sol.fit_info

    return fit_model_result


def fit_spectra_apy(
    data_obj=None,
    input_data=None,
    number_of_cpu_cores=None,
    use_multiprocessing=True,
    fit_params=None,
):
    """
    Fit spectra onto a CSI dataset using the astropy modeling and fitting library.

    Parameters:
    ----------
    input_data : array-like, optional
        CSI data in format [freq - z - x - y - reps - channels].
        - if None or "seq2d", self.seq2d is used
        - if "csi_image", self.csi_image is used.

    number_of_cpu_cores : int, optional
        Number of CPU cores to be used for multiprocessing.
        If None (Default), the number of cores is selected automatically.

    use_multiprocessing : bool, default=True
        If True, use multiprocessing using multiple CPU cores.

    fit_params : dict, optional
        Dictionary containing fit parameters.
        - e.g. bounds: `{"amplitude_1": {"bounds": (0.0, 0.1)} ,"x_0_1": {"bounds": (178, 180)}}`

    Returns
    -------
    fit_results : dict
        Dictionary containing the fitted model results and other relevant parameters.
        - `data`: Model results.
        - `metabs`: List of metabolites.
        - `nrange`: Range of repetitions.
        - `freq_range`: Frequency range.
        - `lb`: Linebroadening.

    Examples
    --------
    Define Gaussian linebroadening:
    >>> linebroadening_Hz = 0

    # Generate linebroadened CSI dataset
    >>> spec_data_00Hz = luca_csi.multi_dim_linebroadening(lb=linebroadening_Hz)

    Specify metabolites for fitting:
    >>> metabs = ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"]

    Initialize fit parameters and bounds:
    >>> fit_params = {
    >>>    "lb": linebroadening_Hz,
    >>>    "fwhm_1.value":0.5, "fwhm_2.value":0.5,
    >>>    "fwhm_3.value":0.5, "fwhm_4.value":0.5,
    >>>    "fwhm_1.bounds":(0.2, 2), "fwhm_2.bounds":(0.2, 2),
    >>>    "fwhm_3.bounds":(0.2, 2), "fwhm_4.bounds":(0.2, 2)
    >>> }

    Perform fitting
    >>> fit_model_results_lb00Hz = luca_csi.fit_spectra_astropy2(
    >>>    metabs=metabs,
    >>>    fit_reps=[0,0],
    >>>    fit_params=fit_params,
    >>>    use_multiprocessing=True
    )
    ```

    """
    # import relevat functions from fitting utility:

    from tqdm.auto import tqdm
    import time

    if input_data is None:
        # ParaVision reco data:
        try:
            spec_data = data_obj.spec
        except:
            try:
                spec_data = data_obj.seq2d_reordered
            except:
                raise ValueError("No input_data provided")
    else:
        spec_data = input_data
        pass

    spec_data = np.abs(spec_data)  # / np.max(np.abs(spec_data))

    if fit_params is None:
        fit_params = {}

    # generate + fill fit_params
    fit_params = generate_default_fit_params_apy(
        data_obj=data_obj, input_data=spec_data, fit_params=fit_params
    )

    freq_range = get_freq_axis(scan=data_obj, unit="ppm", npoints=input_data.shape[0])

    # prepare data (linebroadening):
    ## ------------------------------------------------------------------------------
    lb = fit_params.get("lb", 0)

    # metabolites:
    metabs = fit_params.get("metabs")

    # number of peaks:
    n_peaks = np.size(metabs)

    # set output level (0=silent)
    verblevel = fit_params.get("verblevel", 0)

    # repetitions:
    fit_reps = fit_params.get("fit_reps", [0, -1])

    # all repetitions:
    if fit_reps == [0, -1]:
        # for init size of empty array:
        nr = spec_data.shape[4]

        # range over which to iterate
        nrange = range(nr)

    # only one repetition
    elif fit_reps[0] == fit_reps[1]:
        nrange = range(fit_reps[0], fit_reps[1] + 1)
        nr = 1

    else:
        nr = fit_reps[1] - fit_reps[0] + 1
        nrange = range(fit_reps[0], fit_reps[1] + 1)

    # fitted_model = np.apply_along_axis(wrapper_fit_spec, axis=0, arr=spec_data)
    fit_model_results = np.empty(
        (
            1,
            spec_data.shape[1],
            spec_data.shape[2],
            spec_data.shape[3],
            spec_data.shape[4],
            spec_data.shape[5],
        ),
        dtype=object,
    )
    # Parallel computing to speed up fitting for multidimensional data
    if use_multiprocessing:
        from joblib import Parallel, delayed, cpu_count

        try:
            # try to set the number of usable cpu cores to the amount of
            # available cores
            if number_of_cpu_cores is None:
                # int-divide by 2 because hyperthreading cores don't count/help:
                number_of_cpu_cores = cpu_count() // 2

            else:
                pass
            logger.debug("Using %d cpu cores" % number_of_cpu_cores)
        except:
            use_multiprocessing = False
            number_of_cpu_cores = 1
            logger.warning("Using %d cpu cores" % number_of_cpu_cores)

    # to measure duration of fitting:
    st = time.time()

    if use_multiprocessing is True:
        # reshape data:
        reshaped_data = spec_data[:, :, :, :, nrange, :].reshape(spec_data.shape[0], -1)

        # Fixme tqdm has trouble with CPU bound parallel operations...
        index_list = tqdm(
            range(reshaped_data.shape[1]),
            desc="Fitting progress on %d cores" % number_of_cpu_cores,
            leave=True,
        )

        f_init = generate_fit_init_apy(
            data_obj=data_obj,
            input_data=reshaped_data,
            fit_params=fit_params,
        )

        # Prepare the arguments for each function call
        args = [
            (
                data_obj,
                np.squeeze(reshaped_data[:, i]),
                f_init,
                freq_range,
                verblevel,
                fit_params,
            )
            for i in index_list
        ]

        results = Parallel(n_jobs=number_of_cpu_cores)(
            delayed(fit_single_point_apy)(*arg) for arg in args
        )

        # reshape fit results (still a sol model) to original data shape
        fit_model_results[:, :, :, :, nrange, :] = np.reshape(
            results,
            [
                1,
                spec_data.shape[1],
                spec_data.shape[2],
                spec_data.shape[3],
                nr,
                spec_data.shape[5],
            ],
        )

    else:
        # Initialize the tqdm object for the outermost loop
        pbar = tqdm(
            range(spec_data.shape[1] * spec_data.shape[2] * spec_data.shape[3] * nr),
            desc="Progress: ",
            leave=True,
        )
        from copy import deepcopy

        f_init = generate_fit_init_apy(
            data_obj=data_obj,
            input_data=spec_data,
            fit_params=fit_params,
        )

        for r in nrange:
            for x in range(spec_data.shape[2]):
                for y in range(spec_data.shape[3]):
                    # Check if there are any non-finite values in the data
                    if np.any(~np.isfinite(np.squeeze(spec_data[:, 0, x, y, r, 0]))):
                        logger.error("There are non-finite values in the data.")
                    else:
                        pass

                    fit_model_result = fit_single_point_apy(
                        data_obj=data_obj,
                        data_point=np.squeeze(spec_data[:, 0, x, y, r, 0]),
                        f_init=f_init,
                        freq_range=freq_range,
                        verblevel=verblevel,
                        fit_params=fit_params,
                    )
                    fit_model_results[0, 0, x, y, r, 0] = fit_model_result
                    pbar.update()

        pbar.close()
    if fit_params["show_tqdm"]:
        # print(fit_params)
        # print(f"Fitting took: {np.round(time.time() - st, 2)}s")
        pass
    else:
        pass
    fit_results = {}
    fit_results["data"] = fit_model_results
    fit_results["metabs"] = metabs
    fit_results["nrange"] = nrange
    fit_results["freq_range"] = freq_range
    fit_results["lb"] = lb
    return fit_results


def extract_fit_results_apy(
    data_obj=None, fit_params=None, fit_results=None, metabs=None
):
    """
    Returns the fitted spectra and maps of amplitude, frequency, fwhm, background, AUC and AUC without background for
    the metabolites metabs when the fit_results (output from fit_spectra_astropy2) are passed in

    Parameters
    ----------
    metabs: string
        list of metabolites of which spectra and maps should be generated
    fit_results: ND array
        the fitted model output (output from fit_spectra_astropy2)
    freq_range: 1D array
        array of spectral frequencies that was fitted against (default get_ppm())

    Returns
    -------
    A ND Array of fit spectra (same dimensions as seq2d_reordered, Maps of fit-parameters of the different metabolites

    Examples:
    -------
    # define Gaussian linebroadening:
    linebroadening_Hz = 0

    # Generate linebroadened CSI dataset (for later comparison)
    csi_data_00Hz = luca_csi.multi_dim_linebroadening(lb=linebroadening_Hz)

    # which metabolites should be fit:
    metabs = ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"]

    # init fit parameters + bounds:
    # init fit parameters + bounds:
    fit_params = {"lb": linebroadening_Hz,
                  "fwhm_1.value":0.5,
                  "fwhm_2.value":0.5,
                  "fwhm_3.value":0.5,
                  "fwhm_4.value":0.5,
                  "fwhm_1.bounds":(0.2, 2),
                  "fwhm_2.bounds":(0.2, 2),
                  "fwhm_3.bounds":(0.2, 2),
                  "fwhm_4.bounds":(0.2, 2)}


    # perform fitting
    fit_model_results_lb00Hz = luca_csi.fit_spectra_astropy2(metabs=metabs,
                                                             fit_reps=[0,0],
                                                             fit_params=fit_params,
                                                             use_multiprocessing=True)


    # Extract spectra and fit results (maps of amplitude, frequency, fullwidth-halfmax ...
    fit_spectra_00Hz, fit_res_00Hz = luca_csi.extract_fit_results(metabs=metabs,
                                                                  fit_results=fit_model_results_lb00Hz)

    """
    if fit_results is None:
        return None

    if metabs is None:
        metabs = fit_params["metabs"]
    else:
        if type(metabs) == "list":
            pass
        else:
            metabs = list((metabs,))

    freq_range = fit_params.get("freq_range", get_freq_axis(scan=data_obj, unit="ppm"))

    # exctract data:
    fit_model_results = fit_results["data"]

    # init empty dict to save the fit results per metabolite:
    fit_maps = {}

    # init empty array to fill with fitted spectra:
    fit_spectra = np.zeros(
        (
            freq_range.shape[0],
            fit_model_results.shape[1],
            fit_model_results.shape[2],
            fit_model_results.shape[3],
            fit_model_results.shape[4],
            fit_model_results.shape[5],
        )
    )

    from copy import deepcopy

    # Initialize the dictionary
    for k, m in enumerate(metabs):
        if m == "":
            m = str(k)
        fit_maps[m] = {
            "amplitude": np.zeros(fit_model_results.shape),
            "x_0": np.zeros(fit_model_results.shape),
            "fwhm": np.zeros(fit_model_results.shape),
            "nfloor": np.zeros(fit_model_results.shape),
            "AUC_background": np.zeros(fit_model_results.shape),
            "AUC_no_background": np.zeros(fit_model_results.shape),
        }

        if fit_params["lineshape"] == "Voigt":
            # Gaussian linewidth:
            fit_maps[m]["fwhm_G"] = np.zeros(fit_model_results.shape)

    # Loop through fit_model_results to populate the dictionary
    # Assuming fit_model_results is a numpy array, change the loop accordingly if it's a dictionary
    for c in range(fit_model_results.shape[5]):
        for z in range(fit_model_results.shape[1]):
            for r in range(fit_model_results.shape[4]):
                for x in range(fit_model_results.shape[2]):
                    for y in range(fit_model_results.shape[3]):
                        # You may need to adjust the indices based on your actual data shape
                        try:
                            fit_model = fit_model_results[0, z, x, y, r, c].copy()
                        except:
                            fit_model = None

                        # generate fit maps:
                        for k, m in enumerate(metabs):
                            if m == "":
                                m = str(k)
                            if fit_model is not None:
                                try:
                                    if (fit_params["lineshape"] == "Lorentzian") or (
                                        fit_params["lineshape"] == "sqrtLorentzian"
                                    ):
                                        fit_maps[m]["amplitude"][
                                            0, z, x, y, r, c
                                        ] = getattr(
                                            fit_model, f"amplitude_{k + 1}"
                                        ).value
                                        fit_maps[m]["x_0"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"x_0_{k + 1}"
                                        ).value
                                        fit_maps[m]["fwhm"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"fwhm_{k + 1}"
                                        ).value
                                        fit_maps[m]["nfloor"][
                                            0, z, x, y, r, c
                                        ] = getattr(fit_model, "nfloor").value

                                        f_temp = deepcopy(fit_model)
                                        for kk in range(np.size(metabs)):
                                            # Assuming the attributes are 1-indexed
                                            if k == kk:
                                                continue  # Skip the one you want to keep
                                            setattr(f_temp, f"amplitude_{kk+1}", 0)

                                        fit_maps[m]["AUC_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(f_temp(freq_range))
                                        fit_maps[m]["AUC_no_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(
                                            f_temp(freq_range)
                                            - np.squeeze(
                                                fit_maps[m]["nfloor"][0, z, x, y, r, c]
                                            )
                                        )
                                    elif fit_params["lineshape"] == "Voigt":
                                        fit_maps[m]["amplitude"][
                                            0, z, x, y, r, c
                                        ] = getattr(
                                            fit_model, f"amplitude_{k + 1}"
                                        ).value
                                        fit_maps[m]["x_0"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"x_0_{k + 1}"
                                        ).value
                                        fit_maps[m]["fwhm"][0, z, x, y, r, c] = getattr(
                                            fit_model, f"fwhm_{k + 1}"
                                        ).value
                                        fit_maps[m]["fwhm_G"][
                                            0, z, x, y, r, c
                                        ] = getattr(fit_model, f"fwhm_G_{k + 1}").value
                                        fit_maps[m]["nfloor"][
                                            0, z, x, y, r, c
                                        ] = getattr(fit_model, "nfloor").value

                                        f_temp = deepcopy(fit_model)
                                        for kk in range(np.size(metabs)):
                                            # Assuming the attributes are 1-indexed
                                            if k == kk:
                                                continue  # Skip the one you want to keep
                                            setattr(f_temp, f"amplitude_{kk + 1}", 0)

                                        fit_maps[m]["AUC_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(f_temp(freq_range))
                                        fit_maps[m]["AUC_no_background"][
                                            0, z, x, y, r, c
                                        ] = np.sum(
                                            f_temp(freq_range)
                                            - np.squeeze(
                                                fit_maps[m]["nfloor"][0, z, x, y, r, c]
                                            )
                                        )

                                except AttributeError as e:
                                    print(
                                        f"Attribute not found for metabolite {m} at indices {z}, {x}, {y}, {r}, {c}: {e}"
                                    )
                                    fit_maps[m]["amplitude"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["x_0"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["fwhm"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["nfloor"][0, z, x, y, r, c] = 0.0
                                    fit_maps[m]["AUC_background"][
                                        0, z, x, y, r, c
                                    ] = 0.0
                                    fit_maps[m]["AUC_no_background"][
                                        0, z, x, y, r, c
                                    ] = 0.0
                            else:
                                # print(
                                #     f"fit_model is None at indices {z}, {x}, {y}, {r}, {c}"
                                # )
                                fit_maps[m]["amplitude"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["x_0"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["fwhm"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["nfloor"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["AUC_background"][0, z, x, y, r, c] = 0.0
                                fit_maps[m]["AUC_no_background"][0, z, x, y, r, c] = 0.0

                        # generate fit_spectra:
                        for k, m in enumerate(fit_params["metabs"]):
                            # search independent of lower/uppercase:
                            if m.lower() in [x.lower() for x in metabs]:
                                pass
                            else:
                                # set the amplitude of the non-wanted peaks to 0
                                name = f"amplitude_{k + 1}"
                                getattr(
                                    fit_model,
                                    name,
                                ).value = 0

                        if fit_model is None:
                            fit_spectra[:, z, x, y, r, c] = np.zeros_like(freq_range)
                        else:
                            fit_spectra[:, z, x, y, r, c] = fit_model(freq_range)

    return fit_spectra, fit_maps


def generate_fit_init_apy(
    data_obj=None,
    input_data=None,
    fit_params=None,
):
    """
    Generate initial fit parameters for spectral data based on provided metabolites and constraints.

    This function dynamically sets up Lorentzian functions based on the number of metabolites (peaks)
    specified in the fit_params. It initializes the fitting parameters and sets bounds for each parameter
    based on provided constraints or defaults.

    Parameters
    ----------
    input_data : ndarray, optional
        Spectral data used to derive some of the initial parameters, especially for setting amplitude.
        If not provided, it defaults to class-based CSI data.

    fit_params : dict, optional
        Dictionary containing fit parameters and constraints. It can specify metabolites, linebroadening,
        initial values, bounds, and the expected lineshape. If certain parameters are not provided,
        the function uses default values.

        Example structure:
        {
            "metabs": ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"],
            ...
        }

    Returns
    -------
    f_init : object
        Initialized model for fitting.

    Notes
    -----
    - The function uses Astropy's custom_model for defining the dynamic sum of Lorentzians.
    - It's designed to be flexible with the number of peaks, allowing for easy scalability.

    """
    # define fit functions + init fir params::
    ## ------------------------------------------------------------------------------
    from astropy.modeling import models
    from ..utils.utils_spectroscopy import get_metab_cs_ppm
    from astropy.modeling.models import custom_model
    from ..utils.utils_general import get_gmr

    if input_data is None:
        if data_obj is not None:
            # ParaVision reco data:
            spec_data = data_obj.seq2d_reordered
        else:
            Warning("provide either input_data or spec_data")
    else:
        spec_data = input_data
        pass

    # fill fit_params dictonary with default values:
    fit_params = generate_default_fit_params_apy(
        data_obj=data_obj, input_data=input_data, fit_params=fit_params
    )

    # get metabolites:
    metabs = fit_params.get("metabs", ["", "", "", "", ""])

    # get linebroadening (use 0 if nothing was set)
    lb_Hz = fit_params.get("lb_Hz", 20)  # lb has to be >0 for lineshape Voigt to work

    if fit_params["freq_unit"] == "ppm":
        # linebroadening factor:  Hz --> ppm (13c, @ 7T)
        lb = lb_Hz / (get_gmr("13c") * 7)
    else:
        lb = lb_Hz

    fit_params["lb"] = lb

    # number of peaks:
    n_peaks = len(metabs)

    # Retrieve predefined ppm values or use default values <-> x_0 definition prefered over metab
    x_0_init = [
        fit_params.get(f"x_0_{k + 1}.value", get_metab_cs_ppm(metab=metab))
        for k, metab in enumerate(metabs)
    ]

    # Initialize amplitudes using either provided values or half the max of the data
    amplitude_init = [
        fit_params.get(f"amplitude_{k + 1}.value", np.max(np.abs(spec_data) / 2.0))
        for k in range(n_peaks)
    ]

    # Initialize FWHM (full width at half maximum) using provided values or default to 1
    fwhm_init = [fit_params.get(f"fwhm_{k + 1}.value", 1) for k in range(n_peaks)]

    # Initialize FWHM (full width at half maximum) using provided values or default to 1
    fwhm_G_init = [fit_params.get(f"fwhm_G_{k + 1}.value", lb) for k in range(n_peaks)]

    # Range for the bounds of x_0 values (defaults to 1)
    x_0_bounds_range = [
        fit_params.get(f"x_0_{k + 1}.bounds_range", 1) for k in range(n_peaks)
    ]

    lineshape = fit_params["lineshape"]
    if lineshape == "Lorentzian":
        # Initialize parameters based on predefined values or default ones

        # Retrieve predefined ppm values or use default values <-> x_0 definition prefered over metab
        # Initialize parameters based on predefined values or default ones

        # Define the model: A dynamic sum of Lorentzians
        @custom_model
        def sum_of_lorentzians(
            x,
            amplitude_1=1.0,
            x_0_1=10.0,
            fwhm_1=1.0,
            amplitude_2=1.0,
            x_0_2=30.0,
            fwhm_2=1.0,
            amplitude_3=1.0,
            x_0_3=40.0,
            fwhm_3=1.0,
            amplitude_4=1.0,
            x_0_4=50.0,
            fwhm_4=1.0,
            amplitude_5=1.0,
            x_0_5=60.0,
            fwhm_5=1.0,
            nfloor=0,
        ):
            # Initialize the total value to 0
            total = 0

            # Group parameters into lists for easier iteration
            amplitudes = [
                amplitude_1,
                amplitude_2,
                amplitude_3,
                amplitude_4,
                amplitude_5,
            ]
            x_0s = [x_0_1, x_0_2, x_0_3, x_0_4, x_0_5]
            fwhms = [fwhm_1, fwhm_2, fwhm_3, fwhm_4, fwhm_5]

            # Calculate the total sum of Lorentzians based on the number of peaks
            for i in range(n_peaks):
                total += models.Lorentz1D(
                    amplitude=amplitudes[i], x_0=x_0s[i], fwhm=fwhms[i]
                )(x)

            # Add the floor (baseline) to the total
            return total + nfloor

        # Initialize the model with provided or default parameters
        initial_args = {}
        for i in range(n_peaks):
            initial_args[f"amplitude_{i + 1}"] = amplitude_init[i]
            initial_args[f"x_0_{i + 1}"] = x_0_init[i]
            initial_args[f"fwhm_{i + 1}"] = fwhm_init[i]
        f_init = sum_of_lorentzians(**initial_args)

    elif lineshape == "sqrtLorentzian":
        from ..utils.utils_spectroscopy import squareLorentzian

        # Define the model: A dynamic sum of Lorentzians
        @custom_model
        def sum_of_sqrtlorentzians(
            x,
            amplitude_1=1.0,
            x_0_1=10.0,
            fwhm_1=1.0,
            amplitude_2=1.0,
            x_0_2=30.0,
            fwhm_2=1.0,
            amplitude_3=1.0,
            x_0_3=40.0,
            fwhm_3=1.0,
            amplitude_4=1.0,
            x_0_4=50.0,
            fwhm_4=1.0,
            amplitude_5=1.0,
            x_0_5=60.0,
            fwhm_5=1.0,
            nfloor=0,
        ):
            # Initialize the total value to 0
            total = 0

            # Group parameters into lists for easier iteration
            amplitudes = [
                amplitude_1,
                amplitude_2,
                amplitude_3,
                amplitude_4,
                amplitude_5,
            ]
            x_0s = [x_0_1, x_0_2, x_0_3, x_0_4, x_0_5]
            fwhms = [fwhm_1, fwhm_2, fwhm_3, fwhm_4, fwhm_5]

            # Calculate the total sum of Lorentzians based on the number of peaks
            for i in range(n_peaks):
                total += squareLorentzian(
                    amplitude=amplitudes[i], x_0=x_0s[i], fwhm=fwhms[i]
                )(x)

            # Add the floor (baseline) to the total
            return total + nfloor

        # Initialize the model with provided or default parameters
        initial_args = {}
        for i in range(n_peaks):
            initial_args[f"amplitude_{i + 1}"] = amplitude_init[i]
            initial_args[f"x_0_{i + 1}"] = x_0_init[i]
            initial_args[f"fwhm_{i + 1}"] = fwhm_init[i]
        f_init = sum_of_sqrtlorentzians(**initial_args)

    elif lineshape == "Voigt":
        # Initialize parameters based on predefined values or default ones
        # Define the model: A dynamic sum of Lorentzians
        @custom_model
        def sum_of_voigts(
            x,
            amplitude_1=1.0,
            x_0_1=10.0,
            fwhm_1=1.0,
            fwhm_G_1=1.0,
            amplitude_2=1.0,
            x_0_2=30.0,
            fwhm_2=1.0,
            fwhm_G_2=1.0,
            amplitude_3=1.0,
            x_0_3=40.0,
            fwhm_3=1.0,
            fwhm_G_3=1.0,
            amplitude_4=1.0,
            x_0_4=50.0,
            fwhm_4=1.0,
            fwhm_G_4=1.0,
            amplitude_5=1.0,
            x_0_5=60.0,
            fwhm_5=1.0,
            fwhm_G_5=1.0,
            nfloor=0,
        ):
            # Initialize the total value to 0
            total = 0

            # Group parameters into lists for easier iteration
            amplitudes = [
                amplitude_1,
                amplitude_2,
                amplitude_3,
                amplitude_4,
                amplitude_5,
            ]
            x_0s = [x_0_1, x_0_2, x_0_3, x_0_4, x_0_5]
            # Lorentzian line width:
            fwhm_Ls = [fwhm_1, fwhm_2, fwhm_3, fwhm_4, fwhm_5]
            # Gaussian linewidth:
            fwhm_Gs = [fwhm_G_1, fwhm_G_2, fwhm_G_3, fwhm_G_4, fwhm_G_5]

            # Calculate the total sum of Lorentzians based on the number of peaks
            for i in range(n_peaks):
                total += models.Voigt1D(
                    amplitude_L=amplitudes[i],
                    x_0=x_0s[i],
                    fwhm_L=fwhm_Ls[i],
                    fwhm_G=fwhm_Gs[i],
                )(x)

            # Add the floor (baseline) to the total
            return total + nfloor

        # Initialize the model with provided or default parameters
        initial_args = {}
        for i in range(n_peaks):
            initial_args[f"amplitude_{i + 1}"] = amplitude_init[i]
            initial_args[f"x_0_{i + 1}"] = x_0_init[i]
            initial_args[f"fwhm_{i + 1}"] = fwhm_init[i]
            initial_args[f"fwhm_G_{i + 1}"] = fwhm_G_init[i]
        f_init = sum_of_voigts(**initial_args)

    else:
        Warning(f"unknwon lineshape: {lineshape}")
        return None

    # Set parameter bounds directly on the model's attributes
    for i in range(n_peaks):
        # Fix amplitude values and set them to 0.0 for peaks that have no defined x_0_{i+1} (frequency)
        if x_0_init[i]:
            # Define bounds for amplitude, x_0, and fwhm for each peak
            getattr(f_init, f"amplitude_{i + 1}").bounds = fit_params.get(
                f"amplitude_{i + 1}.bounds", (0.0, np.inf)
            )
            getattr(f_init, f"x_0_{i + 1}").bounds = fit_params.get(
                f"x_0_{i + 1}.bounds",
                (
                    x_0_init[i] - x_0_bounds_range[i],
                    x_0_init[i] + x_0_bounds_range[i],
                ),
            )
            getattr(f_init, f"fwhm_{i + 1}").bounds = fit_params.get(
                f"fwhm_{i + 1}.bounds", (0.1, 10)
            )
            # fix Gaussian linewidth:
            if lineshape == "Voigt":
                getattr(f_init, f"fwhm_G_{i + 1}").fixed = fit_params.get(
                    f"fwhm_G_{i + 1}.fixed", True
                )
                # set Gaussian linewidth value:
                getattr(f_init, f"fwhm_G_{i + 1}").value = fit_params.get(
                    f"fwhm_G_{i + 1}.value", lb
                )

        else:
            getattr(f_init, f"amplitude_{i + 1}").fixed = fit_params.get(
                f"amplitude_{i + 1}.fixed", True
            )
            getattr(f_init, f"amplitude_{i + 1}").value = fit_params.get(
                f"amplitude_{i + 1}.value", 0.0
            )
            # the stuff between the 2 lines is just defined to avoid calling a non-defined parameter:
            # ------------------------------------------------------------
            # Define bounds for amplitude, x_0, and fwhm for each peak
            getattr(f_init, f"amplitude_{i + 1}").bounds = fit_params.get(
                f"amplitude_{i + 1}.bounds", (0.0, np.inf)
            )
            getattr(f_init, f"x_0_{i + 1}").bounds = fit_params.get(
                f"x_0_{i + 1}.bounds",
                (
                    0,  # simple fix, needs better solution later:
                    1,
                ),
            )
            getattr(f_init, f"fwhm_{i + 1}").bounds = fit_params.get(
                f"fwhm_{i + 1}.bounds", (0.1, 10)
            )
            # ------------------------------------------------------------

    # Set bounds for the floor (baseline) if it exists
    if hasattr(f_init, "nfloor"):
        f_init.nfloor.fixed = fit_params.get("nfloor.fixed", False)
    if f_init.nfloor.fixed:  # Set bounds for the floor (baseline) if it exists
        pass
    else:
        f_init.nfloor.bounds = fit_params.get(
            "nfloor.bounds", (0.0, 0.5 * amplitude_init[0])
        )

    return f_init


def generate_default_fit_params_apy(data_obj=None, input_data=None, fit_params=None):
    """
    Generate default parameters for fitting. Also helpful to get an overview of available
    parameters.

    Parameters
    ----------
    data_obj : object, optional
        Object containing data information, mainly used for fetching frequency range.

    input_data : ndarray, optional
        Spectral data used for certain default computations like area of data.

    fit_params : dict, optional
        Initial structure of fit parameters, to which this function can add or modify defaults.

    Returns
    -------
    fit_params : dict
        Updated dictionary with set default values.

    """
    if fit_params is None:
        fit_params = {}

    if input_data is None:
        if data_obj is not None:
            input_data = data_obj.seq2d
        else:
            pass

    # get units (default to ppm)
    fit_params["freq_unit"] = fit_params.get("freq_unit", "ppm")

    # get expected lineshape (use Lorentzian if nothing was set):
    fit_params["lineshape"] = fit_params.get("lineshape", "sqrtLorentzian")

    # decide wether to normalize the data during fit process to area = 1 (highly recommended)
    fit_params["normalize_data"] = fit_params.get("normalize_data", True)

    # get metabolites:
    metabs = fit_params.get("metabs", ["", "", "", "", ""])
    while len(metabs) < 5:
        metabs.append("")

    # write metabolite list into fit_params:
    fit_params["metabs"] = metabs

    if input_data is None:
        if data_obj is not None:
            nfreq_points = data_obj.method["PVM_SpecMatrix"]
        else:
            # default:
            nfreq_points = 200
    else:
        nfreq_points = input_data.shape[0]

    # get frequency range
    freq_range = fit_params.get(
        "fit_params.freq_range",
        get_freq_axis(
            scan=data_obj, unit=fit_params["freq_unit"], npoints=nfreq_points
        ),
    )
    fit_params["freq_range"] = freq_range

    if freq_range is None:
        Warning("freq_range is None, provide either scan or freq_range!")

    # set default fit method to Sequential Least Square Programming
    fit_method = fit_params.get("fit_method", "SLSQPLSQFitter")
    fit_params["fit_method"] = fit_method

    # epsilson (quasi step-size for SLSQP)
    if fit_method == "SLSQPLSQFitter":
        eps = fit_params.get("fit_eps", np.sqrt(np.finfo(float).eps))
        fit_params["fit_eps"] = eps
    else:  # for now use same epsilon for fitting:
        eps = fit_params.get("fit_eps", np.sqrt(np.finfo(float).eps))
        fit_params["fit_eps"] = eps

    # if the area should be "Normalized" This is really helpful when using SLSQPLSQFitter
    fit_params["fit_area_opt"] = fit_params.get("fit_area_opt", True)

    # set default max number of iterations to 200:
    fit_params["fit_maxiter"] = fit_params.get("fit_maxiter", 200)

    # set default linebroadening (in Hz) to 20Hz:
    fit_params["lb_Hz"] = fit_params.get("lb_Hz", 20)

    # whether a second round of fitting with the main peak subtracted should be performde:
    fit_params["fit_subtract_main_peak"] = fit_params.get(
        "fit_subtract_main_peak", False
    )

    # store area of spectrum:
    if input_data is None:
        fit_params["area_data"] = None
    else:
        fit_params["area_data"] = np.sum(input_data)

    # set b0 field strength:
    fit_params["b0"] = fit_params.get("b0", 7)

    # set nucleus:
    fit_params["nuc"] = fit_params.get("nuc", "13c")

    if input_data is None:
        # Maxiumum area value will be 100 (good for SLSQPLSQFitter)
        fit_params["fit_area_opt_factor"] = fit_params.get("fit_area_opt_factor", 1)

    else:
        # Maxiumum area value will be 100 (good for SLSQPLSQFitter)
        fit_params["fit_area_opt_factor"] = fit_params.get(
            "fit_area_opt_factor", np.max(np.sum(np.abs(input_data), axis=0)) / 100.0
        )

    # fit_params["fit_freq_range"] = fit_params.get("fit_freq_range", [fit_params["freq_range"][0], fit_params["freq_range"][-1]] )

    # initialize fit parameters:
    for k in range(len(metabs)):
        # full - width half - max( in ppm)
        fit_params[f"fwhm_{k + 1}.value"] = fit_params.get(f"fwhm_{k + 1}.value", 0.2)
        # bounds for full - width half - max ( in ppm)
        fit_params[f"fwhm_{k + 1}.bounds"] = fit_params.get(
            f"fwhm_{k + 1}.bounds", (0.1, 2)
        )
        # fwhm of Gaussian (in case lineshape is Voigt)
        fit_params[f"fwhm_G_{k + 1}.value"] = fit_params.get(
            f"fwhm_G_{k + 1}.value",
            fit_params["lb_Hz"] / (get_gmr(fit_params["nuc"]) * fit_params["b0"]),
        )
        # Range for the bounds of x_0 values (defaults to 1)
        fit_params[f"x_0_{k + 1}.bounds_range"] = fit_params.get(
            f"x_0_{k + 1}.bounds_range", 1
        )
    return fit_params


def match_fit_init_apy(
    data_obj=None,
    fit_init=None,
    input_data=None,
    fit_params=None,
):
    """
    Adjusts the initial fit parameters to match the input data.
    This is mainly to adjust amplitudes and other initial parameters closer to what's observed in the actual data.

    Parameters
    ----------
    data_obj : object, optional
        Object containing data information.

    fit_init : object
        Initial fitting parameters.

    input_data : ndarray
        1D spectral data.

    freq_range : ndarray, optional
        Frequency range corresponding to the input data.

    fit_params : dict, optional
        Dictionary of fit parameters.

    Returns
    -------
    fit_init : object
        Updated fitting parameters after matching with input data.

    """
    if input_data is None:
        Warning("input_data is None!")
        return fit_init

    metabs = fit_params.get("metabs")

    # get number of peaks
    n_peaks = np.size(metabs)

    # get Gaussian linewidth if it was applied:
    from ..utils.utils_general import get_gmr

    # get nucleus
    nuc = fit_params.get("nuc", "13c")

    # get field strength
    b0 = fit_params.get("b0", 7)

    # turn Hz to ppm
    lb = fit_params.get("lb", 0) / (get_gmr(nuc) * b0)

    freq_range = fit_params.get("freq_range", None)

    # take magnitude of data in case it's complex:
    if isinstance(input_data, complex):
        input_data = np.abs(input_data)

    # get frequency range:
    if freq_range is None:
        freq_range = get_freq_axis(
            scan=data_obj, unit="ppm", npoints=input_data.shape[0]
        )

    fit_params["freq_range"] = freq_range

    # define range in which should be fitted:
    fit_freq_range = fit_params.get("fit_freq_range", [freq_range[0], freq_range[-1]])

    # define range:
    fit_range = [0, 0]
    fit_range[0] = np.argmin(np.abs(freq_range - fit_freq_range[0]))
    fit_range[1] = np.argmin(np.abs(freq_range - fit_freq_range[-1]))
    fit_params["fit_range"] = fit_params.get("fit_range", fit_range)

    # get fitting method:
    fit_method = fit_params.get("fit_method", "SLSQPLSQFitter")

    # set fitting epsilon depending on data size:
    if fit_method == "SLSQPLSQFitter":
        eps = fit_params.get("fit_eps", np.sqrt(np.finfo(float).eps))
        if eps == np.sqrt(np.finfo(float).eps):
            fit_params["fit_eps"] = eps
        else:
            # instead of scaling data, scale stepsze (does not work really well):
            fit_params["fit_eps"] = np.sum(np.abs(input_data)) * eps

    # get expected lineshape (use Lorentzian if nothing was set):
    lineshape = fit_params.get("lineshape", "Lorentzian")

    if (
        lineshape == "Lorentzian"
        or lineshape == "sqrtLorentzian"
        or lineshape == "Voigt"
    ):
        # set bounds for amplitude:
        for k in range(n_peaks):  # Assuming k starts from 1
            if getattr(fit_init, f"amplitude_{k + 1}") > 0:
                # find maximum in range around frequency peak
                f = freq_to_index(
                    freq=getattr(fit_init, f"x_0_{k + 1}"),
                    freq_range=freq_range,
                )
                f_b = (
                    freq_to_index(
                        getattr(getattr(fit_init, f"x_0_{k + 1}"), "bounds")[0],
                        freq_range=freq_range,
                    ),
                    freq_to_index(
                        getattr(getattr(fit_init, f"x_0_{k + 1}"), "bounds")[1],
                        freq_range=freq_range,
                    ),
                )

                # set init value for amplitude:
                setattr(
                    fit_init,
                    f"amplitude_{k + 1}",
                    np.max(
                        input_data[
                            max(0, f - (f_b[1] - f_b[0])) : min(
                                len(freq_range) - 1, f + (f_b[1] - f_b[0])
                            )
                            + 1,
                        ]
                    )
                    - np.min(input_data),  # estimate for background
                )

                if lineshape == "Voigt":
                    # set init value for Gaussian linewidth:
                    setattr(
                        fit_init,
                        f"fwhm_G_{k + 1}",
                        lb,
                    )

        setattr(
            fit_init,
            "nfloor",
            np.min(input_data),
        )

    else:
        pass
    return fit_init

def get_pH_from_OMPD(peak_diff,err_peak_diff):
    """
    Calculates pH for Z-OMPD from the chemical shift difference in OMPD1 and OMPD 5 peaks
    Parameters
    ------
    peak_diff: float, in units of chemical shift / ppm. Difference between OMPD 1 and 5 peak.
    err_peak_diff: float, in unitls of chemical shift. Error in difference between OMPD 1 and 5 peak from fit.
    Returns
    -------
    pH
    """
    pKa = 6.5450
    A = 3.2975
    B = 4.4801
    err_ph = np.abs((A /(np.log(10)*(B-peak_diff)*(A+B-peak_diff)))*err_peak_diff)
    return pKa-np.log10((A/(peak_diff-B))-1),err_ph



def get_pH_from_OMPD(peak_diff,err_peak_diff):
    """
    Calculates pH for Z-OMPD from the chemical shift difference in OMPD1 and OMPD 5 peaks
    Parameters
    ------
    peak_diff: float, in units of chemical shift / ppm. Difference between OMPD 1 and 5 peak.
    err_peak_diff: float, in unitls of chemical shift. Error in difference between OMPD 1 and 5 peak from fit.
    Returns
    -------
    pH
    """
    pKa = 6.5450
    A = 3.2975
    B = 4.4801
    err_ph = np.abs((A /(np.log(10)*(B-peak_diff)*(A+B-peak_diff)))*err_peak_diff)
    return pKa-np.log10((A/(peak_diff-B))-1),err_ph


# -astropy-fitting----astropy-fitting----astropy-fitting----astropy-fitting----astropy-fitting----astropy-fitting----astr
