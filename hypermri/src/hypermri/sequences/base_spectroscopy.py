import logging

from ..brukerexp import BrukerExp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from IPython.display import display
from scipy.optimize import curve_fit
import ipywidgets as widgets
from ..utils.utils_spectroscopy import get_freq_axis
from ..utils.utils_general import calc_sampling_time_axis, calc_timeaxis

# initialize logger
from ..utils.utils_logging import LOG_MODES, init_default_logger

logger = init_default_logger(__name__)

logger.setLevel(LOG_MODES["Warning"])



class BaseSpectroscopy(BrukerExp):
    """ "Base class for most spectroscopy classes.

    This class provides some basic functionality which can be used for most
    spectroscopy sequences. It is encouraged to inherit this class for custom
    spectroscopy sequence classes.


    Attributes
    ----------
    ppm : np.ndarray
        The xaxis of the spectrum in chemical shift units (ppm).

    Hz : np.ndarray
        The xaxis of the spectrum in frequency units (Hz).

    Methods
    --------
    summary() : pd.Dataframe.Styler
        Prints a summary of all frequency related parameters.

    get_Hz(...)
        The underlying function behind the attribute self.Hz. When calling this
        method directly you can specify a cut_off value.
        See method docs for more info.

    get_ppm(...)
        The underlying function behind the attribute self.ppm. When calling this
        method directly you can specify a cut_off value.
        See method docs for more info.

    single_spec_linebroadening(...)
        Linebroadens a given spectrum by LB Herz and returns it.
        See method docs for more info.

    single_spec_find_peaks(...)
        Finds peaks of a given spec using scipy.signal.find_peaks().
        See method docs for more info.

    single_spec_plot(...):
        Plots a single given spectrum. See method docs for more info.
    """

    def __init__(self, path_or_BrukerExpObj, *args, **kwargs):
        """Accepts directory path or BrukerExp object as input."""
        super().__init__(path_or_BrukerExpObj, *args, **kwargs)

    @property
    def sampling_time_axis(self):
        return calc_sampling_time_axis(data_obj=self)

    @property
    def time_axis(self):
        return calc_timeaxis(data_obj=self, start_with_0=True)

    @property
    def ppm(self):
        return get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)

    @property
    def Hz(self):
        return get_freq_axis(scan=self, unit="Hz", cut_off=0, npoints=None)

    def summary(self):
        """Return summary table of frequency related parameter as pd.dataframe."""
        df = pd.DataFrame()

        paramlist = {
            "Reference Frequency [MHz]": "PVM_FrqRef",
            "Reference Frequency [ppm]": "PVM_FrqRefPpm",
            "Working Frequency [MHz]": "PVM_FrqWork",
            "Working Frequency [ppm]": "PVM_FrqWorkPpm",
            "Working Frequency Offset [Hz]": "PVM_FrqWorkOffset",
            "Working Frequency Offset [ppm]": "PVM_FrqWorkOffsetPpm",
        }

        for description, key in paramlist.items():
            try:
                df[description] = list(self.method[key])
            except KeyError:
                pass

        cols = ["Reference Frequency [MHz]"]
        df[cols] = df[df[cols] > 0][cols]
        df = df.dropna().T

        styles = [
            {"selector": "th", "props": [("font-size", "109%")]},  # tabular headers
            {"selector": "td", "props": [("font-size", "107%")]},  # tabular data
        ]

        df_styler = df.style.set_table_styles(styles)

        return df_styler

    def get_ppm(self, cut_off=0):
        """Get frequency axis of given spectroscopy scan in units of ppm.

        Returns ppm axis for spectroscopic measurements given a certain cut_off
        value at which fid will be cut off. The default cut off value of 70 points
        is usually sufficient as there is no signal left.

        Similair to get_Hz_axis() function.

        Parameters
        ----------
        cut_off: Default value is 0. After 'cut_off' points the signal is truncated.

        Returns
        -------
        ppm_axis : np.ndarray
            Frequency axis of measurement in units of ppm.
        """
        logging.critical('Method is outdated, should use ..utils.utils_spectroscopy.get_freq_axis()')
        center_ppm = float(self.method["PVM_FrqWorkPpm"][0])
        BW_ppm = float(self.method["PVM_SpecSW"])
        acq_points = int(self.method["PVM_SpecMatrix"]) - cut_off

        ppm_axis = np.linspace(
            center_ppm - BW_ppm / 2, center_ppm + BW_ppm / 2, acq_points
        )

        return ppm_axis

    def get_Hz(self, cut_off=0):
        """Get frequency axis of given spectroscopy scan in units of Hz.

        Returns Hz axis for spectroscopic measurements given a certain cut_off
        value at which fid will be cut off. The default cut off value of 70 points
        is usually sufficient as there is no signal left.

        Similair to get_Hz_axis() function.

        Parameters
        ----------
        cut_off : int
            Default value is 0. After 'cut_off' points the signal is truncated.

        Returns
        -------
        Hz_axis : np.ndarray
            Frequency axis of measurement in units of Hz.
        """
        BW_Hz = float(self.method["PVM_SpecSWH"])
        acq_points = int(self.method["PVM_SpecMatrix"]) - cut_off

        Hz_axis = np.linspace(-1, 1, acq_points) * (BW_Hz / 2)

        return Hz_axis

    def single_spec_linebroadening(self, spec, LB):
        """Applies linebroadening to a given single spectrum.

        Parameters
        ----------
        LB: linebroadening in Hz
        """
        assert (
            spec.ndim == 1
        ), f"Not a single spectrum! Found shape {spec.shape} instead of a vector."

        acq_time = self.method["PVM_SpecAcquisitionTime"]
        acq_points = self.method["PVM_SpecMatrix"]

        sigma = 2 * np.pi * LB
        time = np.linspace(0, acq_time, acq_points, endpoint=True) / 1000

        def LBconvolution(x):
            return np.abs(np.fft.fft(np.fft.ifft(x) * np.exp(-sigma * time)))

        return LBconvolution(spec)

    def single_spec_find_peaks(
        self,
        spec,
        searchrange=[],
        range_unit="Hz",
        height=0,
        **kwargs,
    ):
        """Wrapper of scipy.signal.find_peaks for a single spectrum.

        Parameter
        ---------
        spec : nd.array
            Single spectrum in which to find peaks.

        searchrange : optional, list of tuples
            If you don't want to search the entire spectrum for a peak but only
            search inside a certain spectal range, you can define here from
            where to where to look for peaks.
            If a single tuple (a,b) is given, this function will look for peaks
            in the range from a to b. The units of a and b are defined in the
            'range_unit' argument.
            Example:                single_spec_find_peaks(
                                        ...
                                        searchrange=(-300, -200),
                                        range_unit="Hz"
                                    )

            Will look for peaks from -300 to -200 Hz.

        range_unit : optional, str {'ppm', 'Hz'}
            In which unit the searchrange is given. Default is "Hz".

        height : optional, float
            Minimum peak height threshold. Default is 0. This is forwarded to the
            scipy.signal.find_peaks function.

        kwargs :
            Keywords forwarded to the scipy.signal.find_peaks function.

        Returns
        -------
        peaks : ndarray
            Indices of peaks in x that satisfy all given conditions.
        properties : dict
            A dictionary containing properties of the returned peaks.


        Note
        ------
        For more information at what arguments the function takes and what it
        returns, have a look at the official scipy.signal.find_peaks
        documentation:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        """
        spec = np.squeeze(spec)

        if spec.ndim > 1:
            raise AssertionError(
                "This method takes a single spectrum only. Found array of shape {spec.shape} instead."
            )
        # get the configured x-axis values
        if range_unit == "ppm":
            xaxis = self.ppm
        elif range_unit == "Hz":
            xaxis = self.Hz
        else:
            xaxis = np.arange(0, self.seq2d.shape[0], 1)

        # see if we have single continuous searchrange or multiple piecewise
        # ranges
        if len(searchrange):

            def find_closest_index(x):
                return np.argmin(np.abs(x - xaxis))

            a = find_closest_index(searchrange[0])
            b = find_closest_index(searchrange[1])

            assert a < b, f"Illegal range! {searchrange}-->i_min={a} > i_max={b}."

        else:
            # that is, x[0, -1] == x[:]
            a, b = 0, -1

        # store info on searchrange and range unit
        self.searchrangeunit = range_unit
        self.searchrange = searchrange
        self.searchrange_idx = [a, b]

        peaks, peakproperties = find_peaks(spec[a:b], height=height, **kwargs)

        return peaks, peakproperties

    def single_spec_plot(self, spec, unit="Hz", ax=None, figsize=(9, 5)):
        """Simple plotting function to plot the spectrum provided."""

        if ax is not None:
            fig, ax = ax.figure, ax
        else:
            fig, ax = plt.subplots(figsize=figsize)

        if unit == "Hz":
            xvals = self.Hz
            xlabel = "Frequency [Hz]"
        else:
            xvals = self.ppm
            xlabel = "Chemical Shift [ppm]"

        ax.plot(xvals, spec)
        ax.set_ylabel("Amplitude [a.u.]")
        ax.set_xlabel(xlabel)
        ax.set_title("")

        fig.tight_layout()

        plt.show()

        return fig, ax

    def multi_spec_plot(self, specs, unit="Hz"):
        """Simple plotting function to plot multiple repetitions."""

        num_reps = len(specs)
        fig, ax = plt.subplots(figsize=(9, 5))

        if unit == "Hz":
            xvals = self.Hz
            xlabel = "Frequency [Hz]"
        else:
            xvals = self.ppm
            xlabel = "Chemical Shift [ppm]"

        def plot_spec(num_rep):
            [line.remove() for line in ax.lines]
            ax.clear()

            ax.plot(xvals, specs[num_rep])
            ax.set_ylabel("Amplitude [a.u.]")
            ax.set_xlabel(xlabel)
            ax.set_title("")

            fig.tight_layout()

        slider_rep = widgets.IntSlider(
            value=num_reps // 2,
            min=0,
            max=num_reps - 1,
            description="Repetition: ",
        )

        out = widgets.interactive_output(
            plot_spec,
            {
                "num_rep": slider_rep,
            },
        )

        # This displays the Hbox containing the slider widgets.
        display(slider_rep, out)
        
    def fit_spectrum_astropy(
        self, data, peaks_to_fit=["Pyruvate", "Lactate"], xscale="ppm"
    ):
        """
        Fit spectrum to lorentzian models using astropy modelling.
        This is the recommended fitting routine.
        Parameters
        ----------
        data: bool or ndarray, 1D spectrum to be fitted
        peaks_to_fit: list of str, options are:
            'Pyruvate', 'Lactate', 'Alanin', 'PyruvateHydrate'
        xscale: str, default is 'ppm'
            options are 'Hz' or 'ppm'
        Returns
        -------
        fitted_model,init_model_fitted
        """
        from astropy.modeling import models, fitting
        import time
        from tqdm.auto import tqdm

        spectral_points = self.method["PVM_SpecMatrix"]
        spec_acq_time = self.method["PVM_SpecAcquisitionTime"]
        # remove 1D axis in self reconstructed data (mostly due to only one slice being present)
        data = np.squeeze(data)

        if xscale == "ppm":
            xaxis = get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)
            xlabel_text = r"$\sigma$ [ppm]"
        elif xscale == "Hz":
            xaxis = get_freq_axis(scan=self, unit="Hz", cut_off=0, npoints=None)
            xlabel_text = r"$\sigma$ [Hz]"
        else:
            xaxis = np.arange(0, self.data.shape[0], 1)
            xlabel_text = "array index"
        ref_frq_hz = self.method["PVM_FrqRef"][0]  # MHz
        ref_frq_ppm = self.method["PVM_FrqRefPpm"][0]
        center_frq_hz = self.method["PVM_FrqWork"][0]  # MHz
        center_frq_ppm = self.method["PVM_FrqWorkPpm"][0]
        center_frq_offset_hz = self.method["PVM_FrqWorkOffset"][0]  # Hz
        center_frq_offset_ppm = self.method["PVM_FrqWorkOffsetPpm"][0]
        # check if offsets are alright
        if center_frq_ppm - center_frq_offset_ppm == ref_frq_ppm:
            pass
        else:
            logger.warning(
                "Center frequency and center frequency offset do not match the reference frequency"
            )
        # Todo implement phasing of data

        # only selecting a specific number of repetitions to fit

        # norming to largest peak height in order to have a sense for the fitting parameters
        max_peak_value = np.max(data)
        pyruvate_params = {
            "amplitude": (0.05, max_peak_value),
            "fwhm": (0.1, 2),
            "x_0": (170, 172),
        }
        lactate_params = {
            "amplitude": (0.05, max_peak_value),
            "fwhm": (0.1, 0.5),
            "x_0": (182, 184),
        }
        alanin_params = {
            "amplitude": (0.05, max_peak_value),
            "fwhm": (0.1, 2),
            "x_0": (175, 177),
        }
        pyrhydrate_params = {
            "amplitude": (0.05, max_peak_value),
            "fwhm": (0.1, 2),
            "x_0": (178, 180),
        }
        # convert the x_0 values from ppm into Hz,in case we have a Hz axis
        all_params = {
            "Pyruvate": pyruvate_params,
            "Lactate": lactate_params,
            "PyruvateHydrate": pyrhydrate_params,
            "Alanin": alanin_params,
        }
        axis_change_flag = False
        if xscale == "Hz":
            # updating default porameters in case we fit in Hz
            conversion_fac = center_frq_offset_hz / center_frq_offset_ppm
            reference_freq = center_frq_offset_hz
            for metabolite in all_params.keys():
                for lorentzian_param in all_params[metabolite]:
                    if lorentzian_param == "x_0":
                        all_params[metabolite][lorentzian_param] = (
                            all_params[metabolite][lorentzian_param][0]
                            * conversion_fac,
                            all_params[metabolite][lorentzian_param][1]
                            * conversion_fac,
                        ) - center_frq_offset_hz
                        axis_change_flag = True
                    elif lorentzian_param == "fwhm":
                        all_params[metabolite][lorentzian_param] = (
                            all_params[metabolite][lorentzian_param][0]
                            * conversion_fac,
                            all_params[metabolite][lorentzian_param][1]
                            * conversion_fac,
                        ) - center_frq_hz
                        axis_change_flag = True
        else:
            pass
        if axis_change_flag is True:
            logger.debug(
                "Changed fitting default params from ppm to Hz. Now looking for Pyruvate at"
                + str(all_params["Pyruvate"]["x_0"])
                + "Hz"
            )

        start_time = time.time()

        models_seperately_fitted = []
        models_init = []
        test_fits = []
        test_models = []
        for peak in peaks_to_fit:
            params = all_params[peak]
            # finding starting params
            test_model = models.Lorentz1D(1, np.mean(params["x_0"]), 0.2)
            test_models.append(test_model)
            test_fitter = fitting.LMLSQFitter()

            test_fit = test_fitter(test_model, xaxis, data)
            test_fits.append(test_fit)
            amp_start, fwhm_start, x_0_start = test_fit.parameters
            if min(params["amplitude"]) <= amp_start <= max(params["amplitude"]):
                pass
            else:
                amp_start = np.mean(params["amplitude"])
            if min(params["fwhm"]) <= fwhm_start <= max(params["fwhm"]):
                pass
            else:
                fwhm_start = np.mean(params["fwhm"])
            if min(params["x_0"]) <= x_0_start <= max(params["x_0"]):
                pass
            else:
                x_0_start = np.mean(params["x_0"])

            model_init = models.Lorentz1D(
                amp_start,
                x_0_start,
                fwhm_start,
                bounds={
                    "amplitude": params["amplitude"],
                    "fwhm": params["fwhm"],
                    "x_0": params["x_0"],
                },
            )
            models_init.append(model_init)

            fitter = fitting.TRFLSQFitter(True)
            model_fitted = fitter(model_init, xaxis, data)
            models_seperately_fitted.append(model_fitted)

        fitter = fitting.TRFLSQFitter(True)

        init_model_fitted = fitter(np.sum(test_models), xaxis, data)
        fitted_model = fitter(np.sum(models_init), xaxis, data)

        # get the execution time
        et = time.time()
        elapsed_time = et - start_time
        # print excution time:
        logger.debug("Execution time: %f seconds" % elapsed_time)
        return fitted_model, init_model_fitted

    def fit_spectrum(
        self,
        spectrum,
        x_axis,
        peak_positions,
        peak_width,
        signal_threshold=1,
        plot=False,
        bg_region_first_spec=[0, -1],
        fit_fun="Lorentzian",
    ):
        """
        Fits lorentzian to a spectrum at the desired peak locations
        Parameters
        ----------
        spectrum: numpy array
            1D spectrum to be fitted.
        x_axis: numpy array
            frequency or ppm axis corresponding to the spectrum to be fitted.
        peak_positions: list
            Contains the peak positions where fits should occur in ppm or Hz depending which axis was chosen.
        peak_width: float
            ppm or Hz-range from peak center in which the peak should be fitted.
        signal_threshold: float
            SNR value below which peaks should not be fitted, default is 5.
        plot: bool
            Select if plotting for checking is needed, default is False.
        bg_region_first_spec: list
            indexes of background region, default is first and last, therefore taking the background of the whole
            spectrum.

        Returns
        -------
        peak_coeff: numpy array
            coefficients for a 3 parametric lorentzian fit model, shape: (NRepetitions, Number_of_peaks, 3)
            peak_coeff[0]: peak FWHM
            peak_coeff[1]: peak position (ppm)
            peak_coeff[2]: peak SNR

        peak_errors: numpy array
            errors for fit coefficients calculated from the covariance matrix, shape: (NRepetitions, Number_of_peaks, 3)
        """

        def find_range(axis, ppm):
            return np.argmin(np.abs(axis - ppm))

        if fit_fun == "Lorentzian":
            from ..utils.utils_spectroscopy import lorentzian

            fit_function = lorentzian
        elif fit_fun == "SqrtLorentzian":
            from ..utils.utils_spectroscopy import sqrt_lorentzian

            fit_function = sqrt_lorentzian
        else:
            from ..utils.utils_spectroscopy import lorentzian

            fit_function = lorentzian
        # Interpolating the ppm axis
        # x_axis_itp = np.linspace(np.min(x_axis), np.max(x_axis), 10000)
        # Norming to SNR
        # use first repetition where there should be no signal as reference region for SNR
        # just baseline correction
        # baseline correction
        # spectrum = spectrum - np.mean(spectrum[bg_region_first_spec[0]: bg_region_first_spec[1]])
        N_peaks = len(peak_positions)
        # defining region in which we will fit
        peak_coeff = np.zeros((N_peaks, 3)) * np.nan
        peak_covariance = np.zeros((N_peaks, 3, 3)) * np.nan
        peak_errors = np.zeros((N_peaks, 3)) * np.nan

        for peak_number, peak_center in enumerate(peak_positions):
            peak_roi = [
                find_range(x_axis, peak_center - peak_width),
                find_range(x_axis, peak_center + peak_width),
            ]
            try:
                (
                    peak_coeff[peak_number],
                    peak_covariance[peak_number],
                ) = curve_fit(
                    fit_function,
                    x_axis[peak_roi[0] : peak_roi[1]],
                    spectrum[peak_roi[0] : peak_roi[1]],
                    bounds=(
                        [
                            0.01,
                            peak_center - peak_width / 2.0,
                            np.min(spectrum),
                        ],
                        [600, peak_center + peak_width / 2.0, np.max(spectrum)],
                    ),
                )
                peak_errors[peak_number] = np.sqrt(
                    np.diag(peak_covariance[peak_number])
                )

            except RuntimeError:
                peak_coeff[peak_number] = np.nan
                peak_covariance[peak_number] = np.nan
                peak_errors[peak_number] = np.nan
            # clean up badly fitted peaks
        for peak in range(peak_coeff.shape[0]):
            peak_snr = peak_coeff[peak][2]
            peak_snr_error = peak_errors[peak][2]
            if peak_snr > signal_threshold:
                # peak needs to have an SNR greater than a certain value
                pass
            else:
                peak_coeff[peak] = [np.nan, np.nan, np.nan]

        if plot is True:
            fig, ax = plt.subplots(1, figsize=(12, 5))

            # Plotting for QA
            combined_fits = 0
            for peak_number, peak in enumerate(peak_positions):
                combined_fits += fit_function(
                    x_axis,
                    peak_coeff[peak_number][0],
                    peak_coeff[peak_number][1],
                    peak_coeff[peak_number][2],
                )
            ax.plot(
                x_axis,
                spectrum,
                linewidth="0.5",
                color="r",
                label="Data",
            )
            ax.plot(
                x_axis,
                combined_fits,
                linestyle="dashed",
                color="k",
                label="Lorentzian fit",
            )
            ax.set_ylim([np.min(spectrum), np.max(spectrum)])
            ax.set_ylabel("I [a.u]")
            ax.set_xlabel(r"$\sigma$[ppm]")

            ax.legend()

        else:
            pass

        return peak_coeff, peak_errors

    def find_phase_shift_dual_channel(
        self, LB=0, cut_off=0, plot=False, fid_ch1=None, fid_ch2=None, signal_domain="temporal"
    ):

        """
        Finds the optimal phase shift between data from two channels of a dual-channel coil to maximize the combined signal.

        This method identifies the voxel/repetition with the highest signal and iteratively adjusts the phase of channel 2
        relative to channel 1 to maximize the combined signal of both channels. This process is crucial for accurate
        spectroscopy data analysis, ensuring that signals from dual-channel coils are correctly aligned in phase.

        Parameters
        ----------
        LB : float, optional
            Line broadening applied to the spectra in Hz. This parameter is used to smooth the spectra, making it easier
            to identify the phase shift that maximizes the combined signal. Default is 0.
        cut_off : int, optional
            The number of initial points to be ignored from the recorded FID (Free Induction Decay) as they are typically
            just noise. This helps in focusing the analysis on the relevant signal part. Default is 0.
        plot : bool, optional
            If True, generates and displays a plot of the phase shift analysis and the resulting spectra for quality
            assurance. Default is False.
        fid_ch1 : ndarray, optional
            The FID data from channel 1. If None, the method attempts to automatically retrieve the FID data for both
            channels. Default is None.
        fid_ch2 : ndarray, optional
            The FID data from channel 2. Similar to `fid_ch1`, if None, the method attempts to automatically retrieve
            the FID data. Default is None.
        signal_domain : str, optional
            Specifies the domain of the provided FID data. Can be "temporal" for time-domain data or another value
            for frequency-domain data, in which case the method will perform an inverse FFT to convert it to the
            time domain. Default is "temporal".

        Returns
        -------
        final_phase : float
            The phase shift in degrees that maximizes the integral of the combined signal from both channels.

        Notes
        -----
        - The method assumes that the input FIDs are aligned in time and only differ by a constant phase shift.
        - The phase shift is determined by maximizing the integral of the absolute value of the combined signal,
          which is a common approach in NMR and MRI spectroscopy.
        - The method provides an option to visualize the effect of the phase correction through generated plots,
          which is useful for verifying the accuracy of the phase correction.
        """
        from ..utils.utils_general import calc_sampling_time_axis
        from ..utils.utils_spectroscopy import get_freq_axis as uts_get_freq_axis
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        time_ax = calc_sampling_time_axis(data_obj=self)
    
        if fid_ch1 is None and fid_ch2 is None:
            fid_ch1, fid_ch2 = self.get_two_channel_fids(cut_off)
        else:
            if signal_domain == "temporal":
                pass
            else:
                fid_ch1 = np.fft.ifft(np.fft.ifftshift(fid_ch1, axes=(0,)), axis=0)
                fid_ch2 = np.fft.ifft(np.fft.ifftshift(fid_ch2, axes=(0,)), axis=0)
        
    
        # finding out which of the two channels has the highest signal at which repetition
        # so that we phase the repetition that has the most signal
        # first integrate each repetition for each of the two channels seperately
        max_val_ch1 = np.max(np.sum(np.abs(fid_ch1), axis=0))
        max_val_ch2 = np.max(np.sum(np.abs(fid_ch2), axis=0))
        
        if max_val_ch1 > max_val_ch2:
            max_flat_index = np.argmax(np.sum(np.abs(fid_ch1), axis=0))
        elif max_val_ch2 > max_val_ch1:
            max_flat_index = np.argmax(np.sum(np.abs(fid_ch2), axis=0))
        else:
            max_flat_index = np.argmax(np.sum(np.abs(fid_ch2), axis=0))
        
        max_multi_index = np.unravel_index(max_flat_index, np.sum(np.abs(fid_ch2), axis=0, keepdims=True).shape)
        
        ch_1 = fid_ch1[:,max_multi_index[1],max_multi_index[2],max_multi_index[3],max_multi_index[4],0]
        ch_2 = fid_ch2[:,max_multi_index[1],max_multi_index[2],max_multi_index[3],max_multi_index[4],0]
        
        # Can apply linebroadening here for nicer plots
        sigma = 2 * np.pi * LB
        ch_1_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_1, axis=0), axes=(0,)))
        ch_2_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_2, axis=0), axes=(0,)))
        
        ppm = uts_get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)
    
        # finding optimal phase shift between channels
        Integrals = []
        phases = np.linspace(0, 360, 1000)
  
        for phase in phases:
            itgl = np.sum(np.abs(ch_1 * np.exp(1j * (phase * np.pi) / 180.0) + ch_2))
            Integrals.append(itgl)
            
        Integrals = np.array(Integrals)
        
        final_phase = phases[np.argmin(np.abs(Integrals - np.max(Integrals)))]
    
        
        # Optional plotting for QA
        if plot is True:
            fig, (ax, ax2) = plt.subplots(1, 2)
            ax.plot(phases, Integrals / np.max(Integrals))
            ax.set_xlabel(r"$\phi$ [rad]")
            ax.set_ylabel("Integral of ch_1 phaseshifted against ch_2")
            ax.vlines(
                final_phase,
                np.min(Integrals / np.max(Integrals)),
                1,
                color="orange",
            )
            ax.set_title(r"$\phi$ = " + str(np.round(final_phase, 1)) + "deg")
    
            # now plot spectra
            best_spec = np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        (ch_1 * np.exp(1j * (final_phase * np.pi) / 180.0) + ch_2)
                        * np.exp(-sigma * time_ax), axis=0), axes=(0,)))
    
            ax2.plot(ppm, ch_1_spec / np.max(best_spec), label="Ch_1 spec")
            ax2.plot(ppm, ch_2_spec / np.max(best_spec), label="Ch_2 spec")
            ax2.plot(
                ppm,
                best_spec / np.max(best_spec),
                label="Both Channels spec",
            )
            ax2.set_xlabel(r"$\sigma$[ppm]")
            ax2.set_ylabel("I[a.u.]")
            ax2.legend(loc="best", ncol=1)
            ax2.set_title("Spectra from dual channel data")
            ax2.set_xlim([np.max(ppm), np.min(ppm)])
            minimum = np.argmin(np.abs(Integrals - np.min(Integrals)))
            fig.suptitle("NSPECT")
            plt.tight_layout()
        else:
            pass
    
        return final_phase
    
    def apply_phase_shift_dual_channel(
        self, phase_shift=None, cut_off=0, fid_ch1=None, fid_ch2=None, signal_domain="temporal", plot=True
    ):
        """
        Applies phase correction by given value.
        Parameters
        ----------
        phase_shift: float,
            in degree, phase by which channel 1 is shifted against channel 2
        LB: float,
            linebroadening applied to spectra
        cut_off: int,
            number of points at beginning of fid that are left out.
        signal_domain: str,
            Define the domain that fid_ch1 and fid_ch2 are in.
        Returns
        -------
        phased_fid
        """
        NR = self.method["PVM_NRepetitions"]
        ac_points = self.method["PVM_SpecMatrix"]
    
        if (fid_ch1 is None) and (fid_ch2 is None):
            ch_1 = self.fid_ch1
            ch_2 = self.fid_ch2
        else:
            ch_1 = fid_ch1
            ch_2 = fid_ch2
        
        if phase_shift is None:
            phase_shift = self.find_phase_shift_dual_channel(
                cut_off=cut_off, plot=plot, fid_ch1=fid_ch1, fid_ch2=fid_ch2, signal_domain=signal_domain,
            )
        else:
            pass
            
        if signal_domain=="temporal":
            pass
        else:
            ch_1 = np.fft.ifft(np.fft.ifftshift(ch_1, axes=(0,)), axis=0)
            ch_2 = np.fft.ifft(np.fft.ifftshift(ch_2, axes=(0,)), axis=0)

        phased_fid = np.zeros(fid_ch1.shape, dtype=complex)
        for n in np.arange(0, NR):
            phased_fid[:, :, :, :, n, :] = (
                ch_1[:, :, :, :, n, :] * np.exp(1j * (phase_shift * np.pi) / 180.0)
                + ch_2[:, :, :, :, n, :]
            )
        if signal_domain == "temporal":
            return phased_fid
        else:
            return np.fft.fftshift(np.fft.fft(phased_fid, axis=0), axes=(0,))
    
    