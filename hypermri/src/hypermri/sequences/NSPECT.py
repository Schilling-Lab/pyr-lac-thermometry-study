from .base_spectroscopy import BaseSpectroscopy
from ..brukerexp import BrukerExp
from ..utils.utils_logging import LOG_MODES, init_default_logger
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from ..utils.utils_spectroscopy import get_freq_axis

# initialize logger
logger = init_default_logger(__name__)

logger.setLevel(LOG_MODES["Critical"])


def NSPECT(path_or_BrukerExp):
    """Wrapper which decides wether to load the single or dualchannel NSPECT class.

    Accepts directory path or BrukerExp object as input.
    """
    if isinstance(path_or_BrukerExp, BrukerExp):
        path_or_BrukerExp = path_or_BrukerExp.path

    tmp_brukerexp = BrukerExp(path_or_BrukerExp, load_data=False)

    if tmp_brukerexp.n_receivers == 2:
        return NSPECT_DualChannel(path_or_BrukerExp)
    else:
        return NSPECT_SingleChannel(path_or_BrukerExp)


class NSPECT_SingleChannel(BaseSpectroscopy):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp)

        self.NR = self.method["PVM_NRepetitions"]
        self.spec, self.fids, self.complex_spec = self.get_spec()

    def get_spec(self, LB=0, cut_off=0):
        """
        Calculates spectra and ppm axis for non localized spectroscopy measurements with repetitions.
        Parameters
        ----------
        LB : float, optional
            linebroadening applied in Hz, default is 0
        cut_off : float, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        spec : array
            magnitude spectra in a n-dimensional array
        fids: array
            linebroadened fids in a n-dimensional array
        complex_spec: array
            complex spectra in n-dimensional array
        """

        # if we dont have rawdatajob0 file we load fid
        if len(self.rawdatajob0) == 0:
            fid = self.fid
        else:
            fid = self.rawdatajob0

        NR = self.method["PVM_NRepetitions"]
        ac_points = self.method["PVM_SpecMatrix"]
        ac_time = self.method["PVM_SpecAcquisitionTime"]

        time_ax = np.linspace(0, ac_time, ac_points)[cut_off:] / 1000
        sigma = 2.0 * np.pi * LB

        fids = np.zeros((NR, ac_points - cut_off), dtype=complex)
        spec = np.zeros((NR, ac_points - cut_off))
        complex_spec = np.zeros((NR, ac_points - cut_off), dtype=complex)

        ffts = lambda x: np.fft.fftshift(np.fft.fft(x))

        for rep in range(NR):
            tmp_spec = ffts(
                fid[cut_off + rep * ac_points : ac_points + rep * ac_points]
                * np.exp(-sigma * time_ax)
            )

            complex_spec[rep, :] = tmp_spec

            fids[rep, :] = fid[
                cut_off + rep * ac_points : ac_points + rep * ac_points
            ] * np.exp(-sigma * time_ax)

        spec = np.abs(complex_spec)

        s = np.squeeze
        return s(spec), s(fids), s(complex_spec)

    # from warnings import deprecated
    # @deprecated("Use get_freq_axis instead") # will be possible to use in 3.13
    def get_ppm(self, cut_off=0, freq_shift_ppm=0):
        """Get frequency axis of given spectroscopy scan in units of ppm.

        Returns ppm axis for spectroscopic measurements given a certain cut_off
        value at which fid will be cut off. The default cut off value of 70 points
        is usually sufficient as there is no signal left.

        Similair to get_Hz_axis() function.

        Parameters
        ----------
        cut_off: int
            after 'cut_off' points the signal is truncated.

        freq_shift_ppm : float
            frequency shift in ppm due to erroneous frequency calibration

        Returns
        -------
        ppm_axis : np.ndarray
            frequency axis of measurement in units of ppm.
        """

        center_ppm = float(self.method["PVM_FrqWorkPpm"][0]) + freq_shift_ppm
        BW_ppm = float(self.method["PVM_SpecSW"])
        acq_points = int(self.method["PVM_SpecMatrix"]) - cut_off

        ppm_axis = np.linspace(
            center_ppm - BW_ppm / 2, center_ppm + BW_ppm / 2, acq_points
        )

        return ppm_axis

    def linebroadening(self, LB):
        if self.NR == 1:
            self.spec = self.single_spec_linebroadening(self.spec, LB)
        else:
            for i, spec in enumerate(self.spec):
                self.spec[i] = self.single_spec_linebroadening(spec, LB)

    def plot(self, unit="Hz", **kwargs):
        if self.NR > 1:
            return self.plot_multi_rep(unit=unit, **kwargs)
        else:
            return self.plot_single_rep(unit=unit, **kwargs)

    # analysis for dual channel
    def plot_multi_rep(self, unit="ppm", linebroadening=None):
        """
        Plots dual channel data for a NSPECT or Singlepulse sequence interactively
        Parameters
        ----------
        linebroadening: float, optional
            Linebroadening applied to spectra in Hz, default is 0.

        Returns
        -------
        """
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]
        time_ax = (
            np.linspace(0, ac_time, ac_points - 0) / 1000
        )  # removed cut_off, wrong calculation anyways

        if linebroadening is not None:
            spec, _, _ = self.get_spec(linebroadening)

            sigma = 2.0 * np.pi * linebroadening
            fids = self.fids * np.exp(-sigma * time_ax)
            print(
                "Linebroadening is only applied for this plot. To apply it permanently use NSPECT.linebroadening(...)"
            )
        else:
            spec = self.spec
            fids = self.fids

        if unit == "Hz":
            xvals = self.Hz
            xlabel = "Frequency [Hz]"
        else:
            xvals = self.ppm
            xlabel = "Chemical Shift [ppm]"

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        @widgets.interact(rep=(0, NR - 1, 1))
        def update(rep=0):
            [l.remove() for l in ax[0].lines]

            ax[0].plot(
                time_ax,
                np.real(fids[rep, :]),
                label="Re",
                color="r",
            )
            ax[0].plot(
                time_ax,
                np.imag(fids[rep, :]),
                label="Im",
                color="b",
            )
            ax[0].set_title("FID")

            [l.remove() for l in ax[1].lines]

            ax[1].plot(xvals, spec[rep, :], label="Spectrum", color="k")
            ax[1].set_title("Spectra")
            ax[1].set_xlabel(xlabel)

        ax[0].legend()
        # ax[1].legend()
        fig.suptitle("Linebroadening " + str(linebroadening) + " Hz")

    def plot_single_rep(self, unit="Hz", **kwargs):
        return self.single_spec_plot(self.spec, unit, **kwargs)

    def phase_correct_fid(self, number=0):
        """
        Interactively allows phasing of fids.
        Parameters
        ----------
        number: int, number of the fid that should be phased, in case there are repetitons.
        """
        fid = self.fids[number, :]
        spec_complex = np.fft.fftshift(np.fft.fft(fid))
        # perform a baseline correction
        spec_base_line_corr = spec_complex - np.mean(spec_complex)
        # phase the real spectrum
        Integrals_th = []
        phases = np.linspace(0, 360, 50)
        for phase in phases:
            itgl = np.sum(
                np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0))
            )
            Integrals_th.append(itgl)
        initial_guess_phase = phases[
            np.argmin(np.abs(Integrals_th - np.max(Integrals_th)))
        ]

        fig, ax = plt.subplots(1, figsize=(12, 4), tight_layout=True)

        (line_real,) = ax.plot(
            np.real(
                spec_base_line_corr * np.exp(1j * (initial_guess_phase * np.pi) / 180.0)
            ),
            label="Real-phased",
            color="k",
        )
        (line_abs,) = ax.plot(
            np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr)),
            label="Magnitude",
            color="r",
        )
        ax.hlines(
            0,
            0,
            len(spec_base_line_corr),
            linestyles="dashed",
            alpha=0.3,
            color="b",
            label="Baseline",
        )
        # ax.fill_between([8000, 9000],np.min(np.real(spec_base_line_corr)),np.max(np.real(spec_base_line_corr)), alpha=0.3, color='C2', label='Background')
        ax.set_xlabel("Points")
        ax.set_ylabel("I [a.u.]")
        ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")
        ax.legend()

        @widgets.interact(phase=(0, 360, 0.1))
        def update(phase):
            line_real.set_ydata(
                np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0))
            )
            line_abs.set_ydata(
                np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr))
            )
            ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")

    def get_one_channel_fids(self, cut_off=70):
        """
        Splits recorded data into two channels for measurements with a 2-channel receiver coil.
        Returns
        -------
        fid_ch1: np.array, shape = (ac_points, 1, 1, 1, NR, 1)
        fid_ch2: np.array, shape = (ac_points, 1, 1, 1, NR, 1)
        """

        if len(self.rawdatajob0) > 0:
            data = self.rawdatajob0
        else:
            data = self.fid

        NR = self.method["PVM_NRepetitions"]
        ac_points = self.method["PVM_SpecMatrix"]

        fid = np.zeros((NR, ac_points - cut_off), dtype=complex)
        for i in range(NR):
            fid[i, :] = data[i * ac_points + cut_off : i * ac_points + ac_points]

        fid = np.reshape(
            np.transpose(fid, (1, 0)), (ac_points - cut_off, 1, 1, 1, NR, 1)
        )
        return fid

    def full_reco(self, cut_off=0):
        """
        Do the full reconstruction with one call
        Parameters
        ----------
        cut_off: where to start processing the fid. 0 for custom sequences, 70 for Bruker sequences

        Returns
        -------

        """
        # get frequency range (in ppm)
        self.freq_range = get_freq_axis(scan=self, unit="ppm", cut_off=cut_off)

        # get frequency range (in ppm)
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]

        # define time axis:
        dt = ac_time / ac_points
        self.time_axis = (
            np.linspace(0, (ac_points - cut_off) * dt, (ac_points - cut_off)) / 1000.0
        )

        # extract FIDs
        self.fid = self.get_one_channel_fids(cut_off=cut_off)

        # get spectra
        (
            self.freq_range,
            self.spec,
        ) = self.plot_spec_non_localized_spectroscopy_one_channel(
            cut_off=cut_off,
            fid=self.fid,
            return_complex=True,
        )

    def plot_spec_non_localized_spectroscopy_one_channel(
        self,
        cut_off=70,
        fid=None,
        return_complex=True,
    ):
        """
        Calculates spectra and ppm axis for non localized spectroscopy measurements with repetitions and dual
        channel data.
        Parameters
        ----------
        LB : float, optional
            linebroadening applied in Hz, default is 0
        cut_off : float, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        ppm_axis : np.array
            ppm-scale for spectra
        spec_phased : np.array
            spectra for each repetition calculated from combining both channels with the
            optimal phase according to find_phase_shift_dual_channel
        spec_ch1 : np.array
            spectra for each repetition for data from channel 1
        spec_ch2 : np.array
            spectra for each repetition for data from channel 2
        """

        ppm_axis = get_freq_axis(scan=self, unit="ppm", cut_off=cut_off, npoints=None)

        ffts = lambda x: np.fft.fftshift(np.fft.fft(x, axis=0), axes=0)

        if fid is None:
            spec = ffts(self.fid)
        else:
            spec = ffts(fid)
        if return_complex:
            return ppm_axis, spec
        else:
            return ppm_axis, np.abs(spec)

    # Analysis functions
    def calculate_T1(
        self,
        first_spec,
        last_spec,
        Sample_ID=0,
        savepath=None,
        peak_ppm=False,
        offset=False,
        input_data=None,
        FA=None,
        integration_width=50,
        guess_T1=70,
        show_all=True,
        ax_to_plot_to=False,
    ):
        """

        Arguments
        ---------

        first_spec : int
            Index of the first point for the T1 fit to use.

        last_spec : int
            Index of the last point for the T1 fit to use.

        show_all : bool, optional
            Shows all data points, points not used for the fit are in gray.
            Default is True. If set to false only the points used for the fit
            are shown.

        savepath : string, optional
            If provided, stores the fit image as follows:
            savepath + "Sample_" + str(Sample_ID) + "_t1_measurement_7T.png"

        Sample_ID : int, optional
            See savepath.
        """

        spec, fids, complex_spec = self.get_spec(LB=0, cut_off=70)
        ppm_axis_hyper = get_freq_axis(scan=self, unit="ppm", cut_off=70, npoints=None)

        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]

        # let's you choose FA to see the influence
        if FA is None:
            FA = float(self.method["ExcPulse1"][2])
        else:
            pass
        TR = float(self.method["PVM_RepetitionTime"]) / 1000

        time_ax = np.arange(TR * first_spec, last_spec * TR, TR)

        if peak_ppm:
            center_hyper = np.squeeze(np.where(spec - peak_ppm == 0))[1]
            center_ppm_hyper = peak_ppm

        else:
            center_hyper = np.squeeze(np.where(spec - np.max(spec) == 0))[1]

            center_ppm_hyper = ppm_axis_hyper[center_hyper]

        lower_bound_integration_ppm_hyper = np.abs(
            ppm_axis_hyper - (center_ppm_hyper - integration_width)
        )
        lower_bound_integration_index_hyper = np.argmin(
            lower_bound_integration_ppm_hyper
            - np.min(lower_bound_integration_ppm_hyper)
        )
        upper_bound_integration_ppm_hyper = np.abs(
            ppm_axis_hyper - (center_ppm_hyper + integration_width)
        )
        upper_bound_integration_index_hyper = np.argmin(
            upper_bound_integration_ppm_hyper
            - np.min(upper_bound_integration_ppm_hyper)
        )
        # from this we calculate the integrated peak region
        # sorted so that lower index is first
        integrated_peak_roi_hyper = [
            lower_bound_integration_index_hyper,
            upper_bound_integration_index_hyper,
        ]
        integrated_peak_roi_hyper.sort()
        SNR = []
        for n in range(first_spec, last_spec, 1):
            SNR.append(
                np.sum(
                    spec[n, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]]
                )
            )
        # replace SNR with input data if something was passed:
        if input_data is None:
            pass
        else:
            # assuming 1D:
            SNR = input_data[first_spec:last_spec]

        if offset == False:

            def exp(x, a, T1, off=0):
                return a * np.exp(-x / T1)

            coeff, err = curve_fit(exp, time_ax, SNR, p0=(np.max(SNR), 185, 0))

        else:

            def exp(x, a, T1, off):
                return a * np.exp(-x / T1) + off

            coeff, err = curve_fit(exp, time_ax, SNR, p0=(np.max(SNR), 70, SNR[-1]))

        T1error = np.sqrt(np.diag(err))[1]

        T1 = 1 / ((1 / coeff[1]) + (np.log(np.cos(FA * np.pi / 180)) / TR))
        print(
            "7T measurement -- T1= ",
            np.round(T1, 1),
            "plus minus ",
            np.round(T1error, 1),
            " s",
        )
        if ax_to_plot_to:
            ax = ax_to_plot_to
        else:
            fig, ax = plt.subplots(1)
        ax.clear()
        ax.scatter(time_ax, SNR, label="Data", color="r")
        ax.plot(
            time_ax,
            exp(time_ax, coeff[0], coeff[1], coeff[2]),
            label="Fit, T1 = "
            + str(np.round(T1, 1))
            + r"$\pm$"
            + str(np.round(T1error, 1))
            + " s",
        )
        ax.set_xlabel("Time since start of experiment [s] ")
        ax.set_ylabel("I [a.u.]")
        ax.set_title(
            "FA ="
            + str(FA)
            + "°, TR ="
            + str(TR)
            + " s, NR = "
            + str(last_spec - first_spec)
        )

        ax.legend()

        if not isinstance(savepath, type(None)):
            plt.savefig(
                savepath + "Sample_" + str(Sample_ID) + "_t1_measurement_7T.png"
            )
        return T1, T1error

    def Interactive_Fitting(self, first_spec, last_spec, ax_obj):
        """
        Returns interactive plot with sliders that enable adjustment of fitting bounds for T1 fit.
        Can be used to find suitable bounds for optimal T1 fit.
        Uses InversionRecovery_Calculate_T1_multi_peak function

         intervalls: list of ints
         Defines start and end point for each peak to integrate them.

         axlist: subplot axes
         Required for interactive plot.
         Number of required axes: 1 + number of peaks to fit

         Returns
         -------

         Interactive plot
        """
        # create sliders to adjust bounds
        a_slider = widgets.FloatRangeSlider(
            value=[0, 100000.0],
            min=0.0,
            max=10000000.0,
            step=1,
            description="a:",
            disabled=False,
            readout=True,
            readout_format=".1f",
        )

        T1_slider = widgets.FloatRangeSlider(
            value=[0, 70.0],
            min=0.0,
            max=10000.0,
            step=0.1,
            description="T1:",
            disabled=False,
            readout=True,
            readout_format=".1f",
        )

        ui_main = widgets.VBox(
            [a_slider, T1_slider],
            layout=widgets.Layout(display="flex"),
        )

        def plot(a_input, T1_input):
            # build bounds dict
            bounds = np.array(
                [
                    [a_input[0], T1_input[0]],
                    [a_input[1], T1_input[1]],
                ]
            )

            output = self.calculate_T1(first_spec, last_spec, bounds, ax_obj=ax_obj)

        out = widgets.interactive_output(
            plot,
            {
                "a_input": a_slider,
                "T1_input": T1_slider,
            },
        )

        display(ui_main, out)


class NSPECT_DualChannel(BaseSpectroscopy):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp)
        # extract FIDs:
        self.fid_ch1, self.fid_ch2 = self.get_two_channel_fids()
        # extract necessary phase shift to combine chanels:
        self.correct_phase = self.find_phase_shift_dual_channel(False)
        #
        self.phased_fid = self.apply_phase_shift_dual_channel(
            phase_shift=self.correct_phase
        )

        (
            self.freq_range,
            self.spec,
            self.spec_ch1,
            self.spec_ch2,
        ) = self.get_spec_non_localized_spectroscopy_dual_channel()

        self.complex_spec = np.zeros((self.fid_ch1.shape[0], self.fid_ch1.shape[1], self.fid_ch1.shape[2], self.fid_ch1.shape[3], self.fid_ch1.shape[4], 2), dtype=complex)

        self.complex_spec[:,:,:,:,:,0]=np.fft.fftshift(np.fft.fft(self.fid_ch1,axis=0),axes=(0,))[:,:,:,:,:,0]
        self.complex_spec[:, :, :, :, :, 1] = np.fft.fftshift(np.fft.fft(self.fid_ch2, axis=0), axes=(0,))[:,:,:,:,:,0]

    def full_reco(self, cut_off=0):
        """
        Do the full reconstruction with one call
        Parameters
        ----------
        cut_off: where to start processing the fid. 0 for custom sequences, 70 for Bruker sequences

        Returns
        -------

        """
        # get frequency range (in ppm)
        self.freq_range = get_freq_axis(scan=self, unit="ppm", cut_off=cut_off)

        # get frequency range (in ppm)
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        self.time_axis = np.linspace(0, ac_time, ac_points - cut_off) / 1000

        # extract FIDs
        self.fid_ch1, self.fid_ch2 = self.get_two_channel_fids(cut_off=cut_off)
        # find phase shift necessary to combine the channels
        self.correct_phase = self.find_phase_shift_dual_channel(
            fid_ch1=self.fid_ch1, fid_ch2=self.fid_ch2, cut_off=cut_off
        )
        # apply phase shift to FID
        self.phased_fid = self.apply_phase_shift_dual_channel(
            self.correct_phase,
            fid_ch1=self.fid_ch1,
            fid_ch2=self.fid_ch2,
            cut_off=cut_off,
        )
        # get spectra
        (
            self.freq_range,
            self.spec,
            self.spec_ch1,
            self.spec_ch2,
        ) = self.get_spec_non_localized_spectroscopy_dual_channel(
            LB=0,
            cut_off=cut_off,
            return_complex=True,
            fid_ch1=self.fid_ch1,
            fid_ch2=self.fid_ch2,
        )

    # dual channel functions
    def get_two_channel_fids(self, cut_off=70):

        """
        Splits recorded data into two channels for measurements with a 2-channel receiver coil.
        Returns
        -------
        fid_ch1: np.array, shape = (ac_points, 1, 1, 1, NR, 1)
        fid_ch2: np.array, shape = (ac_points, 1, 1, 1, NR, 1)
        """

        if len(self.rawdatajob0) > 0:
            data = self.rawdatajob0
        else:
            data = self.fid

        NR = self.method["PVM_NRepetitions"]
        ac_points = self.method["PVM_SpecMatrix"]

        fid_ch1 = np.zeros((NR, ac_points - cut_off), dtype=complex)
        fid_ch2 = np.zeros((NR, ac_points - cut_off), dtype=complex)
        for i in np.arange(0, NR * 2, 2):
            nn = int(i / 2)
            fid_ch1[nn, :] = data[i * ac_points + cut_off : i * ac_points + ac_points]

        # counter to accurately put data at right points, because we can not just divide
        # by 2 like we did for channel 1
        count = 0
        for i in np.arange(1, NR * 2, 2):
            fid_ch2[count, :] = data[
                i * ac_points + cut_off : i * ac_points + ac_points
            ]
            count += 1

        fid_ch1 = np.reshape(
            np.transpose(fid_ch1, (1, 0)), (ac_points - cut_off, 1, 1, 1, NR, 1)
        )
        fid_ch2 = np.reshape(
            np.transpose(fid_ch2, (1, 0)), (ac_points - cut_off, 1, 1, 1, NR, 1)
        )

        return fid_ch1, fid_ch2

    def find_phase_shift_dual_channel(
        self, LB=0, cut_off=70, plot=False, fid_ch1=None, fid_ch2=None
    ):
        """
        Finds the phase shift between data from a dual channel coil

        Parameters
        -------
        lb: float, optional
            Linebroadening applied to spectra in Hz.
        cut_off: int, optional
            Number of points that are left out for recorded fid as they are just noise at the beginning, default is 70.
        plot: bool, optional
            if a plot of the result is wanted for QA, can be turned to True

        Returns
        -------
        final_phase: float, phase in degree that maximizes integral of both channels
        """

        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        time_ax = np.linspace(0, ac_time, ac_points - cut_off) / 1000

        if fid_ch1 is None and fid_ch2 is None:
            fid_ch1, fid_ch2 = self.get_two_channel_fids(cut_off)
        else:
            pass

        # finding out which of the two channels has the highest signal at which repetition
        # so that we phase the repetition that has the most signal
        # first integrate each repetition for each of the two channels seperately
        integral_ch1 = []
        integral_ch2 = []
        for n in range(self.method["PVM_NRepetitions"]):
            integral_ch1.append(np.sum(np.abs(fid_ch1[:, :, :, :, n, :])))
            integral_ch2.append(np.sum(np.abs(fid_ch2[:, :, :, :, n, :])))
        # find out at which repetition the signal is maximized
        max_signal_rep_ch1 = np.where(np.abs(integral_ch1 - np.max(integral_ch1)) == 0)[
            0
        ][0]
        max_signal_rep_ch2 = np.where(np.abs(integral_ch2 - np.max(integral_ch2)) == 0)[
            0
        ][0]
        # check which channel has the larger difference (i.e. more signal)
        # we cant just compare the max values cause the background offset is different
        # i.e. channel 2 could have the same absolute intensity maximum but relatively
        # it has lower signal
        signal_diff_ch1 = np.max(integral_ch1) - np.min(integral_ch1)
        signal_diff_ch2 = np.max(integral_ch2) - np.min(integral_ch2)
        # selecting which index to use
        if signal_diff_ch1 > signal_diff_ch2:
            max_signal_rep = max_signal_rep_ch1
        else:
            max_signal_rep = max_signal_rep_ch2

        ch_1 = fid_ch1[:, :, :, :, max_signal_rep, :]
        ch_2 = fid_ch2[:, :, :, :, max_signal_rep, :]
        # Can apply linebroadening here for nicer plots
        sigma = 2 * np.pi * LB
        ch_1_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_1 * np.exp(-sigma * time_ax))))
        ch_2_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_2 * np.exp(-sigma * time_ax))))

        ppm = get_freq_axis(scan=self, unit="ppm", cut_off=cut_off, npoints=None)

        # finding optimal phase shift between channels
        Integrals = []
        phases = np.linspace(0, 360, 1000)
        for phase in phases:
            itgl = np.sum(np.abs(ch_1 * np.exp(1j * (phase * np.pi) / 180.0) + ch_2))
            Integrals.append(itgl)

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
                        * np.exp(-sigma * time_ax)
                    )
                )
            )

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
        self, phase_shift=None, cut_off=70, fid_ch1=None, fid_ch2=None
    ):
        """
        Applies phase correction by given value.
        Parameters
        ----------
        phase_shift: float, in degree, phase by which channel 1 is shifted against channel 2
        LB: float, linebroadening applied to spectra
        cut_off: int, number of points at beginning of fid that are left out.
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
            pass
        if phase_shift is None:
            phase_shift = self.find_phase_shift_dual_channel(
                cut_off=cut_off, plot=False, fid_ch1=fid_ch1, fid_ch2=fid_ch2
            )
        else:
            pass

        phased_fid = np.zeros((ac_points - cut_off, 1, 1, 1, NR, 1), dtype=complex)
        for n in np.arange(0, NR):
            phased_fid[:, :, :, :, n, :] = (
                ch_1[:, :, :, :, n, :] * np.exp(1j * (phase_shift * np.pi) / 180.0)
                + ch_2[:, :, :, :, n, :]
            )
        return phased_fid

    # FIXME: if one chooses a cut_off that is not 70, the function does not work
    # FIXME:  because the fid file is always cut off to 70 --> get_two_channel_fids
    # FIXME: re write this a bit more smartly
    def get_spec_non_localized_spectroscopy_dual_channel(
        self, LB=0, cut_off=70, return_complex=False, fid_ch1=None, fid_ch2=None
    ):
        """
        Calculates spectra and ppm axis for non localized spectroscopy measurements with repetitions and dual
        channel data.
        Parameters
        ----------
        LB : float, optional
            linebroadening applied in Hz, default is 0
        cut_off : float, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        ppm_axis : np.array
            ppm-scale for spectra
        spec_phased : np.array
            spectra for each repetition calculated from combining both channels with the
            optimal phase according to find_phase_shift_dual_channel
        spec_ch1 : np.array
            spectra for each repetition for data from channel 1
        spec_ch2 : np.array
            spectra for each repetition for data from channel 2
        """
        ffts = lambda x: np.fft.fftshift(np.fft.fft(x))
        ffts_ax1 = lambda x: np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)
        ppm_axis = get_freq_axis(scan=self, unit="ppm", cut_off=cut_off, npoints=None)
        old = False
        if old == False:
            ffts = lambda x: np.fft.fftshift(np.fft.fft(x, axis=0), axes=0)
            if (fid_ch1 is None) and (fid_ch2 is None):
                ch1 = self.fid_ch1
                ch2 = self.fid_ch2
            else:
                ch1 = fid_ch1
                ch2 = fid_ch2

            spec_ch1 = ffts(ch1)
            spec_ch2 = ffts(ch2)
            spec_phased = ffts(self.phased_fid)
        else:
            ac_time = self.method["PVM_SpecAcquisitionTime"]
            ac_points = self.method["PVM_SpecMatrix"]
            NR = self.method["PVM_NRepetitions"]
            time_ax = np.linspace(0, ac_time, ac_points - cut_off) / 1000
            sigma = 2.0 * np.pi * LB

            ppm_axis = get_freq_axis(
                scan=self, unit="ppm", cut_off=cut_off, npoints=None
            )

            # spec_phased = np.zeros((NR, ac_points - cut_off))
            # spec_ch1 = np.zeros((NR, ac_points - cut_off))
            # spec_ch2 = np.zeros((NR, ac_points - cut_off))

            ffts = lambda x: np.fft.fftshift(np.fft.fft(x))
            ffts_ax1 = lambda x: np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)

            spec_ch1, spec_ch2 = ffts_ax1(self.fid_ch1), ffts_ax1(self.fid_ch2)
            spec_phased = ffts_ax1(self.phased_fid)

            for rep_counter in range(0, NR, 1):
                spec_phased[rep_counter, :] = np.abs(
                    ffts(self.phased_fid[rep_counter, :] * np.exp(-sigma * time_ax))
                )
                spec_ch1[rep_counter, :] = np.abs(
                    ffts(self.fid_ch1[rep_counter, :] * np.exp(-sigma * time_ax))
                )
                spec_ch2[rep_counter, :] = np.abs(
                    ffts(self.fid_ch2[rep_counter, :] * np.exp(-sigma * time_ax))
                )
                if return_complex is False:
                    spec_phased[rep_counter, :] = np.abs(
                        np.fft.fftshift(
                            np.fft.fft(
                                self.phased_fid[rep_counter, :]
                                * np.exp(-sigma * time_ax)
                            )
                        )
                    )
                    spec_ch1[rep_counter, :] = np.abs(
                        np.fft.fftshift(
                            np.fft.fft(
                                self.fid_ch1[rep_counter, :] * np.exp(-sigma * time_ax)
                            )
                        )
                    )
                    spec_ch2[rep_counter, :] = np.abs(
                        np.fft.fftshift(
                            np.fft.fft(
                                self.fid_ch2[rep_counter, :] * np.exp(-sigma * time_ax)
                            )
                        )
                    )
                else:
                    spec_phased[rep_counter, :] = np.fft.fftshift(
                        np.fft.fft(
                            self.phased_fid[rep_counter, :] * np.exp(-sigma * time_ax)
                        )
                    )
                    spec_ch1[rep_counter, :] = np.fft.fftshift(
                        np.fft.fft(
                            self.fid_ch1[rep_counter, :] * np.exp(-sigma * time_ax)
                        )
                    )

                    spec_ch2[rep_counter, :] = np.fft.fftshift(
                        np.fft.fft(
                            self.fid_ch2[rep_counter, :] * np.exp(-sigma * time_ax)
                        )
                    )

        return ppm_axis, spec_phased, spec_ch1, spec_ch2

    # analysis for dual channel
    def plot_spec_non_localized_spectroscopy_dual_channel(self, linebroadening=0):
        """
        Plots dual channel data for a NSPECT or Singlepulse sequence interactively
        Parameters
        ----------
        linebroadening: float, optional
            Linebroadening applied to spectra in Hz, default is 0.

        Returns
        -------
        """
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]
        time_ax = np.linspace(0, ac_time, ac_points - 70) / 1000
        sigma = 2.0 * np.pi * linebroadening

        (
            ppm_axis,
            spec_phased,
            spec_ch1,
            spec_ch2,
        ) = self.get_spec_non_localized_spectroscopy_dual_channel(linebroadening, 70)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        @widgets.interact(rep=(0, NR - 1, 1))
        def update(rep=0):
            [l.remove() for l in ax[0].lines]
            [l.remove() for l in ax[0].lines]
            [l.remove() for l in ax[0].lines]

            ax[0].plot(
                time_ax,
                np.real(self.fid_ch1[rep, :] * np.exp(-sigma * time_ax)),
                label="Re(Fid_Ch1)",
                color="r",
            )
            ax[0].plot(
                time_ax,
                np.real(self.fid_ch2[rep, :] * np.exp(-sigma * time_ax)),
                label="Re(Fid_Ch2)",
                color="b",
            )
            ax[0].plot(
                time_ax,
                np.real(self.phased_fid[rep, :] * np.exp(-sigma * time_ax)),
                label="Re(PhasedFid)",
                color="k",
            )
            ax[0].set_title("Fids")

            [l.remove() for l in ax[1].lines]
            [l.remove() for l in ax[1].lines]
            [l.remove() for l in ax[1].lines]

            ax[1].plot(ppm_axis, spec_ch1[rep, :], label="Ch1", color="r")
            ax[1].plot(ppm_axis, spec_ch2[rep, :], label="Ch2", color="b")
            ax[1].plot(
                ppm_axis, spec_phased[rep, :], label="Phased spectrum", color="k"
            )
            ax[1].set_title("Spectra")

        ax[0].legend()
        ax[1].legend()
        fig.suptitle("Linebroadening " + str(linebroadening) + " Hz")

    # Analysis functions
    def calculate_T1(
        self,
        first_spec,
        last_spec,
        peak_ppm=False,
        integration_width=50,
        guess_T1=70,
        show_all=True,
        Sample_ID=0,
        savepath=None,
    ):
        """
        Same method as in single channel nspect! could be inherited
        Arguments
        ---------

        first_spec : int
            Index of the first point for the T1 fit to use.

        last_spec : int
            Index of the last point for the T1 fit to use.

        show_all : bool, optional
            Shows all data points, points not used for the fit are in gray.
            Default is True. If set to false only the points used for the fit
            are shown.

        savepath : string, optional
            If provided, stores the fit image as follows:
            savepath + "Sample_" + str(Sample_ID) + "_t1_measurement_7T.png"

        Sample_ID : int, optional
            See savepath.
        """
        spec, fids, _ = self.get_spec(0)
        ppm_axis_hyper = self.get_ppm(cut_off=0)

        FA = float(self.method["ExcPulse1"][2])
        TR = float(self.method["PVM_RepetitionTime"]) / 1000

        time_ax = np.arange(TR * first_spec, last_spec * TR, TR)

        if peak_ppm:
            center_hyper = np.squeeze(np.where(spec - peak_ppm == 0))[1]
            center_ppm_hyper = peak_ppm
        else:
            center_hyper = np.squeeze(np.where(spec - np.max(spec) == 0))[1]

            center_ppm_hyper = ppm_axis_hyper[center_hyper]

        lower_bound_integration_ppm_hyper = np.abs(
            ppm_axis_hyper - (center_ppm_hyper - integration_width)
        )
        lower_bound_integration_index_hyper = np.argmin(
            lower_bound_integration_ppm_hyper
            - np.min(lower_bound_integration_ppm_hyper)
        )
        upper_bound_integration_ppm_hyper = np.abs(
            ppm_axis_hyper - (center_ppm_hyper + integration_width)
        )
        upper_bound_integration_index_hyper = np.argmin(
            upper_bound_integration_ppm_hyper
            - np.min(upper_bound_integration_ppm_hyper)
        )
        # from this we calculate the integrated peak region
        # sorted so that lower index is first
        integrated_peak_roi_hyper = [
            lower_bound_integration_index_hyper,
            upper_bound_integration_index_hyper,
        ]
        integrated_peak_roi_hyper.sort()
        SNR = []
        for n in range(first_spec, last_spec, 1):
            SNR.append(
                np.sum(
                    spec[n, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]]
                )
            )

        # norm to one for T1
        x = SNR / np.max(SNR)

        y_offset = np.mean(x[-5:])

        # SNR=SNR/SNR[0]
        def exp(x, a, T1):
            return a * np.exp(-x / T1) + y_offset

        coeff, err = curve_fit(exp, time_ax, x, p0=(1, guess_T1))
        T1error = np.sqrt(np.diag(err))[1]

        T1 = 1 / ((1 / coeff[1]) + (np.log(np.cos(FA * np.pi / 180)) / TR))

        print("*" * 50)
        print(f"Flip Angle Corrected T1 = {T1:.2f}±{T1error:.2f}s")
        print("*" * 50)

        fig, ax = plt.subplots(1)

        if first_spec and show_all:
            time_ax_first = np.arange(TR * 0, (first_spec + 1) * TR, TR)
            SNR_first = []
            for n in range(0, first_spec + 1, 1):
                SNR_first.append(
                    np.sum(
                        spec[
                            n,
                            integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1],
                        ]
                    )
                )
            SNR_first /= np.max(SNR)
            ax.scatter(time_ax_first, SNR_first, color="gray", alpha=0.6)

        if last_spec and show_all:
            time_ax_last = np.arange(TR * last_spec, len(spec) * TR, TR)
            SNR_last = []
            for n in range(last_spec, len(spec), 1):
                SNR_last.append(
                    np.sum(
                        spec[
                            n,
                            integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1],
                        ]
                    )
                )
            SNR_last /= np.max(SNR)
            ax.scatter(time_ax_last, SNR_last, color="gray", alpha=0.6)

        ax.scatter(time_ax, x, label="Data", color="r")
        ax.plot(
            time_ax,
            exp(time_ax, coeff[0], coeff[1]),
            label="Fit, T1 = "
            + str(np.round(coeff[1], 1))
            + r"$\pm$"
            + str(np.round(T1error, 1))
            + " s",
        )
        ax.set_xlabel("Time since start of experiment [s] ")
        ax.set_ylabel("I [a.u.]")
        ax.set_title(
            "NSPECT - FA ="
            + str(FA)
            + "°, TR ="
            + str(TR)
            + " s, NR = "
            + str(last_spec - first_spec)
        )

        ax.legend()

        if savepath:
            plt.savefig(
                savepath + "Sample_" + str(Sample_ID) + "_t1_measurement_7T.png"
            )
        return T1, T1error
