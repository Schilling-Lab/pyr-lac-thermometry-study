from ..brukerexp import BrukerExp
from .base_spectroscopy import BaseSpectroscopy
from ..utils import onlycallonce

import numpy as np
import matplotlib.pyplot as plt


class gNSPECT_MS(BaseSpectroscopy):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp, load_data=False)

        self.NSlices = self.method["SliceNum"]

        self.rawdata = self.Load_rawdatajob0_file()

        self.spack_length = len(self.rawdata) // self.NSlices

        self.fids = []
        self.specs = []
        self.complex_specs = []

        for i in range(self.NSlices):
            rawfid = self.rawdata[self.spack_length * i : self.spack_length * (i + 1)]

            spec, fid, cmplx_spec = self.get_spec(rawfid, LB=0)

            self.fids.append(fid)
            self.specs.append(np.squeeze(spec))
            self.complex_specs.append(cmplx_spec)

    def get_spec(self, fid, LB=0, cut_off=0):
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
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]

        time_ax = np.linspace(0, ac_time, ac_points - cut_off) / 1000
        sigma = 2.0 * np.pi * LB

        fids = np.zeros((NR, ac_points - cut_off), dtype=complex)
        spec = np.zeros((NR, ac_points - cut_off))
        complex_spec = np.zeros((NR, ac_points - cut_off), dtype=complex)

        rep_counter = 0
        while rep_counter < NR:
            test_spec = np.fft.fftshift(
                np.fft.fft(
                    fid[
                        cut_off
                        + rep_counter * ac_points : ac_points
                        + rep_counter * ac_points
                    ]
                    * np.exp(-sigma * time_ax)
                )
            )
            spec[rep_counter, :] = np.abs(test_spec)
            complex_spec[rep_counter, :] = test_spec
            fids[rep_counter, :] = fid[
                cut_off + rep_counter * ac_points : ac_points + rep_counter * ac_points
            ] * np.exp(-sigma * time_ax)
            rep_counter += 1

        return spec, fids, complex_spec

    @onlycallonce
    def linebroadening(self, LB):
        for i, spec in enumerate(self.specs):
            self.specs[i] = self.single_spec_linebroadening(spec, LB)

    def plot(self, units="Hz", labels=None):
        fig, ax = plt.subplots(figsize=(9, 5))

        xvals = self.Hz if units == "Hz" else self.ppm

        labels = labels if labels else np.arange(1, len(self.specs) + 1)

        for i, spec in enumerate(self.specs):
            ax.plot(xvals, spec, label=labels[i])

        if units == "Hz":
            ax.set_xlabel("Frequency [Hz]")
        else:
            ax.set_xlabel("Chemical Shift [ppm]")

        ax.set_ylabel("Amplitude [a.u.]")
        ax.legend()
        fig.tight_layout()
        plt.show()

        return fig, ax
