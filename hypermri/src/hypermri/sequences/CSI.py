from ..brukerexp import BrukerExp
from .base_spectroscopy import BaseSpectroscopy
from ..utils.utils_decorator import onlycallonce
from ..utils.utils_logging import LOG_MODES, init_default_logger
from ..utils.utils_spectroscopy import (
    lorentzian,
    freq_to_index,
    get_freq_axis,
    zeropad_data,
)
from ..utils.utils_general import (
    define_imageFOV_parameters,
    define_imagematrix_parameters,
    define_grid,
    get_extent,
    get_plotting_extent,
    img_interp,
    calc_timeaxis,
    load_plot_params,
    shift_anat,
    get_counter,
    add_counter,
)
from ..utils.utils_anatomical import add_scalebar

import numpy as np
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import norm
import cv2
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# initialize logger
logger = init_default_logger(__name__)

logger.setLevel(LOG_MODES["Critical"])


class CSI(BaseSpectroscopy):
    def __init__(self, path_or_BrukerExp, log_mode="warning"):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp)
        if path_or_BrukerExp == "dummy":
            return None

        # set up logging level
        logger.setLevel(LOG_MODES[log_mode])

        # write header file:
        from ..utils.utils_general import init_header, shift_image

        self.header = init_header(data_object=self)

        # reorder kspace points
        try:
            # sort complex k-space data:
            self.kspace_array = self.reorder_kspace(
                data_type="job",
                db=False,
            )

        except Exception as e:
            err_msg, err_type = str(e), e.__class__.__name__

            msg = f"[ExpNum={self.ExpNum}] The following error occured during "
            msg += f"self.reorder_kspace():\n    {err_type}: {err_msg}\n"

            self.kspace_array = None
            logger.critical(msg)

        try:
            # perform 3D Fourier Transform to transform temporal k-space
            # data into spatial-spectral domain
            # echoes-z-x-y-rep-chans (shift has to be perform on temporal direction to avoid strong "ringing" in spectra)
            self.kspace_array = shift_image(
                input_data=self.kspace_array,
                shift_method="phase",
                data_obj=self,
                domain="kspace",
            )

        # print("k-space shifted!")
        except Exception as e:
            err_msg, err_type = str(e), e.__class__.__name__

            msg = f"[ExpNum={self.ExpNum}] The following error occured during "
            msg += f"shift_kspace():\n    {err_type}: {err_msg}\n"

            logger.critical(msg)

        # perform fourier Transform:
        try:
            # perform 3D Fourier Transform to transform temporal k-space
            # data into spatial-spectral domain
            # echoes-z-x-y-rep-chans (shift has to be perform on temporal direction to avoid strong "ringing" in spectra)
            self.csi_image = self.csi_fft(
                input_data=np.fft.fftshift(self.kspace_array, axes=(0,)),
                axes=(0, 2, 3),
                ift=False,
            )
        except Exception as e:
            err_msg, err_type = str(e), e.__class__.__name__

            msg = f"[ExpNum={self.ExpNum}] The following error occured during "
            msg += f"self.csi_fft():\n    {err_type}: {err_msg}\n"

            logger.critical(msg)

        # testing
        try:
            # TODO: should adjust the reordering algorithm to do this...
            self.csi_image = np.roll(
                np.roll(self.csi_image, shift=-1, axis=2), shift=-1, axis=3
            )
        except Exception as e:
            err_msg, err_type = str(e), e.__class__.__name__

            msg = f"[ExpNum={self.ExpNum}] The following error occured during "
            msg += f"rolling():\n    {err_type}: {err_msg}\n"

            logger.critical(msg)

        try:
            # I dont fully understand why this is necessary, but if the matrix size is odd, a half-voxel shift has to be
            # applied

            # check if image dimensions are odd:
            if np.mod(self.csi_image.shape[2], 2) or np.mod(self.csi_image.shape[3], 2):
                from scipy.ndimage import shift

                shift_y = np.mod(self.csi_image.shape[2], 2) / 2.0
                shift_x = np.mod(self.csi_image.shape[3], 2) / 2.0
                self.csi_image = shift(
                    input=self.csi_image,
                    shift=[0.0, 0.0, shift_y, shift_x, 0.0, 0.0],
                    mode="wrap",
                )
        except:
            pass

        self.kspace_array = np.fft.ifftshift(
            np.fft.ifft(
                np.fft.ifftshift(
                    np.fft.ifftshift(
                        np.fft.ifft(np.fft.ifftshift(self.csi_image, axes=2), axis=2),
                        axes=2,
                    ),
                    axes=3,
                ),
                axis=3,
            ),
            axes=3,
        )

        # flip spectral dimension of 2dseq file
        self.seq2d = np.flip(self.seq2d, axis=0)

        # reorder seq2d to match fid - z - x - y - reps - channel order:
        self.seq2d_reordered = self.permute_seq2d(input_data=self.seq2d)
        # reorient complex, from rawdata reconstructed image to match anatomical
        self.csi_image = self.permute_csi_image(input_data=self.csi_image)
        # contains manipulated spectrum after call of self.linebroadening()
        self.seq2d_broadend = None
        # store peak infos, once self.find_peaks() was called
        self.peaks = dict()
        self.peakproperties = dict()

        # amount of search ranges, changed by self.find_peaks_multirange
        self.searchrangeunit = "ppm"
        self.searchranges = dict()
        self.searchranges_idx = dict()

        # add ROI dictionary to class:
        self.ROI = dict()  # keys will be ROI IDs

        self.Nslices = self.get_Nslices()

    @property
    def Nsearchranges(self):
        return len(self.peaks)

    @property
    def data(self):
        if self.seq2d_broadend is None:
            return self.seq2d
        return self.seq2d_broadend

    # needs further work!
    def full_reco(
        self,
        anatomical=None,
        cut_off=0,
        mirror_first_axis=False,
        mirror_first_and_second_axis=False,
    ):
        """
        Perform a full reconstruction of the CSI data, including reorientation and shifting of anatomical images.

        Parameters
        ----------
        anatomical : object, optional
            Anatomical image object that will be shifted (seq2d_oriented) to match the position of the CSI image.
        cut_off : int, optional
            Number of initial points of FID to cut off. Default is 0.
        mirror_first_axis : bool, optional
            Whether to mirror the first axis of the anatomical image. Default is False.
        mirror_first_and_second_axis : bool, optional
            Whether to mirror the first and second axes of the anatomical image. Default is False.

        Returns
        -------
        None

        Notes
        -----
        This function reorients the anatomical image to match the orientation of the CSI image and shifts the axial image.
        If the reconstruction does not work, it attempts to redo the complex CSI reconstruction.

        """
        # reorient anatomical image to match orientation of CSI image:
        from ..utils.utils_general import (
            reorient_anat,
            define_imageFOV_parameters,
            shift_image,
        )

        # reoriented seq2ds will be stored as seq2d_oriented in the data_objs:
        reorient_anat(
            data_obj=anatomical,
            mirror_first_axis=mirror_first_axis,
            mirror_first_and_second_axis=mirror_first_and_second_axis,
        )

        # shift axial image:
        if anatomical is not None:
            anatomical.seq2d_oriented = shift_anat(
                csi_obj=self,
                anat_obj=anatomical,
                use_scipy_shift=True,
                mirror_first_axis=mirror_first_axis,
                mirror_first_and_second_axis=mirror_first_and_second_axis,
            )
            anat_fov = define_imageFOV_parameters(data_obj=anatomical)
            csi_fov = define_imageFOV_parameters(data_obj=self)
            # if anat_fov[1:] != csi_fov[1:]:
            # logger.critical(
            # f"FOV of anatomical image={anat_fov[1:]} is not FOV of CSI: {csi_fov[1:]}!"
            # )
        else:
            pass

        # for some reason this reco is not working, even though it should be the same as above :/
        still_not_working = True
        if still_not_working:
            pass
        else:
            # redo complex csi reco:
            kspace_array = self.reorder_kspace(
                data_type="job",
                db=False,
                cut_off=cut_off,
            )

            self.kspace_array = shift_image(
                input_data=kspace_array,
                shift_method="phase",
                force_shift=True,
                data_obj=self,
                domain="kspace",
            )

            csi_image = self.csi_fft(
                input_data=np.fft.fftshift(self.kspace_array, axes=(0,)),
                axes=(0, 2, 3),
                ift=False,
            )

            csi_image = np.roll(np.roll(csi_image, shift=-1, axis=2), shift=-1, axis=3)
            self.csi_image = self.permute_csi_image(input_data=csi_image)

    def frequency_adjust(
        self,
        input_data=None,
        freq_peaks=None,
        freq_range=[-50, 50],
        start_rep=None,
        stop_rep=None,
        freq_axis=None,
        ppm_hz="Hz",
    ):
        """
        Is used to reduce the effects of B0 inhomogneities. The highest peak in
        the frequency range "range" around a peak "peaks" is shifted to align
        with the peak "peaks"

        Args:
            input_data (_type_, optional): _description_. Defaults to None.
            peaks (_type_, optional): _description_. Defaults to None.
            range (_type_, optional): _description_. Defaults to None.
        """
        # get size parameters:
        fidsz = self.method["PVM_SpecMatrix"]  # spectral points
        nr = self.method["PVM_NRepetitions"]  # number of repetitions
        ysz = self.method["PVM_Matrix"][1]  #
        xsz = self.method["PVM_Matrix"][0]

        if start_rep is None:
            start_rep = 0

        if stop_rep is None:
            stop_rep = nr - 1

        if input_data is None:
            try:
                input_data = self.csi_image
            except:
                input_data = self.csi_fft(self.kspace_array, axes=(0, 2, 3), ift=False)

        if freq_axis is None:
            freq_axis = get_freq_axis(
                scan=self, unit=ppm_hz, npoints=input_data.shape[0]
            )

        # get frequency resolution:
        freq_res = freq_axis[1] - freq_axis[0]

        if freq_peaks is None:
            freq_peaks = 0

        # translate range into indices:
        start_idx = np.argmin(np.abs(freq_axis - (freq_range[0] + freq_peaks)))
        stop_idx = np.argmin(np.abs(freq_axis - (freq_range[1] + freq_peaks)))
        peak_idx = np.argmin(np.abs(freq_axis - freq_peaks))

        # init empty array:
        shifted_data = np.zeros_like(input_data)
        freq_map = np.zeros((xsz, ysz, stop_rep - start_rep + 1))

        # perform shifting:
        for k in range(start_rep, stop_rep + 1, 1):
            for i in range(ysz):
                for j in range(xsz):
                    spec = input_data[:, 0, j, i, k, 0]
                    # get maximum intensity:
                    max_int_idx = start_idx + np.argmax(
                        np.abs(spec[start_idx:stop_idx])
                    )
                    # calculate the shift distance
                    shift_distance = peak_idx - max_int_idx
                    freq_map[j, i, k] = shift_distance * freq_res
                    # shift the spectrum data
                    shifted_data[:, 0, j, i, k, 0] = np.roll(spec, shift_distance)

        return shifted_data, freq_map

    def permute_seq2d(self, input_data=None):
        """This function is supposed to reorder the seq2d data
        The point of this is to have the seq2d in a default order:
        fid - z - x - y - rep - nchans. This makes processing much easier

        Args:
            input_data (array): N-D Array.

        Returns:
            N-D Array: reordered seq2d file
        """

        # decide which data to use:
        if input_data is None:
            seq2d_file = self.seq2d
            # check if seq2d data was already reordered:
            num_seq2d_permute_counts = get_counter(
                data_obj=self,
                counter_name="seq2d_permute_counts",
            )

            # was already reorderd:
            if num_seq2d_permute_counts > 0:
                Warning("seq2d was already reoriented!")
                return seq2d_file
        else:
            seq2d_file = input_data

        # get size parameters:
        # fidsz = self.method["PVM_SpecMatrix"] # spectral points
        # nr = self.method["PVM_NRepetitions"] # number of repetitions
        # ysz = self.method["PVM_Matrix"][1] #
        # xsz = self.method["PVM_Matrix"][0]
        SpatialDim = self.method["PVM_SpatDimEnum"]
        nchans = self.method["PVM_EncNReceivers"]

        # reordering depends on dimension:
        if SpatialDim == "<2D>":
            if nchans == 1:
                reordered_seq2d = seq2d_file[..., np.newaxis, np.newaxis]
            else:
                # change seems to be necessary (dont know exactly why)
                reordered_seq2d = seq2d_file[..., np.newaxis, np.newaxis]

            # fidsz - z - x - y - nr - nchans
            reordered_seq2d = np.transpose(reordered_seq2d, (0, 4, 1, 2, 3, 5))

        else:
            if nchans == 1:
                reordered_seq2d = seq2d_file[..., np.newaxis]
            else:
                reordered_seq2d = seq2d_file[...]
            reordered_seq2d = np.transpose(reordered_seq2d, (0, 4, 1, 2, 3, 5))

        patient_pos = self.acqp["ACQ_patient_pos"]
        read_orient = self.method["PVM_SPackArrReadOrient"]
        slice_orient = self.method["PVM_SPackArrSliceOrient"]

        if patient_pos == "Head_Supine":
            if read_orient == "H_F":
                reordered_seq2d = np.flip(reordered_seq2d, axis=2)

        if patient_pos == "Head_Supine":
            if slice_orient == "axial":
                if read_orient == "L_R":
                    reordered_seq2d = np.flip(np.flip(reordered_seq2d, axis=3), axis=2)

        # add 1 to reorder counter:
        if input_data is None:
            add_counter(
                data_obj=self,
                counter_name="seq2d_permute_counts",
                n_counts=1,
            )

        return reordered_seq2d

    def permute_csi_image(self, input_data=None, force_permute=False):
        """This function is supposed to reorder the kspace data
        The point of this is to have the kspace in a default order:
        fid - z - x - y - rep - nchans. This makes processing much easier

        Args:
            input_data (array): N-D Array.

        Returns:
            N-D Array: reordered seq2d file
        """

        # import counters to keep track of number of permutations:
        from ..utils.utils_general import get_counter, add_counter

        # decide which data to use:
        if input_data is None:
            csi_image = self.csi_image
        else:
            csi_image = input_data

        # check if kspace data was already reordered:
        num_csi_image_permute_counts = get_counter(
            data_obj=self,
            counter_name="csi_image_permute_counts",
        )
        ##print(f"num_csi_image_permute_counts {num_csi_image_permute_counts}")
        if num_csi_image_permute_counts > 0 and force_permute is False:
            return csi_image
        else:
            patient_pos = self.acqp["ACQ_patient_pos"]
            read_orient = self.method["PVM_SPackArrReadOrient"]
            slice_orient = self.method["PVM_SPackArrSliceOrient"]

            # was already reorderd:
            # if num_csi_image_permute_counts > 0:
            #     Warning('csi_image was already reoriented!')
            #     return csi_image

            # get size parameters:
            # fidsz = self.method["PVM_SpecMatrix"] # spectral points
            # nr = self.method["PVM_NRepetitions"] # number of repetitions
            # ysz = self.method["PVM_Matrix"][1] #
            # xsz = self.method["PVM_Matrix"][0]
            SpatialDim = self.method["PVM_SpatDimEnum"]
            nchans = self.method["PVM_EncNReceivers"]

            if SpatialDim == "<2D>":
                zsz = 0
            elif SpatialDim == "<3D>":
                zsz = self.method["PVM_Matrix"][2]
            else:
                pass

            # init:
            reordered_csi_image = csi_image

            # reordering depends on dimension:
            if SpatialDim == "<2D>":
                # kspace has already desired shape, just has to be reoriented  to match
                # the anatomical
                if patient_pos == "Head_Prone":
                    if slice_orient == "axial":
                        if read_orient == "L_R":
                            reordered_csi_image = np.flip(
                                np.flip(csi_image, axis=(2,)), axis=(3,)
                            )
                    if slice_orient == "coronal":
                        if read_orient == "H_F":
                            Warning("Not implemented yet")
                            reordered_csi_image = np.flip(
                                np.flip(csi_image, axis=(2,)), axis=(3,)
                            )

                elif patient_pos == "Head_Supine":
                    if slice_orient == "axial":
                        if read_orient == "L_R":
                            # needs to be tested:
                            reordered_csi_image = np.flip(csi_image, axis=(2,))

            else:
                reordered_csi_image = csi_image
                Warning(
                    "Patient position %s, slice orientation %s, read orientation %s is not implemented yet"
                    % (patient_pos, slice_orient, read_orient)
                )
                pass

            # add 1 to reorder counter:
            add_counter(
                data_obj=self,
                counter_name="csi_image_permute_counts",
                n_counts=1,
            )

            return reordered_csi_image

    def csi_fft(self, input_data=None, axes=(0, 2, 3), ift=False):
        """
        Quickly computes the fftshift(fft(fftshift(input_data))) along the input axis.
        If input_data is not provided, returns the fft along axis of the fid file

        Parameters:
        ----------
        input_data : (complex) ndarray (None)
            data that should be fourier-transformed.
            If blank, self.fid is used

        axes : tuple (0, 2, 3)
            axes along the fourier transform should be performed
            Axes are Spectroscopic, x, y.

        ift : use inverse Fourier transform and inverese fftshifts
        """

        # if no input dat was given, use fid
        if input_data.any():
            fid = input_data
            logger.debug("using input data")
        else:
            fid = self.kspace_array
            logger.debug("using self.kspace_array")

        logger.debug("len(axes) = " + str(len(axes)))

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
                    logger.debug("ift along " + str(axes[d]))
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
                    logger.debug("ft along " + str(axes[d]))
        except:
            Warning("Could not perform ffts!")

        return fid_ft

    def reorder_kspace(
        self,
        data_type="job",
        plot_QA=False,
        cut_off=0,
        LB=0,
        db=False,
    ):
        """
        Used for measurements of type "animal" in paravision..
        Reconstructs complex dataset comparable to 2dseq file from Paravision
        Reads in rawdatajob0 file

        Parameters
        ----------
        data_type: which Bruker raw file type, either .job or .fid
        plot_QA: bool, wether we want Plots that check the reco against the 2dseq file
        cut_off: int, How many entries of each fid are cut of at the beginning, default is 0
        LB: float, Linebroadening applied to reconstruction in Hz

        Returns
        -------
        shifted_final: np.array, shape: fid-points, ysz, xsz

        """
        if data_type == "job":
            try:
                fid = self.rawdatajob0
                logger.debug("loaded job file")
            except:
                logger.warning("couldn't load job file, trying to load .fid file")
                fid = self.fid
                logger.debug("loaded fid file")
        elif data_type == "fid":
            try:
                fid = self.fid
                logger.debug("loaded fid file")
            except:
                logger.warning("couldn't load fid file, trying to load .job file")
                try:
                    fid = self.rawdatajob0
                    logger.debug("loaded job file")
                except:
                    logger.warning("loading fid does not work")
        else:
            logger.warning(
                "unknown data_type parameter, trying to continue with .job file"
            )
            try:
                fid = self.rawdatajob0
                logger.debug("loaded job file")
            except:
                try:
                    fid = self.rawdatajob0
                    logger.debug("loaded job file")
                except:
                    logger.warning("loading fid does not work")

        # planned order is fid-y-x-z-nr-nchannels
        # get size parameters:
        # parameters:

        nr = self.method["PVM_NRepetitions"]
        ysz = self.method["PVM_Matrix"][1]
        xsz = self.method["PVM_Matrix"][0]

        # check if 3D or 2D:
        if self.method["PVM_SpatDimEnum"] == "<2D>":
            zsz = 1
        else:
            zsz = int(self.method["PVM_Matrix"][2])

        fidsz = self.method["PVM_SpecMatrix"]
        nchans = self.method["PVM_EncNReceivers"]
        mat = self.method["PVM_Matrix"]
        nr = self.method["PVM_NRepetitions"]
        num_bins = [mat[1], mat[0]]

        # get CSI phase  info
        enc_order, x_indices, y_indices = self.get_encoding_order()

        # init empty array:
        kspace_array = np.zeros((fidsz, ysz, xsz, zsz, nr, nchans))

        read_orient = self.method[
            "PVM_SPackArrReadOrient"
        ]  # depends how encoding is done either A_P or L_R (default)

        # as_list = np.reshape(old_matrix_coords, ysz * xsz)
        data = fid.copy()

        # reshape rawdata file: OLD FALSE CODING
        # d = np.reshape(data, ((nchans, nr, zsz, ysz, xsz, fidsz)))
        # d = np.transpose(d, (5, 2, 3, 4, 1, 0))

        # reshape rawdata file: NEW CORRECT CODING
        d = np.reshape(data, ((nr, zsz, ysz, xsz, nchans, fidsz)))
        d = np.transpose(d, (5, 1, 2, 3, 0, 4))

        d = np.reshape(d, (fidsz, zsz, xsz * ysz, nr, nchans))

        # get sampling phase encode indices:
        x = list(x_indices.T + 0.5)
        y = list(y_indices.T + 0.5)

        # Define bins of grid:
        binx = list(range(mat[0] + 1))
        biny = list(range(mat[1] + 1))

        # init empty array:
        kspace_array = np.zeros((xsz, ysz, fidsz, nr, zsz, nchans), dtype="complex")

        # now reco differs in case linear or centric is used
        if enc_order == "LINEAR_ENC LINEAR_ENC":
            # do linear encoding reco
            logger.debug("Encoding order", enc_order)
            if self.n_receivers > 1:
                logger.debug(
                    "This is dual channel data, performing phasing first, then csi reco..."
                )
                # do phasing
            else:
                logger.debug("This is single channel data")

                # If nr is 1, add an extra dimension to d
                if nr == 1:
                    d = d[..., np.newaxis]

                # Use binned_statistic_2d to sort the 1D data into a 2D grid based on the x and y coordinates
                from scipy.stats import binned_statistic_2d

                if db is True:
                    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

                    # # Calculate statistics
                    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                        x, y, None, "count", bins=[binx, biny]
                    )

                    plt.figure(figsize=(12, 5))
                    ax = plt.subplot(1, 1, 1)
                    # ax.set_xlim(np.min(binx),np.max(binx))
                    # ax.set_ylim(np.min(biny),np.max(biny))
                    ax.grid(which="both")
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.scatter(x, y)

                for ic in range(nchans):  # number of channels
                    for iz in range(zsz):  # 3rd dimension
                        for it in range(fidsz):
                            for ir in range(nr):  # repetitions
                                # print(
                                #     f"np.squeeze(d[it, iz, :, ir, ic]).shape = {np.squeeze(d[it, iz, :, ir, ic]).shape}"
                                # )

                                (
                                    statistic,
                                    x_edge,
                                    y_edge,
                                    binnumber,
                                ) = binned_statistic_2d(
                                    x,
                                    y,
                                    np.squeeze(d[it, iz, :, ir, ic]),
                                    "mean",
                                    bins=[binx, biny],
                                )
                                kspace_array[:, :, it, ir, iz, ic] = np.reshape(
                                    np.squeeze(statistic), [xsz, ysz]
                                )

                kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

                # TODO: should adjust the reordering algorithm to do this...
                kspace_array = np.roll(
                    np.roll(kspace_array, shift=1, axis=2), shift=1, axis=3
                )

        elif enc_order == "CENTRIC_ENC CENTRIC_ENC":
            # do centric encoding reco
            if self.n_receivers > 1:
                logger.debug(
                    "This is dual channel data, performing phasing first, then csi reco..."
                )
                # do phasing
                # TODO
                # phased_fids = re_ordered_fids_list  # to be implemented

            else:
                logger.debug(
                    "We don't have dual channel data here, proceeding with CSI reco"
                )
                # this now applies to measurement where we don't have two channels
                # e.g. 31mm coil measurements
                # reorder k space according to sampling
                old_matrix_coords = np.zeros((ysz, xsz))

                if read_orient == "L_R":
                    # logger.debug('xsz',xsz,'ysz', ysz)
                    # logger.debug(x_indices, y_indices)
                    c = 0
                    for ny in y_indices:
                        for nx in x_indices:
                            old_matrix_coords[ny, nx] = c
                            c += 1

                    # Create a mesh grid
                    x_grid, y_grid = np.meshgrid(x, y)

                    # Flatten the grids to get 1D arrays
                    x = x_grid.flatten()
                    y = y_grid.flatten()

                    # Use binned_statistic_2d to sort the 1D data into a 2D grid based on the x and y coordinates
                    from scipy.stats import binned_statistic_2d

                    if db is True:
                        logger.critical(f"sampling position x: {x}")
                        logger.critical(f"sampling position y: {y}")

                        logger.critical(f"bin edge x: {binx}")
                        logger.critical(f"bin edge y: {biny}")

                        logger.critical(f"d.shape = {d.shape}")
                        logger.critical(f"k_space_array.shape = {kspace_array.shape}")

                        from matplotlib.ticker import AutoMinorLocator, MultipleLocator

                        # # Calculate statistics
                        statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                            x, y, None, "count", bins=[binx, biny]
                        )

                        print(statistic)
                        plt.figure(figsize=(12, 5))
                        ax = plt.subplot(1, 1, 1)
                        # ax.set_xlim(np.min(binx),np.max(binx))
                        # ax.set_ylim(np.min(biny),np.max(biny))
                        ax.grid(which="both")
                        ax.xaxis.set_major_locator(MultipleLocator(1))
                        ax.yaxis.set_major_locator(MultipleLocator(1))
                        ax.scatter(x, y)

                    for ic in range(nchans):  # number of channels
                        for iz in range(zsz):  # 3rd dimension
                            for it in range(fidsz):
                                for ir in range(nr):  # repetitions
                                    (
                                        statistic,
                                        x_edge,
                                        y_edge,
                                        binnumber,
                                    ) = binned_statistic_2d(
                                        x,
                                        y,
                                        np.squeeze(d[it, iz, :, ir, ic]),
                                        "mean",
                                        bins=[binx, biny],
                                    )

                                    kspace_array[:, :, it, ir, iz, ic] = np.reshape(
                                        np.squeeze(statistic), [xsz, ysz]
                                    )

                elif read_orient == "A_P":
                    c = 0
                    for ny in y_indices:
                        for nx in x_indices:
                            old_matrix_coords[ny, nx] = c
                            c += 1
                    # as_list = np.reshape(old_matrix_coords, ysz * xsz)

                    # extract fids from dataset , i.e. cut
                    get_fids = []
                    for n in np.arange(0, self.rawdatajob0.shape[0], fidsz):
                        # transform the long FID into an array were every entry has the FID of a certain pixel
                        get_fids.append(self.rawdatajob0[cut_off + n : n + fidsz])
                    # make an array
                    # update fidsz in case we decide to loose the first 70 entries of each fid as there is
                    # no signal there
                    fidsz = fidsz - cut_off
                    get_fids = np.array(get_fids)

                    for idx in np.arange(0, get_fids.shape[0], 1):
                        placement_idx = np.where(old_matrix_coords == idx)
                        # found the index where each fid needs to go
                        # the first one in the rawdata file is the center of k space
                        # and so on outwards
                        nx = placement_idx[0][0]
                        ny = placement_idx[1][0]
                        # possibly change the dimensions of k space array
                        kspace_array[:, nx, ny] = get_fids[int(idx), :]
                    kspace_array = np.transpose(kspace_array, [0, 2, 1])
                else:
                    logger.debug(
                        read_orient,
                        " Orientation not know, no k-space reordering performed",
                    )
                logger.debug("k-space_shape", kspace_array.shape)
                # to match echoes - z - x - y - reps - chans
                kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

        elif enc_order == "luca_centric":
            # If nr is 1, add an extra dimension to d
            if nr == 1:
                d = d[..., np.newaxis]

            # Use binned_statistic_2d to sort the 1D data into a 2D grid based on the x and y coordinates
            from scipy.stats import binned_statistic_2d

            if db is True:
                from matplotlib.ticker import AutoMinorLocator, MultipleLocator

                # # Calculate statistics
                statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                    x, y, None, "count", bins=[binx, biny]
                )

                print(statistic)
                plt.figure(figsize=(12, 5))
                ax = plt.subplot(1, 1, 1)
                # ax.set_xlim(np.min(binx),np.max(binx))
                # ax.set_ylim(np.min(biny),np.max(biny))
                ax.grid(which="both")
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.scatter(x, y)
            for ic in range(nchans):  # number of channels
                for iz in range(zsz):  # 3rd dimension
                    for it in range(fidsz):
                        for ir in range(nr):  # repetitions
                            statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                                x,
                                y,
                                np.squeeze(d[it, iz, :, ir, ic]),
                                "mean",
                                bins=[binx, biny],
                            )
                            kspace_array[:, :, it, ir, iz, ic] = np.reshape(
                                np.squeeze(statistic), [xsz, ysz]
                            )
            kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

        elif enc_order == "luca_spiral":
            logger.critical(f"Encoding order {enc_order} used, not tested yet!!!")

            # If nr is 1, add an extra dimension to d
            if nr == 1:
                d = d[..., np.newaxis]

            # Use binned_statistic_2d to sort the 1D data into a 2D grid based on the x and y coordinates
            from scipy.stats import binned_statistic_2d

            if db is True:
                from matplotlib.ticker import AutoMinorLocator, MultipleLocator

                # # Calculate statistics
                statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                    x, y, None, "count", bins=[binx, biny]
                )

                # print(statistic)
                plt.figure(figsize=(12, 5))
                ax = plt.subplot(1, 1, 1)
                # ax.set_xlim(np.min(binx),np.max(binx))
                # ax.set_ylim(np.min(biny),np.max(biny))
                # ax.grid(which="both")
                # ax.xaxis.set_major_locator(MultipleLocator(1))
                # ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.scatter(x, y)
            for ic in range(nchans):  # number of channels
                for iz in range(zsz):  # 3rd dimension
                    for it in range(fidsz):
                        for ir in range(nr):  # repetitions
                            statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                                x,
                                y,
                                np.squeeze(d[it, iz, :, ir, ic]),
                                "mean",
                                bins=[binx, biny],
                            )
                            kspace_array[:, :, it, ir, iz, ic] = np.reshape(
                                np.squeeze(statistic), [xsz, ysz]
                            )
            kspace_array = np.transpose(kspace_array, [2, 4, 0, 1, 3, 5])

        else:
            logger.critical(f"Encoding order {enc_order} used, not implemented yet")

        if plot_QA is True:
            fig, ax = plt.subplots(1, 3, figsize=(12, 5), tight_layout=True)

            @widgets.interact(
                x=(0, xsz - 1, 1),
                y=(0, ysz - 1, 1),
                z=(0, zsz - 1, 1),
                rep=(0, nr - 1, 1),
                channel=(0, nchans - 1, 1),
            )
            def update(x=xsz // 2, y=ysz // 2, z=zsz // 2, rep=0, channel=0):
                ax[0].cla()
                ax[1].cla()
                ax[2].cla()

                mag_csi_reco = np.abs(
                    np.sum(kspace_array[:, :, :, z, rep, channel], axis=0)
                )

                mag_csi_pv = np.abs(np.sum(self.seq2d, axis=0))

                logger.debug(mag_csi_pv.shape, mag_csi_reco.shape)
                if mag_csi_pv.shape == (xsz, ysz, nr):
                    pass
                else:
                    logger.error(
                        str(mag_csi_pv.shape)
                        + " does not correspond to assumed shape "
                        + "(xsz,ysz,nr)-->(%d,%d,%d)" % (xsz, ysz, nr)
                    )
                ax[0].imshow(mag_csi_pv[:, :, rep])
                ax[1].imshow(mag_csi_reco)

                ax[1].set_title("Shifted Reco")
                ax[0].set_title("CSI 2dseq")

                ax[2].set_title("Spectra")

                ppm = get_freq_axis(
                    scan=self, unit="ppm", cut_off=cut_off, npoints=None
                )
                ppm2dseq = get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)

                selfreco_spec = np.abs(
                    np.fft.fftshift(np.fft.fft(kspace_array[:, y, x, z, rep, channel]))
                )

                ax[2].plot(
                    ppm,
                    selfreco_spec,
                    color="r",
                    label="My Reco- LB =" + str(LB) + " Hz",
                )
                pvreco_spec = self.seq2d[:, x, y, rep]
                ax[2].plot(
                    ppm2dseq,
                    pvreco_spec,
                    color="k",
                    label="PV6 2dseq",
                )

                fig.legend()

        else:
            logger.debug("No Plots wanted")

        kspace_array = kspace_array[cut_off:, :, :, :, :, :]
        self.kspace_array = kspace_array

        return kspace_array

    def get_encoding_order(self):
        """

        Returns
        -------
        encoding order (string), x_indices, y_indices
        """
        # get dimensions:
        ysz = self.method["PVM_Matrix"][1]
        xsz = self.method["PVM_Matrix"][0]

        # check for encoding type:
        if self.method["Method"] == "<Bruker:CSI>" or self.method["Method"] == "<User:csi_fix>":
            enc_order = self.method["PVM_EncOrder"]
            if enc_order == "LINEAR_ENC LINEAR_ENC":
                # for later reordering into an 2D array:
                x_indices = self.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                # for later reordering into an 2D array:
                # reorder k space according to sampling

                x_indices = self.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                ##print(f"x_indices = {x_indices}")
                # Create a mesh grid
                x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                ##print(f"x_grid = {x_grid}")
                # Flatten the grids to get 1D arrays
                x_indices = x_grid.flatten()
                ##print(f"x_indices = {x_indices}")
                y_indices = y_grid.flatten()
            elif enc_order == "CENTRIC_ENC CENTRIC_ENC":
                # for later reordering into an 2D array:
                # reorder k space according to sampling

                x_indices = self.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                # Create a mesh grid
                x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                # Flatten the grids to get 1D arrays
                x_indices = x_grid.flatten()
                y_indices = y_grid.flatten()

        elif self.method["Method"] in [
            "<User:lucaCSI>",
            "<User:lucaCSI2>",
            "<User:lucaCSI3>",
            "<User:lucaCSI4>",
        ]:
            if self.method["CentricEncOrder_OnOff"] == "On":
                enc_order = "luca_centric"
                y_indices = self.method["CentricEncOrderMatrixy"]
                x_indices = self.method["CentricEncOrderMatrixx"]
                if np.min(y_indices) < 0:
                    y_indices = y_indices + np.abs(np.min(y_indices))
                    x_indices = x_indices + np.abs(np.min(x_indices))
            elif self.method["SpiralEncOrder_OnOff"] == "On":
                enc_order = "luca_spiral"
                y_indices = self.method["SpiralEncOrderMatrixy"]
                x_indices = self.method["SpiralEncOrderMatrixx"]
                if np.min(y_indices) < 0:
                    y_indices = y_indices + np.abs(np.min(y_indices))
                    x_indices = x_indices + np.abs(np.min(x_indices))
            else:
                enc_order = self.method["PVM_EncOrder"]
                logger.warning(f"Can't determine encoding method, assuming {enc_order}")

                if enc_order == "LINEAR_ENC LINEAR_ENC":
                    # for later reordering into an 2D array:
                    x_indices = self.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                    y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                    # for later reordering into an 2D array:
                    # reorder k space according to sampling

                    x_indices = self.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                    y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                    ##print(f"x_indices = {x_indices}")
                    # Create a mesh grid
                    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                    ##print(f"x_grid = {x_grid}")
                    # Flatten the grids to get 1D arrays
                    x_indices = x_grid.flatten()
                    ##print(f"x_indices = {x_indices}")
                    y_indices = y_grid.flatten()
                elif enc_order == "CENTRIC_ENC CENTRIC_ENC":
                    # for later reordering into an 2D array:
                    # reorder k space according to sampling

                    x_indices = self.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
                    y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
                    # Create a mesh grid
                    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

                    # Flatten the grids to get 1D arrays
                    x_indices = x_grid.flatten()
                    y_indices = y_grid.flatten()

        # same as stock
        elif self.method["Method"] == "<User:geoffCSI>":
            enc_order = self.method["PVM_EncOrder"]
            # for later reordering into an 2D array:
            y_indices = self.method["PVM_EncSteps1"] + int(np.floor(ysz / 2.0))
            x_indices = self.method["PVM_EncSteps0"] + int(np.floor(xsz / 2.0))
        else:
            logger.warning("Can't determine encoding method")
            enc_order = None
            x_indices = np.linspace(0, xsz - 1, xsz)
            y_indices = np.linspace(0, ysz - 1, ysz)
        return enc_order, x_indices, y_indices

    def shift_anat(
        self,
        input_data=None,
        mat_csi=None,
        mat_anat=None,
        csi_obj=None,
        anat_obj=None,
        fov_csi=None,
        fov_anat=None,
        apply_fft=True,
        shift_vox=None,
        keep_intensity=True,
        use_scipy_shift=True,
        shift_dims=2,
        force_shift=False,
    ):
        """
        Shift anatomical image data to align with CSI data. This has to be done because the anatomical and CSI image
        have different resolutions. This causes the images to be misaligned, even if they are acquired in the same
        position.
        This is because to the (0,0) point, that is used to position the imaging matrix is set in the center of a
        corner voxel. If the matrix size difference is big enough (like for a high-res anatomical image and a low-res
        CSI image), this offset becomes relevant and has to be corrected for.

        Parameters
        ----------
        input_data : ndarray, optional
            The anatomical image data to be shifted. If None, data from `anat_obj` will be used.
        mat_csi : list, optional
            The matrix dimensions of the CSI data.
        mat_anat : list, optional
            The matrix dimensions of the anatomical data.
        csi_obj : object, optional
            The CSI object containing CSI data and parameters.
        anat_obj : object, optional
            The anatomical object containing anatomical data and parameters.
        fov_csi : list, optional
            The field-of-view of the CSI data.
        fov_anat : list, optional
            The field-of-view of the anatomical data.
        apply_fft : bool, optional
            Whether to apply FFT to the data. Default is True.
        shift_vox : list, optional
            The voxel shifts to be applied. If None, shifts will be calculated.
        keep_intensity : bool, optional
            Whether to keep the intensity of the image. Default is True.
        use_scipy_shift : bool, optional
            Whether to use scipy's shift function. Default is True.
        shift_dims : int, optional
            The number of dimensions to shift. Default is 2.
        force_shift : bool, optional
            Whether to force the shift even if it has been applied before. Default is False.

        Returns
        -------
        ndarray
            The shifted anatomical image data.
        """
        # to avoid misalignment, add a checker that checks if the image as
        #  already been rotated:
        from ..utils.utils_general import (
            get_counter,
            add_counter,
            define_imageFOV_parameters,
            define_imagematrix_parameters,
            reorient_anat,
        )

        if anat_obj is not None:
            # use 2dseq from anat_obj if no input_data was passed:
            if input_data is None:
                try:
                    # check if data was already reoriented to match bssfp orientation:
                    reorient_anat_counts = get_counter(
                        data_obj=anat_obj, counter_name="reorient_counter"
                    )
                    # if not --> reorient:
                    if reorient_anat_counts == 0:
                        reorient_anat(data_obj=anat_obj)
                        input_data = anat_obj.seq2d_oriented
                    else:
                        input_data = anat_obj.seq2d_oriented
                    reorient_anat_counts = get_counter(
                        data_obj=anat_obj, counter_name="reorient_counter"
                    )
                except:
                    raise Exception("No input")

            # if you want to perform the shift anyway:
            if force_shift is False:
                num_shift_counts = get_counter(
                    data_obj=anat_obj,
                    counter_name="shift_counter",
                )
                if num_shift_counts == 0:
                    pass
                else:
                    logger.critical(
                        "object has already been shifted --> leaving function"
                    )
                    return input_data

        # get parameters from object if passed:
        if csi_obj is not None:
            fov_csi = define_imageFOV_parameters(data_obj=csi_obj)
            mat_csi = define_imagematrix_parameters(data_obj=csi_obj)
            patient_pos_csi = csi_obj.acqp["ACQ_patient_pos"]
            slice_orient_csi = csi_obj.method["PVM_SPackArrSliceOrient"]
            read_orient_csi = csi_obj.method["PVM_SPackArrReadOrient"]
        else:
            patient_pos_csi = self.acqp["ACQ_patient_pos"]
            slice_orient_csi = self.method["PVM_SPackArrSliceOrient"]
            read_orient_csi = self.method["PVM_SPackArrReadOrient"]
            pass

        # get parameters from object if passed:
        if anat_obj is not None:
            fov_anat = define_imageFOV_parameters(data_obj=anat_obj)
            mat_anat = define_imagematrix_parameters(data_obj=anat_obj)

            patient_pos_anat = anat_obj.acqp["ACQ_patient_pos"]
            slice_orient_anat = anat_obj.method["PVM_SPackArrSliceOrient"]
            read_orient_anat = anat_obj.method["PVM_SPackArrReadOrient"]

            # use 2dseq from anat_obj if no input_data was passed:
            if input_data is None:
                try:
                    input_data = anat_obj.seq2d
                except:
                    raise Exception("No input")

        # assume same orientation as we dont have more info:
        else:
            patient_pos_anat = patient_pos_csi
            slice_orient_anat = slice_orient_csi
            read_orient_anat = read_orient_csi
            pass

        # if no input dat was given, stop
        if input_data is None:
            return
        else:
            anat_data = input_data

        # init array:
        anat_data_shifted = input_data

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

        # calculate the resultions ():
        res_csi = [a / b for a, b in zip(fov_csi, mat_csi)]
        res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]

        # calc necessary shift in both directions:
        shift_vox_list = [
            d * (a - b) / 2 / c
            for a, b, c, d in zip(res_csi, res_anat, fov_anat, mat_anat)
        ]
        # logger.critical(res_csi)
        # logger.critical(res_anat)
        # logger.critical(fov_anat)
        # logger.critical(mat_anat)
        # logger.critical(shift_vox_list)
        # logger.critical(slice_orient_csi)
        # logger.critical(slice_orient_anat)
        # logger.critical(read_orient_csi)
        # logger.critical(read_orient_anat)
        # logger.critical(patient_pos_anat)
        # logger.critical(anat_data.shape)

        # shift_vox = [4 / f for f in mat_csi]
        # calculate necessary shift if none was passed:
        if shift_vox is None:
            # init empty array:
            shift_vox = [0, 0, 0, 0, 0, 0]
        if patient_pos_anat == "Head_Prone":
            if read_orient_csi == read_orient_anat == "L_R":
                # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[2] = shift_vox_list[1]
                # y-dim of array echoes-z-x-y-rep-chans
                shift_vox[3] = shift_vox_list[2]
                # shift_vox = [0, 0]

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
                    logger.critical(
                        f"csi read orient:{read_orient_csi} and anat read orient:{read_orient_anat} are"
                        f"not implemented yet!"
                    )
            else:
                logger.critical(
                    f"csi slice orient:{slice_orient_csi} and anat slice orient:{slice_orient_anat} are"
                    f"not implemented yet!"
                )

        # scipy shift package:
        if use_scipy_shift:
            from scipy.ndimage import shift

            # init empty array:
            anat_data_shifted = np.zeros_like(anat_data)

            # shift image:
            anat_data_shifted = shift(anat_data, shift_vox, mode="wrap")

            anat_obj.seq2d_oriented = anat_data_shifted

            # add 1 to shift counter:
            add_counter(
                data_obj=anat_obj,
                counter_name="shift_counter",
                n_counts=1,
            )

        return anat_data_shifted

    def get_Nslices(self) -> int:
        """Get number of slices from both 2D-multislice and 3D acquisitions."""
        SpatialDim = self.method["PVM_SpatDimEnum"]

        if SpatialDim == "<2D>":
            out = self.method["PVM_SPackArrNSlices"]
        elif SpatialDim == "<3D>":
            out = self.method["PVM_Matrix"][-1]
        else:
            raise NotImplementedError(
                f"PVM_SpatDimEnum has an unknown value: {SpatialDim}."
            )

        return out

    # from warnings import deprecated
    # @deprecated("Use get_freq_axis instead") # will be possible to use in 3.13
    def get_ppm(self, cut_off=0):
        """Get frequency axis of given spectroscopy scan in units of ppm.

        Returns ppm axis for spectroscopic measurements given a certain cut_off
        value at which fid will be cut off. The default cut off value of 70 points
        is usually sufficient as there is no signal left.

        Similair to get_Hz_axis() function.

        Parameters
        ----------
        cut_off: Default value is 70. After 'cut_off' points the signal is truncated.

        Returns
        -------
        ppm_axis : np.ndarray
            Frequency axis of measurement in units of ppm.
        """
        center_ppm = float(self.method["PVM_FrqWorkPpm"][0])
        BW_ppm = float(self.method["PVM_SpecSW"])
        acq_points = int(self.method["PVM_SpecMatrix"]) - cut_off

        ppm_axis = np.linspace(
            center_ppm - BW_ppm / 2, center_ppm + BW_ppm / 2, acq_points
        )

        return ppm_axis

    # from warnings import deprecated
    # @deprecated("Use get_freq_axis instead") # will be possible to use in 3.13
    def get_Hz(self, cut_off=0):
        """Get frequency axis of given spectroscopy scan in units of Hz.

        Returns Hz axis for spectroscopic measurements given a certain cut_off
        value at which fid will be cut off. The default cut off value of 70 points
        is usually sufficient as there is no signal left.

        Similair to get_Hz_axis() function.

        Parameters
        ----------
        cut_off : int
            Default value is 70. After 'cut_off' points the signal is truncated.

        Returns
        -------
        Hz_axis : np.ndarray
            Frequency axis of measurement in units of Hz.
        """
        BW_Hz = float(self.method["PVM_SpecSWH"])
        acq_points = int(self.method["PVM_SpecMatrix"]) - cut_off

        Hz_axis = np.linspace(-1, 1, acq_points) * (BW_Hz / 2)

        return Hz_axis

    def reset(self):
        """Resest to intital values."""
        # reset index for @onlycallonce
        self.__SNR_scaling_HasBeenCalled = False

        self.seq2d = self.Load_2dseq_file()
        self.seq2d_broadend = np.copy(self.seq2d)

    def get_idx_list(self):
        """Put indices of all spectra in a list. Alternative to nested for-loops."""
        if self.seq2d.ndim == 5:
            Nspec, Nx, Ny, Nz, Nreps = self.seq2d.shape
            return list(np.ndindex((Nx, Ny, Nz, Nreps)))
        elif self.seq2d.ndim == 4:
            Nspec, Nx, Ny, Nreps = self.seq2d.shape
            return list(np.ndindex((Nx, Ny, Nreps)))
        else:
            raise NotImplementedError(("Unexpected seq2d shape."))

    def apply_to_every_spec(
        self, func, auto=True, mode="index", input_data=None, idx_list=None
    ):
        """Generator applying 'func' to every spectrum, yields tuple (idx, func(spectrum)).

        Assume we want to apply the following function to every CSI spectrum:

           def double_height(spectrum):
               return spectrum * 2

        'apply_to_every_spec` makes this super easy:

           for idx, spec in self.apply_to_every_spec(double_height, mode='index'):
               self.seq2d_broadend[(..., *idx)] = spec

        to avoid the cumbersome (..., *idx) expression we can use mode='slice':

           for idx, spec in self.apply_to_every_spec(double_height, mode='slice'):
               self.seq2d_broadend[idx] = spec

        Parameters
        ----------
        func : function
            The function to be applied, must accept a single spectrum as input.

        auto : bool, optional
            Use self.data instead of self.seq2d. Default is True.

        mode : {'index', 'slice'}, optional
            This determines what type of index is returned. Assuming e.g. a 2D
            csi with a seq2d of shape (4080, 16, 16):

            'index':
                Returns the spectrum position inside the 2D array: e.g. (2, 3)

            'slice':
                Returns the slice of the spectrum: e.g. (..., 2, 3)

        Returns
        ----------
        A generator yielding the following tuple for every CSI spectrum:

                            (idx, func(data[idx]))

        idx: Index the function was applied to. See parameter 'mode' for more.
        func: The function provided as an argument.
        data: self.seq2d or self.data, see parameter 'auto' for more.
        """
        # gives possibilty to pass custom index list:
        if idx_list is None:
            idx_list = self.get_idx_list()
        else:
            pass

        if input_data is None:
            data = self.data if auto else self.seq2d
        else:
            data = input_data

        if mode == "index":
            for idx in idx_list:
                yield idx, func(data[(..., *idx)])

        elif mode == "slice":
            for idx in idx_list:
                yield (..., *idx), func(data[(..., *idx)])

        else:
            raise ValueError(f"'{mode}' is no valid mode, choose 'index' or 'slice'.")

    def linebroadening(self, lb, input_data=None, idx_list=None):
        """Applies linebroadening to the (resconstructed) seq2d data.
        Parameters
        ----------
        lb: linebroadening in Hz
        """

        # if there was no data passed:
        if input_data is None:
            input_data = self.seq2d
            # the broadend array is stored in here, we don't touch the original
            self.seq2d_broadend = np.zeros_like(input_data)

            acq_time = self.method["PVM_SpecAcquisitionTime"]
            acq_points = self.method["PVM_SpecMatrix"]

            sigma = 2 * np.pi * lb
            time = np.linspace(0, acq_time, acq_points) / 1000

            def lbconvolution(x):
                return np.abs(np.fft.fft(np.fft.ifft(x) * np.exp(-sigma * time)))

            iterator = self.apply_to_every_spec(
                lbconvolution, auto=False, mode="slice", input_data=input_data
            )

            for idx, spec in iterator:
                self.seq2d_broadend[idx] = spec

        # if there was data passed:
        else:
            logger.critical("starting lb")
            input_data_broadend = np.zeros_like(input_data)

            acq_time = self.method["PVM_SpecAcquisitionTime"]
            acq_points = self.method["PVM_SpecMatrix"]

            sigma = 2 * np.pi * lb
            time = np.linspace(0, acq_time, acq_points) / 1000

            def lbconvolution(x):
                return np.abs(np.fft.fft(np.fft.ifft(x) * np.exp(-sigma * time)))

            if idx_list is None:
                iterator = self.apply_to_every_spec(
                    lbconvolution, auto=False, mode="slice", input_data=input_data
                )
            else:
                iterator = self.apply_to_every_spec(
                    lbconvolution,
                    auto=False,
                    mode="slice",
                    input_data=input_data,
                    idx_list=idx_list,
                )

            for idx, spec in iterator:
                input_data_broadend[idx] = spec

            return input_data_broadend

    def multi_dim_zeropadding(
        self,
        input_data=None,
        num_zp=0,
        fid_start=0,
        fid_end=0,
        input_domain="spectral",
    ):
        """
        Zeropad multidimensional data along spectral domain
        Parameters
        ----------
        input_data: N-D array
            echoes - z - x -y -rep - chans
        num_zp: int
            number of points to add to the spectrum
        fid_start: Integer
            Start index
        fid_end: Integer
            End index
        input_domain: string
            whether data is in spectral or temporal (FID) domain

        Returns
        -------
        zeropadded data ([echoes + num_zp] - z - x -y -rep - chans)

        """
        if input_data is None:
            csi_data = self.seq2d_reordered
        else:
            csi_data = input_data

        def wrapper_zeropad_data(arr):
            return zeropad_data(
                input_data=arr,
                num_zp=num_zp,
                fid_start=fid_start,
                fid_end=fid_end,
                input_domain=input_domain,
            )

        # Apply along the first axis (axis=0)
        csi_data = np.apply_along_axis(wrapper_zeropad_data, axis=0, arr=csi_data)
        return csi_data

    @onlycallonce
    def SNR_scaling(self, N=500):
        """Scale seq2d to SNR. Pick the outermost N values on both sides as noise."""

        def scale(x):
            noisy_area = np.concatenate((x[:N], x[-N:]))
            return x / np.std(noisy_area)

        for idx, spec in self.apply_to_every_spec(scale, auto=False, mode="slice"):
            self.seq2d[idx] = spec

    def intensity_map(self):
        return np.sum(self.data, axis=0)

    def find_peaks(
        self, range_unit="ppm", searchrange=[], sr_index=0, height=0, **kwargs
    ):
        self.peaks[sr_index] = dict()
        self.peakproperties[sr_index] = dict()

        # get the configured x-axis values
        if range_unit == "ppm":
            xaxis = get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)
        elif range_unit == "Hz":
            xaxis = get_freq_axis(
                scan=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
        else:
            xaxis = np.arange(0, self.seq2d.shape[0], 1)

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
        self.searchranges[sr_index] = searchrange
        self.searchranges_idx[sr_index] = [a, b]

        def custom_find_peaks(spec):
            peaks, info = find_peaks(spec[a:b], height=height, **kwargs)
            peaks += a
            return peaks, info

        for idx, res in self.apply_to_every_spec(custom_find_peaks):
            tmp_peaks, tmp_properties = res

            # store peak location and peak properties
            self.peaks[sr_index][idx] = tmp_peaks
            self.peakproperties[sr_index][idx] = tmp_properties

    def find_peaks_multirange(self, searchranges, range_unit="ppm", **kwargs):
        """Given a list of ranges, find peaks in each of them seperately."""

        for i, sr in enumerate(searchranges):
            self.find_peaks(range_unit=range_unit, searchrange=sr, sr_index=i, **kwargs)

    def get_N_peaks(self, idx, N, sr_index=0, sortby="peak_heights"):
        """Returns the first N peak indices, sorted by the sortby property if given"""
        if len(self.peaks[sr_index][idx]):
            out_idx = self.peakproperties[sr_index][idx][sortby].argsort()
            return [self.peaks[sr_index][idx][i] for i in out_idx][:N]
        return np.NaN

    def peak_map(self, sr_index=0, map_type="peak_loc", xscale="Hz"):
        map_img = np.zeros(self.seq2d.shape[1:])

        xaxis = (
            get_freq_axis(
                scan=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
            if xscale == "Hz"
            else get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)
        )

        sr_peaks = self.peaks[sr_index]
        sr_props = self.peakproperties[sr_index]

        if map_type == "peak_loc":

            def get(i):
                x_idx = self.get_N_peaks(idx=i, N=1, sr_index=sr_index)
                return np.NaN if x_idx is np.NaN else xaxis[x_idx[0]]

        elif map_type == "peak_loc_strict":

            def get(i):
                try:
                    return xaxis[sr_peaks[i][0]]
                except:
                    ##print(f"No peak found for index: {i}")
                    return np.NaN

        elif map_type in next(iter(sr_props.values())).keys():

            def get(i):
                out = sr_props[i][map_type]
                return out if out.size > 0 else np.NaN

        else:
            raise ValueError(
                f"Not a valid map_type. Use 'peak_loc', 'peak_loc_strict' or from {list(next(iter(sr_props.values())).keys())}"
            )

        for idx, peaks in sr_peaks.items():
            map_img[idx] = get(idx)

        return map_img

    def calc_roi(self, key=0, extent=None, axial_image_shape=None):
        """
        Calculates the drawn ROI

        Returns:
            _type_: _description_
        """
        from ..utils.utils_general import translate_coordinates

        if "mask_anat_x_indices" not in self.ROI[key]:
            self.ROI[key]["mask_anat_x_indices"] = []
            self.ROI[key]["mask_anat_y_indices"] = []

        self.ROI[key]["mask_anat_x_indices"] = extent[0]
        self.ROI[key]["mask_anat_y_indices"] = extent[1]

        # translate coordinates into integers:
        x_coordinates = translate_coordinates(
            original_coords=self.ROI[key]["mask_anat_coords"][0],
            fov=extent[0],
            matrix_size=axial_image_shape[0],
        )

        y_coordinates = translate_coordinates(
            original_coords=self.ROI[key]["mask_anat_coords"][1],
            fov=extent[1],
            matrix_size=axial_image_shape[1],
        )
        self.ROI[key]["mask_anat_x_indices"] = x_coordinates
        self.ROI[key]["mask_anat_y_indices"] = y_coordinates
        vertices = np.array(
            [[[x, y] for x, y in zip(y_coordinates, x_coordinates)]], dtype=np.int32
        )

        # Create an empty mask with the same dimensions as the target image
        mask = np.zeros((axial_image_shape), dtype=np.uint8)

        # Fill the polygon in the mask
        self.ROI[key]["mask_anat"] = np.fliplr(cv2.fillPoly(mask, vertices, 1))
        self.ROI[key]["mask_csi"] = np.fliplr(
            cv2.resize(
                self.ROI[key]["mask_anat"],
                [self.method["PVM_Matrix"][1], self.method["PVM_Matrix"][0]],
            )
        )
        return True

    def plot2d(
        self,
        axlist,
        axial_image=None,
        display_ui=True,
        csi_data=None,
        csi_fit_data=None,
        fig=None,
        plot_params=None,
        save_fig=False,
    ):
        """
        Plots a 2D Chemical Shift Imaging (CSI) dataset, optionally overlaid onto an anatomical image.
        This function provides a comprehensive interface for visualization adjustments including
        data interpolation, display settings, and dynamic plotting through interactive UI elements in Jupyter.

        Parameters
        ----------
        axlist : list of matplotlib.axes.Axes
            List of Axes objects where the plots will be drawn.
        axial_image : hypermri.sequence object or ndarray, optional
            An object containing the anatomical image data or a 6D numpy array (same dimensions as `csi_data`)
            to be used as a background for overlays. If None, no anatomical image is displayed.
        display_ui : bool, optional
            If True (default), display interactive UI elements for dynamic plotting adjustments.
        csi_data : ndarray, optional
            6D array representing the CSI data to be plotted. If None, internal CSI data will be used.
            Dimensions should match those of `axial_image` if it is a numpy array.
        csi_fit_data : ndarray, optional
            6D array of CSI fitted data to be plotted alongside the original data for comparison.
            This array should have the same dimensions as `csi_data`.
        fig : matplotlib.figure.Figure, optional
            Figure object where the plots will be drawn. If None, the current figure will be used.
        plot_params : dict, optional
            Dictionary containing plot parameters for customization of the plot appearances such as color maps and overlays.
            Can be used to regenerated images.
        save_fig : bool, optional
            If True, the generated plot will be saved to file. This requires 'fig' and 'plot_params' to be set.

        Returns
        -------
        None or object
            The function does not return any values but renders matplotlib plots directly to the provided Axes objects.
            If 'display_ui' is False, it returns configured UI elements for external handling.


        Example
        -------
        >>> csi_scan.reorient_anat(anatomical_obj_ax=flash);
        >>> fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(5,5))
        >>> csi_scan.plot2d(axlist=(ax1, ax2),
        >>>     axial_image=flash,
        >>>     display_ui=True,
        >>>     csi_data=csi_scan.csi_image, # or csi_scan.csi_image
        >>>     fig=fig,
        >>>     plot_params=None,
        >>>     save_fig=False)

        """
        # FIXME metab_ratio produces an error when Fum/Mal is selected
        # FIXME saving of plots does not work intuitively
        # FIXME documentation is missing

        from matplotlib.ticker import StrMethodFormatter
        from matplotlib import rcParams

        # check if there is data to plot
        if csi_data is None:
            if not isinstance(self.csi_image, np.ndarray):
                logger.critical(
                    "There is no reconstructed data available. "
                    + "Please call the function bSSFP.reconstruction first!"
                    + "--> Aborting"
                )
                return False
            else:
                # get reconstructed data:
                data = self.csi_image
        else:
            # use input data:
            data = csi_data

        if np.ndim(csi_fit_data) == 7:
            csi_fit_data = np.sum(csi_fit_data, axis=-1, keepdims=True)

        # define CSI parameters:
        # ------------------------------------------------------------
        self.data_to_plot = data
        self.org_data_to_plot = data
        nechoes_csi = data.shape[0]
        reps_csi = data.shape[4]
        chans_csi = data.shape[5]

        mm_read_csi, mm_phase_csi, mm_slice_csi = define_imageFOV_parameters(
            data_obj=self
        )
        dim_read_csi, dim_phase_csi, dim_slice_csi = define_imagematrix_parameters(
            data_obj=self
        )

        # check if csi_data is interpolated compared to the raw data
        if (np.array(csi_data.shape) == np.array(self.csi_image.shape)).all() == False:
            logger.critical(
                "csi_data.shape does not match self.csi_image.shape. Assuming input data was interpolated."
            )
            if self.csi_image is not None:
                interpolation_matrix = np.array(
                    np.array(csi_data.shape) / np.array(self.csi_image.shape), dtype=int
                )
                read_itpl_factor = interpolation_matrix[1]
                phase_itpl_factor = interpolation_matrix[2]
                slice_itpl_factor = interpolation_matrix[3]
                echo_spectral_itpl_factor = interpolation_matrix[0]

                logger.critical(
                    "Changed CSI matrix size according to interpolation matrix: "
                    + str(interpolation_matrix)
                )
                logger.critical(
                    "dim_read_csi changed to " + str(dim_read_csi * read_itpl_factor)
                )
                logger.critical(
                    "dim_phase_csi changed to " + str(dim_phase_csi * phase_itpl_factor)
                )
                logger.critical(
                    "dim_slice_csi changed to " + str(dim_slice_csi * slice_itpl_factor)
                )
                logger.critical(
                    "nechoes_csi changed to "
                    + str(nechoes_csi * echo_spectral_itpl_factor)
                )
                logger.critical(
                    "Changed CSI matrix size according to interpolation matrix: "
                    + str(interpolation_matrix)
                )
                logger.warning("dim_read_csi changed to " + str(dim_read_csi))
                logger.warning("dim_phase_csi changed to " + str(dim_phase_csi))
                logger.warning("dim_slice_csi changed to " + str(dim_slice_csi))
                logger.warning("nechoes_csi changed to " + str(nechoes_csi))
            elif self.seq2d_reordered:
                interpolation_matrix = np.array(csi_data.shape) / np.array(
                    self.seq2d_reordered.shape
                )
                read_itpl_factor = interpolation_matrix[1]
                phase_itpl_factor = interpolation_matrix[2]
                slice_itpl_factor = interpolation_matrix[3]
                echo_spectral_itpl_factor = interpolation_matrix[0]
                logger.critical(
                    "Changed CSI shape according to interpolation matrix: "
                    + str(interpolation_matrix)
                )
                logger.critical(
                    "dim_read_csi changed to " + str(dim_read_csi * read_itpl_factor)
                )
                logger.critical(
                    "dim_phase_csi changed to " + str(dim_phase_csi * phase_itpl_factor)
                )
                logger.critical(
                    "dim_slice_csi changed to " + str(dim_slice_csi * slice_itpl_factor)
                )
                logger.critical(
                    "nechoes_csi changed to "
                    + str(nechoes_csi * echo_spectral_itpl_factor)
                )
                logger.critical(
                    "Changed CSI shape according to interpolation matrix: "
                    + str(interpolation_matrix)
                )
                logger.warning("dim_read_csi changed to " + str(dim_read_csi))
                logger.warning("dim_phase_csi changed to " + str(dim_phase_csi))
                logger.warning("dim_slice_csi changed to " + str(dim_slice_csi))
                logger.warning("nechoes_csi changed to " + str(nechoes_csi))
            else:
                interpolation_matrix = np.ones_like(np.array(csi_data.shape))
                read_itpl_factor = 1
                phase_itpl_factor = 1
                slice_itpl_factor = 1
                echo_spectral_itpl_factor = 1
                logger.critical(
                    "Could not find self.csi_image or self.seq2d_reordered to determine if csi_data was interpolated beforehand."
                )
        else:
            interpolation_matrix = np.ones_like(np.array(csi_data.shape))
            read_itpl_factor = 1
            phase_itpl_factor = 1
            slice_itpl_factor = 1
            echo_spectral_itpl_factor = 1

        dim_read_csi, dim_phase_csi, dim_slice_csi, nechoes_csi = (
            dim_read_csi * read_itpl_factor,
            dim_phase_csi * phase_itpl_factor,
            dim_slice_csi * slice_itpl_factor,
            nechoes_csi * echo_spectral_itpl_factor,
        )

        read_orient_csi = self.method["PVM_SPackArrReadOrient"]
        # read_offset_csi = self.method["PVM_SPackArrReadOffset"]
        # phase_offset_csi = self.method["PVM_SPackArrPhase1Offset"]
        # slice_offset_csi = self.method["PVM_SPackArrSliceOffset"]

        csi_grid = define_grid(
            mat=np.array((dim_read_csi, dim_phase_csi, dim_slice_csi)),
            fov=np.array((mm_read_csi, mm_phase_csi, mm_slice_csi)),
        )

        # define axial image parameters:
        # ------------------------------------------------------------
        # No Data - No Overlay:
        if axial_image is None:
            pass
        # object contains no info about extent, assume same dimensions:
        elif isinstance(axial_image, np.ndarray):
            pass
        # object contains info about extent:
        elif axial_image.__module__.startswith("hypermri."):
            mm_read_ax, mm_phase_ax, mm_slice_ax = define_imageFOV_parameters(
                data_obj=axial_image
            )
            (
                dim_anat_read_ax,
                dim_anat_phase_ax,
                dim_anat_slice_ax,
            ) = define_imagematrix_parameters(data_obj=axial_image)

            # read_orient_ax = self.method["PVM_SPackArrReadOrient"]
            # read_offset_ax = self.method["PVM_SPackArrReadOffset"]
            # phase_offset_ax = self.method["PVM_SPackArrPhase1Offset"]
            # slice_offset_ax = self.method["PVM_SPackArrSliceOffset"]

            # get grid describing the coronal slices: A_P
            # ax_grid = define_grid(
            #     mat=np.array((dim_anat_read_ax, dim_anat_phase_ax, dim_anat_slice_ax)),
            #     fov=np.array((mm_read_ax, mm_phase_ax, mm_slice_ax)),
            # )

            axial_image.seq2d_oriented = shift_anat(
                csi_obj=self,
                anat_obj=axial_image,
                use_scipy_shift=True,
            )

            anat_image = axial_image.seq2d_oriented
            pass
        else:
            pass

        # define axial image parameters:
        # ------------------------------------------------------------
        # No Data - No overlay:

        # dim_read_plot = len(csi_grid[0])

        # FIXME if no anatomical is loaded there is an error here
        dim_read_anat = anat_image.shape[1]
        dim_slice_plot = len(csi_grid[2])

        # define plotting extents:
        plotting_extent = get_plotting_extent(data_obj=self)
        if (interpolation_matrix == 1).all():
            self.plotting_grid = self.get_plotting_grid(csi_interpolation_matrix=None)
        else:
            self.plotting_grid = self.get_plotting_grid(
                csi_interpolation_matrix=interpolation_matrix
            )

        # define extent of datasets in space:
        ax_csi_ext, sag_csi_ext, cor_csi_ext = get_extent(data_obj=self)
        ax_anat_ext, sag_anat_ext, cor_anat_ext = get_extent(data_obj=axial_image)

        ax_csi_ext_phase = np.linspace(
            ax_csi_ext[0] - (ax_csi_ext[0] / self.data_to_plot.shape[2]),
            ax_csi_ext[1] - (ax_csi_ext[1] / self.data_to_plot.shape[2]),
            self.data_to_plot.shape[2],
        )

        ax_csi_ext_slice = np.linspace(
            ax_csi_ext[2] - (ax_csi_ext[2] / self.data_to_plot.shape[3]),
            ax_csi_ext[3] - (ax_csi_ext[3] / self.data_to_plot.shape[3]),
            self.data_to_plot.shape[3],
        )

        # for checking if the axial slice is still in the CSI slice range:
        cor_anat_range = np.linspace(cor_anat_ext[2], cor_anat_ext[3], dim_anat_read_ax)

        if not read_orient_csi == "H_F":
            # checking if we have the right read orientation
            # if this error comes up you have to implement different orientations possibly
            logger.debug(
                "Careful this function is not evaluated for this read orientation"
            )
        else:
            pass

        # get time axis
        time_axis = calc_timeaxis(data_obj=self, start_with_0=True)

        if plot_params is not None:
            try:
                plot_params = load_plot_params(param_file=plot_params, data_obj=self)
            except:
                plot_params = {}
            if "fig_name" in plot_params:
                fig_name = plot_params["fig_name"]
            else:
                fig_name = None

        # define image plotting function
        def plot_img(
            slice_nr_ax=0,
            slice_nr_cor=0,
            slice_nr_sag=0,
            freq_start=0,
            freq_end=0,
            rep1=0,
            chan=0,
            alpha_overlay=0.5,
            rep2=None,
            rep_avg=False,
            freq_avg=False,
            plot_style="Abs",
            experiment_style="Spectropic",
            anat_overlay="Metab",
            metab_clim=[0, 1],
            cmap="plasma",
            interp_method="none",
            interp_factor=1,
            freq_units="ppm",
            background=0.0,
            lb=0.0,
            freq_1=0.0,
            freq_2=0.0,
            freq_3=0.0,
            freq_4=0.0,
            ROI=0,
            metab_ratio="1 (1 Metab)",
            zero_as_Nan=False,
        ):
            """
            This is the plotting function that is called whenever one of it's
            parameters is changed:
            Parameters:
            slice_nr_ax: axial slice number
            slice_nr_cor: coronal slice number
            slice_nr_sag: sagtial slice number (often not used)
            freq_start: echo number (for me-bSSFP)
            rep1: repetition that is shown. if rep_avg = True, start of range that is averaged (see rep2)
            chan: Channel (for multichannel coil experiments)
            alpha_overlay: the intenstiy of the overlay of the metabolic images onto the anatomical (0-1)
            rep2: if rep_avg==True, this is the end of the range that is averaged (see rep1)
            rep_avg: if True, the range from rep1 to rep2 is averaged
            plot_style: can be magnitude, real, imaginary of phase
            pyr_lac: which metabolite to show (default pyruvate was acquired first
            anat_overlay: Can be metabolite only, overlay or anatomical only
            metab_clim: range of values that is displayed (smaller are set to NaN)
            cmap: colormap of metabolic images
            interp_method: interpolation method with which the images are displayed
            background: background signal that will be subtracted
            """
            # init range:
            rep_range = rep1
            # get frequency axis:
            # if freq_units == "ppm":
            #     freq_range = self.get_ppm()
            # elif freq_units == "Hz":
            #     freq_range = self.get_Hz()
            # else:
            #     freq_range = np.linspace(
            #         0, self.data_to_plot.shape[0] - 1, self.data_to_plot.shape[0]
            #     )

            freq_range = self.freq_range

            # generate a custom cmap:
            # map_name, cmap = self.generate_custom_cmap(cmap_name=cmap)

            # clear axis:
            for ax in axlist:
                ax.clear()

            # to avoid confusion as this is probably not universal (different read directions
            # and patient_pos)... define this here so it can be made depending on those paramters:
            # slice_nr_ax_csi = -(slice_nr_ax + 1)
            slice_nr_ax_csi = slice_nr_ax

            # parameters for interpolation:
            interp_params = {}
            interp_params["interp_method"] = interp_method
            interp_params["interp_factor"] = interp_factor
            interp_params["interp_cmap"] = cmap
            interp_params["interp_threshold"] = metab_clim[0]

            # get metabolic image:
            metab_image = self.prep_metab_image(
                data=None,
                plot_style=plot_style,
                rep_avg=rep_avg,
                rep1=rep1,
                rep2=rep2,
                freq_units=freq_units,
                freq_avg=freq_avg,
                freq_start=freq_start,
                freq_end=freq_end,
                slice_nr_ax_csi=slice_nr_ax_csi,
                background=background,
                chan=chan,
                metab_ratio=metab_ratio,
                freqs=[freq_1, freq_2, freq_3, freq_4],
                interp_params=interp_params,
            )

            # get spectrum:
            csi_spec = self.prep_spectrum(
                data=None,
                plot_style=plot_style,
                rep_avg=rep_avg,
                rep1=rep1,
                rep2=rep2,
                freq_units=freq_units,
                freq_avg=freq_avg,
                freq_start=freq_start,
                freq_end=freq_end,
                ix=slice_nr_sag,
                iy=slice_nr_cor,
                slice_nr_ax_csi=slice_nr_ax_csi,
                background=background,
                chan=chan,
                current_key=rangeslider_choose_ROI.value,
            )

            # get spectrum:
            csi_fit_spec = self.prep_spectrum(
                data=csi_fit_data,
                plot_style=plot_style,
                rep_avg=rep_avg,
                rep1=rep1,
                rep2=rep2,
                freq_units=freq_units,
                freq_avg=freq_avg,
                freq_start=freq_start,
                freq_end=freq_end,
                ix=slice_nr_sag,
                iy=slice_nr_cor,
                slice_nr_ax_csi=slice_nr_ax_csi,
                background=background,
                chan=chan,
                current_key=rangeslider_choose_ROI.value,
            )

            # get time curve:
            freq_range_temp = freq_end - freq_start
            csi_time = []
            csi_time.append(
                self.prep_timecurve(
                    data=None,
                    plot_style=plot_style,
                    rep_avg=rep_avg,
                    rep1=rep1,
                    rep2=rep2,
                    freq_units=freq_units,
                    freq_avg=freq_avg,
                    freq_start=freq_start,
                    freq_end=freq_end,
                    ix=slice_nr_sag,
                    iy=slice_nr_cor,
                    slice_nr_ax_csi=slice_nr_ax_csi,
                    background=background,
                    chan=chan,
                )
            )

            csi_fit_time = []
            csi_fit_time.append(
                self.prep_timecurve(
                    data=csi_fit_data,
                    plot_style=plot_style,
                    rep_avg=rep_avg,
                    rep1=rep1,
                    rep2=rep2,
                    freq_units=freq_units,
                    freq_avg=freq_avg,
                    freq_start=freq_start,
                    freq_end=freq_end,
                    ix=slice_nr_sag,
                    iy=slice_nr_cor,
                    slice_nr_ax_csi=slice_nr_ax_csi,
                    background=background,
                    chan=chan,
                )
            )

            text_freq_values = [
                text_freq_1.value,
                text_freq_2.value,
                text_freq_3.value,
                text_freq_4.value,
            ]
            for k in range(4):
                csi_time.append(
                    self.prep_timecurve(
                        data=None,
                        plot_style=plot_style,
                        rep_avg=rep_avg,
                        rep1=rep1,
                        rep2=rep2,
                        freq_units=freq_units,
                        freq_avg=freq_avg,
                        freq_start=text_freq_values[k] - freq_range_temp / 2,
                        freq_end=text_freq_values[k] + freq_range_temp / 2,
                        ix=slice_nr_sag,
                        iy=slice_nr_cor,
                        slice_nr_ax_csi=slice_nr_ax_csi,
                        background=background,
                        chan=chan,
                    )
                )
                csi_fit_time.append(
                    self.prep_timecurve(
                        data=csi_fit_data,
                        plot_style=plot_style,
                        rep_avg=rep_avg,
                        rep1=rep1,
                        rep2=rep2,
                        freq_units=freq_units,
                        freq_avg=freq_avg,
                        freq_start=freq_start,
                        freq_end=freq_end,
                        ix=slice_nr_sag,
                        iy=slice_nr_cor,
                        slice_nr_ax_csi=slice_nr_ax_csi,
                        background=background,
                        chan=chan,
                    )
                )

            # find the points in the axial and coronal that are closest to the
            # bssfp location:
            # ax_slice_ind = sag_slice_ind = cor_slice_ind = None

            ax_slice_ind = slice_nr_ax_csi
            mask_ax = np.ones((self.data_to_plot.shape[1], self.data_to_plot.shape[2]))

            # metabolic axial image:
            metab_image = np.rot90(np.squeeze((metab_image)))

            # if the colorange shoud be set automcatically:
            if checkbox_auto_set_crange.value:
                # stupid but should do the trick for now:
                metab_clim = list(metab_clim)
                metab_clim[0] = np.nanmin(metab_image)
                metab_clim[1] = np.nanmax(metab_image)

                # don't show negative number in case of magntiude display style:
                if plot_style == "Abs":
                    metab_clim[0] = 0
                metab_clim = tuple(metab_clim)

            # axial view: -----------------------------------------------------
            if (anat_overlay == "Metab + Anat") and (anat_image is not None):
                # anatomical axial image:
                axlist[0].imshow(
                    np.squeeze(np.rot90(anat_image[0, ax_slice_ind, :, :, 0, 0])),
                    extent=plotting_extent,
                    cmap="bone",
                )

                im_ax = axlist[0].imshow(
                    metab_image,
                    extent=plotting_extent,
                    cmap=cmap,
                    alpha=alpha_overlay,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )

                # only plot ROI if checkbox_draw_roi.value is True:
                if (
                    rangeslider_choose_ROI.value in self.ROI
                    and "mask_anat_coords" in self.ROI[rangeslider_choose_ROI.value]
                    and self.ROI[rangeslider_choose_ROI.value]["mask_anat_coords"][0]
                ):
                    axlist[0].plot(
                        self.ROI[rangeslider_choose_ROI.value]["mask_anat_coords"][0],
                        self.ROI[rangeslider_choose_ROI.value]["mask_anat_coords"][1],
                        color=self.ROI[rangeslider_choose_ROI.value]["color"],
                    )

            elif anat_overlay == "Metab":
                # metabolic axial image:
                im_ax = axlist[0].imshow(
                    metab_image,
                    extent=plotting_extent,
                    cmap=cmap,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )

            elif (anat_overlay == "Anat") and (anat_image is not None):
                # anatomical axial image:
                axlist[0].imshow(
                    np.squeeze(np.rot90(anat_image[0, ax_slice_ind, :, :, 0, 0])),
                    extent=plotting_extent,
                    cmap="bone",
                )

            else:  # default show metab image
                im_ax = axlist[0].imshow(
                    metab_image,
                    extent=plotting_extent,
                    cmap=cmap,
                    alpha=alpha_overlay * mask_ax,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )

            # generate colorbar if fig object was passed:
            if fig is None:
                # colorbar needs figure:
                pass

            elif (fig is not None) and (anat_overlay != "Anat"):
                # nasty but otherwise plots new colorbar each time a paramter is changed :/
                global cbar
                try:
                    cbar.remove()
                except:
                    pass
                cbar = fig.colorbar(im_ax, ax=axlist[0], fraction=0.033, pad=0.04)
                cbar.remove()
                cbar = fig.colorbar(
                    im_ax,
                    ax=axlist[0],
                    fraction=0.033,
                    pad=0.04,
                )
                # set colorbar range, use 3 ticks (min, mean, max)
                md = (metab_clim[1] + metab_clim[0]) / 2
                cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
                cbar.set_ticklabels(
                    np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
                )
            else:
                # dont plot colorbar if no metabolic data:
                try:
                    # skip for now because this changes the size of the plotted images
                    # which can be quite annoying:
                    pass
                    # remove if left over from previos plot:
                    # cbar.remove()
                except:
                    pass

            # plot lines:
            axlist[0].axhline(
                ax_csi_ext_slice[slice_nr_cor],
                color="white",
                linewidth=0.25,
                linestyle="solid",
            )
            axlist[0].axvline(
                ax_csi_ext_phase[
                    slice_nr_sag
                ],  # mm_phase_csi / 2 - slice_nr_sag * slice_thick_sag_csi,
                color="white",
                linewidth=0.25,
                linestyle="dashed",
            )

            # plot labels:
            axlist[0].set_xlabel("mm (read)")
            axlist[0].set_ylabel("mm (phase)")

            if cor_csi_ext[2] <= cor_anat_range[slice_nr_ax] <= cor_csi_ext[3]:
                axlist[0].set_title("Axial", color="black")
            else:
                axlist[0].set_title("Axial", color="red")

            # spectral view: -------------------------------------------
            if len(axlist) > 1:
                axlist[1].plot(
                    freq_range, csi_spec, linewidth=1, color="black", label="Data"
                )
                # add fitted spectrum to plot
                if csi_fit_data is not None:
                    axlist[1].plot(
                        freq_range,
                        csi_fit_spec,
                        linewidth=1,
                        color="red",
                        label="Fit",
                    )
                axlist[1].axvline(
                    freq_start,
                    color="black",
                    linewidth=1,
                    linestyle="dashed",
                )
                axlist[1].axhline(
                    0,
                    color="black",
                    linewidth=0.25,
                    linestyle="dashed",
                )
                axlist[1].legend()
                # draw 2nd line:
                if (freq_end > freq_start) and (freq_avg_checkbox.value):
                    axlist[1].axvline(
                        freq_end,
                        color="black",
                        linewidth=1,
                        linestyle="dashed",
                    )
                elif (freq_end > freq_start) and (freq_avg_checkbox.value is False):
                    axlist[1].axvline(
                        freq_end,
                        color="black",
                        linewidth=0.25,
                        linestyle="dashed",
                    )

                # flip frequency axis if units are ppm
                if plot_opts_freq_units.value == "ppm":
                    axlist[1].invert_xaxis()
                else:
                    pass

                if rangeslider_choose_ROI.value in self.ROI:
                    if "mask_anat_coords" in self.ROI[rangeslider_choose_ROI.value]:
                        axlist[1].set_title(
                            self.ROI[rangeslider_choose_ROI.value]["name"]
                        )
                    else:
                        axlist[1].set_title(str(slice_nr_sag) + " " + str(slice_nr_cor))
                else:
                    axlist[1].set_title(str(slice_nr_sag) + " " + str(slice_nr_cor))

                axlist[1].set_xlabel(f"[{plot_opts_freq_units.value}]")
                freqs_on_off = [
                    checkbox_freq_1.value,
                    checkbox_freq_2.value,
                    checkbox_freq_3.value,
                    checkbox_freq_4.value,
                ]
                freqs_value = [
                    text_freq_1.value,
                    text_freq_2.value,
                    text_freq_3.value,
                    text_freq_4.value,
                ]
                for k in range(4):
                    if freqs_on_off[k]:
                        axlist[1].axvline(
                            freqs_value[k],
                            linewidth=1,
                            color=rcParams["axes.prop_cycle"].by_key()["color"][k],
                        )

            # plot time curve ---------------------------------------------
            if len(axlist) > 2:
                # axlist[2].plot(time_axis, csi_time, linewidth=1)
                axlist[2].plot(
                    time_axis, csi_time[0], linewidth=1, color="black", label="Cursor"
                )
                for k, data_set in enumerate(csi_time[1::]):
                    if freqs_on_off[k]:
                        axlist[2].plot(
                            time_axis,
                            data_set,
                            linewidth=1,
                            color=rcParams["axes.prop_cycle"].by_key()["color"][k],
                            label=f"freq {k}",
                        )

                    axlist[2].axvline(
                        time_axis[rep1],
                        color="black",
                        linewidth=1,
                        linestyle="dashed",
                    )
                    # draw 2nd line:
                    if (rep2 > rep1) and (rep_avg_checkbox.value):
                        axlist[2].axvline(
                            time_axis[rep2],
                            color="black",
                            linewidth=1,
                            linestyle="dashed",
                        )
                    elif (rep2 > rep1) and (rep_avg_checkbox.value is False):
                        axlist[2].axvline(
                            time_axis[rep2],
                            color="black",
                            linewidth=0.25,
                            linestyle="dashed",
                        )
                    else:
                        pass
                    axlist[2].set_xlabel("t [s]")
                    axlist[2].legend()
                if csi_fit_data is not None:
                    axlist[2].plot(
                        time_axis,
                        csi_fit_time[0],
                        linewidth=1,
                        color="red",
                        label="Fit",
                    )

            labelpad = 50  # Adjust this value as needed

            # Function to format y-axis tick labels
            # Function to format y-axis tick labels with fixed width
            def format_y_tick_labels(ax, width=10):
                ticks = ax.get_yticks()
                formatted_labels = []
                for tick in ticks:
                    if "e" in f"{tick:.1e}":
                        formatted_label = f"{tick:.1e}"
                    else:
                        formatted_label = f"{tick:>{width}.4g}"
                    formatted_labels.append(formatted_label)
                ax.set_yticklabels(formatted_labels, family="monospace")

            # Apply custom formatting to each subplot
            for ax in axlist[1:-1]:
                format_y_tick_labels(ax)

        # --------------------------------------------------

        global rep_avg_checkbox_option

        # create interactive sliders for  slices in all dimensions
        rep_avg_checkbox = widgets.Checkbox(
            description="Avg. reps",
            tooltip="Take average over repetitions",
            layout=widgets.Layout(width="100px"),
        )
        # rep_avg_checkbox_option = widgets.VBox(layout=widgets.Layout(display="flex"))

        freq_avg_checkbox = widgets.Checkbox(
            description="Avg. freqs",
            tooltip="Take average over frequency range",
            layout=widgets.Layout(width="100px"),
        )

        slice_slider_rep1 = widgets.IntSlider(
            value=0,
            min=0,
            max=reps_csi - 1,
            description="Rep. Start: ",
            # layout=widgets.Layout(width="25%"),
        )
        slice_slider_rep2 = widgets.IntSlider(
            value=0,
            min=0,
            max=reps_csi - 1,
            description="Rep. End: ",
            # layout=widgets.Layout(width="25%"),
        )

        # set the minimum of the end reptition to the value of the start reptition
        def rep_range_checker(args):
            if slice_slider_rep1.value > slice_slider_rep2.value:
                slice_slider_rep2.value = slice_slider_rep1.value
            else:
                pass
            return True

        slice_slider_rep1.observe(rep_range_checker, names="value")
        slice_slider_rep2.observe(rep_range_checker, names="value")

        ## Different plot options:
        # lets you choose the data shown:
        plot_opts_part = widgets.Dropdown(
            options=["Abs", "Real", "Imag", "Phase"],
            value="Abs",
            description="Plot style:",
            disabled=False,
            layout=widgets.Layout(width="20%"),
        )

        # lets you choose wether to show anatomical, metabolic or ratio:
        plot_opts_anat_overlay = widgets.Dropdown(
            options=["Metab", "Metab + Anat", "Anat"],
            value="Metab + Anat",
            description="Overlay",
            disabled=False,
            layout=widgets.Layout(width="20%"),
        )

        # lets you choose the overlay of the metabolic on the
        # anatomical:
        plot_opts_alpha_overlay = widgets.BoundedFloatText(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            disabled=False,
            description="alpha:",
            continuous_update=False,
            layout=widgets.Layout(width="10%"),
        )

        # set plot colorrange of metabolites:
        plot_opts_metab_clim = widgets.FloatRangeSlider(
            min=-10.0,
            value=[0, np.max(data)],
            max=np.max(data),
            description="Crange:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            tooltip="Displayed colorrange",
            readout=True,
            readout_format=".1f",
            layout=widgets.Layout(width="35%"),
        )

        # set frequency axis' units:
        plot_opts_freq_units = widgets.Dropdown(
            options=["ppm", "Hz", "index"],
            value="index",
            description="Freq. units:",
            disabled=False,
            layout=widgets.Layout(width="15%"),
        )

        # on frequney unit:
        echo_slider_start = widgets.FloatSlider(
            description="Freq. Start: ",
            continuous_update=False,  # layout=widgets.Layout(width="200px")
        )
        echo_slider_end = widgets.FloatSlider(
            description="Freq. End: ",
            continuous_update=False,  # layout=widgets.Layout(width="200px")
        )

        def set_freq_slider(args):
            # only valid if freq slider had value before:
            if hasattr(args, "old"):
                freq_start_val_old = echo_slider_start.value
                freq_end_val_old = echo_slider_end.value
                # to get freq_range of previous setting:
                if args["old"] == "ppm":
                    freq_range_old = get_freq_axis(
                        scan=self, unit="ppm", npoints=self.data_to_plot.shape[0]
                    )
                    self.freq_unit = "ppm"
                # Hz frequency range:
                elif args["old"] == "Hz":
                    freq_range_old = get_freq_axis(
                        scan=self,
                        unit="Hz",
                        npoints=self.data_to_plot.shape[0],
                    )
                    self.freq_unit = "Hz"
                # index frequency rang:
                if args["old"] == "index":
                    freq_range_old = np.linspace(0, nechoes_csi - 1, nechoes_csi)
                    self.freq_unit = "index"
            # use index rang:
            else:
                args["old"] = args["new"]
                freq_start_val_old = 0
                freq_end_val_old = 1
                freq_range_old = np.linspace(0, nechoes_csi - 1, nechoes_csi)

            # get index of previous settings:
            echo_start_index = freq_to_index(
                freq=freq_start_val_old, freq_range=freq_range_old
            )
            echo_end_index = freq_to_index(
                freq=freq_end_val_old, freq_range=freq_range_old
            )

            ppm_axis = get_freq_axis(
                scan=self, unit="ppm", npoints=self.data_to_plot.shape[0]
            )
            hz_axis = get_freq_axis(
                scan=self, unit="Hz", npoints=self.data_to_plot.shape[0]
            )

            # set freq_range of new settings:
            if args["new"] == "ppm":
                freq_range_new = ppm_axis
            elif args["new"] == "Hz":
                freq_range_new = hz_axis
            elif args["new"] == "index":
                freq_range_new = np.linspace(0, nechoes_csi - 1, nechoes_csi).astype(
                    int
                )
            else:
                pass

            # this is quite messy but reliably changes the frequency range and keeps the set frequency values:
            if args["new"] == args["old"]:
                echo_slider_end.max = freq_range_new[-1]
                echo_slider_end.min = freq_range_new[0]
                echo_slider_end.value = freq_range_new[echo_end_index]
                echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                echo_slider_start.max = freq_range_new[-1]
                echo_slider_start.min = freq_range_new[0]
                echo_slider_start.value = freq_range_new[echo_start_index]
                echo_slider_start.step = freq_range_new[1] - freq_range_new[0]
                self.freq_range = freq_range_new
                return True

            # set sliders accordingly to new settings:
            # from "small" to "big"
            if args["new"] == "Hz" and args["old"] in ("ppm", "index"):
                echo_slider_end.max = freq_range_new[-1]
                echo_slider_end.min = freq_range_new[0]
                echo_slider_end.value = freq_range_new[echo_end_index]
                echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                echo_slider_start.max = freq_range_new[-1]
                echo_slider_start.min = freq_range_new[0]
                echo_slider_start.value = freq_range_new[echo_start_index]
                echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

            # "ppm bigger than number of echo index:"
            if args["new"] == "ppm" and args["old"] == "index":
                if ppm_axis[-1] > nechoes_csi:
                    echo_slider_end.max = freq_range_new[-1]
                    echo_slider_end.value = freq_range_new[echo_end_index]
                    echo_slider_end.min = freq_range_new[0]
                    echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                    echo_slider_start.max = freq_range_new[-1]
                    echo_slider_start.value = freq_range_new[echo_start_index]
                    echo_slider_start.min = freq_range_new[0]
                    echo_slider_start.step = freq_range_new[1] - freq_range_new[0]
                elif ppm_axis[-1] < nechoes_csi:
                    echo_slider_end.value = freq_range_new[echo_end_index]
                    echo_slider_end.max = freq_range_new[-1]
                    echo_slider_end.min = freq_range_new[0]
                    echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                    echo_slider_start.value = freq_range_new[echo_start_index]
                    echo_slider_start.max = freq_range_new[-1]
                    echo_slider_start.min = freq_range_new[0]
                    echo_slider_start.step = freq_range_new[1] - freq_range_new[0]
                else:
                    echo_slider_end.value = freq_range_new[echo_end_index]
                    echo_slider_end.max = freq_range_new[-1]
                    echo_slider_end.min = freq_range_new[0]
                    echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                    echo_slider_start.value = freq_range_new[echo_start_index]
                    echo_slider_start.max = freq_range_new[-1]
                    echo_slider_start.min = freq_range_new[0]
                    echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

            elif args["new"] == "ppm" and args["old"] == "Hz":
                echo_slider_start.value = freq_range_new[echo_start_index]
                echo_slider_start.min = freq_range_new[0]
                echo_slider_start.max = freq_range_new[-1]
                echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

                echo_slider_end.value = freq_range_new[echo_end_index]
                echo_slider_end.min = freq_range_new[0]
                echo_slider_end.max = freq_range_new[-1]
                echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

            elif args["new"] == "index" and args["old"] == "ppm":
                if nechoes_csi > ppm_axis[-1]:
                    echo_slider_end.min = freq_range_new[0]
                    echo_slider_end.max = freq_range_new[-1]
                    echo_slider_end.value = freq_range_new[echo_end_index]
                    echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                    echo_slider_start.min = freq_range_new[0]
                    echo_slider_start.max = freq_range_new[-1]
                    echo_slider_start.value = freq_range_new[echo_start_index]
                    echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

                elif ppm_axis[-1] > nechoes_csi:
                    echo_slider_end.value = freq_range_new[echo_end_index]
                    echo_slider_end.min = freq_range_new[0]
                    echo_slider_end.max = freq_range_new[-1]
                    echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                    echo_slider_start.value = freq_range_new[echo_start_index]
                    echo_slider_start.min = freq_range_new[0]
                    echo_slider_start.max = freq_range_new[-1]
                    echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

                else:
                    echo_slider_end.value = freq_range_new[echo_end_index]
                    echo_slider_end.min = freq_range_new[0]
                    echo_slider_end.max = freq_range_new[-1]
                    echo_slider_end.step = freq_range_new[1] - freq_range_new[0]

                    echo_slider_start.value = freq_range_new[echo_start_index]
                    echo_slider_start.min = freq_range_new[0]
                    echo_slider_start.max = freq_range_new[-1]
                    echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

            elif args["new"] == "index" and args["old"] == "Hz":
                if nechoes_csi < hz_axis[-1]:
                    echo_slider_end.value = int(freq_range_new[echo_end_index])
                    echo_slider_end.min = int(freq_range_new[0])
                    echo_slider_end.max = int(freq_range_new[-1])
                    echo_slider_end.step = int(freq_range_new[1] - freq_range_new[0])

                    echo_slider_start.value = int(freq_range_new[echo_start_index])
                    echo_slider_start.min = int(freq_range_new[0])
                    echo_slider_start.max = int(freq_range_new[-1])
                    echo_slider_start.step = int(freq_range_new[1] - freq_range_new[0])

                elif hz_axis[-1] > nechoes_csi:
                    echo_slider_end.max = int(freq_range_new[-1])
                    echo_slider_end.value = int(freq_range_new[echo_end_index])
                    echo_slider_end.min = int(freq_range_new[0])
                    echo_slider_end.step = int(freq_range_new[1] - freq_range_new[0])

                    echo_slider_start.max = int(freq_range_new[-1])
                    echo_slider_start.value = int(freq_range_new[echo_start_index])
                    echo_slider_start.min = int(freq_range_new[0])
                    echo_slider_start.step = int(freq_range_new[1] - freq_range_new[0])

                else:
                    echo_slider_end.max = int(freq_range_new[-1])
                    echo_slider_end.value = int(freq_range_new[echo_end_index])
                    echo_slider_end.min = int(freq_range_new[0])
                    echo_slider_end.step = int(freq_range_new[1] - freq_range_new[0])

                    echo_slider_start.max = freq_range_new[-1]
                    echo_slider_start.value = freq_range_new[echo_start_index]
                    echo_slider_start.min = freq_range_new[0]
                    echo_slider_start.step = freq_range_new[1] - freq_range_new[0]

            # translate freq1, freq2, ...
            text_freq_1.value = np.round(
                freq_range_new[
                    freq_to_index(freq=text_freq_1.value, freq_range=freq_range_old)
                ],
                4,
            )
            text_freq_1.step = freq_range_new[1] - freq_range_new[0]
            text_freq_2.value = np.round(
                freq_range_new[
                    freq_to_index(freq=text_freq_2.value, freq_range=freq_range_old)
                ],
                4,
            )
            text_freq_2.step = freq_range_new[1] - freq_range_new[0]
            text_freq_3.value = np.round(
                freq_range_new[
                    freq_to_index(freq=text_freq_3.value, freq_range=freq_range_old)
                ],
                4,
            )
            text_freq_3.step = freq_range_new[1] - freq_range_new[0]
            text_freq_4.value = np.round(
                freq_range_new[
                    freq_to_index(freq=text_freq_4.value, freq_range=freq_range_old)
                ],
                4,
            )
            text_freq_4.step = freq_range_new[1] - freq_range_new[0]

            self.freq_range = freq_range_new

        # change frequency range dependent on frequney unit:
        plot_opts_freq_units.observe(set_freq_slider, names="value")

        set_freq_slider({"new": "index"})

        # lets you ccoose colormaps:
        plot_opts_cmap = widgets.Dropdown(
            options=[
                "cividis",
                "inferno",
                "magma",
                "plasma",
                "viridis",
                "jet",
                "Blues",
                "BuGn",
                "BuPu",
                "GnBu",
                "Greens",
                "Greys",
                "OrRd",
                "Oranges",
                "PuBu",
                "PuBuGn",
                "PuRd",
                "Purples",
                "RdPu",
                "Reds",
                "Wistia",
                "YlGn",
                "YlGnBu",
                "YlOrBr",
                "YlOrRd",
                "afmhot",
                "autumn",
                "binary",
                "bone",
                "cool",
                "copper",
                "gist_gray",
                "gist_heat",
                "gist_yarg",
                "rainbow",
                "gray",
                "hot",
                "pink",
                "spring",
                "summer",
                "winter",
                "custom_cold",
            ],
            value="plasma",
            description="CM:",
            tooltip="Colormap",
            disabled=False,
            layout=widgets.Layout(width="15%"),
        )

        # lets you ccoose interpolation methods:
        plot_opts_interpolation_method = widgets.Dropdown(
            options=[
                "none",
                "antialiased",
                "nearest",
                "bilinear",
                "bicubic",
                "spline16",
                "spline36",
                "hanning",
                "hamming",
                "hermite",
                "kaiser",
                "quadric",
                "catrom",
                "gaussian",
                "bessel",
                "mitchell",
                "sinc",
                "lanczos",
                "blackman",
            ],
            value="bilinear",
            description="IM:",
            tooltip="Interpolation method",
            disabled=False,
            layout=widgets.Layout(width="15%"),
        )

        plot_opts_ratio = widgets.Dropdown(
            options=["1 (1 Metab)", "2/1 (Lac/Pyr)", "1/(2+3) (Mal/Fum)"],
            value="1 (1 Metab)",
            description="Ratio: ",
            tooltip="Choose if you want to plot the ratio of metabolites",
        )

        # interpolation factor:
        plot_opts_interpolation_factor = widgets.BoundedIntText(
            value=1,
            min=1,
            max=100,
            step=1,
            disabled=False,
            description="IF:",
            tooltip="Interpolation factor",
            layout=widgets.Layout(width="10%"),
        )

        # name of figure to save:
        text_save_fig_name = widgets.Text(
            value="fig_name.svg",
            placeholder="Type something",
            description="Name:",
            disabled=False,
            layout=widgets.Layout(width="25%"),
        )

        # save figure (in cwd):
        button_save_fig = widgets.Button(
            description="Save fig.",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Save the figure",
            icon="check",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(width="50px"),
        )

        # Draw ROI in anatomical image:
        checkbox_draw_roi = widgets.Checkbox(
            description="Draw ROI",
            value=False,
            tooltip="Lets you draw the ROI in anatomical image",
            layout=widgets.Layout(width="20%"),
        )

        # Draw ROI in anatomical image:
        checkbox_auto_set_crange = widgets.Checkbox(
            description="ACR",
            value=True,
            tooltip="automatically sets the colorange to min <-> max of available values",
            layout=widgets.Layout(width="10%"),
        )

        # Draw ROI in anatomical image:
        rangeslider_choose_ROI = widgets.BoundedIntText(
            value=0,
            min=0,
            max=100,
            step=1,
            disabled=False,
            tooltip="Choose ROI.",
            description="ROI #:",
            layout=widgets.Layout(width="15%"),
        )
        # Draw ROI in anatomical image:
        text_ROI_name = widgets.Text(
            value="ROI_name",
            placeholder="Name of ROI #",
            description="ROI Name:",
            disabled=False,
            layout=widgets.Layout(width="20%"),
        )

        # perfrom frequecy adjustment(in cwd):
        button_ROI_clear = widgets.Button(
            description="Clear ROI",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Clear the drawn ROI",
            layout=widgets.Layout(width="10%"),
        )

        # perfrom frequecy adjustment(in cwd):
        button_freq_adj = widgets.Button(
            description="Freq Adj.",
            disabled=True,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Aligns all frequecencies in the chosen frequency range",
            icon="check",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(width="15%"),
        )

        # Background that is subtracted:
        text_background = widgets.FloatText(
            value=0.0,
            description="BG:",
            tooltip="Background (will be subtracted)",
            continuous_update=False,
            disabled=False,
            layout=widgets.Layout(width="12%"),
        )

        # Linebroadening [Hz]
        text_lb = widgets.FloatText(
            value=0.0,
            description="LB:",
            disabled=False,
            continuous_update=False,
            tooltip="Linebroadening peak [Hz]",
            layout=widgets.Layout(width="12%"),
        )

        # frequency 1
        text_freq_1 = widgets.FloatText(
            value=0.0,
            description="Freq1:",
            disabled=False,
            layout=widgets.Layout(width="15%"),
        )
        checkbox_freq_1 = widgets.Checkbox(
            value=False,
            tooltip="Plot data from frequency 1",
        )
        text_freq_2 = widgets.FloatText(
            value=0.0,
            description="Freq2:",
            disabled=False,
            layout=widgets.Layout(width="15%"),
        )
        checkbox_freq_2 = widgets.Checkbox(
            value=False,
            tooltip="Plot data from frequency 2",
        )
        text_freq_3 = widgets.FloatText(
            value=0.0,
            description="Freq3:",
            disabled=False,
            layout=widgets.Layout(width="15%"),
        )
        checkbox_freq_3 = widgets.Checkbox(
            value=False,
            tooltip="Plot data from frequency 3",
        )
        text_freq_4 = widgets.FloatText(
            value=0.0,
            description="Freq4:",
            disabled=False,
            layout=widgets.Layout(width="15%"),
        )
        checkbox_freq_4 = widgets.Checkbox(
            value=False,
            tooltip="Plot data from frequency 4",
        )

        def perform_lb(args):
            from ..utils.utils_spectroscopy import multi_dim_linebroadening

            # linebroadening [Hz]
            lb = args["new"]

            # check if linebroadening values has changed:
            if args["new"] is not args["old"]:
                input_data = self.org_data_to_plot.copy()

                # perform linebroadening
                self.data_to_plot = multi_dim_linebroadening(
                    input_data=input_data,
                    lb=lb,
                    input_domain="spectral",
                    data_obj=self,
                )

                # ------------------------ plotting ------------------------
                # get plot parameters:
                plot_img_params = set_plot_params(self)

                # Extract the values from the widgets
                plot_img_values = {
                    key: widget.value for key, widget in plot_img_params.items()
                }

                # Call the plot_img function with the extracted values
                plot_img(**plot_img_values)
            else:
                pass

        text_lb.observe(perform_lb, names="value")

        def choose_ROI_name(args):
            """Either initialize the ROI name with the "ROI_name_" + ROI ID, or
            sets the field to the already existing ROI name with the ROI ID.
            """
            key = args["new"]

            # check if a mask with this key exists:
            if key not in self.ROI:
                self.ROI[key] = {}

            # check if this key has a mask:
            if "name" not in self.ROI[key]:
                self.ROI[key]["name"] = "ROI_name_" + str(key)
            else:
                text_ROI_name.value = self.ROI[key]["name"]

            plot_img_params = set_plot_params(self)
            plot_img_params["ROI"].value = key
            set_plot_params(self, params=plot_img_params)

        # change ROI name deciding on which ROI is chosen:
        rangeslider_choose_ROI.observe(choose_ROI_name, names="value")

        def set_ROI_name(args):
            """Sets the ROI name to the already existing ROI name with the ROI ID."""
            # get the ROI ID:
            key = rangeslider_choose_ROI.value

            # check if a mask with this key exists:
            if key not in self.ROI:
                self.ROI[key] = {}

            # check if this key has a mask:
            self.ROI[key]["name"] = text_ROI_name.value

        # change ROI name deciding on which ROI is chosen:
        text_ROI_name.observe(set_ROI_name, names="value")

        # init sliders:
        slice_slider_ax = widgets.IntSlider(
            value=dim_read_anat // 2,
            min=0,
            max=dim_read_anat - 1,
            description="Slice Axial: ",
            style=dict(text_color="red"),  # doesnt work
        )
        slice_slider_cor = widgets.IntSlider(
            value=dim_slice_plot // 2,
            min=0,
            max=dim_slice_plot - 1,
            description="Slice Coronal: ",
        )
        slice_slider_sag = widgets.IntSlider(
            value=dim_phase_csi // 2,
            min=0,
            max=dim_phase_csi - 1,
            description="Slice sagittal: ",
        )
        slice_slider_chan = widgets.IntSlider(
            value=0,
            min=0,
            max=chans_csi - 1,
            description="Chan.:",
            tooltip="Receive Channel",
            layout=widgets.Layout(width="20%"),
        )

        def adjust_colorrange(args):
            # Should adjust the colorrange depening on the chosen data type
            # (real, imag, phase, magnitude)
            input_data = self.org_data_to_plot
            if args["new"] == "Abs":
                plot_opts_metab_clim.max = 1.2 * np.nanmax(np.abs(input_data))
                plot_opts_metab_clim.min = 1.2 * np.nanmin(np.abs(input_data))
            if args["new"] == "Real":
                plot_opts_metab_clim.max = 1.2 * np.nanmax(np.real(input_data))
                plot_opts_metab_clim.min = 1.2 * np.nanmin(np.real(input_data))
            if args["new"] == "Imag":
                plot_opts_metab_clim.max = 1.2 * np.nanmax(np.imag(input_data))
                plot_opts_metab_clim.min = 1.2 * np.nanmin(np.imag(input_data))
            if args["new"] == "Phase":
                plot_opts_metab_clim.max = 2 * np.nanmax(np.angle(input_data))
                plot_opts_metab_clim.min = 2 * np.nanmin(np.angle(input_data))

        plot_opts_part.observe(adjust_colorrange, names="value")

        # combine widgets into a vertical box:
        ui_colors = widgets.HBox(
            [
                plot_opts_alpha_overlay,
                plot_opts_metab_clim,
                checkbox_auto_set_crange,
                plot_opts_cmap,
                plot_opts_interpolation_method,
                plot_opts_interpolation_factor,
                text_lb,
                text_background,
            ],
        )

        # # combine widgets into a horizontal box::
        ui_up_left = widgets.HBox(
            [
                plot_opts_anat_overlay,
                plot_opts_part,
                plot_opts_ratio,
                plot_opts_freq_units,
                button_freq_adj,
                text_save_fig_name,
                button_save_fig,
            ]
        )
        ui_up = ui_up_left
        ui_rep1 = widgets.HBox(
            [
                slice_slider_rep1,
                slice_slider_chan,
                slice_slider_ax,
            ]
        )
        ui_ROI = widgets.HBox(
            [
                slice_slider_rep2,
                checkbox_draw_roi,
                rangeslider_choose_ROI,
                text_ROI_name,
                button_ROI_clear,
            ]
        )

        ui_reps = widgets.VBox([ui_rep1, ui_ROI])
        ui_echoes = widgets.VBox([echo_slider_start, echo_slider_end])
        ui_checkboxes = widgets.VBox([rep_avg_checkbox, freq_avg_checkbox])
        ui_reps_echoes_chans = widgets.HBox([ui_echoes, ui_checkboxes, ui_reps])
        ui_freqs = widgets.HBox(
            [
                text_freq_1,
                checkbox_freq_1,
                text_freq_2,
                checkbox_freq_2,
                text_freq_3,
                checkbox_freq_3,
                text_freq_4,
                checkbox_freq_4,
            ]
        )

        def clear_roi(args):
            # clear the values that are stored in the ROI dictonary under the current chosen key + update the plot

            # set dictonary to empty:
            self.ROI[rangeslider_choose_ROI.value] = {}
            # replace old ROI name with default name:
            text_ROI_name.value = "ROI_name_" + str(rangeslider_choose_ROI.value)

            # ------------------------ plotting ------------------------
            # get plot parameters:
            plot_img_params = set_plot_params(self)

            # Extract the values from the widgets
            plot_img_values = {
                key: widget.value for key, widget in plot_img_params.items()
            }

            # Call the plot_img function with the extracted values
            plot_img(**plot_img_values)

        button_ROI_clear.on_click(clear_roi)

        def draw_roi_finish(args):
            """calcuate the ROI when the drawing of the ROI has finished (tickbox off)"""
            if args["new"]:
                pass
            else:
                self.calc_roi(
                    key=rangeslider_choose_ROI.value,
                    extent=[axlist[0].get_xlim(), axlist[0].get_ylim()],
                    axial_image_shape=anat_image.shape[1:3],
                )

                # ------------------------ plotting ------------------------
                # get plot parameters:
                plot_img_params = set_plot_params(self)

                # Extract the values from the widgets
                plot_img_values = {
                    key: widget.value for key, widget in plot_img_params.items()
                }

                # Call the plot_img function with the extracted values
                plot_img(**plot_img_values)

        checkbox_draw_roi.observe(draw_roi_finish, names="value")

        global plot_img_params

        def set_plot_params(self, params=None):
            """
            Set the plot parameters. Sets first the widget values to the parameter (if
            ther are loaded values). Either way, after that the plot parameters are set to
            the widget values
            """
            # sets the sliders to a specific value in case a parameter file was loaded
            try:
                # set this first because it affects the repetition setting:
                slice_slider_ax.value = params["slice_nr_ax"]
                slice_slider_cor.value = params["slice_nr_cor"]
                slice_slider_sag.value = params["slice_nr_sag"]
                plot_opts_freq_units.value = params["freq_units"]
                echo_slider_start.value = params["freq_start"]
                echo_slider_end.value = params["freq_end"]
                slice_slider_rep1.value = params["rep1"]
                slice_slider_chan.value = params["chan"]
                plot_opts_alpha_overlay.value = params["alpha_overlay"]
                slice_slider_rep2.value = params["rep2"]
                rep_avg_checkbox.value = params["rep_avg"]
                freq_avg_checkbox.value = params["freq_avg"]
                plot_opts_part.value = params["plot_style"]
                plot_opts_anat_overlay.value = params["anat_overlay"]
                plot_opts_metab_clim.value = params["metab_clim"]
                plot_opts_cmap.value = params["cmap"]
                plot_opts_interpolation_method.value = params["interp_method"]
                text_background.value = params["background"]
                text_lb.value = params["lb"]
                plot_opts_interpolation_factor.value = params["interp_factor"]
                rangeslider_choose_ROI.value = params["ROI"]
                text_freq_1.value = params["freq_1"]
                text_freq_2.value = params["freq_2"]
                text_freq_3.value = params["freq_3"]
                text_freq_4.value = params["freq_4"]
                plot_opts_ratio.value = params["metab_ratio"]
                # plot_opts_0_as_NaN.value = params["zero_as_NaN"]
            except:
                pass

            # set plotting parameters to slider values:
            plot_img_params = {
                "slice_nr_ax": slice_slider_ax,
                "slice_nr_cor": slice_slider_cor,
                "slice_nr_sag": slice_slider_sag,
                "freq_units": plot_opts_freq_units,
                "freq_start": echo_slider_start,
                "freq_end": echo_slider_end,
                "chan": slice_slider_chan,
                "alpha_overlay": plot_opts_alpha_overlay,
                "plot_style": plot_opts_part,
                "rep1": slice_slider_rep1,
                "rep2": slice_slider_rep2,
                "rep_avg": rep_avg_checkbox,
                "freq_avg": freq_avg_checkbox,
                "anat_overlay": plot_opts_anat_overlay,
                "metab_clim": plot_opts_metab_clim,
                "cmap": plot_opts_cmap,
                "interp_method": plot_opts_interpolation_method,
                "interp_factor": plot_opts_interpolation_factor,
                "background": text_background,
                "lb": text_lb,
                "ROI": rangeslider_choose_ROI,
                "freq_1": text_freq_1,
                "freq_2": text_freq_2,
                "freq_3": text_freq_3,
                "freq_4": text_freq_4,
                "metab_ratio": plot_opts_ratio,
                # "zero_as_NaN": plot_opts_0_as_NaN,
            }
            return plot_img_params

        # Define the on_click function
        def on_click(event):
            if event.inaxes in axlist:
                # axial image:
                if event.inaxes == axlist[0]:
                    if checkbox_draw_roi.value:
                        # take click position:
                        slice_x = event.xdata
                        slice_y = event.ydata
                        key = rangeslider_choose_ROI.value

                        # check if a mask with this key exists:
                        if key not in self.ROI:
                            self.ROI[key] = {}

                        # check if this key has a mask:
                        if "mask_anat_coords" not in self.ROI[key]:
                            self.ROI[key]["mask_anat_coords"] = ([], [])

                        # check if this key has a mask_anat_coords:
                        if "mask_anat" not in self.ROI[key]:
                            self.ROI[key]["mask_anat"] = []

                        # append click position to mask_anat_coords:
                        self.ROI[key]["mask_anat_coords"][0].append(slice_x)
                        self.ROI[key]["mask_anat_coords"][1].append(slice_y)

                        if "color" not in self.ROI[key]:
                            self.ROI[key]["color"] = plt.rcParams[
                                "axes.prop_cycle"
                            ].by_key()["color"][key]

                        # get plot parameters:
                        plot_img_params = set_plot_params(self)

                        # Extract the values from the widgets
                        plot_img_values = {
                            key: widget.value for key, widget in plot_img_params.items()
                        }

                        # Call the plot_img function with the extracted values
                        plot_img(**plot_img_values)

                    else:
                        # get plot parameters:
                        plot_img_params = set_plot_params(self)

                        # take click position:
                        slice_x = event.xdata
                        slice_y = event.ydata

                        # define position:
                        if self.acqp["ACQ_patient_pos"] == "Head_Prone":
                            if self.method["PVM_SPackArrSliceOrient"] == "axial":
                                if self.method["PVM_SPackArrReadOrient"] == "L_R":
                                    # csi_grid_plot = [csi_grid[1], csi_grid[2]]
                                    pos = (0, slice_x, slice_y)
                                    pos = [pos[1], pos[2]]
                            elif self.method["PVM_SPackArrSliceOrient"] == "sagittal":
                                if self.method["PVM_SPackArrReadOrient"] == "H_F":
                                    # csi_grid_plot = [csi_grid[0], csi_grid[2]]
                                    pos = (slice_x, 0, slice_y)
                                    pos = [pos[0], pos[2]]
                            elif self.method["PVM_SPackArrSliceOrient"] == "coronal":
                                if self.method["PVM_SPackArrReadOrient"] == "H_F":
                                    # csi_grid_plot = [csi_grid[0], csi_grid[1]]
                                    pos = (slice_x, slice_y, 0)
                                    pos = [pos[0], pos[1]]

                        # get csi grid:
                        sag_min_ind = np.argmin(np.abs(self.plotting_grid[0] - slice_x))
                        cor_min_ind = np.argmin(np.abs(self.plotting_grid[1] - slice_y))

                        # set the 2 plot parameters:
                        plot_img_params["slice_nr_sag"].value = sag_min_ind
                        plot_img_params["slice_nr_cor"].value = cor_min_ind

                        # update plot parameter dict:
                        plot_img_params = set_plot_params(self, params=plot_img_params)
                # metabolic image:
                elif event.inaxes == axlist[1]:
                    # get plot parameters:
                    plot_img_params = set_plot_params(self)

                    # take click position:
                    freq_val = event.xdata
                    sig_int = event.ydata

                    # define position:
                    # freq_1_before = freq_to_index(freq_val, self.freq_range)
                    freq_start_before = plot_img_params["freq_start"].value
                    freq_end_before = plot_img_params["freq_end"].value

                    freq_diff_before = np.abs(freq_end_before - freq_start_before)

                    if (freq_val + freq_diff_before) > self.freq_range[-1]:
                        freq_end_after = self.freq_range[-1]
                    else:
                        freq_end_after = freq_val + freq_diff_before

                    # set the plot parameters:
                    plot_img_params["freq_start"].value = freq_val
                    plot_img_params["freq_end"].value = freq_end_after

                    # update plot parameter dict:
                    plot_img_params = set_plot_params(self, params=plot_img_params)
                # time curve:
                elif event.inaxes == axlist[2]:
                    # get plot parameters:
                    plot_img_params = set_plot_params(self)

                    # so rep2 can also be shifted:
                    rep1_before = plot_img_params["rep1"].value
                    rep2_before = plot_img_params["rep2"].value

                    # take click position:
                    time1_after = event.xdata

                    rep1_after = np.argmin(np.abs(time_axis - time1_after))
                    if rep1_after < 0:
                        rep1_after = 0

                    # calculate difference:
                    rep_diff = rep1_after - rep1_before

                    # apply difference to rep2
                    rep1 = rep1_after
                    rep2 = rep2_before + rep_diff

                    # set the 2 plot parameters:
                    plot_img_params["rep1"].value = rep1
                    plot_img_params["rep2"].value = rep2

                    # update plot parameter dict:
                    plot_img_params = set_plot_params(self, params=plot_img_params)

                else:
                    pass

        # Connect the click event to the on_click function
        fig.canvas.mpl_connect("button_press_event", on_click)
        # define plot parameters:
        plot_img_params = set_plot_params(self, params=plot_params)
        # create a link to the plot_img function that is called whenever on the parameters in the {} list is changed:
        out = widgets.interactive_output(
            plot_img,
            plot_img_params,
        )
        # main_ui = widgets.HBox([ui_ax, ui_cor, ui_sag])
        main_ui = widgets.VBox([ui_up, ui_colors, ui_reps_echoes_chans, ui_freqs])

        # function that is called when save fig. button is pressed:
        # has to be defined after plot_img_params were defined:
        def save_figure(args, fig_name=None):
            import os

            if fig_name is None:
                fig_name = text_save_fig_name.value[0:-4]
            else:
                pass

            import matplotlib.pyplot as plt

            # get path to folder on above acquisition data:
            if hasattr(self, "savepath"):
                # FIXME this does not work properly
                path_parentfolder = self.savepath
            else:
                # get path to folder on above acquisition data::
                path_parentfolder = str(self.path)[0 : str(self.path).rfind("\\")]
            print(path_parentfolder)
            # make it path if not:
            if path_parentfolder[-1] != "/":
                path_to_save = path_parentfolder + "/"

            logger.debug(path_parentfolder)

            # save figure:
            # fig = plt.gcf()
            if 0 == 1:
                import pickle

                # with open(os.path.join("test.pickle")) as f:
                #     pickle.dumps(fig, f, "wb")

                with open(
                    os.path.join(path_parentfolder, fig_name + ".pickle"),
                    "wb",
                ) as f:
                    pickle.dump(fig, f)

            # save figure as vector graphic:
            fig.savefig(os.path.join(path_parentfolder, fig_name + ".svg"))

            # also save as png for preview:
            fig.savefig(
                os.path.join(path_parentfolder, fig_name + ".png"),
                format="png",
            )

            # save plot infos (txt):
            # with open(
            #     os.path.join(path_parentfolder, text_save_fig_name.value[0:-3] + "txt"),
            #     "w",
            # ) as file:
            #     for key, value in plot_img_params.items():
            #         file.write(f"{key}: {value.value}\n")

            # save plot infos (dict):
            # init empty dict

            plot_img_params_dict = {}
            # fill dict:
            for key, value in plot_img_params.items():
                # print(value.value)
                plot_img_params_dict[key] = value.value

            # get figure size:
            plot_img_params_dict["fig_size"] = (
                fig.get_size_inches() * fig.dpi
            ).tolist()

            # save plotting parameters in json:
            import json

            json.dump(
                plot_img_params_dict,
                open(
                    os.path.join(path_parentfolder, fig_name + ".json"),
                    "w",
                ),
            )

        import functools

        # define callback for save figure button:
        button_save_fig.on_click(functools.partial(save_figure, fig_name=None))

        def run_frequency_adjust(
            args, input_data, freq_peaks, freq_range, start_rep, stop_rep, ppm_hz
        ):
            self.data_to_plot, freq_map = self.frequency_adjust(
                input_data=input_data,
                freq_peaks=freq_peaks,
                freq_range=freq_range,
                start_rep=start_rep,
                stop_rep=stop_rep,
                ppm_hz=plot_opts_freq_units.value,
            )

            # get plot parameters:
            plot_img_params = set_plot_params(self)

            # Extract the values from the widgets
            plot_img_values = {
                key: widget.value for key, widget in plot_img_params.items()
            }

            # Call the plot_img function with the extracted values
            plot_img(**plot_img_values)
            return True

        button_freq_adj.on_click(
            functools.partial(
                run_frequency_adjust,
                input_data=self.data_to_plot,
                freq_peaks=None,
                freq_range=[-50, 50],
                start_rep=None,
                stop_rep=None,
                ppm_hz="hz",
            )
        )

        # # Perform 1 initial plot:
        # if (plot_opts_anat_overlay.value == "Metab + Anat") and (
        #     anat_image is not None
        # ):
        #     # anatomical axial image:
        #     im_ax_anat axlist[0].imshow(
        #         np.squeeze(np.random.randn(10, 10)),
        #         extent=plotting_extent,
        #         cmap="bone",
        #     )
        #
        #     im_ax = axlist[0].imshow(
        #         np.squeeze(np.random.randn(10, 10)),
        #         extent=plotting_extent,
        #         cmap="jet",
        #         alpha=0.5,
        #         vmin=0,
        #         vmax=1,
        #     )

        # This displays the Hbox containing the slider widgets.
        if display_ui:
            display(main_ui, out)
            if save_fig:
                save_figure(self, fig_name=fig_name)
        else:
            return main_ui

    def plot(
        self, axes=None, xscale="ppm", parameter_dict=None, display_ui=True, img=None
    ):
        """
        Plots 2dseq file from a csi measurement by summing over each spectrum
        Returns
        -------
        Interactive plot
        """
        if axes == None:
            fig, axes = plt.subplots(nrows=2)
        else:
            assert len(axes) >= 2, "You have to provide a list of at least two axes."
            fig = axes[0].get_figure()

        # get the configured x-axis values
        if xscale == "ppm":
            xaxis = get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)
            xlabel_text = r"$\sigma$ [ppm]"
        elif xscale == "Hz":
            xaxis = get_freq_axis(
                scan=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
            xlabel_text = r"$\sigma$ [Hz]"
        else:
            xaxis = np.arange(0, self.seq2d.shape[0], 1)
            xlabel_text = "array index"

        # use default paramter dictionary if none is provided
        if not parameter_dict:
            parameter_dict = {"cmap": "viridis"}

        if self.Nsearchranges > 0:
            plot_peaks = True
        else:
            plot_peaks = False

        self.__layout_adjusted = False  # ugly patch for fig.tight_layout & widgets

        # define image plotting function
        def plot_imgs(x_pos, y_pos, slice_nr, searchrange_nr, map_type):
            if map_type == "intensity" or self.Nsearchranges == 0:
                img = self.intensity_map()
            else:
                img = self.peak_map(searchrange_nr, map_type, xscale)

            [patch.remove() for patch in axes[0].patches]
            [line.remove() for line in axes[1].lines]
            axes[0].clear()
            axes[1].clear()

            # plot csi signal intensity image and current voxel indicator
            axes[0].imshow(img[:, :, slice_nr], **parameter_dict)

            axes[0].add_patch(
                Rectangle((x_pos - 0.5, y_pos - 0.5), 1, 1, fc="r", ec="w", alpha=0.5)
            )

            # remove x and y ticks
            axes[0].set_xticks([])
            axes[0].set_ylabel(f"{img.shape[0]} x {img.shape[1]}", fontsize=10)
            axes[0].set_yticks([])

            # plot csi spectrum for selected voxel
            axes[1].plot(
                xaxis,
                self.data[:, y_pos, x_pos, slice_nr],
                color="k",
            )
            axes[1].set_ylim(
                [
                    np.min(self.data[:, y_pos, x_pos, slice_nr]),
                    np.max(self.data[:, y_pos, x_pos, slice_nr]) * 1.1,
                ]
            )
            axes[1].set_xlabel(xlabel_text)
            # axes[1].set_title(f"CSI X: {x_pos} Y: {y_pos} Slice: {slice_nr}")

            # in case self.find_peaks() was called, display peaks
            if plot_peaks:
                if self.Nslices == 1:
                    idx = (y_pos, x_pos, 0)
                else:
                    idx = (y_pos, x_pos, slice_nr, 0)
                peaks_pos = self.peaks[searchrange_nr][idx]

                for p in peaks_pos:
                    axes[1].scatter(
                        xaxis[p],
                        self.data[p, y_pos, x_pos, slice_nr],
                    )

            if not self.__layout_adjusted:
                fig.tight_layout()
                self.__layout_adjusted = True

        Nx, Ny = self.seq2d.shape[2], self.seq2d.shape[1]

        # create interactive slider for  slices
        x_slider = widgets.IntSlider(
            value=Nx // 2, min=0, max=Nx - 1, description="X: "
        )

        # create interactive slider for echoes
        y_slider = widgets.IntSlider(
            value=Ny // 2, min=0, max=Ny - 1, description="Y: "
        )

        # create interactive slider for echoes
        slice_slider = widgets.IntSlider(
            value=self.Nslices // 2, min=0, max=self.Nslices - 1, description="Slice: "
        )

        searchrange_selector_options = []
        for i in range(self.Nsearchranges):
            srange = self.searchranges[i]

            srange_name = (
                f"{srange[0]:>4.0f} to {srange[1]:>4.0f} {self.searchrangeunit}"
            )
            searchrange_selector_options.append((srange_name, i))

        if len(searchrange_selector_options) == 0:
            searchrange_selector_options = [0]

        searchrange_selector = widgets.Dropdown(
            options=searchrange_selector_options,
            value=0,
            description="Search Range:",
            style={"description_width": "initial"},
        )

        map_selector_options = ["intensity", "peak_loc"]

        map_selector = widgets.Dropdown(
            options=map_selector_options,
            value="intensity",
            description="Image Map:",
            style={"description_width": "initial"},
        )

        # put both sliders inside a HBox for nice alignment  etc.
        row1 = widgets.HBox([x_slider, y_slider, slice_slider])
        row2 = widgets.HBox(
            [searchrange_selector, map_selector],
        )
        ui = widgets.VBox(
            [row1, row2],
            layout=widgets.Layout(display="flex"),
        )

        # connect plotting function with sliders
        sliders = widgets.interactive_output(
            plot_imgs,
            {
                "x_pos": x_slider,
                "y_pos": y_slider,
                "slice_nr": slice_slider,
                "searchrange_nr": searchrange_selector,
                "map_type": map_selector,
            },
        )

        # This displays the Hbox containing the slider widgets.
        if display_ui:
            display(ui, sliders)
        else:
            return (ui, sliders)

    # def save_figure(args, fig_name=None):
    #     if fig_name is None:
    #         fig_name = text_save_fig_name.value[0:-4]
    #     else:
    #         pass

    #     import matplotlib.pyplot as plt

    #     # get path to folder on above acquisition data::
    #     path_parentfolder = str(self.path)[0 : str(self.path).rfind("\\")]

    #     # save figure:
    #     # fig = plt.gcf()
    #     if 0 == 1:
    #         import pickle

    #         # with open(os.path.join("test.pickle")) as f:
    #         #     pickle.dumps(fig, f, "wb")

    #         with open(
    #             os.path.join(path_parentfolder, fig_name + ".pickle"),
    #             "wb",
    #         ) as f:
    #             pickle.dump(fig, f)

    #     # save figure as vector graphic:
    #     fig.savefig(os.path.join(path_parentfolder, fig_name + ".svg"))

    #     # also save as png for preview:
    #     fig.savefig(
    #         os.path.join(path_parentfolder, fig_name + ".png"),
    #         format="png",
    #     )

    #     # save plot infos (txt):
    #     # with open(
    #     #     os.path.join(path_parentfolder, text_save_fig_name.value[0:-3] + "txt"),
    #     #     "w",
    #     # ) as file:
    #     #     for key, value in plot_img_params.items():
    #     #         file.write(f"{key}: {value.value}\n")

    #     # save plot infos (dict):
    #     # init empty dict

    #     plot_img_params_dict = {}
    #     # fill dict:
    #     for key, value in plot_img_params.items():
    #         print(value.value)
    #         plot_img_params_dict[key] = value.value

    #     # get figure size:
    #     plot_img_params_dict["fig_size"] = (fig.get_size_inches() * fig.dpi).tolist()

    #     # save plotting parameters in json:
    #     import json

    #     json.dump(
    #         plot_img_params_dict,
    #         open(
    #             os.path.join(path_parentfolder, fig_name + ".json"),
    #             "w",
    #         ),
    #     )

    def prep_metab_image(
        self,
        data=None,
        plot_style="Abs",
        rep_avg=False,
        rep1=0,
        rep2=1,
        freq_units="index",
        freq_avg=False,
        freq_start=0,
        freq_end=1,
        slice_nr_ax_csi=0,
        background=0,
        chan=0,
        metab_ratio="1 (1 Metab)",
        freqs=[0, 0, 0, 0],
        interp_params={},
    ):
        """
        Prepares the metabolite image for plotting.

        Parameters
        ----------
        data : array_like, optional
            The data to plot. If None, the data to plot is taken from
            self.data_to_plot. The default is None.
        plot_style : str, optional
            The style of the plot. The default is "Abs".
        rep_avg : bool, optional
            If True, the data is averaged over repetitions. The default is False.
        rep1 : int, optional
            The first repetition to average over. The default is 0.
        rep2 : int, optional
            The second repetition to average over. The default is 1.
        freq_units : str, optional
            The units of the frequency axis. The default is "index".
        freq_avg : bool, optional
            If True, the data is averaged over echoes. The default is False.
        freq_start : int, optional
            The first frequency to average over to freq_end. The default is 0.
        freq_end : int, optional
            The last echo to average over. The default is 0.


        """

        # get interpolation parameters:

        from ..utils.utils_general import img_interp

        # interpoaltion factor (default = 1
        interp_factor = interp_params.get("interp_factor", 1)
        interp_cmap = interp_params.get("cmap", "jet")
        interp_method = interp_params.get("interp_method", "bilinear")
        interp_threshold = interp_params.get("interp_threshold", -np.inf)

        # choose plot style:
        if data is None:
            data_to_plot = self.data_to_plot
        else:
            data_to_plot = data

        ppm_axis = get_freq_axis(
            scan=self, unit="ppm", npoints=self.data_to_plot.shape[0]
        )
        hz_axis = get_freq_axis(
            scan=self,
            unit="Hz",
            npoints=self.data_to_plot.shape[0],
        )
        # get frequency axis:
        if freq_units == "ppm":
            freq_range = ppm_axis
        elif freq_units == "Hz":
            freq_range = hz_axis
        else:
            freq_range = np.linspace(
                0, data_to_plot.shape[0] - 1, data_to_plot.shape[0]
            )

        # translate frequency from units to index:
        freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
        freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)

        # for repetitions:
        if rep_avg:
            if rep2 > rep1:
                rep_range = slice(
                    rep1, rep2 + 1
                )  # +1 because the stop index is exclusive
            else:
                rep_range = slice(rep1, rep1 + 1)  # single value, but still a slice
        else:
            rep_range = slice(rep1, rep1 + 1)  # single value, but still a slice

        # for echoes:
        if freq_avg:
            if freq_end > freq_start:
                echo_range = slice(
                    freq_start_index, freq_end_index + 1
                )  # +1 because the stop index is exclusive
            else:
                echo_range = slice(
                    freq_start_index, freq_start_index + 1
                )  # single value, but still a slice
        else:
            echo_range = slice(
                freq_start_index, freq_start_index + 1
            )  # single value, but still a slice

        # hardcode for now:
        slice_nr_ax_csi = 0
        csi_image_data = np.mean(
            data_to_plot[
                echo_range,
                slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
                :,
                :,
                rep_range,
                slice(chan, chan + 1),
            ],
            axis=4,
            keepdims=True,
        )

        if plot_style == "Abs":
            csi_image_data = np.absolute(csi_image_data)
        elif plot_style == "Real":
            csi_image_data = np.real(csi_image_data)
        elif plot_style == "Imag":
            csi_image_data = np.imag(csi_image_data)
        elif plot_style == "Phase":
            csi_image_data = np.angle(csi_image_data)
        else:
            # default
            csi_image_data = np.absolute(csi_image_data)

        # subbract background:
        csi_image_data = csi_image_data - background

        # 1 Metabolite:
        if metab_ratio == "1 (1 Metab)":
            csi_image = np.mean(
                csi_image_data,
                axis=0,
                keepdims=True,
            )

            # Interpolate the image by a factor:
            csi_image = img_interp(
                metab_image=np.squeeze(csi_image),
                interp_factor=interp_factor,
                cmap=interp_cmap,
                interp_method=interp_method,
                threshold=interp_threshold,
                overlay=True,
            )

        elif metab_ratio == "2/1 (Lac/Pyr)":
            freq_range_temp = freq_end - freq_start
            csi_image = []
            for k in range(2):
                freq_start = (freqs[k] - freq_range_temp / 2,)
                freq_end = (freqs[k] + freq_range_temp / 2,)
                freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
                freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)
                # for echoes:
                if freq_avg:
                    if freq_end > freq_start:
                        echo_range = slice(
                            freq_start_index, freq_end_index + 1
                        )  # +1 because the stop index is exclusive
                    else:
                        echo_range = slice(
                            freq_start_index, freq_start_index + 1
                        )  # single value, but still a slice
                else:
                    echo_range = slice(
                        freq_start_index, freq_start_index + 1
                    )  # single value, but still a slice

                csi_image.append(
                    np.mean(
                        csi_image_data[
                            echo_range,
                            slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
                            :,
                            :,
                            :,
                            slice(chan, chan + 1),
                        ],
                        axis=0,
                        keepdims=True,
                    )
                )

            csi_image_interp = []
            for k in range(2):
                # Interpolate the image by a factor:
                csi_image_interp.append(
                    img_interp(
                        metab_image=np.squeeze(csi_image[k]),
                        interp_factor=interp_factor,
                        cmap=interp_cmap,
                        interp_method=interp_method,
                        threshold=interp_threshold,
                        overlay=True,
                    )
                )
            csi_image = csi_image_interp[1] / csi_image_interp[0]

        else:
            pass

        return csi_image

    def prep_spectrum(
        self,
        data=None,
        plot_style="Abs",
        rep_avg=False,
        rep1=0,
        rep2=1,
        freq_units="index",
        freq_avg=False,
        freq_start=0,
        freq_end=1,
        slice_nr_ax_csi=0,
        ix=0,
        iy=0,
        background=0,
        chan=0,
        current_key=0,
    ):
        # choose plot style:
        if data is None:
            data_to_plot = self.data_to_plot
        else:
            data_to_plot = data

        # subtract background:
        csi_image_data = data_to_plot

        ppm_axis = get_freq_axis(
            scan=self, unit="ppm", npoints=self.data_to_plot.shape[0]
        )
        hz_axis = get_freq_axis(
            scan=self,
            unit="Hz",
            npoints=self.data_to_plot.shape[0],
        )
        # get frequency axis:
        if freq_units == "ppm":
            freq_range = ppm_axis
        elif freq_units == "Hz":
            freq_range = hz_axis
        else:
            freq_range = np.linspace(
                0, csi_image_data.shape[0] - 1, csi_image_data.shape[0]
            )

        # translate frequency from units to index:
        freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
        freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)

        # for repetitions:
        if rep_avg:
            if rep2 > rep1:
                rep_range = slice(
                    rep1, rep2 + 1
                )  # +1 because the stop index is exclusive
            else:
                rep_range = slice(rep1, rep1 + 1)  # single value, but still a slice
        else:
            rep_range = slice(rep1, rep1 + 1)  # single value, but still a slice

        # hardcode for now:
        slice_nr_ax_csi = 0

        # don't apply mask, take mean over repetitions:
        csi_spec = np.mean(
            csi_image_data[
                :,
                slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
                slice(ix, ix + 1),
                slice(iy, iy + 1),
                rep_range,
                slice(chan, chan + 1),
            ],
            axis=4,
            keepdims=True,
        )

        if plot_style == "Abs":
            csi_spec = np.absolute(csi_spec)
        elif plot_style == "Real":
            csi_spec = np.real(csi_spec)
        elif plot_style == "Imag":
            csi_spec = np.imag(csi_spec)
        elif plot_style == "Phase":
            csi_spec = np.angle(csi_spec)
        else:
            # default
            csi_spec = np.absolute(csi_spec)

        # lazy fix:
        if np.ndim(csi_image_data) > 6:  # probably fit data ...
            csi_image_data = csi_image_data[:, :, :, :, :, :, 0]

        # whether to apply mask:
        if current_key in self.ROI:
            if "mask_csi" in self.ROI[current_key]:
                if np.sum(self.ROI[current_key]["mask_csi"]) > 0:
                    # apply mask:
                    csi_image_data = (
                        csi_image_data
                        * self.ROI[current_key]["mask_csi"][
                            np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis
                        ]
                    )
                    csi_spec = np.mean(
                        np.mean(
                            np.mean(
                                csi_image_data[
                                    :,
                                    slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
                                    :,
                                    :,
                                    rep_range,
                                    slice(chan, chan + 1),
                                ],
                                axis=4,
                                keepdims=True,
                            ),
                            axis=3,
                            keepdims=True,
                        ),
                        axis=2,
                        keepdims=True,
                    )

            else:
                # dont apply mask, take mean over repetitions:
                csi_spec = np.mean(
                    csi_image_data[
                        :,
                        slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
                        slice(ix, ix + 1),
                        slice(iy, iy + 1),
                        rep_range,
                        slice(chan, chan + 1),
                    ],
                    axis=4,
                    keepdims=True,
                )
        else:
            pass

        csi_spec = csi_spec - -background

        return np.squeeze(csi_spec)

    def prep_timecurve(
        self,
        data=None,
        plot_style="Abs",
        rep_avg=False,
        rep1=0,
        rep2=1,
        freq_units="index",
        freq_avg=False,
        freq_start=0,
        freq_end=1,
        slice_nr_ax_csi=0,
        ix=0,
        iy=0,
        background=0,
        chan=0,
    ):
        # choose plot style:
        if data is None:
            data_to_plot = self.data_to_plot
        else:
            data_to_plot = data

        if plot_style == "Abs":
            csi_image_data = np.absolute(data_to_plot)
        elif plot_style == "Real":
            csi_image_data = np.real(data_to_plot)
        elif plot_style == "Imag":
            csi_image_data = np.imag(data_to_plot)
        elif plot_style == "Phase":
            csi_image_data = np.angle(data_to_plot)
        else:
            # default
            csi_image_data = np.absolute(data_to_plot)

        # subbract background:
        csi_image_data = csi_image_data - background

        ppm_axis = get_freq_axis(
            scan=self, unit="ppm", npoints=self.data_to_plot.shape[0]
        )
        hz_axis = get_freq_axis(
            scan=self,
            unit="Hz",
            npoints=self.data_to_plot.shape[0],
        )
        # get frequency axis:
        if freq_units == "ppm":
            freq_range = ppm_axis
        elif freq_units == "Hz":
            freq_range = hz_axis
        else:
            freq_range = np.linspace(
                0, csi_image_data.shape[0] - 1, csi_image_data.shape[0]
            )

        # translate frequency from units to index:
        freq_start_index = freq_to_index(freq=freq_start, freq_range=freq_range)
        freq_end_index = freq_to_index(freq=freq_end, freq_range=freq_range)

        # for echoes:
        if freq_avg:
            if freq_end > freq_start:
                echo_range = slice(
                    freq_start_index, freq_end_index + 1
                )  # +1 because the stop index is exclusive
            else:
                echo_range = slice(
                    freq_start_index, freq_start_index + 1
                )  # single value, but still a slice
        else:
            echo_range = slice(
                freq_start_index, freq_start_index + 1
            )  # single value, but still a slice

        # hardcode for now:
        slice_nr_ax_csi = 0

        csi_spec = np.mean(
            csi_image_data[
                echo_range,
                slice(slice_nr_ax_csi, slice_nr_ax_csi + 1),
                slice(ix, ix + 1),
                slice(iy, iy + 1),
                :,
                slice(chan, chan + 1),
            ],
            axis=0,
            keepdims=True,
        )

        return np.squeeze(csi_spec)

    def integrate_intervall(
        self, start_intervall, end_intervall, xscale="Hz", min_intensity=0.25
    ):
        if xscale == "Hz":
            xaxis = get_freq_axis(
                scan=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
        elif xscale == "ppm":
            xaxis = get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)

        # get points that lie in interval
        index_in_interval = np.where(
            np.logical_and(xaxis >= start_intervall, xaxis <= end_intervall),
        )

        # extract
        values_in_intervall = np.squeeze(self.seq2d[index_in_interval[0], :, :, 0])

        values_in_intervall = np.where(
            values_in_intervall > min_intensity, values_in_intervall, 0
        )

        # integrate over index
        integral = np.sum(values_in_intervall, axis=0)

        return integral

    def fit_spectra(self, fit_params, data, xscale="ppm", plot=False, cut_off=0):
        """
        Fitting lorentzian peak functions to reconstructed CSI data.

        This function used scipy.curvefit to fit lorentzians to found peaks in a 2dseq file from a csi experiment.

        Parameters
        ----------
        fit_params: dict
            has following keys:
            peak_positions: list of float, contains the positions of the peaks to be fitted in ppm (TODO implement for Hz as well)
            peak_heights: list of float, maximum height to which the lorentzian can be fitted
            peak_widths: float, upper bound for FWHM of peak to be fitted
            signal_threshold: float, everything greater than this SNR value is considered a peak, lower peaks will be discarded
            background_region: list of float, contains ppm values of a background region for baseline correction (TODO implement for Hz)
            reps_to_fit: list of int, numbers of the repetitons to be fitted. If set to True, all reps are fitted.

        data: nd.array
            Multi-dimensional array with CSI data to be fitted. As a default one can use the 2dseq file.
            Like:
             scans = hypermri.BrukerDir(path_to_day_folder)
             csi = scans[5]
             csi.fit_spectra(fit_params, csi.seq2d)
        xscale: str, default is ppm
            can also be Hz or None, then the index is chosen
        plot: bool, default is False.
            Wether or not to show the results.
        Returns
        -------

        """
        if xscale == "ppm":
            xaxis = get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)
            xlabel_text = r"$\sigma$ [ppm]"
        elif xscale == "Hz":
            xaxis = get_freq_axis(
                scan=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
            xlabel_text = r"$\sigma$ [Hz]"
        else:
            xaxis = np.arange(0, self.seq2d.shape[0], 1)
            xlabel_text = "array index"

        # we assume csi 2dseq data of form:
        # spectral_acq_points, x-dimension, y-dimension, repetitions
        logger.debug("Assuming input data of format")

        # Todo implement phasing of data

        NR = self.method["PVM_NRepetitions"]
        matrix = self.method["PVM_Matrix"]
        xdim = matrix[0]
        ydim = matrix[1]
        spectral_points = self.method["PVM_SpecMatrix"]
        spec_acq_time = self.method["PVM_SpecAcquisitionTime"]

        if data.shape == (spectral_points, xdim, ydim, NR):
            logger.debug(
                "Assuming input data from 2dseq file of format (spectral_acq_points, x-dimension, y-dimension, repetitions)"
            )
        else:
            logger.critical(
                "Input data does not have format (spectral_acq_points, x-dimension, y-dimension, repetitions)"
            )

        # only selecting a specific number of repetitions to fit
        reps_to_fit = fit_params["reps_to_fit"]
        logger.debug(
            "csi.fit_spectra() -- Changed number of repetitions to fit to %d" + str(NR)
        )
        # fitting methods
        fit_method = fit_params["fit_method"]
        if fit_method not in ["curve_fit"]:
            logger.error(
                "Fit method "
                + str(fit_params["fit_method"])
                + " is not a valid fit method"
            )
        else:
            pass

        def find_range(axis, ppm):
            return np.argmin(np.abs(axis - ppm))

        background_region_indices = [
            find_range(xaxis, fit_params["background_region"][0]),
            find_range(xaxis, fit_params["background_region"][1]),
        ]
        # Interpolating the ppm axis
        x_axis_itp = np.linspace(np.min(xaxis), np.max(xaxis), 10000)

        N_peaks = len(fit_params["peak_positions"])
        #
        peak_coeff = np.zeros((N_peaks, 3, xdim, ydim, NR))
        peak_covariance = np.zeros((N_peaks, 3, 3, xdim, ydim, NR))
        peak_errors = np.zeros((N_peaks, 3, xdim, ydim, NR))
        # TODO make this parallel using multiprocessing
        for rep in tqdm(range(NR), desc="Fitting repetitions"):
            for x_coord in range(xdim):
                for y_coord in range(ydim):
                    # baseline correct each spectrum
                    spec_norm = data[:, x_coord, y_coord, rep] - np.mean(
                        data[
                            background_region_indices[0] : background_region_indices[1],
                            x_coord,
                            y_coord,
                            rep,
                        ]
                    )
                    # fit each peak
                    for peak_number, peak_center in enumerate(
                        fit_params["peak_positions"]
                    ):
                        peak_roi = [
                            find_range(xaxis, peak_center - fit_params["peak_width"]),
                            find_range(xaxis, peak_center + fit_params["peak_width"]),
                        ]
                        if rep in reps_to_fit:
                            try:
                                # Actual fitting happens here
                                if fit_method == "curve_fit":
                                    (
                                        peak_coeff[
                                            peak_number, :, x_coord, y_coord, rep
                                        ],
                                        peak_covariance[
                                            peak_number, :, :, x_coord, y_coord, rep
                                        ],
                                    ) = curve_fit(
                                        lorentzian,
                                        xaxis[peak_roi[0] : peak_roi[1]],
                                        spec_norm[peak_roi[0] : peak_roi[1]],
                                        bounds=(
                                            [
                                                0.01,
                                                peak_center
                                                - fit_params["peak_width"] / 2.0,
                                                0,
                                            ],
                                            [
                                                fit_params["peak_width"] * 2,
                                                peak_center
                                                + fit_params["peak_width"] / 2.0,
                                                fit_params["peak_heights"][peak_number],
                                            ],
                                        ),
                                    )
                                    peak_errors[
                                        peak_number, :, x_coord, y_coord, rep
                                    ] = np.sqrt(
                                        np.diag(
                                            peak_covariance[
                                                peak_number, :, :, x_coord, y_coord, rep
                                            ]
                                        )
                                    )
                                # TODO implement different fitting
                                else:
                                    pass
                            except RuntimeError:
                                peak_coeff[peak_number, :, x_coord, y_coord, rep] = 0
                                peak_covariance[
                                    peak_number, :, :, x_coord, y_coord, rep
                                ] = 0
                                peak_errors[peak_number, :, x_coord, y_coord, rep] = 0
                        else:
                            peak_coeff[peak_number, :, x_coord, y_coord, rep] = [
                                0,
                                0,
                                0,
                            ]
                        # clean up badly fitted peaks
                    # for peak in range(peak_coeff.shape[0]):
                    # FIXME this removes all fits so far
                    #    peak_height = peak_coeff[peak,:,x_coord,y_coord,rep][2]
                    #    peak_height_error = peak_errors[peak,:,x_coord,y_coord,rep][2]
                    #    if peak_height > fit_params['signal_threshold']:
                    #        # peak needs to have an SNR greater than a certain value
                    #        pass
                    #    else:
                    #        peak_coeff[peak,:,x_coord,y_coord,rep] = [0, 0, 0]

        if plot is True:
            plt.close("all")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            @widgets.interact(
                x_coord=(0, xdim - 1, 1), y_coord=(0, ydim - 1, 1), rep=(0, NR - 1, 1)
            )
            def update(x_coord=xdim // 2, y_coord=ydim // 2, rep=0):
                ax[1].cla()
                ax[0].cla()
                displayed_spectrum = data[:, x_coord, y_coord, rep] - np.mean(
                    data[
                        background_region_indices[0] : background_region_indices[1],
                        x_coord,
                        y_coord,
                        rep,
                    ]
                )

                displayed_image = np.sum(data, axis=0)[:, :, rep]
                ax[0].imshow(displayed_image, cmap="plasma")
                ax[0].axhline(x_coord, color="w", linestyle="dashed")
                ax[0].axvline(y_coord, color="c", linestyle="dashed")
                ax[0].set_title("CSI summed over the spectrum")
                ax[0].set_xlabel("y_coord")
                ax[0].set_ylabel("x_coord")

                # Plotting for QA
                combined_fits = 0
                for peak_number, peak in enumerate(fit_params["peak_positions"]):
                    combined_fits += lorentzian(
                        x_axis_itp,
                        peak_coeff[peak_number, :, x_coord, y_coord, rep][0],
                        peak_coeff[peak_number, :, x_coord, y_coord, rep][1],
                        peak_coeff[peak_number, :, x_coord, y_coord, rep][2],
                    )
                    print(
                        "Peak at "
                        + str(peak)
                        + " has fit params: "
                        + str(peak_coeff[peak_number, :, x_coord, y_coord, rep])
                    )

                ax[1].plot(
                    xaxis,
                    displayed_spectrum,
                    linewidth="0.5",
                    color="r",
                    label="Data",
                )

                # ax[1].set_xlim([np.max(xaxis),np.min(xaxis)])
                # ax[1].set_ylim([np.min(displayed_spectrum-5),np.max(displayed_spectrum+5)])
                ax[1].plot(
                    x_axis_itp,
                    combined_fits,
                    linestyle="dashed",
                    color="k",
                    label="Lorentzian fit",
                )

                ax[1].set_ylabel("I [a.u]")
                ax[1].set_xlabel(xlabel_text)
                ax[1].legend()

        else:
            pass

        return peak_coeff, peak_errors

    def get_plotting_grid(self, csi_interpolation_matrix=None):
        """
        returns the grid that is used to plot the data
        """
        # patient position (usually Head_Prone)
        patient_pos = self.acqp["ACQ_patient_pos"]
        read_orient = self.method["PVM_SPackArrReadOrient"]
        slice_orient = self.method["PVM_SPackArrSliceOrient"]
        mm_slice_gap = self.method["PVM_SPackArrSliceGap"]

        mm_read_csi, mm_phase_csi, mm_slice_csi = define_imageFOV_parameters(
            data_obj=self
        )
        dim_read_csi, dim_phase_csi, dim_slice_csi = define_imagematrix_parameters(
            data_obj=self
        )
        if csi_interpolation_matrix is not None:
            dim_read_csi, dim_phase_csi, dim_slice_csi = (
                dim_read_csi * csi_interpolation_matrix[1],
                dim_phase_csi * csi_interpolation_matrix[2],
                dim_slice_csi * csi_interpolation_matrix[3],
            )
            logger.critical(
                "Updated plotting grid according to interpolation matrix: "
                + str(csi_interpolation_matrix)
            )
            csi_grid = define_grid(
                mat=np.array((dim_read_csi, dim_phase_csi, dim_slice_csi)),
                fov=np.array((mm_read_csi, mm_phase_csi, mm_slice_csi)),
            )
        else:
            csi_grid = define_grid(
                mat=np.array((dim_read_csi, dim_phase_csi, dim_slice_csi)),
                fov=np.array((mm_read_csi, mm_phase_csi, mm_slice_csi)),
            )

        # offsets [mm]
        read_offset = self.method["PVM_SPackArrReadOffset"]
        phase_offset = self.method["PVM_SPackArrPhase1Offset"]
        slice_offset = self.method["PVM_SPackArrSliceOffset"]

        csi_grid = list(csi_grid)

        if patient_pos == "Head_Prone":
            if slice_orient == "axial":
                if read_orient == "L_R":
                    csi_grid[1] = csi_grid[1] + read_offset
                    csi_grid[2] = csi_grid[2] + phase_offset
                    csi_grid = [csi_grid[1], csi_grid[2]]
                elif read_orient == "A_P":
                    csi_grid[1] = csi_grid[1] + phase_offset
                    csi_grid[2] = csi_grid[2] + read_offset
                    csi_grid = [csi_grid[2], csi_grid[1]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )
                    pass

            # for now do the same to get values:
            elif slice_orient == "sagittal":
                if read_orient == "H_F":
                    csi_grid[2] = list(csi_grid)[2] + phase_offset
                    csi_grid[0] = list(csi_grid)[0] + read_offset
                    csi_grid = [csi_grid[0], csi_grid[2]]

                elif read_orient == "A_P":
                    csi_grid[2] = list(csi_grid)[2] + read_offset
                    csi_grid[0] = list(csi_grid)[0] + phase_offset
                    csi_grid = [csi_grid[2], csi_grid[0]]

                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )
                    pass

            # for now do the same to get values:
            elif slice_orient == "coronal":
                if read_orient == "H_F":
                    csi_grid[1] = list(csi_grid)[1] + phase_offset
                    csi_grid[0] = list(csi_grid)[0] + read_offset
                    csi_grid = [csi_grid[0], csi_grid[1]]
                elif read_orient == "L_R":
                    csi_grid[1] = list(csi_grid)[1] + read_offset
                    csi_grid[0] = list(csi_grid)[0] + phase_offset
                    csi_grid = [csi_grid[1], csi_grid[0]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )
                    pass

            # for now do the same to get values:
            else:
                raise Exception("unknown slice orientation: " + slice_orient)
                pass

        # same as for head_prone, has to be changed later!!!!
        elif patient_pos == "Head_Supine":
            logger.critical("Head_Supine is not fully implemented yet.")
            if slice_orient == "axial":
                if read_orient == "L_R":
                    csi_grid = [csi_grid[1], csi_grid[2]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )
                    pass

            # for now do the same to get values:
            elif slice_orient == "sagittal":
                if read_orient == "H_F":
                    csi_grid = [csi_grid[0], csi_grid[2]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )
                    pass

            # for now do the same to get values:
            elif slice_orient == "coronal":
                if read_orient == "H_F":
                    csi_grid = [csi_grid[0], csi_grid[1]]
                else:
                    raise Exception(
                        "unknown read orientation: "
                        + read_orient
                        + "for slice_orient: "
                        + slice_orient
                    )
                    pass

            # for now do the same to get values:
            else:
                raise Exception("unknown slice orientation: " + slice_orient)
                pass

        else:
            raise Exception("unknown patient_position: " + patient_pos)
        return csi_grid

    def fit_spectra_astropy(
        self,
        data=None,
        peaks_to_fit=["Pyruvate", "Lactate"],
        fit_repetitions="all",
        number_of_cpu_cores=None,
        use_multiprocessing=True,
        xscale="ppm",
    ):
        """

        Parameters
        ----------
        data: bool or ndarray
            if None, 2dseq data is fitted
        peaks_to_fit = Available options: ["Pyruvate", "Lactate", "PyruvateHydrate", "Alanin"],
        fit_repetitions: "all" or [start, end]
        number_of_cpu_cores
        use_multiprocessing
        xscale: str, default is 'ppm'
        Returns
        -------

        """
        if data is None:
            data = self.seq2d
        else:
            # TODO implement passing of self reconstructed data
            pass

        data = data / np.max(np.abs(data))

        # define parameters:
        NR = self.method["PVM_NRepetitions"]
        matrix = self.method["PVM_Matrix"]
        xdim = matrix[0]
        ydim = matrix[1]
        slices = self.method["PVM_NSPacks"]
        spectral_points = self.method["PVM_SpecMatrix"]
        spec_acq_time = self.method["PVM_SpecAcquisitionTime"]
        ref_frq_hz = self.method["PVM_FrqRef"][0]  # MHz
        ref_frq_ppm = self.method["PVM_FrqRefPpm"][0]
        center_frq_hz = self.method["PVM_FrqWork"][0]  # MHz
        center_frq_ppm = self.method["PVM_FrqWorkPpm"][0]
        center_frq_offset_hz = self.method["PVM_FrqWorkOffset"][0]  # Hz
        center_frq_offset_ppm = self.method["PVM_FrqWorkOffsetPpm"][0]

        # remove 1D axis in self reconstructed data (mostly due to only one slice being present)
        # data = np.squeeze(data)
        if slices > 1:
            raise Exception(
                "Careful - This data has %d slices, this is not implemented yet"
                % slices
            )

        if data.shape == (spectral_points, xdim, ydim, NR):
            logger.debug(
                "Assuming input data from 2dseq file of format (spectral_acq_points, x-dimension, y-dimension, repetitions)"
            )
        else:
            logger.critical(
                "Input data does not have format (spectral_acq_points, x-dimension, y-dimension, repetitions)"
            )

        # check unit of passed frequency range:
        if xscale == "ppm":
            xaxis = get_freq_axis(scan=self, unit="ppm", cut_off=0, npoints=None)
            xlabel_text = r"$\sigma$ [ppm]"
        elif xscale == "Hz":
            xaxis = get_freq_axis(
                scan=self,
                unit="Hz",
                cut_off=0,
                npoints=None,
            )
            xlabel_text = r"$\sigma$ [Hz]"
        else:
            xaxis = np.arange(0, self.data.shape[0], 1)
            xlabel_text = "array index"

        # check if offsets are alright
        if center_frq_ppm - center_frq_offset_ppm == ref_frq_ppm:
            pass
        else:
            logger.warning(
                "Center frequency and center frequency offset do not match the reference frequency"
            )
        # Todo implement phasing of data

        # only selecting a specific number of repetitions to fit
        if fit_repetitions == "all":
            fit_repetitions = range(NR)
        else:
            fit_repetitions = range(fit_repetitions[0], fit_repetitions[-1] + 1)
            NR = len(fit_repetitions)

        logger.debug(
            "csi.fit_spectra() -- Changed number of repetitions to fit to " + str(NR)
        )

        # Parallel computing to speed up fitting for multidimensional data
        from astropy.modeling import models, fitting
        import time
        from tqdm.auto import tqdm
        from joblib import Parallel, delayed, cpu_count

        if use_multiprocessing:
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

        # fitting starting parameters:
        pyruvate_params = {"amplitude": (0.01, 1), "fwhm": (0.1, 1), "x_0": (170, 172)}
        lactate_params = {"amplitude": (0.01, 1), "fwhm": (0.1, 1), "x_0": (182, 184)}
        alanin_params = {"amplitude": (0.01, 1), "fwhm": (0.1, 1), "x_0": (175, 177)}
        pyrhydrate_params = {
            "amplitude": (0.01, 1),
            "fwhm": (0.1, 0.8),
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
            for metabolite in all_params.keys():
                for lorentzian_param in all_params[metabolite]:
                    if lorentzian_param in ["x_0", "fwhm"]:
                        all_params[metabolite][lorentzian_param] = (
                            all_params[metabolite][lorentzian_param][0]
                            * conversion_fac,
                            all_params[metabolite][lorentzian_param][1]
                            * conversion_fac,
                        )
                        axis_change_flag = True
        else:
            pass
        if axis_change_flag is True:
            logger.debug(
                "Changed fitting default params from ppm to Hz. Now looking for Pyruvate at"
                + str(all_params["Pyruvate"]["x_0"] + "Hz")
            )

        # parallelising repetitions
        start_time = time.time()

        def fitting_routine(repetition):
            sum_of_models_fitted = []
            for x_coord in tqdm(range(xdim), leave=False):
                for y_coord in tqdm(range(ydim), leave=False):
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

                        # find repetition with most signal, pixel value
                        largest_peak_indices = np.unravel_index(
                            np.argmax(np.sum(np.abs(data), axis=0, keepdims=True)),
                            shape=data.shape,
                        )
                        test_fit = test_fitter(
                            test_model,
                            xaxis,
                            np.abs(
                                data[
                                    :,
                                    largest_peak_indices[1],
                                    largest_peak_indices[2],
                                    largest_peak_indices[3],
                                    largest_peak_indices[4],
                                    largest_peak_indices[5],
                                ]
                            ),
                        )
                        test_fits.append(test_fit)
                        amp_start, fwhm_start, x_0_start = test_fit.parameters
                        if (
                            min(params["amplitude"])
                            <= amp_start
                            <= max(params["amplitude"])
                        ):
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

                        # generate a Lorentzian model:
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

                        # perform test fit:
                        fitter = fitting.TRFLSQFitter(True)
                        model_fitted = fitter(
                            model_init,
                            xaxis,
                            data[:, 0, x_coord, y_coord, repetition, 0],
                        )
                        models_seperately_fitted.append(model_fitted)

                    fitter = fitting.TRFLSQFitter(True)
                    sum_of_models_fitted.append(
                        fitter(
                            np.sum(models_init),
                            xaxis,
                            data[:, 0, x_coord, y_coord, repetition, 0],
                        )
                    )
                    # sum_of_init_models_fitted = fitter(
                    #     np.sum(test_models),
                    #     xaxis,
                    #     data[
                    #         :,
                    #         largest_peak_indices[1],
                    #         largest_peak_indices[2],
                    #         largest_peak_indices[3],
                    #         largest_peak_indices[4],
                    #         largest_peak_indices[5],
                    #     ],
                    # )

            return sum_of_models_fitted

        if use_multiprocessing:
            # FIXME The tqdm just runs through and does not provide an accurate representation of the time the calculation takes
            # why is this the case? maybe try this out
            index_list = tqdm(
                fit_repetitions,
                desc="Fitting progress on %d cores" % number_of_cpu_cores,
                leave=True,
            )
            # actual parallelizing of the process
            fitted_models_all_repetitions = Parallel(n_jobs=number_of_cpu_cores)(
                delayed(fitting_routine)(it) for it in index_list
            )
        else:
            fitted_models_all_repetitions = []
            for rep in tqdm(
                fit_repetitions, desc="Fitting without multiprocessing ", leave=True
            ):
                fitted_models_all_repetitions.append(fitting_routine(rep))

        # self.fit = fitted_models_all_repetitions
        # get the execution time
        et = time.time()
        elapsed_time = et - start_time
        # print excution time:
        logger.debug("Execution time: %f seconds" % elapsed_time)
        return fitted_models_all_repetitions

    def fit_spectra_astropy2(
        self,
        input_data=None,
        number_of_cpu_cores=None,
        use_multiprocessing=True,
        fit_params=None,
        use_old_method=True,
    ):
        """
        Fit spectra onto a CSI dataset using the astropy modeling and fitting library.

        Parameters
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
        ```python
        # Define Gaussian linebroadening:
        linebroadening_Hz = 0

        # Generate linebroadened CSI dataset
        csi_data_00Hz = luca_csi.multi_dim_linebroadening(lb=linebroadening_Hz)

        # Specify metabolites for fitting:
        metabs = ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"]

        # Initialize fit parameters and bounds:
        fit_params = {
            "lb": linebroadening_Hz,
            "fwhm_1.value":0.5, "fwhm_2.value":0.5,
            "fwhm_3.value":0.5, "fwhm_4.value":0.5,
            "fwhm_1.bounds":(0.2, 2), "fwhm_2.bounds":(0.2, 2),
            "fwhm_3.bounds":(0.2, 2), "fwhm_4.bounds":(0.2, 2)
        }

        # Perform fitting
        fit_model_results_lb00Hz = luca_csi.fit_spectra_astropy2(
            metabs=metabs,
            fit_reps=[0,0],
            fit_params=fit_params,
            use_multiprocessing=True
        )
        ```

        """
        # import relevat functions from fitting utility:
        if use_old_method is False:
            from ..utils.utils_fitting import fit_spectra_apy

            fit_results = fit_spectra_apy(
                data_obj=self,
                input_data=input_data,
                number_of_cpu_cores=number_of_cpu_cores,
                use_multiprocessing=use_multiprocessing,
                fit_params=fit_params,
            )
            return fit_results
        else:
            pass
        from ..utils.utils_fitting import (
            generate_fit_init_apy,
            generate_default_fit_params_apy,
            fit_single_point_apy,
        )
        from tqdm.auto import tqdm
        import time

        if input_data is None:
            # ParaVision reco data:
            csi_data = self.seq2d_reordered
        else:
            csi_data = input_data
            pass

        csi_data = np.abs(csi_data)  # / np.max(np.abs(csi_data))

        if fit_params is None:
            fit_params = {}

        # generate + fill fit_params
        fit_params = generate_default_fit_params_apy(
            data_obj=self, input_data=csi_data, fit_params=fit_params
        )

        freq_range = get_freq_axis(scan=self, unit="ppm", npoints=input_data.shape[0])

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
            nr = csi_data.shape[4]

            # range over which to iterate
            nrange = range(nr)

        # only one repetition
        elif fit_reps[0] == fit_reps[1]:
            nrange = range(fit_reps[0], fit_reps[1] + 1)
            nr = 1

        else:
            nr = fit_reps[1] - fit_reps[0] + 1
            nrange = range(fit_reps[0], fit_reps[1] + 1)

        # fitted_model = np.apply_along_axis(wrapper_fit_spec, axis=0, arr=csi_data)
        fit_model_results = np.empty(
            (
                1,
                csi_data.shape[1],
                csi_data.shape[2],
                csi_data.shape[3],
                csi_data.shape[4],
                csi_data.shape[5],
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
            reshaped_data = csi_data[:, :, :, :, nrange, :].reshape(
                csi_data.shape[0], -1
            )

            # Fixme tqdm has trouble with CPU bound parallel operations...
            index_list = tqdm(
                range(reshaped_data.shape[1]),
                desc="Fitting progress on %d cores" % number_of_cpu_cores,
                leave=True,
            )

            f_init = generate_fit_init_apy(
                data_obj=self,
                input_data=reshaped_data,
                fit_params=fit_params,
            )

            # Prepare the arguments for each function call
            args = [
                (
                    self,
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
                    csi_data.shape[1],
                    csi_data.shape[2],
                    csi_data.shape[3],
                    nr,
                    csi_data.shape[5],
                ],
            )

        else:
            # Initialize the tqdm object for the outermost loop
            pbar = tqdm(
                range(csi_data.shape[1] * csi_data.shape[2] * csi_data.shape[3] * nr),
                desc="Progress: ",
                leave=True,
            )
            from copy import deepcopy

            f_init = generate_fit_init_apy(
                data_obj=self,
                input_data=csi_data,
                fit_params=fit_params,
            )

            for r in nrange:
                for x in range(csi_data.shape[2]):
                    for y in range(csi_data.shape[3]):
                        # Check if there are any non-finite values in the data
                        if np.any(~np.isfinite(np.squeeze(csi_data[:, 0, x, y, r, 0]))):
                            logger.error("There are non-finite values in the data.")
                        else:
                            pass

                        fit_model_result = fit_single_point_apy(
                            data_obj=self,
                            data_point=np.squeeze(csi_data[:, 0, x, y, r, 0]),
                            f_init=f_init,
                            freq_range=freq_range,
                            verblevel=verblevel,
                            fit_params=fit_params,
                        )
                        fit_model_results[0, 0, x, y, r, 0] = fit_model_result
                        pbar.update()

            pbar.close()
        ##print(f"Fitting took: {np.round(time.time() - st, 2)}s")
        fit_results = {}
        fit_results["data"] = fit_model_results
        fit_results["metabs"] = metabs
        fit_results["nrange"] = nrange
        fit_results["freq_range"] = freq_range
        fit_results["lb"] = lb
        return fit_results

    def extract_fit_results(self, fit_params=None, fit_results=None, metabs=None):
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
        ---------
        define Gaussian linebroadening:
        >>> linebroadening_Hz = 0

        Generate linebroadened CSI dataset (for later comparison)
        >>> csi_data_00Hz = luca_csi.multi_dim_linebroadening(lb=linebroadening_Hz)

        which metabolites should be fit:
        >>> metabs = ["Pyruvate","Lactate", "Alanine", "PyruvateHydrate"]

        init fit parameters + bounds:
        >>> fit_params = {"lb": linebroadening_Hz,
        ...              "fwhm_1.value":0.5,
        ...              "fwhm_2.value":0.5,
        ...              "fwhm_3.value":0.5,
        ...              "fwhm_4.value":0.5,
        ...              "fwhm_1.bounds":(0.2, 2),
        ...              "fwhm_2.bounds":(0.2, 2),
        ...              "fwhm_3.bounds":(0.2, 2),
        ...              "fwhm_4.bounds":(0.2, 2)}


        perform fitting
        >>> fit_model_results_lb00Hz = luca_csi.fit_spectra_astropy2(metabs=metabs,
        ...                                                         fit_reps=[0,0],
        ...                                                         fit_params=fit_params,
        ...                                                         use_multiprocessing=True)
        ...

        Extract spectra and fit results (maps of amplitude, frequency, fullwidth-halfmax ...
        >>> fit_spectra_00Hz, fit_res_00Hz = luca_csi.extract_fit_results_apy(metabs=metabs,
        ...                                                              fit_results=fit_model_results_lb00Hz)

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

        freq_range = fit_params.get("freq_range", get_freq_axis(scan=self, unit="ppm"))

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
                                        if (
                                            fit_params["lineshape"] == "Lorentzian"
                                        ) or (
                                            fit_params["lineshape"] == "sqrtLorentzian"
                                        ):
                                            fit_maps[m]["amplitude"][
                                                0, z, x, y, r, c
                                            ] = getattr(
                                                fit_model, f"amplitude_{k + 1}"
                                            ).value
                                            fit_maps[m]["x_0"][
                                                0, z, x, y, r, c
                                            ] = getattr(fit_model, f"x_0_{k + 1}").value
                                            fit_maps[m]["fwhm"][
                                                0, z, x, y, r, c
                                            ] = getattr(
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
                                                setattr(
                                                    f_temp, f"amplitude_{kk + 1}", 0
                                                )

                                            fit_maps[m]["AUC_background"][
                                                0, z, x, y, r, c
                                            ] = np.sum(f_temp(freq_range))
                                            fit_maps[m]["AUC_no_background"][
                                                0, z, x, y, r, c
                                            ] = np.sum(
                                                f_temp(freq_range)
                                                - np.squeeze(
                                                    fit_maps[m]["nfloor"][
                                                        0, z, x, y, r, c
                                                    ]
                                                )
                                            )
                                        elif fit_params["lineshape"] == "Voigt":
                                            fit_maps[m]["amplitude"][
                                                0, z, x, y, r, c
                                            ] = getattr(
                                                fit_model, f"amplitude_{k + 1}"
                                            ).value
                                            fit_maps[m]["x_0"][
                                                0, z, x, y, r, c
                                            ] = getattr(fit_model, f"x_0_{k + 1}").value
                                            fit_maps[m]["fwhm"][
                                                0, z, x, y, r, c
                                            ] = getattr(
                                                fit_model, f"fwhm_{k + 1}"
                                            ).value
                                            fit_maps[m]["fwhm_G"][
                                                0, z, x, y, r, c
                                            ] = getattr(
                                                fit_model, f"fwhm_G_{k + 1}"
                                            ).value
                                            fit_maps[m]["nfloor"][
                                                0, z, x, y, r, c
                                            ] = getattr(fit_model, "nfloor").value

                                            f_temp = deepcopy(fit_model)
                                            for kk in range(np.size(metabs)):
                                                # Assuming the attributes are 1-indexed
                                                if k == kk:
                                                    continue  # Skip the one you want to keep
                                                setattr(
                                                    f_temp, f"amplitude_{kk + 1}", 0
                                                )

                                            fit_maps[m]["AUC_background"][
                                                0, z, x, y, r, c
                                            ] = np.sum(f_temp(freq_range))
                                            fit_maps[m]["AUC_no_background"][
                                                0, z, x, y, r, c
                                            ] = np.sum(
                                                f_temp(freq_range)
                                                - np.squeeze(
                                                    fit_maps[m]["nfloor"][
                                                        0, z, x, y, r, c
                                                    ]
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
                                    fit_maps[m]["AUC_background"][
                                        0, z, x, y, r, c
                                    ] = 0.0
                                    fit_maps[m]["AUC_no_background"][
                                        0, z, x, y, r, c
                                    ] = 0.0

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
                                fit_spectra[:, z, x, y, r, c] = np.zeros_like(
                                    freq_range
                                )
                            else:
                                fit_spectra[:, z, x, y, r, c] = fit_model(freq_range)

        return fit_spectra, fit_maps

    def plot_fit_spectra(
        self,
        fit_spectra=None,
        fit_results=None,
        fit_params=None,
        rep=None,
        csi_data=None,
        plot_params=None,
    ):
        """
        Plot an overlay of the measured and the fit spectra.

        Parameters
        ----------
        fit_spectra : ndarray
            The fitted spectra.
        fit_results : dict
            The fit results (output from fit_spectra_astropy2).
        rep : int or range
            Which repetition should be plotted.
        csi_data : ndarray
            The CSI data that should be plotted in overlay with the fit_spectra.
        plot_params : dict
            Some plot parameters.

        Returns
        -------
        matplotlib.figure.Figure
            A multisubplot figure.

        Examples
        --------
        define Gaussian linebroadening:
        >>> linebroadening_Hz = 0

        Generate linebroadened CSI dataset (for later comparison)
        >>> csi_data_00Hz = luca_csi.multi_dim_linebroadening(lb=linebroadening_Hz)

        which metabolites should be fit:
        >>> metabs = ["Pyruvate", "Lactate", "Alanine", "PyruvateHydrate"]

        init fit parameters + bounds:
        >>> fit_params = {"lb": linebroadening_Hz,
        ...               "fwhm_1.value": 0.5,
        ...               "fwhm_2.value": 0.5,
        ...               "fwhm_3.value": 0.5,
        ...               "fwhm_4.value": 0.5,
        ...               "fwhm_1.bounds": (0.2, 2),
        ...               "fwhm_2.bounds": (0.2, 2),
        ...               "fwhm_3.bounds": (0.2, 2),
        ...               "fwhm_4.bounds": (0.2, 2)}

        perform fitting
        >>> fit_model_results_lb00Hz = luca_csi.fit_spectra_astropy2(metabs=metabs,
        ...                                                          fit_reps=[0, 0],
        ...                                                          fit_params=fit_params,
        ...                                                          use_multiprocessing=True)
        >>> plot_params = {"ylim": "max", "order": "orgdata"}
        >>> luca_csi.plot_fit_spectra(fit_results=fit_model_results_lb00Hz,
        ...                           rep=0,
        ...                           plot_params=plot_params)

        Can also be used with pseudo-inv fitting method's results:
        >>> plt.close('all')
        >>> for r in range(10):
        ...     plot_params = {"ylim": "max", "order": "orgdata", "plot_diff": False, "plot_title": False, "savefig": True, "savepath": luca_csi.savepath}
        ...     luca_csi.plot_fit_spectra(fit_spectra=np.abs(np.sum(fit_spectrums, axis=-1, keepdims=True)),
        ...                               csi_data=np.abs(luca_csi.csi_image),
        ...                               rep=r,
        ...                               fit_params=fit_params,
        ...                               plot_params=plot_params)
        """

        # load spectra (from fit results or directly)
        if fit_spectra is None:
            if fit_results is not None:
                fit_spectra, fit_maps = self.extract_fit_results(
                    fit_params=fit_params, fit_results=fit_results
                )
            else:
                pass
        else:
            pass

        # load CSI data (passed or from self)
        if csi_data is None:
            return False

        if fit_spectra is None:
            fit_spectra = np.zeros_like(csi_data)
            # set parameters to have nice plot without fit data:
            linewidth_fit = plot_params.get("linewidth_fit", 0)
            plot_params["linewidth_fit"] = linewidth_fit
            alpha_fit = plot_params.get("alpha_fit", 0)
            plot_params["alpha_fit"] = alpha_fit
            linewidth_measurement = plot_params.get("linewidth_measurement", 0.5)
            plot_params["linewidth_measurement"] = linewidth_measurement
            alpha_measurement = plot_params.get("alpha_measurement", 1)
            plot_params["alpha_measurement"] = alpha_measurement

        # check if data is complex, if yes apply ||
        if np.issubdtype(csi_data.dtype, np.complexfloating):
            csi_data = np.abs(csi_data)

        if np.issubdtype(fit_spectra.dtype, np.complexfloating):
            fit_spectra = np.abs(fit_spectra)

        # assume that the data is real/imaginary part in case it goes below zero:
        if np.min(csi_data) < 0 or np.min(fit_spectra) < 0:
            plot_min = min(np.min(csi_data), np.min(fit_spectra))
        else:
            plot_min = 0

        # decide which repetitions to plot:
        if rep is None:
            if fit_results is not None:
                rep = fit_results["nrange"]
            else:
                rep = range(0, 1)
        elif type(rep) == int:
            rep = range(rep, rep + 1)

        # extract plot parameters:
        if plot_params is None:
            plot_params = {}
        xvals = plot_params.get(
            "freq_range",
            get_freq_axis(scan=self, unit="ppm", npoints=csi_data.shape[0]),
        )
        xlim = plot_params.get("xlim", (None, None))
        ylim = plot_params.get("ylim", None)
        plot_diff = plot_params.get("plot_diff", False)
        plot_xy_title = plot_params.get("plot_xy_title", False)

        # toggle if voxel indices should be plotted:
        plot_title = plot_params.get("plot_title", True)
        save_fig = plot_params.get("save_fig", False)
        save_path = plot_params.get("save_path", self.path)
        save_dpi = plot_params.get("dpi", None)
        spine_width = plot_params.get("spine_width", 0.1)
        linewidth_measurement = plot_params.get("linewidth_measurement", 1)
        linewidth_fit = plot_params.get("linewidth_fit", 0.25)
        alpha_fit = plot_params.get("alpha_fit", 1)
        alpha_measurement = plot_params.get("alpha_measurement", 0.5)
        fig_transparent = plot_params.get("fig_transparent", False)

        order = plot_params.get(
            "order", "orgdata"
        )  # "orgdata" (like the data that was fit) or
        # "plotdata" (like it will  be plotted, (rotated by 90 degree)

        # define range:
        x_dim = fit_spectra.shape[2]
        y_dim = fit_spectra.shape[3]
        for r in rep:
            if order == "orgdata":
                fig, axes = plt.subplots(y_dim, x_dim, figsize=(7, 7))
            else:
                fig, axes = plt.subplots(x_dim, y_dim, figsize=(7, 7))

            # Remove margins between subplots
            plt.subplots_adjust(wspace=0, hspace=0)

            for i in range(x_dim):
                for j in range(y_dim):
                    # Remove axis labels
                    if order == "orgdata":
                        ax = axes[y_dim - 1 - j, i]
                    else:
                        ax = axes[i, j]
                    # Remove ticks and tick labels
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                    # Add title
                    if plot_title:
                        ax.set_title(f"({i}, {j})", fontsize=5)

                    # Plot the spectral data in each subplot
                    # axes[i, j].plot(csi_data_00Hz[:,0,i, j, 0,0]-fit_results_lb00Hz[:,0,i, j, 0,0])
                    if plot_diff:
                        ax.plot(
                            xvals,
                            csi_data[:, 0, i, j, r, 0] - fit_spectra[:, 0, i, j, r, 0],
                            linewidth=1,
                        )
                    else:
                        ax.plot(
                            xvals,
                            csi_data[:, 0, i, j, r, 0],
                            linewidth=linewidth_measurement,
                            alpha=alpha_measurement,
                        )
                        ax.plot(
                            xvals,
                            fit_spectra[:, 0, i, j, r, 0],
                            linewidth=linewidth_fit,
                            alpha=alpha_fit,
                        )
                    if ylim == "max":
                        ax.set_ylim((plot_min, np.max(csi_data[:, 0, :, :, r, 0])))
                    else:
                        ax.set_ylim((plot_min, ylim))

                    ax.set_xlim(xlim)

            for ax in axes.flatten():
                for spine in ax.spines.values():
                    spine.set_linewidth(spine_width)
            plt.show()
            if save_fig:
                import os

                save_name = plot_params.get(
                    "save_name", "spectra_fit_overlayed_" + str(r)
                )
                save_format = plot_params.get("save_format", "png")
                if isinstance(save_format, str):
                    pass
                else:
                    save_format = str(save_format)
                plt.savefig(
                    os.path.join(save_path, save_name + "." + save_format),
                    transparent=fig_transparent,
                    dpi=save_dpi,
                )

    def plot_fit_spectra_overlay(
        self,
        fit_spectra=None,
        rep=None,
        csi_data=None,
        marked_voxel=None,
        anatomical_images=None,
        anatomical_image_number=None,
        plot_params={},
    ):
        """
        Plot an overlay of the measured and the fit spectra.

        Parameters
        ----------
        fit_spectra : ndarray
            The fitted spectra.
        rep : int or range
            Which repetition should be plotted.
        csi_data : ndarray
            The CSI data that should be plotted in overlay with the fit_spectra.
        marked_voxel : tuple of int
            Indices of marked spectrum in overlay.
        anatomical_image : BrukerExp instance
            Contains anatomical reference images.
        anatomical_image_number : int
            Number of slice chosen for overlay (indexing starts at 0).
        plot_params : dict
            Some plot parameters.

        Returns
        -------
        matplotlib.figure.Figure
            A multisubplot figure.

        Examples
        --------
        >>> plot_params = {}
        >>> plot_params['figsize'] = (15, 5)  # default
        >>> plot_params['ylim'] = 'max'  # default = None --> all voxels individual
        >>> plot_params['show_tqdm'] = False  # shows progress, default = True
        >>> plot_params['plot_title'] = "Hyperpolarized 13C-Pyruvate"
        >>> plot_params['linecolor_measurement'] = 'yellow'
        >>> plot_params['linecolor_fit'] = 'r'
        >>> plot_params['alpha_measurement'] = 1
        >>> plot_params['alpha_fit'] = 0  # --> don't show fit in small subplots

        >>> plot_params['marked_voxel'] = [5, 5]  # which voxel to highlight
        >>> mv = plot_params['marked_voxel']  # This is done automatically in the function as well
        >>> plot_params['marked_voxel_title'] = f"Spectrum from voxel {mv}"
        >>> plot_params['marked_voxel_ylim'] = (0, 4e6)  # None --> Same ylim as all spectra
        >>> plot_params['marked_voxel_linecolor_measurement'] = 'gray'
        >>> plot_params['marked_voxel_linecolor_fit'] = 'r'
        >>> plot_params['marked_voxel_alpha_measurement'] = 0.5
        >>> plot_params['marked_voxel_alpha_fit'] = 1

        >>> plot_params['marked_voxel_linewidth_fit'] = 1  # --> don't show fit in small subplots
        >>> plot_params['marked_voxel_linewidth_measurement'] = 2  # --> don't show fit in small subplots

        >>> fit_spectra = np.sum(fit_spectrums, axis=-1, keepdims=True)  # sum along metabolite axis
        >>> # fit_spectra = np.take(fit_spectrums, indices=0, axis=-1, keepdims=True)  # Take first metabolite fit result

        >>> csi_object.plot_fit_spectra_overlay(
        >>>     csi_data=csi_object.csi_image,
        >>>     anatomical_images=axial_object,
        >>>     fit_spectra=fit_spectra,
        >>>     anatomical_image_number=3,
        >>>     marked_voxel=(5, 5),
        >>>     plot_params=plot_params
        >>> )
        """
        if csi_data is None and fit_spectra is None:
            raise ValueError("Pass either csi_data or/and fit_spectra!")

        if fit_spectra is None:
            fit_spectra_passed = False
            fit_spectra = np.zeros_like(csi_data)
            # set parameters to have nice plot without fit data:
            linewidth_fit = plot_params.get("linewidth_fit", 0)
            plot_params["linewidth_fit"] = linewidth_fit
            alpha_fit = plot_params.get("alpha_fit", 0)
            plot_params["alpha_fit"] = alpha_fit
            linewidth_measurement = plot_params.get("linewidth_measurement", 0.5)
            plot_params["linewidth_measurement"] = linewidth_measurement
            alpha_measurement = plot_params.get("alpha_measurement", 1)
            plot_params["alpha_measurement"] = alpha_measurement
        else:
            fit_spectra_passed = True

        if csi_data.shape[2:4] != fit_spectra.shape[2:4]:
            raise ValueError(
                f"csi_data and fit_spectra need to have same x-y dimensions but have"
                f"csi_data: {csi_data.shape[2:4]} and "
                f"fit_spectra: {fit_spectra.shape[2:4]}"
            )
        if anatomical_images is None:
            raise ValueError("Please supply anatomical_images")

        # check if data is complex, if yes apply ||
        if np.issubdtype(csi_data.dtype, np.complexfloating):
            csi_data = np.abs(csi_data)

        if np.issubdtype(fit_spectra.dtype, np.complexfloating):
            fit_spectra = np.abs(fit_spectra)

        # assume that the data is real/imaginary part in case it goes below zero:
        if np.min(csi_data) < 0 or np.min(fit_spectra) < 0:
            plot_min = min(np.min(csi_data), np.min(fit_spectra))
        else:
            plot_min = 0

        # decide which repetitions to plot:
        if rep is None:
            rep = range(0, 1)
        elif type(rep) == int:
            rep = range(rep, rep + 1)

        # extract plot parameters:
        if plot_params is None:
            plot_params = {}

        chemical_shift_axis_ppm = plot_params.get(
            "freq_range",
            get_freq_axis(scan=self, unit="ppm", npoints=csi_data.shape[0]),
        )
        xlim = plot_params.get("xlim", (None, None))
        ylim = plot_params.get("ylim", None)
        plot_xy_title = plot_params.get("plot_xy_title", False)

        if marked_voxel is None:
            marked_voxel = plot_params.get("marked_voxel", None)
            if marked_voxel is None:
                marked_voxel = (0, 0)
                plot_params["marked_voxel"] = marked_voxel
            if not isinstance(marked_voxel, tuple):
                try:
                    marked_voxel = tuple(marked_voxel)
                except:
                    pass

        # toggle if voxel indices should be plotted:
        plot_title = plot_params.get("plot_title", True)
        save_fig = plot_params.get("save_fig", False)
        save_path = plot_params.get("save_path", self.path)
        save_dpi = plot_params.get("dpi", None)
        spine_width = plot_params.get("spine_width", 0.1)
        linewidth_measurement = plot_params.get("linewidth_measurement", 0.5)
        linewidth_fit = plot_params.get("linewidth_fit", 0.25)
        marked_voxel_linewidth_measurement = plot_params.get(
            "marked_voxel_linewidth_measurement", 1
        )
        marked_voxel_linewidth_fit = plot_params.get("marked_voxel_linewidth_fit", 1)
        linecolor_measurement = plot_params.get("linecolor_measurement", "y")
        linecolor_fit = plot_params.get("linecolor_fit", "r")
        marked_voxel_linecolor_measurement = plot_params.get(
            "marked_voxel_linecolor_measurement", "y"
        )
        marked_voxel_linecolor_fit = plot_params.get("marked_voxel_linecolor_fit", "r")
        alpha_measurement = plot_params.get("alpha_measurement", 1)
        alpha_fit = plot_params.get("alpha_fit", 1)
        marked_voxel_alpha_measurement = plot_params.get(
            "marked_voxel_alpha_measurement", 1
        )
        marked_voxel_alpha_fit = plot_params.get("marked_voxel_alpha_fit", 1)
        mvt = f"Spectrum from voxel {marked_voxel}"
        marked_voxel_title = plot_params.get("marked_voxel_title", mvt)
        marked_voxel_ylim = plot_params.get("marked_voxel_ylim", "auto")
        fig_transparent = plot_params.get("fig_transparent", False)
        figsize = plot_params.get("figsize", (15, 5))
        show_tqdm = plot_params.get("show_tqdm", True)
        extent = plot_params.get("extent_image", get_extent(data_obj=self))
        vmin = plot_params.get("vmin_image", None)
        vmax = plot_params.get("vmax_image", None)
        showticks = plot_params.get("showticks", False)
        scalebar_factor = plot_params.get("scalebar_factor", 1.0)
        if len(extent) > 1 and len(extent) == 4:
            pass
        else:
            extent = extent[0]

        if ylim == "max":
            ylim = (0, np.max(csi_data))
        else:
            ylim = (0, plot_params.get("ylim", None))

        if marked_voxel_ylim is not None:
            pass
        else:
            marked_voxel_ylim = ylim

        order = plot_params.get(
            "order", "orgdata"
        )  # "orgdata" (like the data that was fit) or
        # "plotdata" (like it will  be plotted, (rotated by 90 degree)

        # define range:
        x_dim = fit_spectra.shape[2]
        y_dim = fit_spectra.shape[3]

        # load anatomical image stack
        image_stack = anatomical_images.seq2d_oriented
        # pixel_size_mm = anatomical_images.method["PVM_SpatResol"]
        from ..utils.utils_general import (
            define_imageFOV_parameters,
            define_imagematrix_parameters,
        )

        # get matrix size and FOV:
        mat_anat = define_imagematrix_parameters(data_obj=anatomical_images)
        fov_anat = define_imageFOV_parameters(data_obj=anatomical_images)
        # calculate resolution:
        res_anat = [f / m for f, m in zip(fov_anat, mat_anat)]
        # z-x-y --> 1 = "x"
        pixel_size_mm = res_anat[1]
        for r in rep:
            fig, ax = plt.subplots(figsize=figsize)

            ax.imshow(
                np.rot90(np.squeeze(image_stack[0, anatomical_image_number])),
                cmap="bone",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )

            if not showticks:
                ax.axis("off")

            ax.set_title(plot_title)
            add_scalebar(
                px=1e-1,
                ax=ax,
                units="cm",
                fixed_value=1,
                color="w",
                box_alpha=0.0,
                box_color="w",
                location="lower right",
                frameon=True,
                pad=0.075,
                length_fraction=None,
                border_pad=0.075,
                fontsize=12,
            )

            height = (np.abs(extent[2]) + np.abs(extent[3])) / csi_data.shape[3]
            colorlist = ["dodgerblue", "magenta"]
            irange = np.arange(
                extent[2],
                extent[3],
                height,
            )[::-1]
            width = (np.abs(extent[0]) + np.abs(extent[1])) / csi_data.shape[2]
            jrange = np.arange(
                extent[0],
                extent[1],
                width,
            )

            for i in tqdm(range(csi_data.shape[3]), disable=not show_tqdm):
                for j in tqdm(range(csi_data.shape[2]), disable=not show_tqdm):
                    # bottom_anchor = i / csi_data.shape[2]
                    # left_anchor = j / csi_data.shape[3]
                    bottom_anchor = irange[i]
                    left_anchor = jrange[j]
                    # width = 1 / csi_data.shape[3]
                    # height = 1 / csi_data.shape[2]
                    # print(f"bottom_anchor={bottom_anchor}, left_anchor={left_anchor}")

                    axin = ax.inset_axes(
                        [left_anchor, bottom_anchor, width, height],
                        transform=ax.transData,
                    )  # [left, bottom, width, height]

                    axin.plot(
                        chemical_shift_axis_ppm,
                        np.rot90(np.squeeze(csi_data[:, 0, :, :, r]), axes=(1, 2))[
                            :, i, j
                        ],
                        color=linecolor_measurement,
                        linewidth=linewidth_measurement,
                        alpha=alpha_measurement,
                    )
                    if fit_spectra is not None:
                        axin.plot(
                            chemical_shift_axis_ppm,
                            np.rot90(
                                np.squeeze(fit_spectra[:, 0, :, :, r]), axes=(1, 2)
                            )[:, i, j],
                            color=linecolor_fit,
                            linewidth=linewidth_fit,
                            alpha=alpha_fit,
                        )

                    axin.invert_xaxis()

                    axin.set_xlim(xlim)
                    axin.set_ylim(ylim)

                    axin.set_xticks([])
                    axin.set_yticks([])

                    axin.patch.set_alpha(0)

                    if (j, i) == marked_voxel:
                        axin.spines["bottom"].set_color(colorlist[1])
                        axin.spines["top"].set_color(colorlist[1])
                        axin.spines["left"].set_color(colorlist[1])
                        axin.spines["right"].set_color(colorlist[1])

                        axin.spines["bottom"].set_linewidth(1.5)
                        axin.spines["top"].set_linewidth(1.5)
                        axin.spines["left"].set_linewidth(1.5)
                        axin.spines["right"].set_linewidth(1.5)

                    else:
                        axin.spines["bottom"].set_color(colorlist[0])
                        axin.spines["top"].set_color(colorlist[0])
                        axin.spines["left"].set_color(colorlist[0])
                        axin.spines["right"].set_color(colorlist[0])

                        axin.spines["bottom"].set_linewidth(0.5)
                        axin.spines["top"].set_linewidth(0.5)
                        axin.spines["left"].set_linewidth(0.5)
                        axin.spines["right"].set_linewidth(0.5)

            axin = ax.inset_axes([1.2, 0.2, 0.4, 0.4])

            axin.plot(
                chemical_shift_axis_ppm,
                np.rot90(np.squeeze(csi_data[:, 0, :, :, r]), axes=(1, 2))[
                    :, marked_voxel[1], marked_voxel[0]
                ],
                color=marked_voxel_linecolor_measurement,
                linewidth=marked_voxel_linewidth_measurement,
                label=marked_voxel_title,
                alpha=marked_voxel_alpha_measurement,
            )
            if fit_spectra_passed:
                axin.plot(
                    chemical_shift_axis_ppm,
                    np.rot90(np.squeeze(fit_spectra[:, 0, :, :, r]), axes=(1, 2))[
                        :, marked_voxel[1], marked_voxel[0]
                    ],
                    color=marked_voxel_linecolor_fit,
                    linewidth=marked_voxel_linewidth_fit,
                    alpha=marked_voxel_alpha_fit,
                    label="fit",
                )
                axin.legend()

            axin.invert_xaxis()

            axin.set_xlim(xlim)
            if marked_voxel_ylim == "auto":
                ymax_measured = np.max(
                    np.rot90(np.squeeze(csi_data[:, 0, :, :, r]), axes=(1, 2))[
                        :, marked_voxel[1], marked_voxel[0]
                    ]
                )
                if fit_spectra_passed:
                    ymax_fit = np.max(
                        np.rot90(np.squeeze(fit_spectra[:, 0, :, :, r]), axes=(1, 2))[
                            :, marked_voxel[1], marked_voxel[0]
                        ]
                    )
                else:
                    ymax_fit = 0
                ymax = 1.1 * max(ymax_measured, ymax_fit)
                axin.set_ylim((0, ymax))
            else:
                axin.set_ylim(marked_voxel_ylim)

            axin.set_xlabel(r"chemical shift [ppm]")
            axin.set_ylabel(r"signal magnitude [a.u.]")

            axin.spines["bottom"].set_color(colorlist[1])
            axin.spines["top"].set_color(colorlist[1])
            axin.spines["left"].set_color(colorlist[1])
            axin.spines["right"].set_color(colorlist[1])

            leg = axin.legend(bbox_to_anchor=(1.0, 1.7), edgecolor="k")

            if save_fig:
                import os

                save_name = plot_params.get("save_name", "spectra_overlayed_" + str(r))
                save_format = plot_params.get("save_format", "png")
                if isinstance(save_format, str):
                    pass
                else:
                    save_format = str(save_format)
                plt.savefig(
                    os.path.join(save_path, save_name + "." + save_format),
                    transparent=fig_transparent,
                    dpi=save_dpi,
                )

    def plot_fit_results_interactive(
        self,
        axlist,
        anat=None,
        fig=None,
        fit_maps=None,
        fit_results=None,
        metabs=["Pyruvate", "Lactate"],
        ratio_map=True,
        plot_params=None,
        fit_params=None,
        animal_mask=None,
        display_ui=True,
    ):
        from ipywidgets import widgets, interactive, HBox, VBox, FloatSlider, Layout
        import os

        # Initialize list to keep track of colorbar references
        if not hasattr(self, "colorbars_plotting"):
            self.colorbars_plotting = [
                None,
                None,
                None,
            ]  # Assuming three axes for simplicity

        def remove_existing_colorbars():
            # Remove existing colorbars if they exist
            for i, cbar in enumerate(self.colorbars_plotting):
                if cbar is not None:
                    cbar.remove()
            self.colorbars_plotting = [None, None, None]  # Reset colorbar references

            # shift anat to match CSI:

        if plot_params is None:
            plot_params = {}

        if fit_maps is None:
            if fit_results is not None:
                # extract fit data from fit model:
                fit_spectra, fit_maps = self.extract_fit_results(
                    fit_params=fit_params, fit_results=fit_results
                )
            else:
                return False

        def plot_img(
            fit_maps=None,
            plot_params=None,
            metab_1_threshold=0,
            metab_2_threshold=0,
            ratio_threshold=0,
        ):
            remove_existing_colorbars()  # Call function to remove existing colorbars

            # extract plot parameters:
            vmin_metab1 = plot_params.get("vmin_metab1", None)
            vmax_metab1 = plot_params.get("vmax_metab1", None)
            vmin_metab2 = plot_params.get("vmin_metab2", None)
            vmax_metab2 = plot_params.get("vmax_metab2", None)
            vmin_ratio = plot_params.get("vmin_ratio", None)
            vmax_ratio = plot_params.get("vmax_ratio", None)
            axial_slice = plot_params.get("axial_slice", None)
            sum_metabs = plot_params.get("sum_metabs", None)
            figsize = plot_params.get("figsize", (10, 10))
            cmap = plot_params.get("cmap", "jet")
            alpha = plot_params.get("alpha", 0.5)
            rep = plot_params.get("rep", 0)
            map_type = plot_params.get("map_type", "ratio_map")
            interp_factor = plot_params.get("interp_factor", 2)
            interp_method = plot_params.get("interp_method", "lanczos")

            # generate masks:
            threshold_metab = np.zeros((3,))

            # lower threshold:
            threshold_metab[0] = plot_params.get("threshold_metab1", metab_1_threshold)
            threshold_metab[1] = plot_params.get("threshold_metab2", metab_2_threshold)
            threshold_metab[2] = plot_params.get("threshold_ratio", ratio_threshold)
            if isinstance(fit_maps, dict):
                pass
            else:
                fit_maps_temp = {}
                # print(f"fit_maps_temp={fit_maps_temp}")
                fit_maps_temp[metabs[0]] = {}
                fit_maps_temp[metabs[0]] = fit_maps[:, :, :, :, :, :, 0]
                fit_maps_temp[metabs[1]] = {}
                fit_maps_temp[metabs[1]] = fit_maps[:, :, :, :, :, :, 1]
                # print(f"fit_maps_temp={fit_maps_temp}")
                fit_maps = fit_maps_temp

            metabs_mask = np.ones(
                (
                    2,
                    fit_maps[metabs[0]].shape[2],
                    fit_maps[metabs[0]].shape[3],
                )
            )
            metabs_mask[0] = np.where(
                np.squeeze(fit_maps[metabs[0]][:, :, :, :, rep, :]) < metab_1_threshold,
                0,
                1,
            )

            metabs_mask[1] = np.where(
                np.squeeze(fit_maps[metabs[1]][:, :, :, :, rep, :]) < metab_2_threshold,
                0,
                1,
            )

            # get lac pyr ratio map:
            metab_map = np.zeros(
                (
                    np.size(metabs),
                    fit_maps[metabs[0]].shape[3],
                    fit_maps[metabs[0]].shape[2],
                )
            )

            interpolated = np.zeros(
                (
                    np.size(metabs),
                    fit_maps[metabs[0]].shape[3] * interp_factor,
                    fit_maps[metabs[0]].shape[2] * interp_factor,
                )
            )

            if anat is None:
                anat_image = None
                anat_overlay = 0
                alpha = 1
            else:
                try:
                    anat_image = anat.seq2d_oriented
                except:
                    anat_image = shift_anat(
                        anat_obj=anat,
                        csi_obj=self,
                        use_scipy_shift=True,
                    )

                anat_overlay = 1

            for k, m in enumerate(metabs):
                metab_map[k, :, :] = np.squeeze(fit_maps[m][0, 0, :, :, rep, 0])

                # apply mask:
                metab_map[k, :, :] = metab_map[k, :, :] * metabs_mask[k, :, :]

                # Interpolate the image by a factor:
                interpolated[k, :, :] = img_interp(
                    metab_image=metab_map[k, :, :],
                    interp_factor=interp_factor,
                    cmap=cmap,
                    interp_method=interp_method,
                    threshold=threshold_metab[k],
                    overlay=True,
                )

            if ratio_map is True:
                denominator = interpolated[0, :, :]
                numerator = interpolated[1, :, :]

                # Initialize an array of the same shape with zeros or NaNs
                map_ratio = np.zeros(denominator.shape)

                # Only perform division where denominator is not zero
                non_zero_indices = denominator != 0
                map_ratio[non_zero_indices] = (
                    numerator[non_zero_indices] / denominator[non_zero_indices]
                )

            if animal_mask is not None:
                from ..utils.utils_spectroscopy import make_NDspec_6Dspec

                interpolated_temp = make_NDspec_6Dspec(
                    input_data=interpolated, provided_dims=["channels", "x", "y"]
                )
                from ..utils.utils_general import apply_mask

                interpolated_temp_masked = apply_mask(
                    mask=animal_mask,
                    input_data=interpolated_temp,
                    return_nans=True,
                    bitwise=False,
                    mask_slice_ind=axial_slice,
                )

                interpolated_temp_masked = np.transpose(
                    np.squeeze(
                        interpolated_temp_masked,
                    ),
                    (2, 0, 1),
                )

                interpolated = interpolated_temp_masked

                if ratio_map is True:
                    print(map_ratio.shape)
                    map_ratio_temp = make_NDspec_6Dspec(
                        input_data=map_ratio, provided_dims=["x", "y"]
                    )
                    print(map_ratio_temp.shape)

                    map_ratio_temp_masked = apply_mask(
                        mask=animal_mask,
                        input_data=map_ratio_temp,
                        return_nans=True,
                        bitwise=False,
                        mask_slice_ind=axial_slice,
                    )
                    #
                    # map_ratio_temp_masked = np.transpose(
                    #     np.squeeze(
                    #         map_ratio_temp_masked,
                    #     ),
                    #     (0, 1),
                    # )
                    map_ratio = np.squeeze(map_ratio_temp_masked)

            # define extent:
            from ..utils.utils_general import get_extent

            ax_anat_ext, _, _ = get_extent(data_obj=anat)
            ax_csi_ext, _, _ = get_extent(data_obj=self)
            if axial_slice is None:
                axial_range = range(anat_image.shape[1])
            else:
                axial_range = range(axial_slice, axial_slice + 1)

            plot_params["save_fig"] = plot_params.get("save_fig", False)
            plot_params["save_path"] = plot_params.get("save_path", self.path)
            plot_params["save_format"] = plot_params.get("save_format", "png")
            plot_params["save_name"] = plot_params.get("save_name", None)

            for k in axial_range:
                if ratio_map is True:
                    if anat_image is not None:
                        axlist[0].imshow(
                            np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                            cmap="bone",
                            extent=ax_anat_ext,
                            alpha=anat_overlay,
                        )
                    im_ax = axlist[0].imshow(
                        np.rot90(interpolated[0, :, :]),
                        alpha=alpha,
                        cmap=cmap,
                        extent=ax_csi_ext,
                        vmin=vmin_metab1,
                        vmax=vmax_metab1,
                    )
                    if fig is not None:
                        self.colorbars_plotting[0] = fig.colorbar(
                            im_ax,
                            ax=axlist[0],
                            fraction=0.033,
                            pad=0.04,
                        )
                    # set colorbar range, use 3 ticks (min, mean, max)
                    # cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
                    # cbar.set_ticklabels(
                    #    np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
                    # )
                    axlist[0].set_title(metabs[0])
                    if anat_image is not None:
                        axlist[1].imshow(
                            np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                            cmap="bone",
                            extent=ax_anat_ext,
                            alpha=anat_overlay,
                        )
                    im_ax = axlist[1].imshow(
                        np.rot90(interpolated[1, :, :]),
                        alpha=alpha,
                        cmap=cmap,
                        extent=ax_csi_ext,
                        vmin=vmin_metab2,
                        vmax=vmax_metab2,
                    )
                    if fig is not None:
                        self.colorbars_plotting[1] = fig.colorbar(
                            im_ax,
                            ax=axlist[1],
                            fraction=0.033,
                            pad=0.04,
                        )

                    axlist[1].set_title(metabs[1])

                    if anat_image is not None:
                        plt.imshow(
                            np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                            cmap="bone",
                            extent=ax_anat_ext,
                            alpha=anat_overlay,
                        )

                    axlist[2].set_title(metabs[1] + "/" + metabs[0])
                    im_ax = axlist[2].imshow(
                        np.rot90(map_ratio),
                        alpha=alpha,
                        cmap=cmap,
                        extent=ax_csi_ext,
                        vmin=vmin_ratio,
                        vmax=vmax_ratio,
                    )
                    if fig is not None:
                        self.colorbars_plotting[2] = fig.colorbar(
                            im_ax,
                            ax=axlist[2],
                            fraction=0.033,
                            pad=0.04,
                        )
                    plt.tight_layout()

                    if plot_params["save_fig"]:
                        if plot_params["save_name"] is not None:
                            save_name = plot_params["save_name"] + "_" + str(k).zfill(2)
                        else:
                            save_name = f"{metabs[0]}_{metabs[1]}_{str(k).zfill(2)}"
                        plt.savefig(
                            os.path.join(
                                plot_params["save_path"],
                                save_name + "." + plot_params["save_format"],
                            )
                        )

        @widgets.interact
        def update(
            metab_1_threshold=(
                0.0,
                np.max(fit_maps[:, :, :, :, :, :, 0]),
                np.max(fit_maps[:, :, :, :, :, :, 0]) / 1000,
            ),
            metab_2_threshold=(
                0.0,
                np.max(fit_maps[:, :, :, :, :, :, 1]),
                np.max(fit_maps[:, :, :, :, :, :, 1]) / 1000,
            ),
            ratio_threshold=(
                0.0,
                20,
                201,
            ),
        ):
            plot_img(
                fit_maps,
                plot_params,
                metab_1_threshold,
                metab_2_threshold,
                ratio_threshold,
            )

    def plot_fit_results(
        self,
        anat=None,
        fit_maps=None,
        fit_results=None,
        metabs=["Pyruvate", "Lactate"],
        map_type="AUC_no_background",
        rep=0,
        ratio_map=True,
        interp_method="lanczos",
        plot_params=None,
        fit_params=None,
        interp_factor=1,
    ):
        """
        Visualize the fit results by plotting the specified maps alongside an anatomical image.

        Parameters
        ----------
        anat : object, optional
            Axial image object used as background for overlaying fit maps.
        fit_maps : dict, optional
            Dictionary containing fit results maps for different metabolites.
        fit_results : object, optional
            Fit results that can be converted to fit_maps if fit_maps is not provided.
        metabs : list of str
            List of metabolites to be plotted. Default is ["Pyruvate", "Lactate"].
        map_type : str
            Specifies which fit data results to plot. Default is "AUC_no_background".
        rep : int
            Specifies which repetition of the fitting to plot. Default is 0.
        ratio_map : bool
            If True, plots a ratio of metab[1] to metab[0]. Default is True.
        interp_method : str
            Specifies the interpolation method to be applied. Default is "lanczos".
        plot_params : dict, optional
            Dictionary containing additional plotting parameters like vmin and vmax for the color scale.
        fit_params : dict, optional
            Parameters from the fitting operation.
        interp_factor : int
            Factor by which the image will be interpolated.

        Returns
        -------
        None
        """
        import os

        # shift anat to match CSI:
        if plot_params is None:
            plot_params = {}

        if fit_maps is None:
            if fit_results is not None:
                # extract fit data from fit model:
                fit_spectra, fit_maps = self.extract_fit_results(
                    fit_params=fit_params, fit_results=fit_results
                )
            else:
                return False

        # extract plot parameters:
        vmin_metab1 = plot_params.get("vmin_metab1", None)
        vmax_metab1 = plot_params.get("vmax_metab1", None)
        vmin_metab2 = plot_params.get("vmin_metab2", None)
        vmax_metab2 = plot_params.get("vmax_metab2", None)
        vmin_ratio = plot_params.get("vmin_ratio", None)
        vmax_ratio = plot_params.get("vmax_ratio", None)
        axial_slice = plot_params.get("axial_slice", None)
        sum_metabs = plot_params.get("sum_metabs", None)
        figsize = plot_params.get("figsize", (10, 10))
        cmap = plot_params.get("cmap", "jet")
        alpha = plot_params.get("alpha", 0.5)

        # generate masks:
        threshold_metab = np.zeros((3,))

        # lower threshold:
        threshold_metab[0] = plot_params.get("threshold_metab1", 0)
        threshold_metab[1] = plot_params.get("threshold_metab2", 0)
        threshold_metab[2] = plot_params.get("threshold_ratio", 0)

        if isinstance(fit_maps, dict):
            pass
        else:
            fit_maps_temp = {}
            # print(f"fit_maps_temp={fit_maps_temp}")
            fit_maps_temp[metabs[0]] = {}
            fit_maps_temp[metabs[0]][map_type] = fit_maps[:, :, :, :, :, :, 0]
            fit_maps_temp[metabs[1]] = {}
            fit_maps_temp[metabs[1]][map_type] = fit_maps[:, :, :, :, :, :, 1]
            # print(f"fit_maps_temp={fit_maps_temp}")
            fit_maps = fit_maps_temp

        metabs_mask = np.ones(
            (
                2,
                fit_maps[metabs[0]][map_type].shape[2],
                fit_maps[metabs[0]][map_type].shape[3],
            )
        )
        metabs_mask[0][
            np.squeeze(fit_maps[metabs[0]][map_type][:, :, :, :, rep, :])
            < 0  # threshold_metab[0]
        ] = np.nan
        metabs_mask[1][
            np.squeeze(fit_maps[metabs[0]][map_type][:, :, :, :, rep, :])
            < 0  # threshold_metab[1]
        ] = np.nan

        # get lac pyr ratio map:
        metab_map = np.zeros(
            (
                np.size(metabs),
                fit_maps[metabs[0]][map_type].shape[3],
                fit_maps[metabs[0]][map_type].shape[2],
            )
        )

        interpolated = np.zeros(
            (
                np.size(metabs),
                fit_maps[metabs[0]][map_type].shape[3] * interp_factor,
                fit_maps[metabs[0]][map_type].shape[2] * interp_factor,
            )
        )

        if anat is None:
            anat_image = None
            anat_overlay = 0
            alpha = 1
        else:
            try:
                anat_image = anat.seq2d_oriented
            except:
                anat_image = shift_anat(
                    anat_obj=anat,
                    csi_obj=self,
                    use_scipy_shift=True,
                )

            anat_overlay = 1
        for k, m in enumerate(metabs):
            metab_map[k, :, :] = np.rot90(
                np.squeeze(fit_maps[m][map_type][0, 0, :, :, rep, 0])
            )

            # apply mask:
            metab_map[k, :, :] = metab_map[k, :, :] * np.rot90(metabs_mask[k, :, :])

            # Interpolate the image by a factor:
            interpolated[k, :, :] = img_interp(
                metab_image=metab_map[k, :, :],
                interp_factor=interp_factor,
                cmap=cmap,
                interp_method=interp_method,
                threshold=threshold_metab[k],
                overlay=True,
            )

        if ratio_map is True:
            denominator = interpolated[0, :, :]
            numerator = interpolated[1, :, :]

            # Initialize an array of the same shape with zeros or NaNs
            map_ratio = np.zeros(denominator.shape)

            # Only perform division where denominator is not zero
            non_zero_indices = denominator != 0
            map_ratio[non_zero_indices] = (
                numerator[non_zero_indices] / denominator[non_zero_indices]
            )

        # define extent:
        from ..utils.utils_general import get_extent

        ax_anat_ext, _, _ = get_extent(data_obj=anat)
        ax_csi_ext, _, _ = get_extent(data_obj=self)
        if axial_slice is None:
            axial_range = range(anat_image.shape[1])
        else:
            axial_range = range(axial_slice, axial_slice + 1)

        plot_params["save_fig"] = plot_params.get("save_fig", False)
        plot_params["save_path"] = plot_params.get("save_path", self.path)
        plot_params["save_format"] = plot_params.get("save_format", "png")
        plot_params["save_name"] = plot_params.get("save_name", None)

        for k in axial_range:
            plt.figure(figsize=figsize, tight_layout=True)
            fig = plt.gcf()
            if ratio_map is True:
                ax_1 = plt.subplot(1, 3, 1)
                if anat_image is not None:
                    plt.imshow(
                        np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                        cmap="bone",
                        extent=ax_anat_ext,
                        alpha=anat_overlay,
                    )
                im_ax = plt.imshow(
                    interpolated[0, :, :],
                    alpha=alpha,
                    cmap=cmap,
                    extent=ax_csi_ext,
                    vmin=vmin_metab1,
                    vmax=vmax_metab1,
                )
                cbar = fig.colorbar(
                    im_ax,
                    ax=ax_1,
                    fraction=0.033,
                    pad=0.04,
                )
                # set colorbar range, use 3 ticks (min, mean, max)
                # cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
                # cbar.set_ticklabels(
                #    np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
                # )
                ax_1.set_title(metabs[0])
                ax_2 = plt.subplot(1, 3, 2)
                if anat_image is not None:
                    plt.imshow(
                        np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                        cmap="bone",
                        extent=ax_anat_ext,
                        alpha=anat_overlay,
                    )
                im_ax = plt.imshow(
                    interpolated[1, :, :],
                    alpha=alpha,
                    cmap=cmap,
                    extent=ax_csi_ext,
                    vmin=vmin_metab2,
                    vmax=vmax_metab2,
                )
                cbar = fig.colorbar(
                    im_ax,
                    ax=ax_2,
                    fraction=0.033,
                    pad=0.04,
                )
                ax_2.set_title(metabs[1])
                ax_3 = plt.subplot(1, 3, 3)
                if anat_image is not None:
                    plt.imshow(
                        np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                        cmap="bone",
                        extent=ax_anat_ext,
                        alpha=anat_overlay,
                    )
                ax_3.set_title(metabs[1] + "/" + metabs[0])
                im_ax = plt.imshow(
                    map_ratio,
                    alpha=alpha,
                    cmap=cmap,
                    extent=ax_csi_ext,
                    vmin=vmin_ratio,
                    vmax=vmax_ratio,
                )
                cbar = fig.colorbar(
                    im_ax,
                    ax=ax_3,
                    fraction=0.033,
                    pad=0.04,
                )
                if plot_params["save_fig"]:
                    if plot_params["save_name"] is not None:
                        save_name = plot_params["save_name"] + "_" + str(k).zfill(2)
                    else:
                        save_name = f"Ratio_{metabs[1]}_{metabs[0]}_{str(k).zfill(2)}"
                    plt.savefig(
                        os.path.join(
                            plot_params["save_path"],
                            save_name + "." + plot_params["save_format"],
                        )
                    )
            elif sum_metabs is True:
                ax_1 = plt.subplot(1, 3, 1)
                if anat_image is not None:
                    plt.imshow(
                        np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                        cmap="bone",
                        extent=ax_anat_ext,
                        alpha=anat_overlay,
                    )

                # set colorbar range, use 3 ticks (min, mean, max)
                # cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
                # cbar.set_ticklabels(
                #    np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
                # )
                ax_1.set_title("Anatomical")
                ax_2 = plt.subplot(1, 3, 2)
                if anat_image is not None:
                    plt.imshow(
                        np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                        cmap="bone",
                        extent=ax_anat_ext,
                        alpha=anat_overlay,
                    )
                im_ax = plt.imshow(
                    np.nan_to_num(interpolated[1, :, :])
                    + np.nan_to_num(interpolated[0, :, :]),
                    alpha=alpha,
                    cmap=cmap,
                    extent=ax_csi_ext,
                    vmin=vmin_metab1,
                    vmax=vmax_metab1,
                )
                cbar = fig.colorbar(
                    im_ax,
                    ax=ax_2,
                    fraction=0.033,
                    pad=0.04,
                )

                ax_2.set_title(metabs[0] + " + " + metabs[1])
            else:
                ax_1 = plt.subplot(1, 2, 1)
                plt.imshow(
                    np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                    cmap="bone",
                    extent=ax_anat_ext,
                    alpha=anat_overlay,
                )
                im_ax = plt.imshow(
                    interpolated[0, :, :],
                    alpha=alpha,
                    cmap=cmap,
                    extent=ax_csi_ext,
                    vmin=vmin_metab1,
                    vmax=vmax_metab1,
                )
                cbar = fig.colorbar(
                    im_ax,
                    ax=ax_1,
                    fraction=0.033,
                    pad=0.04,
                )
                ax_1.set_title(metabs[0])
                ax_2 = plt.subplot(1, 2, 2)
                im_ax = plt.imshow(
                    np.rot90(np.squeeze(anat_image[0, k, :, :, 0, 0])),
                    cmap="bone",
                    extent=ax_anat_ext,
                    alpha=anat_overlay,
                )
                im_ax = plt.imshow(
                    interpolated[1, :, :],
                    alpha=alpha,
                    cmap=cmap,
                    extent=ax_csi_ext,
                    vmin=vmin_metab2,
                    vmax=vmax_metab2,
                )
                cbar = fig.colorbar(
                    im_ax,
                    ax=ax_2,
                    fraction=0.033,
                    pad=0.04,
                )
                ax_2.set_title(metabs[1])
                if plot_params["save_fig"]:
                    if plot_params["save_name"] is not None:
                        save_name = plot_params["save_name"] + "_" + str(k).zfill(2)
                    else:
                        save_name = f"{metabs[0]}_{metabs[1]}_{str(k).zfill(2)}"
                    plt.savefig(
                        os.path.join(
                            plot_params["save_path"],
                            save_name + "." + plot_params["save_format"],
                        )
                    )
            ##print(f"Slice: {k}")

    def analyze_freq_maps(
        self,
        mask,
        frequency_map,
        anat_ref_img,
        anat_ref_slice,
        temperature_map=None,
        colormap="plasma",
        plot_csi_anat_ref=False,
        savepath=None,
        colorbarticks=None,
        binsize=0.5,
        plot_params={},
    ):
        """
        This functions analyzes frequency maps retrieved from fitting peaks for the temperature dependency study of pyruvate and lactate.
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
        t_vmin = plot_params.get("t_vmin", 30.0)
        t_vmax = plot_params.get("t_vmax", 42.0)
        t_alpha = plot_params.get("t_alpha", 0.5)

        def gaussian(x, A, mu, sigma):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        displayed_data = np.squeeze(np.abs(np.sum(self.seq2d_reordered, axis=0)))

        masked_frq_map = np.rot90(frequency_map * mask)
        meaned_frq_all = np.nanmean(masked_frq_map)
        std_frq_all = np.nanstd(masked_frq_map)

        # find out if frequency input is in ppm or hz
        if np.nanmean(masked_frq_map) > 30:
            frequency_string = "Hz"
        else:
            frequency_string = "ppm"

        if temperature_map is not None:
            masked_temp_map = np.rot90(temperature_map * mask)
            meaned_temp_all = np.nanmean(masked_temp_map)
            std_temp_all = np.nanstd(masked_temp_map)

        else:
            masked_temp_map = np.zeroes_like(frequency_map)
            meaned_temp_all = 0
            std_temp_all = 0
            logger.warning("No temperature map passed")

        if plot_csi_anat_ref is True:
            fig, ax = plt.subplots(2, 2, figsize=(13, 5), tight_layout=True)
            im0 = ax[0, 0].imshow(np.rot90(displayed_data), cmap=colormap)
            divider = make_axes_locatable(ax[0, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im0, cax=cax, orientation="vertical", label="SNR [a.u.]")
            ax[0, 1].imshow(
                np.rot90(anat_ref_img.seq2d_oriented[0, anat_ref_slice, :, :, 0, 0]),
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
                im2,
                cax=cax,
                orientation="vertical",
                label="f [" + str(frequency_string) + "]",
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

        if temperature_map is not None:
            fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(12, 3))
            ax[0].imshow(
                np.rot90(anat_ref_img.seq2d_oriented[0, anat_ref_slice, :, :, 0, 0]),
                cmap="bone",
                extent=get_plotting_extent(data_obj=self),
            )
            im1 = ax[0].imshow(
                masked_temp_map,
                interpolation="None",
                extent=get_plotting_extent(data_obj=self),
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
                + str(frequency_string)
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

            x_ax_freq_hist = np.linspace(
                np.nanmin(masked_frq_map), np.nanmax(masked_frq_map), 500
            )

            ax[1].plot(
                x_ax_freq_hist,
                norm.pdf(x_ax_freq_hist, loc=meaned_frq_all, scale=std_frq_all),
                label="Probability distribution",
                color="C1",
            )

            # Normalise the histogram values
            x_hist_freq = np.ravel(masked_frq_map)[~np.isnan(np.ravel(masked_frq_map))]
            weights = np.ones_like(x_hist_freq) / len(x_hist_freq)

            bins = int(np.round((np.max(x_hist_freq) - np.min(x_hist_freq)) / binsize))

            ax[1].hist(
                x_hist_freq,
                weights=weights,
                alpha=0.7,
                color="C0",
                edgecolor="k",
                bins=bins,
            )

            ax[1].vlines(
                meaned_frq_all,
                0,
                norm.pdf(meaned_frq_all, loc=meaned_frq_all, scale=std_frq_all),
                color="k",
                linestyle="dashed",
                label="Mean",
            )
            ax[1].set_title(
                r"$\mu=$"
                + str(np.round(meaned_frq_all, 1))
                + r"$ / \sigma =$"
                + str(np.round(std_frq_all, 2))
                + r"$Hz$"
            )
            ax[1].set_xlabel("f [" + str(frequency_string) + "]")
            ax[1].legend()
            ax[1].set_ylabel("Probability")
            print("bin size Frequency", binsize)

            ### temperature
            x_ax_temp_hist = np.linspace(
                np.nanmin(masked_temp_map), np.nanmax(masked_temp_map), 500
            )

            ax[2].plot(
                x_ax_temp_hist,
                norm.pdf(x_ax_temp_hist, loc=meaned_temp_all, scale=std_temp_all),
                label="Probability distribution",
                color="C1",
            )

            # Normalise the histogram values
            x_hist_temp = np.ravel(masked_temp_map)[
                ~np.isnan(np.ravel(masked_temp_map))
            ]
            weights = np.ones_like(x_hist_temp) / len(x_hist_temp)
            bins = int(np.round((np.max(x_hist_temp) - np.min(x_hist_temp)) / binsize))

            ax[2].hist(
                x_hist_temp,
                weights=weights,
                alpha=0.7,
                color="C0",
                edgecolor="k",
                bins=bins,
            )

            ax[2].vlines(
                meaned_temp_all,
                0,
                norm.pdf(meaned_temp_all, loc=meaned_temp_all, scale=std_temp_all),
                color="k",
                linestyle="dashed",
                label="Mean",
            )
            ax[2].set_title(
                r"$\mu=$"
                + str(np.round(meaned_temp_all, 1))
                + r"$ / \sigma =$"
                + str(np.round(std_temp_all, 2))
                + r"[$^{\circ}$C]"
            )
            ax[2].set_xlabel("T [$^{\circ}$C]")
            ax[2].legend()
            ax[2].set_ylabel("Probability")
            print("bin size Temperature", binsize)

            output_dict = {
                "meaned_frq_all_pixels": meaned_frq_all,
                "std_frq_all_pixels": std_frq_all,
                "meaned_temp_all_pixels": meaned_temp_all,
                "std_temp_all_pixels": std_temp_all,
                "masked_frq_map": masked_frq_map,
                "masked_temp_map": masked_temp_map,
            }
        else:
            fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(8, 3))
            ax[0].imshow(
                np.rot90(anat_ref_img.seq2d_oriented[0, anat_ref_slice, :, :, 0, 0]),
                cmap="bone",
                extent=get_plotting_extent(data_obj=self),
            )
            im1 = ax[0].imshow(
                masked_frq_map,
                alpha=0.7,
                interpolation="None",
                extent=get_plotting_extent(data_obj=self),
                cmap=colormap,
            )
            ax[0].set_title(
                " °C - f=" + str(np.round(meaned_frq_all, 2)) + str(frequency_string)
            )
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(
                im1,
                cax=cax,
                orientation="vertical",
                label="f [" + str(frequency_string) + "]",
                ticks=colorbarticks,
            )

            x_ax_freq_hist = np.linspace(
                np.nanmin(masked_frq_map), np.nanmax(masked_frq_map), 500
            )

            ax[1].plot(
                x_ax_freq_hist,
                norm.pdf(x_ax_freq_hist, loc=meaned_frq_all, scale=std_frq_all),
                label="Probability distribution",
                color="C1",
            )

            # Normalise the histogram values
            x_hist_freq = np.ravel(masked_frq_map)[~np.isnan(np.ravel(masked_frq_map))]
            weights = np.ones_like(x_hist_freq) / len(x_hist_freq)
            bins = int(np.round((np.max(x_hist_freq) - np.min(x_hist_freq)) / binsize))
            ax[1].hist(
                x_hist_freq,
                weights=weights,
                alpha=0.7,
                color="C0",
                edgecolor="k",
                bins=bins,
            )

            ax[1].vlines(
                meaned_frq_all,
                0,
                norm.pdf(meaned_frq_all, loc=meaned_frq_all, scale=std_frq_all),
                color="k",
                linestyle="dashed",
                label="Mean",
            )
            ax[1].set_title(
                r"$\mu=$"
                + str(np.round(meaned_frq_all, 1))
                + r"$ / \sigma =$"
                + str(np.round(std_frq_all, 2))
                + r"$Hz$"
            )
            ax[1].set_xlabel("f [" + str(frequency_string) + "]")
            ax[1].legend()
            ax[1].set_ylabel("Probability")
            print("bin size Frequency", binsize)

            output_dict = {
                "meaned_frq_all_pixels": meaned_frq_all,
                "std_frq_all_pixels": std_frq_all,
                "masked_frq_map": masked_frq_map,
            }
        if savepath:
            plt.savefig(savepath)
        return output_dict
