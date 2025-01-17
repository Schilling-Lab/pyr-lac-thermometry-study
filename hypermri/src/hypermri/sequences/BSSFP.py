from ..brukerexp import BrukerExp
from ..utils.utils_anatomical import orient_coronal_and_bssfp_for_plot
from ..utils.utils_logging import LOG_MODES, init_default_logger
from ..utils import plot_colorbar
from ..utils.utils_general import (
    Get_Hist,
    get_counter,
    add_counter,
    load_plot_params,
    shift_anat,
)
from ..utils import utils_anatomical
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from copy import deepcopy
import functools


logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


class BSSFP(BrukerExp):
    def __init__(self, path_or_BrukerExp, log_mode="critical"):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp)
        # globally change log level, default is critical
        logger.setLevel(LOG_MODES[log_mode])

        self.Nslices = self.get_matrixDimensions()
        # print(self.method)
        self.Nechoes = self.get_Nechoes()
        self.Reconstructed_data = None
        # add extent of bssfp images to the class
        try:
            self.get_extent_bssfp()
        except:
            pass

    def get_matrixDimensions(self) -> int:
        """Get dimensions of acquisition matrix"""
        SpatialDim = self.method["PVM_SpatDimEnum"]

        if SpatialDim == "<2D>":
            out = self.method["PVM_SPackArrNSlices"]
        elif SpatialDim == "<3D>":
            out = self.method["PVM_Matrix"][-1]
        else:
            raise NotImplementedError(
                f"PVM_SpatDimEnum as an unknown value: {SpatialDim}."
            )

        return out

    def get_Nechoes(self) -> int:
        if "PVM_NEchoImages" in self.method:
            return self.method["PVM_NEchoImages"]
        else:
            return self.method["PVM_NMovieFrames"]

    def reconstruction(self, db=False, seq2d=None):
        """
        Function that reconstructs images from complex data provided in the fid file.

        ---------
        Parameters:
        db: bool (False)
            gives debugging output info if set true

        Returns
        ----------
        self.Reconstructed_data : np.ndarray
            Reconstructed image data stored in a numpy arra with dimensions [echoes, z, y, x].
            If reconstruction function not called self.Reconstructed_data is initialized as None

        """
        matrixDims = self.method["PVM_Matrix"]
        n_reps = self.method["PVM_NRepetitions"]
        n_chans = self.method["PVM_EncNReceivers"]

        # requires postprocessing of fid data, where padded zeros are removed from data
        reshaped_fid_data = self.postprocess_fid_file()

        if db:
            logger.debug(
                "1. Shape of reshaped_fid_data is" + str(reshaped_fid_data.shape)
            )
            logger.debug("Repetitions: " + str(n_reps))
            logger.debug("matrixDims: " + str(matrixDims))
            logger.debug("n_chans: " + str(n_chans))
            logger.debug("self.Nechoes: " + str(self.Nechoes))
            logger.debug(
                "reshaped_fid_data.shape[1]: " + str(reshaped_fid_data.shape[1])
            )

        if (
            self.method["Method"] == "<User:fisp_grad_1metab_CS>"
            or self.method["Method"] == "<User:fisp_grad_1metab_CS2>"
            or self.method["Method"] == "<User:fisp_gr_1_met_CS_me>"
            or self.method["Method"] == "<User:fisp_me_backup>"
            or self.method["Method"] == "<User:altPowRx_G_FISP>"
            or self.method["Method"] == "<User:altPow_jgs_alt_FISP>"
        ):
            if (
                self.method["PVM_SpatDimEnum"] == "<3D>"
                and np.sum(self.method["PVM_EncPft"]) == 3
            ) or (
                self.method["PVM_SpatDimEnum"] == "<2D>"
                and np.sum(self.method["PVM_EncPft"]) == 2
            ):
                # reshape complex data accordingly --> dimensions: repetitions,
                # slice, phase encode, echo, read direction, number of channels
                if self.method["PVM_SpatDimEnum"] == "<3D>":
                    reshaped_fid_data = np.reshape(
                        reshaped_fid_data,
                        (
                            n_reps,
                            matrixDims[2],
                            matrixDims[1],
                            self.Nechoes,
                            n_chans,
                            matrixDims[0],
                        ),
                    )
                elif self.method["PVM_SpatDimEnum"] == "<2D>":
                    reshaped_fid_data = np.reshape(
                        reshaped_fid_data,
                        n_chans,
                        n_reps,
                        matrixDims[1],
                        self.Nechoes,
                        reshaped_fid_data.shape[1],
                    )
                    np.expand_dims(reshaped_fid_data, axis=0)

                logger.debug(
                    "Order of data is repetitions, slice, phase, nechoes, nchannes, read"
                )
                logger.debug(
                    "2. Shape of reshaped_fid_data is" + str(reshaped_fid_data.shape)
                )
                # reshape to echo, dimensions read (usually z), phase (usually y), slice (usually x)
                # repetition, channel

                reshaped_fid_data = np.transpose(reshaped_fid_data, (3, 5, 2, 1, 0, 4))

                # check if Echo_Mode parameter exists in method dict
                if "Echo_Mode" in self.method:
                    # in case of symmetric readout every second echo has to be flipped
                    if self.method["Echo_Mode"] == "All_Echoes":
                        for idx, data in enumerate(
                            reshaped_fid_data[1::2, :, :, :, 0, 0]
                        ):
                            reshaped_fid_data[idx * 2 + 1, :, :, :, 0, 0] = np.flip(
                                data, axis=0
                            )

                logger.debug(
                    "Order of data is: Echoes, Read, phase, slice, repetitions, channels"
                )
                logger.debug(
                    "3. Shape of reshaped_fid_data is" + str(reshaped_fid_data.shape)
                )
                # perform fft in all dimensions, typical k-space layout requires fftshift to center 0 frequency in the middle
                reshaped_fid_data = self.bssfp_fft(
                    input_data=reshaped_fid_data, axes=(1, 2, 3)
                )
                self.Reconstructed_data = reshaped_fid_data
            else:
                self.Reconstructed_data = reshaped_fid_data

        # that's what it was before:
        else:
            # reshape complex data accordingly --> dimensions: slice, phase encode, echo, read direction
            if self.method["PVM_SpatDimEnum"] == "<3D>":
                reshaped_fid_data = np.reshape(
                    reshaped_fid_data,
                    (
                        matrixDims[2],
                        matrixDims[1],
                        self.Nechoes,
                        reshaped_fid_data.shape[1],
                    ),
                )
            elif self.method["PVM_SpatDimEnum"] == "<2D>":
                reshaped_fid_data = np.reshape(
                    reshaped_fid_data,
                    (matrixDims[1], self.Nechoes, reshaped_fid_data.shape[1]),
                )
                np.expand_dims(reshaped_fid_data, axis=0)

            # reshape to echo, dimensions read (usually z), phase (usually y), slice (usually x)
            reshaped_fid_data = np.transpose(reshaped_fid_data, (2, 3, 1, 0))

            # perform fft in all dimensions, typical k-space layout requires fftshift to center 0 frequency in the middle
            self.Reconstructed_data = np.fft.fftshift(
                np.fft.fft(
                    np.fft.fftshift(
                        np.fft.fftshift(
                            np.fft.fft(
                                np.fft.fftshift(
                                    np.fft.fftshift(
                                        np.fft.fft(
                                            np.fft.fftshift(reshaped_fid_data, 3),
                                            axis=3,
                                        ),
                                        3,
                                    ),
                                    1,
                                ),
                                axis=1,
                            ),
                            1,
                        ),
                        2,
                    ),
                    axis=2,
                ),
                2,
            )

        # use seq2d stored in bssfp object if no seq2d was passed:
        if seq2d is None:
            bssfp_seq2d = self.seq2d

            # doesnt make sense to use 2dseq if phase encodes were scrambled:
            if self.method["Method"] == "<User:fisp_grad_1metab_CS2>":
                if self.method["ScramblePhaseEncodesYesNo"]:
                    bssfp_seq2d = self.seq2d

            num_reco_counts = get_counter(
                data_obj=self,
                counter_name="reco_counter",
            )

        else:
            bssfp_seq2d = seq2d
            num_reco_counts = 0

        logger.debug(
            "shape of self.Reconstructed_data = " + str(self.Reconstructed_data.shape)
        )
        logger.debug("shape of seq2d = " + str(bssfp_seq2d.shape))
        # reshape the data so it matches the .fid reco orientation:

        if num_reco_counts > 0:
            return self.Reconstructed_data, bssfp_seq2d
        else:
            # measurement probably missing repetitions and channels
            # (might need to be improved in the future)
            if np.ndim(bssfp_seq2d) == 3:
                bssfp_seq2d = np.reshape(
                    bssfp_seq2d,
                    (((self.Nechoes,) + bssfp_seq2d.shape + (1,) + (1,))),
                )
            # measurement probably missing either repetitions or channels
            # (might need to be improved in the future)
            elif np.ndim(bssfp_seq2d) == 4:
                if (
                    self.method["Method"] == "<User:fisp_gr_1_met_CS_me>"
                    or self.method["Method"] == "<User:fisp_me_backup>"
                ):
                    bssfp_seq2d = np.transpose(bssfp_seq2d, (3, 0, 1, 2))
                    bssfp_seq2d = np.expand_dims(bssfp_seq2d, axis=-1)
                    bssfp_seq2d = np.expand_dims(bssfp_seq2d, axis=-1)
                else:
                    bssfp_seq2d = np.reshape(
                        bssfp_seq2d, (((self.Nechoes,) + bssfp_seq2d.shape) + (1,))
                    )
            elif np.ndim(bssfp_seq2d) == 5:
                bssfp_seq2d = np.transpose(bssfp_seq2d, (3, 0, 1, 2, 4))
                bssfp_seq2d = np.expand_dims(bssfp_seq2d, axis=-1)
            logger.debug("shape of seq2d = " + str(bssfp_seq2d.shape))

        if (
            self.method["Method"] == "<User:fisp_gr_1_met_CS_me>"
            or self.method["Method"] == "<User:fisp_me_backup>"
        ):
            # check if Echo_Mode parameter exists in method dict
            if "Echo_Mode" in self.method:
                # in case of symmetric readout every second echo has to be flipped
                if self.method["Echo_Mode"] == "All_Echoes":
                    for idx, data in enumerate(bssfp_seq2d[1::2, :, :, :, 0, 0]):
                        bssfp_seq2d[idx * 2 + 1, :, :, :, 0, 0] = np.flip(data, axis=1)

                        for idx_1 in range(data.shape[0]):
                            for idx_2 in range(data.shape[2]):
                                bssfp_seq2d[
                                    idx * 2 + 1, idx_1, :, idx_2, 0, 0
                                ] = np.roll(data[idx_1, :, idx_2], 1)

            self.Reconstructed_data = np.flip(self.Reconstructed_data, axis=2)
            self.Reconstructed_data = np.flip(self.Reconstructed_data, axis=1)
            self.Reconstructed_data = np.flip(self.Reconstructed_data, axis=3)

        # To Do: implemented test for read orientation and patient position:
        # add 1 to reco counter:
        if seq2d is None:
            # overwrite old seq2d:
            self.seq2d = bssfp_seq2d
            add_counter(
                data_obj=self,
                counter_name="reco_counter",
                n_counts=1,
            )

        else:
            pass

        # To Do: implemented test for read orientation and patient position:
        return self.Reconstructed_data, bssfp_seq2d

    def reco_scrambled_phase(self, input_data=None, path_to_coord=None, plot_db=False):
        """
        This script let's you reconstruct bSSFP data that was acquired
        with scramblde phase encode order

        Input: input_data: .fid file of the acquired data:

        """
        if input_data is None:
            input_data = self.fid
        else:
            pass

        if path_to_coord is None:
            raise Exception("path_to_coord parameter is empty")
        else:
            pass

        # load path_to_coord:
        import pandas as pd

        # Load the coordinates
        try:
            df = pd.read_csv(path_to_coord, sep=" ", header=None, names=["x", "y"])
        except:
            raise Exception("Coordinates could not be loaded!")

        # parameters:
        mat = self.method["PVM_Matrix"]
        nr = self.method["PVM_NRepetitions"]

        data = input_data.copy()

        # not sure wether it's mat[2], mat[1] or mat[1], mat[2]:
        # factor of 4 will be removed later:
        d = np.reshape(data, [nr, mat[2], mat[1], 4, mat[0]])

        # only use first out of 4 entries (rest should be empty):

        # If nr is 1, add an extra dimension to d
        if nr == 1:
            d = d[..., np.newaxis]

        # not sure wether it's mat[2], mat[1] or mat[1], mat[2]:
        num_bins = [mat[2], mat[1]]
        matrix = np.zeros_like(d)

        if plot_db is True:
            import matplotlib.colors as colors
            # Create a colormap
            cmap = plt.get_cmap("jet")
            norm = colors.Normalize(vmin=0, vmax=len(df) - 1)

            # Create the plot
            fig, ax = plt.subplots()

            # Scatter plot with points connected and colored based on their index
            for i in range(len(df) - 1):
                ax.plot(
                    df.iloc[i : i + 2, 0], df.iloc[i : i + 2, 1], color=cmap(norm(i))
                )

            # Display the plot
            plt.show()

        # Use binned_statistic_2d to sort the 1D data into a 2D grid based on the x and y coordinates
        from scipy.stats import binned_statistic_2d

        if nr == 1:
            for r in range(nr):
                for k in range(mat[0]):
                    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                        df["y"],
                        df["x"],
                        values=np.squeeze(d[r, :, :, 0, k]).flatten(),
                        statistic="mean",
                        bins=num_bins,
                    )
                    matrix[r, :, :, 0, k] = statistic[..., np.newaxis]
        else:
            for r in range(nr):
                for k in range(mat[0]):
                    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
                        df["y"],
                        df["x"],
                        values=np.squeeze(d[r, :, :, 0, k]).flatten(),
                        statistic="mean",
                        bins=num_bins,
                    )
                    matrix[r, :, :, 0, k] = statistic

        num_unscramle_counts = get_counter(
            data_obj=self,
            counter_name="unscramble",
        )

        # if was not saved yet:
        if num_unscramle_counts == 0:
            # save sorted matrix as fid
            matrix = np.reshape(matrix, (mat[1] * mat[2] * nr, -1))
            matrix = self.postprocess_fid_file(input_data=matrix)
            self.fid = matrix

            add_counter(
                data_obj=self,
                counter_name="unscramble",
                n_counts=1,
            )
        else:
            # reshape data
            matrix = np.reshape(matrix, (mat[1] * mat[2] * nr, -1))
            # remove 0s
            matrix = self.postprocess_fid_file(input_data=matrix)
        
        # bring data into [echoes, z, y, x, repetitions, channels] format
        matrix = np.reshape(matrix, (self.method['PVM_NRepetitions'], self.method['PVM_Matrix'][1], self.method['PVM_Matrix'][2], self.method['PVM_Matrix'][0]))
        print(matrix.shape)
        matrix = np.expand_dims(matrix, axis=(4, 5))
        print(matrix.shape)
        matrix = np.transpose(matrix, (4, 3, 1, 2, 0, 5))

        self.image = self.bssfp_fft(input_data=matrix, axes=(1, 2, 3))

        return matrix

    def full_reco(
        self,
        axial=None,
        sagittal=None,
        coronal=None,
        fieldmap=None,
        interp=False,
        interp_method="nearest",
        interp_use_multiprocessing=True,
        custom_seq2d=None,
    ):
        """
        Perform multiple reco steps like permutation of dimensions and shifting
        """
        # permutes the bssfp data:
        self.reconstruction()

        # reorients bssfp and anatomical images to match in space:
        # if custom_seq2d is None:
        #     self.reorient_reco(
        #         anatomical_obj_ax=axial,
        #         anatomical_obj_cor=coronal,
        #         anatomical_obj_sag=sagittal,
        #         fieldmap_obj_cor=fieldmap,
        #     )
        #     self.reorient_reco()
        # else:
        #     bssfp_custom_data,
        #     bssfp_pv_data,
        #     anatomical_ax,
        #     anatomical_sag,
        #     anatomical_cor,
        #     (fieldmap_cor,) = self.reorient_reco(
        #         anatomical_obj_ax=axial,
        #         anatomical_obj_cor=coronal,
        #         anatomical_obj_sag=sagittal,
        #         fieldmap_obj_cor=fieldmap,
        #         bssfp_custom=custom_seq2d,
        #     )
        #     self.seq2d = bssfp_custom_data

        from ..utils.utils_general import reorient_anat

        # reoriented seq2ds will be stored as seq2d_oriented in the data_objs:
        reorient_anat(data_obj=axial)
        reorient_anat(data_obj=sagittal)
        reorient_anat(data_obj=coronal)
        reorient_anat(data_obj=fieldmap)

        self.reorient_reco()

        # combine channels:
        self.seq2d = self.combine_multichannel(input_data=self.seq2d)

        # shift axial image:
        if axial is not None:
            axial.seq2d_oriented = shift_anat(
                csi_obj=self,
                anat_obj=axial,
                use_scipy_shift=True,
            )
        else:
            pass

        # shift coronal image:
        if sagittal is not None:
            sagittal.seq2d_oriented = shift_anat(
                csi_obj=self,
                anat_obj=sagittal,
                use_scipy_shift=True,
            )
        else:
            pass

        # shift coronal image:
        if coronal is not None:
            coronal.seq2d_oriented = shift_anat(
                csi_obj=self,
                anat_obj=coronal,
                use_scipy_shift=True,
            )
        else:
            pass

        # shift fieldmap image:
        if fieldmap is not None:
            fieldmap.seq2d_oriented = shift_anat(
                csi_obj=self,
                anat_obj=fieldmap,
                use_scipy_shift=True,
            )
        else:
            pass

        # Interpolate bssfp Data
        if interp is True:
            num_interp_counts = get_counter(
                data_obj=self, counter_name="interp_counter"
            )
            if num_interp_counts == 0:
                bssfp_itp = self.interpolate_bssfp(
                    bssfp_data=self.seq2d,
                    interp_size=(
                        1,  # echoes
                        round(coronal.seq2d.shape[1]),  # anatomical
                        round(coronal.seq2d.shape[0]),  # anatomical
                        coronal.seq2d.shape[2],  # slices
                        self.seq2d.shape[4],  # repetitions
                        1,
                    ),  # channels
                    interp_method=interp_method,  # interpolation method
                    use_multiprocessing=interp_use_multiprocessing,  # uses multiple cores
                )  # if =None and use_multiprocessing=True, automatically calcuates the nuber of CPU cores
                # save interpolated data:
                self.seq2d_interp = bssfp_itp
                add_counter(
                    data_obj=self,
                    counter_name="interp_counter",
                    n_counts=1,
                )
                if self.seq2d_interp.shape[1:4] == coronal.seq2d_oriented.shape[1:4]:
                    pass
                else:
                    logger.critical(
                        "interpolation shape (read, phase, slice) does not match coronal.seq2d_oriented shape"
                    )
        else:
            pass

    def find_closest_slice2(
        self,
        grid1=None,
        grid2=None,
        pos1=None,
        pos2=None,
        grid1_obj=None,
        grid2_obj=None,
    ):
        """
        Finds the closest slice in a grid based on provided position data.

        Given one position (either pos1 or pos2) and its respective grid (either grid1 or grid2),
        this function computes the closest slice in the other grid. If both positions are provided,
        the behavior is undefined.

        Args:
            grid1 (list, optional): Grid coordinates for the first dataset. If not provided, it's
                                    derived from `object_grid1`. Defaults to None.
            grid2 (list, optional): Grid coordinates for the second dataset. If not provided, it's
                                    derived from `object_grid2`. Defaults to None.
            pos1 (list, optional): Position in the first dataset. If not provided, the function will
                                   return the closest slice for the second dataset. Defaults to None.
            pos2 (list, optional): Position in the second dataset. If not provided, the function will
                                   return the closest slice for the first dataset. Defaults to None.
            grid1_obj (object, optional): Data object from which grid1 can be derived if `grid1`
                                             is not provided. Defaults to None.
            grid2_obj (object, optional): Data object from which grid2 can be derived if `grid2`
                                             is not provided. Defaults to None.

        Returns:
            list: Indices of the closest slice in the grid that doesn't have its position provided.
                  If both positions are None, returns a tuple of None values.

        Raises:
            ValueError: If both `pos1` and `pos2` are provided.
        """
        from hypermri.utils.utils_general import define_grid as ut_define_grid

        # get both positions:
        if pos1 is None and pos2 is None:
            return None, None, None

        # get both grids:
        if grid1 is None:
            grid1 = ut_define_grid(data_obj=grid1_obj)
        if grid2 is None:
            grid2 = ut_define_grid(data_obj=grid2_obj)

        if pos1 is None:
            pos2_mm = [grid2[0][pos2[0]], grid2[1][pos2[1]], grid2[2][pos2[2]]]
            pos1_mm = [None, None, None]
        else:
            pos1_mm = [grid1[0][pos1[0]], grid1[1][pos1[1]], grid1[2][pos1[2]]]
            pos2_mm = [None, None, None]

        # find closest slice:
        if pos1 is None:
            pos1 = [None, None, None]
            for k, p in enumerate(pos2_mm):
                pos1[k] = np.argmin(np.abs(grid1[k] - p))
            return pos1
        else:
            pos2 = [None, None, None]
            for k, p in enumerate(pos1_mm):
                pos2[k] = np.argmin(np.abs(grid2[k] - p))
            return pos2

    def find_closest_slice(
        self,
        axial_anat_grid=None,
        sagittal_anat_grid=None,
        coronal_anat_grid=None,
        bssfp_grid=None,
        bssfp_pos=None,
        axial_obj=None,
        sagittal_obj=None,
        coronal_obj=None,
        plot_result=False,
        pos_bssfp_anat="bssfp",
    ):
        """
        if pos_bssfp_anat == 'bssfp:
        Calculates the slice in ax_grid and cor_grid that is the
        closest in the bssfp_grid to the index pos (bssfp[pos])
        Can also be used with axial +/- coronal object without specifying the grid

        if pos_bssfp_anat == 'anat':
        Calculates the slice in bssfp_grid that is the
        closest in the anatomical_grid to the index pos (anatomical_grid[pos])
        Can also be used with axial +/- coronal object without specifying the grid

        Parameters
        ----------

        pos_bssfp_anat:
            if the input position is a position in the bssfp_grid or in the anatomical grid

        Examples
        --------
        Example 1:
        
        # generate a grid:
        
        >>> cor_grid = self.define_grid(
        >>>         mat=np.array((dim_read_cor, dim_phase_cor, dim_slice_cor)),
        >>>         fov=np.array((mm_read_cor, mm_phase_cor, mm_slice_cor)),
        >>>     )
        
        or:
        
        >>> cor_grid = dnp_bssfp.define_grid(
        >>>         mat=dnp_bssfp.method['PVM_Matrix'],
        >>>         fov=dnp_bssfp.method['PVM_Fov'])

        >>> ax_slice_ind, cor_slice_ind = self.find_closest_slice(
        >>>         ax_grid=ax_grid,
        >>>         cor_grid=cor_grid,
        >>>         bssfp_grid=bssfp_grid,
        >>>         pos=[
        >>>             slice_nr_ax_bssfp,
        >>>             slice_nr_sag,
        >>>             slice_nr_cor,
        >>>         ],
        >>>     )

        Example 2:

        >>> ax_ind, cor_ind = XX.find_closest_slice(ax_grid=hypermri.utils.utils_anatomical.define_grid(data_obj = axial),
        >>>     cor_grid=hypermri.utils.utils_anatomical.define_grid(data_obj = coronal),
        >>>     bssfp_grid=hypermri.utils.utils_anatomical.define_grid(data_obj = dnp_bssfp),
        >>>     pos=[-1,0,-1]
        >>>     )
        >>> print(ax_ind, cor_ind)

        """
        # init return values as Nones:
        ax_min_ind = sag_min_ind = cor_min_ind = None

        # Do not continue if position is not specified
        if bssfp_pos is None:
            logger.critical("no position input.")
            return ax_min_ind, sag_min_ind, cor_min_ind
        # Do not continue if bssfp grid is not specified
        if bssfp_grid is None:
            # define grid:
            bssfp_grid = self.define_grid()

        # Do not continue if neither axial nor coronal grid are defied
        if (
            axial_anat_grid is None
            and sagittal_anat_grid is None
            and coronal_anat_grid is None
            and axial_obj is None
            and sagittal_obj is None
            and coronal_obj is None
        ):
            return ax_min_ind, sag_min_ind, cor_min_ind

        # Warning if axial grid is too many times defined:
        if axial_anat_grid is not None and axial_obj is not None:
            Warning("Axial grid is defined twice, using axial object")

        # Warning if sagittal grid is too many times defined:
        if sagittal_anat_grid is not None and sagittal_obj is not None:
            Warning("sagittal grid is defined twice, using axial object")

        # Warning if coronal grid is too many times defined:
        if coronal_anat_grid is not None and coronal_obj is not None:
            Warning("Coronal grid is defined twice, using axial object")
        from hypermri.utils.utils_general import define_grid as ut_define_grid

        # define grid (returns None if data_obj is None):
        if axial_anat_grid is None and axial_obj is not None:
            axial_anat_grid = ut_define_grid(data_obj=axial_obj)
        if sagittal_anat_grid is None and sagittal_obj is not None:
            sagittal_anat_grid = ut_define_grid(data_obj=sagittal_obj)
        if coronal_anat_grid is None and coronal_obj is not None:
            coronal_anat_grid = ut_define_grid(data_obj=coronal_obj)

        # logger.critical("ax_grid" + str(axial_anat_grid))
        # logger.critical("sag_grid" + str(sagittal_anat_grid))
        # logger.critical("cor_grid" + str(coronal_anat_grid))
        # logger.critical("bssfp_grid" + str(bssfp_grid))
        # logger.critical("pos" + str(bssfp_pos))
        # logger.critical("pos_bssfp_anat" + pos_bssfp_anat)
        if pos_bssfp_anat == "anat":
            # find closest axial and coronal slice to position:
            if axial_anat_grid is not None:
                if all(x is not None for x in axial_anat_grid):
                    ax_min_ind = np.argmin(
                        np.abs(bssfp_grid[0] - axial_anat_grid[0][bssfp_pos[0]])
                    )
                if sagittal_anat_grid is None:
                    # if all(x is None for x in sagittal_anat_grid):
                    sag_min_ind = np.argmin(
                        np.abs(bssfp_grid[1] - axial_anat_grid[1][bssfp_pos[1]])
                    )
                if coronal_anat_grid is None:
                    # if all(x is None for x in coronal_anat_grid):
                    cor_min_ind = np.argmin(
                        np.abs(bssfp_grid[2] - axial_anat_grid[1][bssfp_pos[2]])
                    )

            if sagittal_anat_grid is not None:
                if all(x is not None for x in sagittal_anat_grid):
                    sag_min_ind = np.argmin(
                        np.abs(bssfp_grid[1] - sagittal_anat_grid[1][bssfp_pos[1]])
                    )
                if axial_anat_grid is None:
                    #                    if all(x is None for x in axial_anat_grid):
                    ax_min_ind = np.argmin(
                        np.abs(bssfp_grid[0] - sagittal_anat_grid[0][bssfp_pos[0]])
                    )
                if sagittal_anat_grid is None:
                    # if all(x is None for x in coronal_anat_grid):
                    cor_min_ind = np.argmin(
                        np.abs(bssfp_grid[2] - sagittal_anat_grid[1][bssfp_pos[2]])
                    )

            if coronal_anat_grid is not None:
                if all(x is not None for x in coronal_anat_grid):
                    cor_min_ind = np.argmin(
                        np.abs(bssfp_grid[2] - coronal_anat_grid[2][bssfp_pos[2]])
                    )
                if axial_anat_grid is None:
                    # if all(x is None for x in axial_anat_grid):
                    ax_min_ind = np.argmin(
                        np.abs(bssfp_grid[0] - coronal_anat_grid[0][bssfp_pos[0]])
                    )
                if sagittal_anat_grid is None:
                    # if all(x is None for x in sagittal_anat_grid):
                    sag_min_ind = np.argmin(
                        np.abs(bssfp_grid[1] - coronal_anat_grid[1][bssfp_pos[1]])
                    )

        else:
            # find closest axial and coronal slice to position:
            if axial_anat_grid is not None:
                if all(x is not None for x in axial_anat_grid):
                    ax_min_ind = np.argmin(
                        np.abs(axial_anat_grid[0] - bssfp_grid[0][bssfp_pos[0]])
                    )
                if sagittal_anat_grid is None:
                    #    if all(x is None for x in sagittal_anat_grid):
                    sag_min_ind = np.argmin(
                        np.abs(axial_anat_grid[1] - bssfp_grid[1][bssfp_pos[1]])
                    )
                if coronal_anat_grid is None:
                    #    if all(x is None for x in coronal_anat_grid):
                    cor_min_ind = np.argmin(
                        np.abs(axial_anat_grid[2] - bssfp_grid[2][bssfp_pos[2]])
                    )
            if sagittal_anat_grid is not None:
                if all(x is not None for x in sagittal_anat_grid):
                    sag_min_ind = np.argmin(
                        np.abs(sagittal_anat_grid[1] - bssfp_grid[1][bssfp_pos[1]])
                    )
                if axial_anat_grid is None:
                    #    if all(x is not None for x in axial_anat_grid):
                    ax_min_ind = np.argmin(
                        np.abs(sagittal_anat_grid[0] - bssfp_grid[0][bssfp_pos[0]])
                    )
                if coronal_anat_grid is None:
                    #    if all(x is None for x in coronal_anat_grid):
                    cor_min_ind = np.argmin(
                        np.abs(sagittal_anat_grid[2] - bssfp_grid[2][bssfp_pos[2]])
                    )

            if coronal_anat_grid is not None:
                if all(x is not None for x in coronal_anat_grid):
                    cor_min_ind = np.argmin(
                        np.abs(coronal_anat_grid[2] - bssfp_grid[2][bssfp_pos[2]])
                    )
                if axial_anat_grid is None:
                    #                    if all(x is not None for x in axial_anat_grid):
                    ax_min_ind = np.argmin(
                        np.abs(coronal_anat_grid[0] - bssfp_grid[0][bssfp_pos[0]])
                    )
                if sagittal_anat_grid is None:
                    #                   if all(x is not None for x in sagittal_anat_grid):
                    sag_min_ind = np.argmin(
                        np.abs(coronal_anat_grid[1] - bssfp_grid[1][bssfp_pos[1]])
                    )

        if plot_result:
            # cant plot if no anatomical images are shown:
            if axial_obj is None and sagittal_obj is None and coronal_obj is None:
                pass
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(3, 2)
            if axial_obj is not None:
                ax[0, 0].imshow(
                    np.rot90(
                        np.sum(
                            np.abs(
                                self.Reconstructed_data[0, bssfp_pos[0], :, :, :, 0]
                            ),
                            axis=2,
                        )
                    )
                )
                ax[0, 0].set_title("bSSFP# " + str(bssfp_pos[0]))

                ax[0, 1].imshow(axial_obj.seq2d[ax_min_ind, :, :], cmap="bone")
                ax[0, 1].set_title("Axial# " + str(ax_min_ind))
            else:
                ax[0, 0].axis("off")
                ax[0, 1].axis("off")

            if sagittal_obj is not None:
                ax[1, 0].imshow(
                    np.sum(
                        np.abs(self.Reconstructed_data[0, :, bssfp_pos[1], :, :, 0]),
                        axis=2,
                    )
                )
                ax[1, 0].set_title("bSSFP# " + str(bssfp_pos[1]))

                ax[1, 1].imshow(sagittal_obj.seq2d[:, sag_min_ind, :], cmap="bone")
                ax[1, 1].set_title("sagittal# " + str(sag_min_ind))

            else:
                ax[1, 0].axis("off")
                ax[1, 1].axis("off")

            if coronal_obj is not None:
                ax[2, 0].imshow(
                    np.sum(
                        np.abs(self.Reconstructed_data[0, :, :, bssfp_pos[2], :, 0]),
                        axis=2,
                    )
                )
                ax[2, 0].set_title("bSSFP# " + str(bssfp_pos[2]))

                ax[2, 1].imshow(coronal_obj.seq2d[:, :, cor_min_ind], cmap="bone")
                ax[2, 1].set_title("Coronal# " + str(cor_min_ind))
            else:
                ax[2, 0].axis("off")
                ax[2, 1].axis("off")

        return ax_min_ind, sag_min_ind, cor_min_ind

    def plot3D_new2(
        self,
        axlist,
        coronal_image=None,
        axial_image=None,
        bssfp_data=None,
        parameter_dict=None,
        display_ui=True,
        pyr_ind=0,
        fig=None,
        plot_params=None,
        save_fig=False,
        thirdMetab=False,
        fieldmap=None,
    ):
        """

        An interactive 3D plotting function for visualizing and analyzing bSSFP data.

        Parameters
        ----------
        axlist : list of matplotlib.axes.Axes
            List of axes to draw on. Should contain 3 axes for 3D plot:
            1st for axial, 2nd for coronal, 3rd for time curves.
        coronal_image : BrukerExp instance, optional
            Contains anatomical reference images, usually coronal.
        axial_image : BrukerExp instance, optional
            Contains anatomical reference images, usually axial.
        bssfp_data : ndarray, optional
            Array to plot. Data should be sorted in the order:
            Echoes, Read, Phase, Slice, Repetitions, Channels.
            If None, uses self.Reconstructed_data.
        parameter_dict : dict, optional
            Dictionary of keyword arguments to pass to ax.imshow.
        display_ui : bool, default True
            If True, directly renders interactive sliders using IPython.display.
            If False, returns the UI object instead.
        pyr_ind : int, default 0
            Index indicating if pyruvate was acquired first (before lactate).
        fig : matplotlib.figure.Figure, optional
            Figure object created during 'fig, ax = plt.subplots()' call.
        plot_params : dict or str, optional
            Plot parameters. Can be a dictionary or a path to a JSON file.
        save_fig : bool, default False
            If True, saves the figure immediately. Useful for batch processing.
        thirdMetab : bool, default False
            If True, plots a third metabolite (expected repetition order: Pyr, 3rd metabolite, Lac).
        fieldmap : ndarray, optional
            Fieldmap data for B0 visualization.

        Returns
        -------
        widgets.HBox or None
            If display_ui is False, returns the ipywidgets HBox containing interactive sliders.
            Otherwise, returns None.

        Notes
        -----
        This function provides an interactive interface for exploring 3D bSSFP data.
        It allows for visualization of anatomical and metabolic data, with options
        for overlay, color mapping, and time series analysis.

        The function supports various plot customizations including:
        - Slice selection in axial, coronal, and sagittal planes
        - Echo and repetition selection
        - Metabolite selection (Pyruvate, Lactate, ratio)
        - Display style (Magnitude, Real, Imaginary, Phase)
        - Color mapping and range adjustment
        - Anatomical overlay options

        Interactive features:
        - Click on images to update slice positions
        - Adjust repetition range for averaging
        - Save figures with current settings

        Examples
        --------
        >>> # Load data
        >>> dirpath = 'path/to/data/'
        >>> scans = hypermri.BrukerDir(dirpath)
        >>> scan_id = '20231213_XX_mmt_'
        >>> savepath = dirpath
        >>> 
        >>> # Load and process data
        >>> bssfp = scans[40]
        >>> coronal = scans[41]
        >>> axial = scans[31]
        >>> 
        >>> # Reconstruct data
        >>> bssfp.full_reco(axial=axial, coronal=coronal, interp=True)
        >>> 
        >>> # Create figure and plot
        >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(12,5))
        >>> bssfp.plot3D_new2(
        ...     coronal_image=coronal,
        ...     axial_image=axial,
        ...     axlist=[ax1, ax2, ax3],
        ...     plot_params='pyruvate_signal_mean_high_interp.json',
        ...     fig=fig
        ... )

        See Also
        --------
        BrukerExp : Class for handling Bruker experiment data
        load_plot_params : Function to load plot parameters from a JSON file
        """
        from IPython.display import clear_output
        from hypermri.utils.utils_general import define_grid as ut_define_grid
        from hypermri.utils.utils_general import (
            define_imagematrix_parameters as ut_define_imagematrix_parameters,
        )
        from hypermri.utils.utils_general import (
            define_imageFOV_parameters as ut_define_imageFOV_parameters,
        )
        from hypermri.utils.utils_general import img_interp as ut_img_interp

        # import cProfile

        clear_output()
        # list of axes has to be passed:
        if axlist is None:
            logger.critical(
                "Need to pass axlist (use fig, (ax1, ax2, ax3) = plt.subplots(1, 3) \
                        to generate and call function with: \
                        axlist=[ax1, ax2, ax3])"
            )
            return False

        # # at least 3 axes have to be passed:
        # if len(axlist) < 3:
        #     logger.critical(
        #         "Not enough ax objects in axlist for 3D plot, please give 3 axes!"
        #     )
        #     return False

        # ony can plot 3D data so far:
        if self.method["PVM_SpatDimEnum"] != "<3D>":
            logger.critical(
                "3D print function but no 3D data available!" + "--> Aborting"
            )
            return False

        # check if there is data to plot
        if bssfp_data is None:
            if not isinstance(self.Reconstructed_data, np.ndarray):
                logger.critical(
                    "There is no reconstructed data available. "
                    + "Please call the function bSSFP.reconstruction first!"
                    + "--> Aborting"
                )
                return False
            else:
                # get reconstructed data:
                data = self.seq2d
        else:
            # use input data:
            data = bssfp_data

        # define bSSFP parameters:
        # ------------------------------------------------------------
        self.data_to_plot = data
        # read image dimensions --- bSSFP
        dim_read_bssfp = data.shape[1]  # was z
        dim_phase_bssfp = data.shape[2]  # was y
        dim_slice_bssfp = data.shape[3]  # was x

        mm_read_bssfp = self.method["PVM_Fov"][0]
        mm_phase_bssfp = self.method["PVM_Fov"][1]
        mm_slice_bssfp = self.method["PVM_Fov"][2]

        reps_bssfp = data.shape[4]
        chans_bssfp = data.shape[5]
        nechoes_bssfp = data.shape[0]

        # slice_thick_cor_bssfp = mm_slice_bssfp / dim_slice_bssfp
        # slice_thick_ax_bssfp = mm_read_bssfp / dim_read_bssfp
        # slice_thick_sag_bssfp = mm_phase_bssfp / dim_phase_bssfp

        read_orient_bssfp = self.method["PVM_SPackArrReadOrient"]
        # read_offset_bssfp = self.method["PVM_SPackArrReadOffset"]
        # phase_offset_bssfp = self.method["PVM_SPackArrPhase1Offset"]
        # slice_offset_bssfp = self.method["PVM_SPackArrSliceOffset"]

        bssfp_grid = ut_define_grid(data_obj=self)

        # ------AXIAL ANATOMICAL ------------------------------
        # ------------------------------------------------------------
        # init empty grid:
        ax_grid = None
        if axial_image is None:
            axial_obj = None
        # object contains no info about extent, assume same dimensions:
        elif isinstance(axial_image, np.ndarray):
            pass

        # object contains info about extent:
        elif axial_image.__module__.startswith("hypermri."):
            # get grid describing the coronal slices:
            ax_grid = ut_define_grid(data_obj=axial_image)

            # save axial object before overwriting:
            axial_obj = axial_image
            # axial_image = axial_image.seq2d
            ax_shift_counter = get_counter(
                data_obj=axial_obj,
                counter_name="shift_counter",
            )
            if ax_shift_counter > 0:
                pass
            else:
                axial_obj.seq2d_oriented = shift_anat(
                    csi_obj=self,
                    anat_obj=axial_obj,
                    use_scipy_shift=True,
                )

            axial_image = axial_obj.seq2d_oriented

            pass
        else:
            pass

        # ------CORONAL ANATOMICAL ------------------------------
        # init empty grid:
        cor_grid = None
        if coronal_image is None:
            coronal_obj = None
        # object contains no info about extent, assume same dimensions:
        elif isinstance(coronal_image, np.ndarray):
            pass
        # object contains info about extent:
        elif coronal_image.__module__.startswith("hypermri."):
            # prepare coupling of coronal and bssfp data:
            # get grid describing the coronal slices:
            cor_grid = ut_define_grid(data_obj=coronal_image)

            # save coronal object before overwriting:
            coronal_obj = coronal_image
            # coronal_image = coronal_image.seq2d
            # axial_image = axial_image.seq2d
            cor_shift_counter = get_counter(
                data_obj=coronal_obj,
                counter_name="shift_counter",
            )

            if cor_shift_counter == 0:
                coronal_obj.seq2d_oriented = shift_anat(
                    csi_obj=self,
                    anat_obj=coronal_obj,
                    use_scipy_shift=True,
                )
            coronal_image = coronal_obj.seq2d_oriented

        else:
            pass

        # FOV - anat:
        if axial_obj is not None:
            fov_anat_read_plot, _, _ = ut_define_imageFOV_parameters(data_obj=axial_obj)

            # matrix - anat:
            mat_anat_read_plot, _, _ = ut_define_imagematrix_parameters(
                data_obj=axial_obj
            )
            res_anat_read_plot = fov_anat_read_plot / mat_anat_read_plot
        else:
            res_anat_read_plot = None
        if coronal_obj is not None:
            (
                mat_anat_slice_plot_z,
                mat_anat_slice_plot_x,
                mat_anat_slice_plot,
            ) = ut_define_imagematrix_parameters(data_obj=coronal_obj)
            (
                fov_anat_slice_plot_z,
                fov_anat_slice_plot_x,
                fov_anat_slice_plot,
            ) = ut_define_imageFOV_parameters(data_obj=coronal_obj)
            res_anat_slice_plot = fov_anat_slice_plot / mat_anat_slice_plot
        else:
            res_anat_slice_plot = None

        # calculate resolution - anat:
        if res_anat_read_plot is None:
            res_anat_read_plot = fov_anat_slice_plot_z / mat_anat_slice_plot_z
        if res_anat_slice_plot is None:
            pass

        # FOV - bssfp:
        fov_bssfp_read_plot, _, fov_bssfp_slice_plot = ut_define_imageFOV_parameters(
            data_obj=self
        )
        # matrix - bssfp:
        mat_bssfp_read_plot, _, mat_bssfp_slice_plot = ut_define_imagematrix_parameters(
            data_obj=self
        )

        if not read_orient_bssfp == "H_F":
            # checking if we have the right read orientation
            # if this error comes up you have to implement different orientations possibly
            logger.debug(
                "Careful this function is not evaluated for this read orientation"
            )
        else:
            pass

        # ------FIELDMAP------------------------------
        if fieldmap is not None:
            from ..utils.utils_general import get_gmr

            # define gyromagnetic ratios:
            gmr_1h = get_gmr(nucleus="1h")  # MHz/T
            gmr_13c = get_gmr(nucleus="13c")  # MHz/T
            output_nuc = "13c"
            fm = fieldmap.seq2d_oriented
            if output_nuc == "1h":
                pass
            elif output_nuc == "13c":
                fm = fm * gmr_13c / gmr_1h
            else:
                pass
            fm = np.flip(fm, axis=1)
            # define grid of fieldmap:
            fieldmap_grid = ut_define_grid(data_obj=fieldmap)

        # use default paramter dictionary if none is provided
        if not parameter_dict:
            parameter_dict = {"cmap": "magma"}

        if plot_params is not None:
            if isinstance(plot_params, dict):
                pass
            elif isinstance(plot_params, str):
                try:
                    plot_params = load_plot_params(
                        param_file=plot_params, data_obj=self
                    )
                except:
                    plot_params = {}
                    pass
            else:
                plot_params = {}
                pass

            if "fig_name" in plot_params:
                fig_name = plot_params["fig_name"]
            else:
                fig_name = None

        else:
            fig_name = None

        # ------------DEFINE EXTENT -----------------
        # calc vals for hline and vline
        cor_ext_phase = np.linspace(
            self.cor_ext[0] - (self.cor_ext[0] / self.data_to_plot.shape[2]),
            self.cor_ext[1] - (self.cor_ext[1] / self.data_to_plot.shape[2]),
            self.data_to_plot.shape[2],
        )

        cor_ext_read = np.linspace(
            self.cor_ext[2] - (self.cor_ext[2] / self.data_to_plot.shape[1]),
            self.cor_ext[3] - (self.cor_ext[3] / self.data_to_plot.shape[1]),
            self.data_to_plot.shape[1],
        )

        ax_ext_phase = np.linspace(
            self.ax_ext[0] - (self.ax_ext[0] / self.data_to_plot.shape[2]),
            self.ax_ext[1] - (self.ax_ext[1] / self.data_to_plot.shape[2]),
            self.data_to_plot.shape[2],
        )
        ax_ext_slice = np.flip(
            np.linspace(
                self.ax_ext[2] - (self.ax_ext[2] / self.data_to_plot.shape[3]),
                self.ax_ext[3] - (self.ax_ext[3] / self.data_to_plot.shape[3]),
                self.data_to_plot.shape[3],
            )
        )

        # define image plotting function
        def plot_img(
            slice_nr_ax=0,
            slice_nr_cor=0,
            slice_nr_sag=0,
            echo_nr=0,
            rep1=0,
            chan=0,
            alpha_overlay=0.5,
            rep2=None,
            rep_avg=False,
            plot_style="Abs",
            experiment_style="Perfusion",
            pyr_lac="Pyr",
            anat_overlay="Metab",
            metab_clim=[0, 1],
            cmap="plasma",
            interp_method="none",
            interp_factor=1,
            background=0.0,
            **kwargs,
        ):
            # cProfile.run("time-consuming_processing()")

            """
            This is the plotting function that is called whenever one of it's
            parameters is changed:
            Parameters:
            slice_nr_ax: axial slice number
            slice_nr_cor: coronal slice number
            slice_nr_sag: sagtial slice number (often not used)
            echo_nr: echo number (for me-bSSFP)
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
            # choose plot style:
            if plot_style == "Abs":
                bssfp_image_data = np.absolute(self.data_to_plot)
            elif plot_style == "Real":
                bssfp_image_data = np.real(self.data_to_plot)
            elif plot_style == "Imag":
                bssfp_image_data = np.imag(self.data_to_plot)
            elif plot_style == "Phase":
                bssfp_image_data = np.angle(self.data_to_plot)
            else:
                # default
                bssfp_image_data = np.absolute(self.data_to_plot)

            # set color range:
            # bssfp_image_data[bssfp_image_data < metab_clim[0]] = np.nan
            # bssfp_image_data[bssfp_image_data > metab_clim[1]] = metab_clim[1]
            # init empty arrays:
            global pyr_data, lac_data
            # deterMine experiment style:
            if experiment_style == "Perfusion":
                pyr_data = bssfp_image_data[:, :, :, :, :, :] - background
            elif experiment_style == "Pyr + Lac" and not thirdMetab:
                if pyr_lac == "Lac/Pyr":
                    # background subtraction is handled further down
                    pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
                    lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]
                else:
                    pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :] - background
                    lac_data = (
                        bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :] - background
                    )

            elif thirdMetab:
                if pyr_lac == "Lac/Pyr":
                    # background subtraction is handled further down
                    pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
                    lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]
                else:
                    pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::3, :] - background
                    thirdMetab_data = (
                        bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 3, :] - background
                    )
                    lac_data = (
                        bssfp_image_data[:, :, :, :, pyr_ind + 2 :: 3, :] - background
                    )
            else:
                Warning("unknown measurement style " + str(plot_opts_measurement.value))

            # determine which contrast to use:
            if pyr_lac == "Pyr":
                bssfp_image_data = pyr_data
            elif (pyr_lac == "Lac") and (
                (experiment_style == "Pyr + Lac")
                or (experiment_style == "3 Metabolites")
            ):
                bssfp_image_data = lac_data
            elif (pyr_lac == "3rd Metabolite") and (
                experiment_style == "3 Metabolites"
            ):
                bssfp_image_data = thirdMetab_data
            elif (pyr_lac == "Lac/Pyr") and (experiment_style == "Pyr + Lac"):
                if rep_avg:
                    # valid range:
                    if rep2 > rep1:
                        rep_range = range(rep1, rep2)
                        # assuming the order Echoes, Read, phase, slice, repetitions, channels in the data:
                        # filter low signal regions:
                        bssfp_image_data = self.data_to_plot

                        pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
                        lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]

                        lac_data = lac_data
                        pyr_data = pyr_data

                        lac_data = np.abs(
                            np.nanmean(
                                lac_data[:, :, :, :, rep_range, :],
                                axis=4,
                                keepdims=True,
                            )
                        )
                        pyr_data = np.abs(
                            np.nanmean(
                                pyr_data[:, :, :, :, rep_range, :],
                                axis=4,
                                keepdims=True,
                            )
                        )

                        pyr_signal_mask = pyr_data > background

                        bssfp_image_data = lac_data / pyr_data
                        bssfp_image_data = bssfp_image_data * pyr_signal_mask
                        rep_range = 0
                    # use start repetition
                    else:
                        rep_range = rep1
                # take start repetition value:
                else:
                    rep_range = rep1
                    bssfp_image_data = lac_data / pyr_data
            else:
                bssfp_image_data = pyr_data
                Warning("unknown Contrast" + str(plot_opts_metabolite.value))

            # takes mean over both repetitions:
            if rep_avg:
                # valid range:
                if rep2 > rep1:
                    # if ratio is chosen, the averaging has already been done
                    if (pyr_lac == "Lac/Pyr") and (experiment_style == "Pyr + Lac"):
                        pass

                    else:
                        rep_range = range(rep1, rep2)
                        # assuming the order Echoes, Read, phase, slice, repetitions, channels in the data:
                        bssfp_image_data = np.mean(
                            bssfp_image_data[:, :, :, :, rep_range, :],
                            axis=4,
                            keepdims=True,
                        )
                        rep_range = 0
                # use start repetition
                else:
                    rep_range = rep1
            # take start repetition value:
            else:
                rep_range = rep1

            # generate a custom cmap:
            map_name, cmap = self.generate_custom_cmap(cmap_name=cmap)

            # clear axis:
            for ax in axlist:
                ax.clear()

            # to avoid confusion as this is probably not universal (different read directions
            # and patient_pos)... define this here so it can be made depending on those paramters:
            # logger.critical(slice_nr_ax)
            # slice_nr_ax_bssfp = ax_slice_calc(
            #     nr_slices=axial_image.shape[1], input_index=slice_nr_ax
            # )

            # this has to be done to match the orientation of the coronal images. Unfortunately this is a bit confusing
            slice_nr_ax_bssfp = -slice_nr_ax - 1
            if axial_obj is None:
                # TODO find better default:
                anat_ax_slice_ind = None
            else:
                (
                    anat_ax_slice_ind,
                    _,
                    _,
                ) = self.find_closest_slice2(
                    grid1=bssfp_grid,
                    pos1=[
                        slice_nr_ax_bssfp,
                        slice_nr_sag,
                        slice_nr_cor,
                    ],
                    grid2_obj=axial_obj,
                )

            if coronal_obj is None:
                anat_cor_slice_ind = None
            else:
                (
                    _,
                    _,
                    anat_cor_slice_ind,
                ) = self.find_closest_slice2(
                    grid1=bssfp_grid,
                    pos1=[
                        slice_nr_ax_bssfp,
                        slice_nr_sag,
                        slice_nr_cor,
                    ],
                    grid2_obj=coronal_obj,
                )

            # if axial and or coronal grid are not defined, use bssfp indices:
            if anat_ax_slice_ind is None:
                # roughly choose similar slice position
                if axial_image is not None:
                    anat_ax_slice_ind = int(
                        slice_nr_ax_bssfp
                        * axial_image.shape[1]
                        / bssfp_image_data.shape[1]
                    )
                else:
                    anat_ax_slice_ind = bssfp_image_data.shape[1]
            if anat_cor_slice_ind is None:
                if coronal_image is not None:
                    # roughly choose similar slice position
                    anat_cor_slice_ind = int(
                        slice_nr_cor
                        * coronal_image.shape[3]
                        / bssfp_image_data.shape[3]
                    )
                else:
                    anat_cor_slice_ind = bssfp_image_data.shape[3]

            # flip index: dim_slice_bssfp
            # logger.critical(coronal_image.shape)
            # anat_cor_slice_ind = coronal_image.shape[3] - anat_cor_slice_ind + 1
            # slice_nr_cor = dim_slice_bssfp - slice_nr_cor + 1

            # generate mask for axial image:
            mask_ax = np.rot90(
                np.squeeze(
                    (
                        bssfp_image_data[
                            echo_nr,
                            slice_nr_ax_bssfp,
                            :,
                            :,
                            rep_range,
                            chan,
                        ]
                    )
                )
            ).copy()
            mask_ax[mask_ax < metab_clim[0]] = 0
            mask_ax[mask_ax > metab_clim[0]] = 1

            # generate mask for axial image:
            mask_cor = np.squeeze(
                (bssfp_image_data[echo_nr, :, :, slice_nr_cor, rep_range, chan])
            ).copy()

            mask_cor[mask_cor < metab_clim[0]] = 0
            mask_cor[mask_cor > metab_clim[0]] = 1

            # function to perform image interpolation:
            def img_interp(
                metab_image=None, interp_factor=None, cmap=None, overlay=True
            ):
                # skip interpolation if interpolation factor is 1
                if interp_factor == 1:
                    interpolated = metab_image
                else:
                    # partially taken from:
                    # https://stackoverflow.com/questions/52419509/get-interpolated-data-from-imshow
                    from matplotlib.image import _resample
                    from matplotlib.transforms import Affine2D
                    import matplotlib.pyplot as plt

                    # define new image size:
                    out_dimensions = (
                        metab_image.shape[0] * interp_factor,
                        metab_image.shape[1] * interp_factor,
                    )

                    # needs to be scale later:
                    metab_image_max = np.max(metab_image)
                    # normalize (necessary for _resample)
                    metab_image = metab_image / metab_image_max

                    # generate axis object
                    _, axs = plt.subplots(1)
                    transform = Affine2D().scale(interp_factor, interp_factor)
                    img = axs.imshow(
                        metab_image, interpolation=interp_method, cmap=cmap
                    )
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
                        interpolated[interpolated <= metab_clim[0]] = np.nan

                return interpolated

            if fieldmap is not None:
                (
                    ax_fm_ind,
                    sag_fm_ind,
                    cor_fm_ind,
                ) = self.find_closest_slice2(
                    grid1=bssfp_grid,
                    grid2=fieldmap_grid,
                    pos1=[slice_nr_ax_bssfp, slice_nr_sag, slice_nr_cor],
                )

                df_b0 = np.squeeze(fm[0, ax_fm_ind, sag_fm_ind, cor_fm_ind, 0, 0])
            else:
                df_b0 = None

            # metabolic axial image:
            metab_image = np.flip(
                np.squeeze(
                    (
                        bssfp_image_data[
                            echo_nr,
                            slice_nr_ax_bssfp,
                            :,
                            :,
                            rep_range,
                            chan,
                        ]
                    )
                ),
                axis=1,
            )

            # Interpolate the image by a factor:
            interpolated = img_interp(
                metab_image=metab_image,
                interp_factor=interp_factor,
                cmap=cmap,
                overlay=True,
            )

            # if the colorange shoud be set automcatically:
            if checkbox_auto_set_crange.value:
                # stupid but should do the trick for now:
                metab_clim = list(metab_clim)
                metab_clim[0] = np.nanmin(interpolated)
                metab_clim[1] = np.nanmax(interpolated)

                # don't show negative number in case of magntiude display style:
                if plot_style == "Abs":
                    metab_clim[0] = 0
                metab_clim = tuple(metab_clim)

            # axial view: -----------------------------------------------------
            if (anat_overlay == "Metab + Anat") and (axial_image is not None):
                # anatomical axial image:
                axlist[0].imshow(
                    np.squeeze(np.rot90(axial_image[0, anat_ax_slice_ind, :, :, 0, 0])),
                    extent=self.ax_ext,
                    cmap="bone",
                )

                im_ax = axlist[0].imshow(
                    np.rot90(interpolated),
                    extent=self.ax_ext,
                    cmap=cmap,
                    alpha=alpha_overlay,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )

            elif anat_overlay == "Metab":
                # metabolic axial image:
                im_ax = axlist[0].imshow(
                    np.rot90(interpolated),
                    extent=self.ax_ext,
                    cmap=cmap,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )

            elif (anat_overlay == "Anat") and (axial_image is not None):
                # anatomical axial image:
                axlist[0].imshow(
                    np.rot90(np.squeeze(axial_image[0, anat_ax_slice_ind, :, :, 0, 0])),
                    extent=self.ax_ext,
                    cmap="bone",
                )

            else:  # default show metab image
                im_ax = axlist[0].imshow(
                    np.rot90(interpolated),
                    extent=self.ax_ext,
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
                ax_ext_slice[slice_nr_cor],
                color="white",
                linewidth=0.25,
                linestyle="solid",
            )
            # plot lines:
            axlist[0].axhline(
                y=ax_ext_slice[slice_nr_cor] - res_anat_slice_plot / 2,
                xmin=0,  # from 0 to 1 (1= all to the right)
                xmax=0.1,  # from 0 to 1 (1= all to the right)
                color="white",
                linewidth=0.45,
                linestyle="--",
            )
            # plot lines:
            axlist[0].axhline(
                ax_ext_slice[slice_nr_cor] + res_anat_slice_plot / 2,
                xmin=0,  # from 0 to 1 (1= all to the right)
                xmax=0.1,  # from 0 to 1 (1= all to the right)
                color="white",
                linewidth=0.45,
                linestyle="--",
            )

            axlist[0].axvline(
                ax_ext_phase[
                    slice_nr_sag
                ],  # mm_phase_bssfp / 2 - slice_nr_sag * slice_thick_sag_bssfp,
                color="white",
                linewidth=0.25,
                linestyle="dashed",
            )

            # plot labels:
            axlist[0].set_xlabel("mm (phase)")
            axlist[0].set_ylabel("mm (slice)")
            if df_b0 is not None:
                axlist[0].set_title(
                    "Axial "
                    + str(anat_ax_slice_ind)
                    + " df = "
                    + str(round(df_b0, 1))
                    + " Hz"
                )

            else:
                axlist[0].set_title("Axial ")

            # sagittal view:
            # axlist[1].imshow(
            #     np.squeeze(
            #         (
            #             bssfp_image_data[
            #                 echo_nr, :, slice_nr_sag, :, rep_range, chan
            #             ]
            #         )
            #     ),
            #     extent=self.sag_ext,
            #     **parameter_dict,
            # )
            # axlist[1].set_xlabel("mm (read)")
            # axlist[1].set_ylabel("mm (slice)")

            # axlist[1].set_title("Sagittal")
            # # axlist[0].set_aspect(dim_y / dim_z)

            # axlist[1].axhline(
            #     mm_slice_bssfp / 2 - slice_nr_cor * slice_thick_cor_bssfp,
            #     color="white",
            #     linewidth=0.25,
            #     linestyle="solid",
            # )
            # axlist[1].axvline(
            #     mm_read_bssfp / 2 - slice_nr_ax * slice_thick_ax_bssfp,
            #     color="white",
            #     linewidth=0.25,
            #     linestyle="dashed",
            # )
            # metabolic coronal image:
            metab_image = np.squeeze(
                (bssfp_image_data[echo_nr, :, :, slice_nr_cor, rep_range, chan])
            )

            # Interpolate the image by a factor:
            interpolated = img_interp(
                metab_image=metab_image,
                interp_factor=interp_factor,
                cmap=cmap,
                overlay=True,
            )

            # if the colorange shoud be set automcatically:
            if checkbox_auto_set_crange.value:
                # stupid but should do the trick for now:
                metab_clim = list(metab_clim)
                metab_clim[0] = np.nanmin(interpolated)
                metab_clim[1] = np.nanmax(interpolated)

                # don't show negative number in case of magntiude display style:
                if plot_style == "Abs":
                    metab_clim[0] = 0
                metab_clim = tuple(metab_clim)

            # coronal view: -------------------------------------------
            if (anat_overlay == "Metab + Anat") and (coronal_image is not None):
                # anatomical coronal image:
                axlist[1].imshow(
                    np.squeeze((coronal_image[0, :, :, anat_cor_slice_ind, 0, 0])),
                    extent=self.cor_ext,
                    cmap="bone",
                )

                axlist[1].imshow(
                    interpolated,
                    extent=self.cor_ext,
                    alpha=alpha_overlay,
                    cmap=cmap,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )
            elif anat_overlay == "Metab":
                axlist[1].imshow(
                    interpolated,
                    extent=self.cor_ext,
                    cmap=cmap,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )
            elif (anat_overlay == "Anat") and (coronal_image is not None):
                # anatomical coronal image:
                axlist[1].imshow(
                    np.squeeze((coronal_image[0, :, :, anat_cor_slice_ind, 0, 0])),
                    extent=self.cor_ext,
                    cmap="bone",
                )
            else:  # default show metabolic image
                axlist[1].imshow(
                    interpolated,
                    extent=self.cor_ext,
                    alpha=alpha_overlay * mask_cor,
                    cmap=cmap,
                    vmin=metab_clim[0],
                    vmax=metab_clim[1],
                )

            # plot lines:
            axlist[1].axhline(
                cor_ext_read[
                    slice_nr_ax
                ],  # mm_read_bssfp / 2 - slice_nr_ax * slice_thick_ax_bssfp - 0.5,
                color="white",
                linewidth=0.25,
                linestyle="solid",
            )
            # plot lines:
            axlist[1].axhline(
                y=cor_ext_read[slice_nr_ax] - res_anat_read_plot / 2,
                xmin=0,  # from 0 to 1 (1= all to the right)
                xmax=0.1,  # from 0 to 1 (1= all to the right)
                color="white",
                linewidth=0.45,
                linestyle="--",
            )
            # plot lines:
            axlist[1].axhline(
                cor_ext_read[slice_nr_ax] + res_anat_read_plot / 2,
                xmin=0,  # from 0 to 1 (1= all to the right)
                xmax=0.1,  # from 0 to 1 (1= all to the right)
                color="white",
                linewidth=0.45,
                linestyle="--",
            )
            axlist[1].axvline(
                cor_ext_phase[
                    slice_nr_sag
                ],  # mm_phase_bssfp / 2 -  * slice_thick_sag_bssfp,
                color="white",
                linewidth=0.25,
                linestyle="dashed",
            )

            # set labels:
            axlist[1].set_xlabel("mm (phase)")
            axlist[1].set_ylabel("mm (read)")
            axlist[1].set_title("Coronal - " + str(anat_cor_slice_ind))

            # plot time curve ---------------------------------------------
            if len(axlist) > 2:
                # pyruvate signal only:
                if pyr_lac == "Pyr":
                    axlist[2].plot(
                        np.squeeze(
                            (
                                pyr_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=2,
                    )
                    # only possible if pyruvate and lactate were measured
                    if experiment_style == "Pyr + Lac":
                        # add lactate:
                        axlist[2].plot(
                            np.squeeze(
                                (
                                    lac_data[
                                        echo_nr,
                                        slice_nr_ax_bssfp,
                                        slice_nr_sag,
                                        slice_nr_cor,
                                        :,
                                        chan,
                                    ]
                                )
                            ),
                            linewidth=1,
                        )
                    if experiment_style == "3 Metabolites":
                        # add lactate:
                        axlist[2].plot(
                            np.squeeze(
                                (
                                    lac_data[
                                        echo_nr,
                                        slice_nr_ax_bssfp,
                                        slice_nr_sag,
                                        slice_nr_cor,
                                        :,
                                        chan,
                                    ]
                                )
                            ),
                            linewidth=1,
                        )

                        # add PyruvateHydrate:
                        axlist[2].plot(
                            np.squeeze(
                                (
                                    thirdMetab_data[
                                        echo_nr,
                                        slice_nr_ax_bssfp,
                                        slice_nr_sag,
                                        slice_nr_cor,
                                        :,
                                        chan,
                                    ]
                                )
                            ),
                            linewidth=1,
                        )
                # only possible if pyruvate and lactate were measured
                elif (
                    experiment_style == "Pyr + Lac"
                    or experiment_style == "3 Metabolites"
                ) and (pyr_lac == "Lac"):
                    axlist[2].plot(
                        np.squeeze(
                            (
                                pyr_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=1,
                    )
                    # highlight lactate (linewidth):
                    axlist[2].plot(
                        np.squeeze(
                            (
                                lac_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=2,
                    )

                    if experiment_style == "3 Metabolites":
                        axlist[2].plot(
                            np.squeeze(
                                (
                                    thirdMetab_data[
                                        echo_nr,
                                        slice_nr_ax_bssfp,
                                        slice_nr_sag,
                                        slice_nr_cor,
                                        :,
                                        chan,
                                    ]
                                )
                            ),
                            linewidth=1,
                        )

                # Plot ratio:
                elif (experiment_style == "Pyr + Lac") and (pyr_lac == "Lac/Pyr"):
                    axlist[2].plot(
                        np.squeeze(
                            (
                                lac_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        )
                        / np.squeeze(
                            (
                                pyr_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=1,
                    )

                # if also PyruvateHydrate was measured
                elif (experiment_style == "3 Metabolites") and (
                    pyr_lac == "3rd Metabolite"
                ):
                    axlist[2].plot(
                        np.squeeze(
                            (
                                pyr_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=1,
                    )

                    axlist[2].plot(
                        np.squeeze(
                            (
                                lac_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=1,
                    )

                    axlist[2].plot(
                        np.squeeze(
                            (
                                thirdMetab_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=2,
                    )

                else:
                    Warning("dont know what to do")
                    axlist[2].plot(
                        np.squeeze(
                            (
                                pyr_data[
                                    echo_nr,
                                    slice_nr_ax_bssfp,
                                    slice_nr_sag,
                                    slice_nr_cor,
                                    :,
                                    chan,
                                ]
                            )
                        ),
                        linewidth=2,
                    )

                axlist[2].axvline(
                    rep1,
                    color="black",
                    linewidth=1,
                    linestyle="dashed",
                )
                # draw 2nd line:
                if (rep2 > rep1) and (rep_avg_checkbox.value):
                    axlist[2].axvline(
                        rep2,
                        color="black",
                        linewidth=1,
                        linestyle="dashed",
                    )
                else:
                    axlist[2].axvline(
                        rep2,
                        color="black",
                        linewidth=0.25,
                        linestyle="dashed",
                    )

                # axlist[2].plot(0, metab_clim[0], color="black", marker=".")
                # axlist[2].plot(0, metab_clim[1], color="black", marker=".")
                # axlist[2].set_ylim()
                axlist[2].axhline(
                    0,
                    color="black",
                    linewidth=0.5,
                    linestyle="dashed",
                )
                axlist[2].set_box_aspect(1)

        # --------------------------------------------------

        global rep_avg_checkbox_option

        # create interactive sliders for  slices in all dimensions
        rep_avg_checkbox = widgets.Checkbox(description="Mean Reps")
        rep_avg_checkbox_option = widgets.VBox(layout=widgets.Layout(display="flex"))
        plot_images_individually_checkbox = widgets.Checkbox(
            description="Ind.", layout=widgets.Layout(display="flex")
        )
        show_b0val_checkbox = widgets.Checkbox(description="Show B0")

        # create interactive slider for echoes
        echo_slider = widgets.IntSlider(
            value=0, min=0, max=nechoes_bssfp - 1, description="Echo: "
        )
        slice_slider_rep1 = widgets.IntSlider(
            value=0,
            min=0,
            max=reps_bssfp - 1,
            description="Rep. Start: ",
            layout=widgets.Layout(width="350px"),
        )
        slice_slider_rep2 = widgets.IntSlider(
            value=0,
            min=0,
            max=reps_bssfp - 1,
            description="Rep. End: ",
            layout=widgets.Layout(width="350px"),
        )

        # set the minimum of the end reptition to the value of the start reptition

        # set the minimum of the end reptition to the value of the start reptition
        def rep_range_checker(args):
            if slice_slider_rep1.value > slice_slider_rep2.value:
                slice_slider_rep2.value = slice_slider_rep1.value
            else:
                pass
            return True

        slice_slider_rep1.observe(rep_range_checker, names="value")
        slice_slider_rep2.observe(rep_range_checker, names="value")

        if thirdMetab:
            options_measurements = ["Perfusion", "Pyr + Lac", "3 Metabolites"]
        else:
            options_measurements = ["Perfusion", "Pyr + Lac"]

        ## Different plot options:
        # lets you choose the type of experiment:
        plot_opts_measurement = widgets.Dropdown(
            options=options_measurements,
            value="Perfusion",  # Defaults to 'pineapple'
            description="Experiment:",
            disabled=False,
            layout=widgets.Layout(width="20%"),
        )
        # lets you choose the data shown:
        plot_opts_part = widgets.Dropdown(
            options=["Abs", "Real", "Imag", "Phase"],
            value="Abs",
            description="Plot style:",
            disabled=False,
            layout=widgets.Layout(width="10%"),
        )

        # lets you choose the metabolite shown:
        plot_opts_metabolite = widgets.Dropdown(
            options=["Pyr", "Lac", "3rd Metabolite"],
            value="Pyr",
            description="Metabolite:",
            disabled=True,  # activate if plot_opts_measurement is set to Pyr+Lac
            layout=widgets.Layout(width="15%"),
        )
        # lets you choose wether to show anatomical, metabolic or ratio:
        if (coronal_image is None) and (axial_image is None):
            plot_opts_anat_overlay = widgets.Dropdown(
                options=["Metab", "Metab + Anat", "Anat"],
                value="Metab",
                description="Anat overlay",
                disabled=True,
                layout=widgets.Layout(width="20%"),
            )
        else:
            plot_opts_anat_overlay = widgets.Dropdown(
                options=["Metab", "Metab + Anat", "Anat"],
                value="Metab",
                description="Anat overlay",
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
            layout=widgets.Layout(width="15%"),
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
            readout=True,
            readout_format=".1f",
            layout=widgets.Layout(width="35%"),
        )
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
            description="Colormap:",
            disabled=False,
            layout=widgets.Layout(width="20%"),
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
            description="Interp.method:",
            disabled=False,
            layout=widgets.Layout(width="20%"),
        )

        save_opts_fileformat_select = widgets.Select(
            options=["svg", "png", "pickle"],
            value="svg",
            # rows=10,
            description="FF:",
            layout=widgets.Layout(width="8%"),
            disabled=False,
        )

        plot_opts_interpolation_factor = widgets.BoundedIntText(
            value=1,
            min=1,
            max=100,
            step=1,
            disabled=False,
            description="Interp. fact:",
            layout=widgets.Layout(width="10%"),
        )
        plot_opts_0_as_NaN = widgets.Checkbox(
            value=False, description="NaN", disabled=False
        )
        # name of figure to save:
        text_save_fig_name = widgets.Text(
            value="fig_name.svg",
            placeholder="Type something",
            description="Name:",
            disabled=False,
            tooltip="name of figure (has to end with .png or .svg)",
            layout=widgets.Layout(width="30%"),
        )
        # save figure (in cwd):
        button_save_fig = widgets.Button(
            description="Save fig.",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Save the figure",
            icon="check",  # (FontAwesome names without the `fa-` prefix)
            layout=widgets.Layout(width="100px"),
        )

        text_background = widgets.FloatText(
            value=0.0,
            description="Background:",
            disabled=False,
            layout=widgets.Layout(width="12%"),
        )
        # Draw ROI in anatomical image:
        checkbox_auto_set_crange = widgets.Checkbox(
            description="ACR",
            value=True,
            tooltip="automatically sets the colorange to min <-> max of available values",
            layout=widgets.Layout(width="10%"),
        )

        # def set_metab_clim():
        #     if args["value"] == True:
        #         plot_opts_metab_clim = widgets.FloatRangeSlider(
        #         min=0.0,
        #         value=[0, np.max(data)],
        #         max=np.max(data),
        #         description="M. range:",
        #         disabled=False,
        #         continuous_update=False,
        #         orientation="horizontal",
        #         readout=True,
        #         readout_format=".1f",
        #     else:
        #         plot_opts_metab_clim = widgets.FloatRangeSlider(
        #         min=0.0,
        #         value=[0, np.max(data)],
        #         max=np.max(data),
        #         description="M. range:",
        #         disabled=False,
        #         continuous_update=False,
        #         orientation="horizontal",
        #         readout=True,
        #         readout_format=".1f",
        #     )

        # rep_avg_checkbox.observe(set_metab_clim, names="value")

        # init sliders:
        slice_slider_ax = widgets.IntSlider(
            value=mat_bssfp_read_plot // 2,
            min=0,
            max=mat_bssfp_read_plot - 1,
            description="Slice Axial: ",
        )
        slice_slider_cor = widgets.IntSlider(
            value=mat_bssfp_slice_plot // 2,
            min=0,
            max=mat_bssfp_slice_plot - 1,
            description="Slice Coronal: ",
        )
        slice_slider_sag = widgets.IntSlider(
            value=dim_phase_bssfp // 2,
            min=0,
            max=dim_phase_bssfp - 1,
            description="Slice sagittal: ",
        )
        slice_slider_chan = widgets.IntSlider(
            value=0,
            min=0,
            max=chans_bssfp - 1,
            description="Channel: ",
            layout=widgets.Layout(width="120px"),
        )

        def set_rep_range(args):
            """
            Sets repetition range and contrast options (Pyr, Lac, ...) depending
            on the experiment type
            """
            # changed from perfusion to pyr+lac
            if args["new"] == "Pyr + Lac" or args["new"] == "3 Metabolites":
                if (np.ceil(reps_bssfp / 2) - 1) == 0:
                    slice_slider_rep1.max = np.ceil(reps_bssfp / 2)
                    slice_slider_rep2.max = np.ceil(reps_bssfp / 2)

                    slice_slider_rep1.value = np.ceil(slice_slider_rep1.value / 2)
                    slice_slider_rep2.value = np.ceil(slice_slider_rep2.value / 2)

                    slice_slider_rep1.disabled = True
                    slice_slider_rep2.disabled = True

                else:
                    slice_slider_rep1.disabled = False
                    slice_slider_rep2.disabled = False

                    slice_slider_rep1.max = np.ceil(reps_bssfp / 2) - 1
                    slice_slider_rep2.max = np.ceil(reps_bssfp / 2) - 1

                    slice_slider_rep1.value = np.ceil(slice_slider_rep1.value / 2)
                    slice_slider_rep2.value = np.ceil(slice_slider_rep2.value / 2)

                plot_opts_metabolite.disabled = False

                if args["new"] == "Pyr + Lac":
                    plot_opts_metabolite.options = ["Pyr", "Lac", "Lac/Pyr"]
                else:
                    plot_opts_metabolite.options = ["Pyr", "Lac", "3rd Metabolite"]

            elif args["new"] == "Perfusion":
                slice_slider_rep1.disabled = False
                slice_slider_rep2.disabled = False

                slice_slider_rep1.max = reps_bssfp - 1
                slice_slider_rep2.max = reps_bssfp - 1

                slice_slider_rep1.value = slice_slider_rep1.value * 2
                slice_slider_rep2.value = slice_slider_rep2.value * 2

                plot_opts_metabolite.value = "Pyr"
                plot_opts_metabolite.disabled = True
            else:
                Warning("Unkown experiment type" + str(args["new"]))

        # change options depending on experiment type:
        plot_opts_measurement.observe(set_rep_range, names="value")

        def adjust_colorrange(args):
            # Should adjust the colorrange depening on the chosen data type
            # (real, imag, phase, magnitude)

            if args["new"] == "Abs":
                plot_opts_metab_clim.max = 1.2 * np.max(np.abs(data))
                plot_opts_metab_clim.min = 1.2 * np.min(np.abs(data))
            if args["new"] == "Real":
                plot_opts_metab_clim.max = 1.2 * np.max(np.real(data))
                plot_opts_metab_clim.min = 1.2 * np.min(np.real(data))
            if args["new"] == "Imag":
                plot_opts_metab_clim.max = 1.2 * np.max(np.imag(data))
                plot_opts_metab_clim.min = 1.2 * np.min(np.imag(data))
            if args["new"] == "Phase":
                plot_opts_metab_clim.max = 2 * np.max(np.angle(data))
                plot_opts_metab_clim.min = 2 * np.min(np.angle(data))

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
                text_background,
            ],
        )

        # # combine widgets into a horizontal box::
        ui_up_left = widgets.HBox(
            [
                plot_opts_measurement,
                plot_opts_anat_overlay,
                plot_opts_metabolite,
                plot_opts_part,
                text_save_fig_name,
                button_save_fig,
                plot_images_individually_checkbox,
            ]
        )
        # ui_up_right = widgets.HBox([save_opts_fileformat_select])
        # ui_up = widgets.VBox([ui_up_left, ui_up_right])
        ui_up = ui_up_left
        # # combine widgets and boxes into more boxes:
        ui_slices = widgets.HBox([slice_slider_ax, slice_slider_sag, slice_slider_cor])
        ui_reps = widgets.VBox([slice_slider_rep1, slice_slider_rep2, rep_avg_checkbox])
        ui_reps_echoes_chans = widgets.HBox([echo_slider, slice_slider_chan, ui_reps])

        def set_plot_params(self, params=None):
            """
            Set the plot parameters. Sets first the widget values to the parameter (if
            ther are loaded values). Either way, after that the plot parameters are set to
            the widget values
            """
            # sets the sliders to a specific value in case a parameter file was loaded (second value is default value if no value is defined)
            try:
                # set this first because it affects the repetition setting:
                plot_opts_measurement.value = params.get(
                    "experiment_style", "Perfusion"
                )
                slice_slider_ax.value = params.get("slice_nr_ax", 0)
                slice_slider_cor.value = params.get("slice_nr_cor", 0)
                slice_slider_sag.value = params.get("slice_nr_sag", 0)
                echo_slider.value = params.get("echo_nr", 0)
                slice_slider_rep1.value = params.get("rep1", 0)
                slice_slider_chan.value = params.get("chan", 0)
                plot_opts_alpha_overlay.value = params.get("alpha_overlay", 0.5)
                slice_slider_rep2.value = params.get("rep2", 0)
                rep_avg_checkbox.value = params.get("rep_avg", False)
                plot_opts_part.value = params.get("plot_style", "Abs")
                plot_opts_metabolite.value = params.get("pyr_lac", "Pyr")
                plot_opts_anat_overlay.value = params.get("anat_overlay", "Metab")
                plot_opts_metab_clim.value = params.get("metab_clim", np.inf)
                plot_opts_cmap.value = params.get("cmap", "jet")
                plot_opts_interpolation_method.value = params.get(
                    "interp_method", "bilinear"
                )
                text_background.value = params.get("background", 0)
                plot_opts_interpolation_factor.value = params.get("interp_factor", 1)
                plot_images_individually_checkbox.value = params.get(
                    "plot_images_individually", "False"
                )
            except:
                pass

            # set plotting parameters to slider values:
            plot_img_params = {
                "slice_nr_ax": slice_slider_ax,
                "slice_nr_cor": slice_slider_cor,
                "slice_nr_sag": slice_slider_sag,
                "echo_nr": echo_slider,
                "chan": slice_slider_chan,
                "alpha_overlay": plot_opts_alpha_overlay,
                "plot_style": plot_opts_part,
                "experiment_style": plot_opts_measurement,
                "rep1": slice_slider_rep1,
                "rep2": slice_slider_rep2,
                "rep_avg": rep_avg_checkbox,
                "pyr_lac": plot_opts_metabolite,
                "anat_overlay": plot_opts_anat_overlay,
                "metab_clim": plot_opts_metab_clim,
                "cmap": plot_opts_cmap,
                "interp_method": plot_opts_interpolation_method,
                "interp_factor": plot_opts_interpolation_factor,
                "background": text_background,
                "plot_images_individually": plot_images_individually_checkbox,
                # "zero_as_NaN": plot_opts_0_as_NaN,
            }
            return plot_img_params

        # Define the on_click function
        def on_click(event):
            if event.inaxes in axlist:
                # axial image:
                if event.inaxes == axlist[0]:
                    # get plot parameters:
                    plot_img_params = set_plot_params(self)

                    # take click position:
                    slice_x_sag = event.xdata
                    slice_y_cor = event.ydata

                    # define position:
                    pos = (0, slice_x_sag, -slice_y_cor)

                    # get bssfp grid:
                    sag_min_ind = np.argmin(np.abs(bssfp_grid[1] - pos[1]))
                    cor_min_ind = np.argmin(np.abs(bssfp_grid[2] - pos[2]))

                    # setthe 2 plot parameters:
                    plot_img_params["slice_nr_sag"].value = sag_min_ind
                    plot_img_params["slice_nr_cor"].value = cor_min_ind
                    # update plot parameter dict:
                    plot_img_params = set_plot_params(self, params=plot_img_params)
                # coronal image:
                elif event.inaxes == axlist[1]:
                    # get plot parameters:
                    plot_img_params = set_plot_params(self)

                    # take click position:
                    slice_x_sag = event.xdata
                    slice_y_ax = event.ydata

                    # define position:
                    pos = (slice_y_ax, slice_x_sag, 0)

                    # get bssfp grid:
                    sag_min_ind = np.argmin(np.abs(bssfp_grid[1] - pos[1]))
                    ax_min_ind = np.argmin(np.abs(bssfp_grid[0] - pos[0]))

                    # setthe 2 plot parameters:
                    plot_img_params["slice_nr_sag"].value = sag_min_ind
                    plot_img_params["slice_nr_ax"].value = ax_min_ind
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
                    rep1_after = event.xdata

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
        main_ui = widgets.VBox([ui_up, ui_colors, ui_slices, ui_reps_echoes_chans])

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

            # make it path if not:
            if path_parentfolder[-1] != "/":
                path_to_save = path_parentfolder + "/"

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

            plot_images_individually = plot_images_individually_checkbox.value

            def save_subplot(ax, filename):
                from matplotlib.transforms import Bbox

                # Get the bounding box of the subplot in figure coordinates
                bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
                    fig.dpi_scale_trans.inverted()
                )

                # Pad the bounding box to ensure nothing is cut off
                # You can adjust the padding as necessary
                padding = 0.1  # Padding in inches
                bbox_padded = Bbox.from_bounds(
                    bbox.x0 - padding,
                    bbox.y0 - padding,
                    bbox.width + 2 * padding,
                    bbox.height + 2 * padding,
                )

                # Save just the region of the figure that contains the subplot
                fig.savefig(filename, bbox_inches=bbox_padded, transparent=True)

            if plot_images_individually:
                # Loop through each subplot and save it individually
                for i, ax in enumerate(fig.get_axes()[:3]):
                    # default order, might give wrong results if axlist < 3 elements
                    if i == 0:
                        subplot_name = "_axial"
                    elif i == 1:
                        subplot_name = "_coronal"
                    elif i == 2:
                        subplot_name = "_timecurve"
                    else:
                        subplot_name = f"_subplot_{i}"

                    subplot_filename_svg = os.path.join(
                        path_parentfolder, f"{fig_name}{subplot_name}.svg"
                    )
                    subplot_filename_png = os.path.join(
                        path_parentfolder, f"{fig_name}{subplot_name}.png"
                    )

                    save_subplot(ax, subplot_filename_svg)
                    save_subplot(ax, subplot_filename_png)

            else:
                # Save the whole figure as before
                fig.savefig(os.path.join(path_parentfolder, fig_name + ".svg"))
                fig.savefig(
                    os.path.join(path_parentfolder, fig_name + ".png"), format="png"
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

        # This displays the Hbox containing the slider widgets.
        if display_ui:
            display(main_ui, out)
            if save_fig:
                save_figure(self, fig_name=fig_name)
        else:
            return main_ui

    def generate_custom_cmap(self, cmap_name=None):
        # Generate a custom colormap
        if cmap_name is None:
            return None
        else:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import numpy as np

        # list of predefined colormap:
        options = [
            "cividis",
            "inferno",
            "magma",
            "plasma",
            "viridis",
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
            "gray",
            "hot",
            "pink",
            "spring",
            "summer",
            "winter",
            "jet",
        ]

        # return predefined colormap:
        if cmap_name in options:
            cmap = plt.get_cmap(cmap_name)
            return cmap_name, cmap

        elif cmap_name == "custom_cold":
            # Define the colors in the colormap
            colors = ["black", "blue", "turquoise", "white"]

            # Create a list of normalized values for the colors
            color_values = np.linspace(0, 1, len(colors))
            color_values = [0.0, 0.2, 0.4, 0.8, 1.0]

            # Create a colormap using the defined colors and values
            colormap = mcolors.LinearSegmentedColormap.from_list(
                "custom_colormap", list(zip(color_values, colors))
            )
            # Convert the colormap to a cmap object
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                "custom_cmap", colors, N=256
            )

            return colormap, cmap
        else:
            return None, None

    def define_grid(self, fov=None, mat=None):
        """
        Defines a grid with the points as the centers of the voxels

        --------
        Parameters:

        fov:
        Field of View in mm [fovx, fovy, fovz]

        mat:
        matrix size in [fovx, fovy, fovz]

        -----
        Returns:
        nd-arrays, defining center of voxels
        """

        if fov is None:
            fov = np.array(self.define_imageFOV_parameters())
        if mat is None:
            mat = np.array(self.define_imagematrix_parameters())
        res = fov / mat

        # init:
        ext_1 = ext_2 = ext_3 = None
        if (len(fov) > 0) and (len(mat) > 0):
            ext_1 = np.linspace(
                -fov[0] / 2 + res[0] / 2, fov[0] / 2 - res[0] / 2, mat[0]
            )
        if (len(fov) > 1) and (len(mat) > 1):
            ext_2 = np.linspace(
                -fov[1] / 2 + res[1] / 2, fov[1] / 2 - res[1] / 2, mat[1]
            )
        if (len(fov) > 2) and (len(mat) > 2):
            ext_3 = np.linspace(
                -fov[2] / 2 + res[2] / 2, fov[2] / 2 - res[2] / 2, mat[2]
            )

        return ext_1, ext_2, ext_3

    def define_imagematrix_parameters(self):
        """
        Warning: Does not take into account the orientation and offsets of the
        object (yet)
        Define the imaging matrix in voxel.
        Returns imaging matrix dimensions  as dim_read, dim_phase, dim_slice
        Input
        -----
        self: Sequence object
        """
        dim_read = self.method["PVM_Matrix"][0]  # was z
        dim_phase = self.method["PVM_Matrix"][1]  # was y
        if self.method["PVM_SpatDimEnum"] == "<3D>":
            dim_slice = self.method["PVM_Matrix"][2]  # was x
        else:
            dim_slice = self.method["PVM_SPackArrNSlices"]  # was x
        return dim_read, dim_phase, dim_slice

    def define_imageFOV_parameters(self):
        """
        Warning: Does not take into account the orientation and offsets of the
        object (yet)
        Calculates the FOV in mm.
        Returns FOV in as mm_read, mm_phase, mm_slice.
        Input
        -----
        self: Sequence object
        """
        # FOV:
        mm_read = self.method["PVM_Fov"][0]
        mm_phase = self.method["PVM_Fov"][1]
        mm_slice_gap = self.method["PVM_SPackArrSliceGap"]

        if self.method["PVM_SpatDimEnum"] == "<3D>":
            mm_slice = self.method["PVM_Fov"][2]
        else:
            _, _, dim_slice = self.define_imagematrix_parameters()
            mm_slice = self.method["PVM_SliceThick"]  # was x
            mm_slice = mm_slice * dim_slice + mm_slice_gap * (dim_slice - 1)

        return mm_read, mm_phase, mm_slice

    def interpolate_bssfp(
        self,
        bssfp_data=None,
        interp_size=None,
        interp_method="linear",
        use_multiprocessing=False,
        dtype=None,
        number_of_cpu_cores=None,
    ):
        """
        This function interpolates the input data (bssfp_data) onto a new grid in the same range as
        bssfp_data but points in each dimension are described by interp_size. Uses scipy's RegularGridInterpolator
        method. Multithreading can be activated.

        Parameters
        ----------
        bssfp_data: ND array
            Expected to have the order Echoes, Read, phase, slice, repetitions, channels
        interp_size: tuple
            Describes the interpolated data size.
            Has to be 6D: number of echhoes, points in  Read, phase, slice; number of repetitions, channels
        interp_method: str
            interpolation method that is used. Supported are "linear", "nearest",
            # if scipy==1.10.0: "slinear", "cubic", "quintic" and "pchip".
        use_multiprocessing: bool
            toggles multiprocessing
        dtype: strs
            should give you the option to change the dtype of the to be interpolated array to reduce the size

        number_of_cpu_cores:
            Number of CPU cores to if multi processing is desired

        Examples
        --------
        # interpolate 1st and 2nd dimension onto a 10x higher grid:

        # get extend in x-y-z:

        >>> x, y, z = dnp_bssfp.define_grid()
        >>> interp_factor = 10

        >>> xres = np.linspace(x[0], x[-1], interp_factor*len(x))
        >>> yres = np.linspace(y[0], y[-1], interp_factor*len(y))

        >>> zres = np.linspace(z[0], z[-1], len(z))
        >>> t = np.arange(0,dnp_bssfp_pv_reco.shape[4],1)

       >>> test_nearest = dnp_bssfp.interpolate_bssfp(bssfp_data=dnp_bssfp_pv_reco,
       >>>                     interp_size=(1,len(xres),len(yres),len(zres),len(t),2),
       >>>                    interp_method="nearest")


        >>> plt.figure()
        >>> for k in range(10):
        >>> ax = plt.subplot(3,4,k+1)
        >>> ax.imshow(np.squeeze(np.mean(test_linear[:,:,:,k,:,:], axis=3)))


        ------------------------------------------------------------------------
        # interpolate every 2nd image (lactate) onto a 4x higher time resoltion:

        # get extend in x-y-z:

        >>> x, y, z = dnp_bssfp.define_grid()
        >>> interp_factor = 4

        >>> xres = np.linspace(x[0], x[-1], len(x))
        >>> yres = np.linspace(y[0], y[-1], len(y))
        >>> zres = np.linspace(z[0], z[-1], len(z))

        >>> t = np.arange(0,dnp_bssfp_pv_reco.shape[4],1)
        >>> tres = np.linspace(z[0], t[-1], interp_factor*len(t))


        >>> test_lac_interp = dnp_bssfp.interpolate_bssfp(bssfp_data=dnp_bssfp_pv_reco[:,:,:,:,::2,:],
        >>>                     interp_size=(1,len(xres),len(yres),len(zres),len(tres),2),
        >>>                     interp_method="linear")
        >>> plt.figure()
        >>> reps = range(100)
        >>> for k in reps:
        >>> ax = plt.subplot(10,10,k+1)
        >>> ax.imshow(np.squeeze(test_lac_interp[0,:,:,5,k+150,0]))

        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy.interpolate import interpn

        import numpy as np

        if use_multiprocessing:
            try:
                from tqdm.auto import tqdm
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

        import time

        # init empty output
        input_data = None

        # init empty output
        if bssfp_data is None:
            input_data = self.Reconstructed_data
        else:
            input_data = bssfp_data

        if interp_size is None:
            Warning("Don't know what to do")
            interp_size = input_data.shape

        # check method:
        allowed_interp_methods = ["linear", "nearest", "splinef2d", "cubic"]
        if interp_method not in allowed_interp_methods:
            logger.critical(
                "uknown interpolation method: "
                + interp_method
                + ", using linear instead."
            )
            interp_method = "linear"

        if dtype is None:
            dtype = bssfp_data.dtype

        # duration:
        t = np.arange(0, input_data.shape[4], 1)
        t_num = input_data.shape[4]

        # cant loop over repetitions if this also should
        # be interpolated ...
        if len(t) == interp_size[4]:
            pass
        else:
            use_multiprocessing = False

        # get extend in x-y-z:
        x, y, z = self.define_grid()
        # number of echoes:
        e_num = input_data.shape[0]
        # range of echoes:
        e = np.linspace(1, e_num, input_data.shape[0])
        # number of channels:
        c_num = input_data.shape[5]
        # range of channels:
        c = np.linspace(1, c_num, input_data.shape[5])

        # points to interpolate onto:
        # echoes:
        eres = np.linspace(1, e[-1], interp_size[0])
        # points in space:
        xres = np.linspace(x[0], x[-1], interp_size[1])
        yres = np.linspace(y[0], y[-1], interp_size[2])
        zres = np.linspace(z[0], z[-1], interp_size[3])
        # repetitions:
        tres = np.linspace(t[0], t[-1], interp_size[4])

        # channels:
        if e_num == 1:
            eres = np.linspace(1, e_num, input_data.shape[0])
        else:
            eres = np.linspace(1, e[-1], interp_size[0])

        # channels:
        if c_num == 1:
            cres = np.linspace(1, c_num, input_data.shape[5])
        else:
            cres = c

        # get the start time
        st = time.time()

        if use_multiprocessing:
            # generate grids (to interpolate onto)
            egres, xgres, ygres, zgres, cgres = np.meshgrid(
                eres, xres, yres, zres, cres, indexing="ij"
            )
            # init empty array:
            interpolated_data = np.zeros(
                (len(eres), len(xres), len(yres), len(zres), len(t), len(cres)),
                dtype=dtype,
            )

            # index list (time range)
            index_list = list(np.ndindex(len(t)))

            # create progress bar
            index_list = tqdm(index_list, desc="interpolation progress", leave=True)

            # define interpolationm functionm
            def interpolate_image(it):
                # the conditions speed up the interpolation quite a bit
                # if there are less dimensions (1 echo e.g.)

                # if more than 1 echo and more than 1 channel:
                if e_num > 1 and c_num > 1:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(e, x, y, z, c),
                            values=np.squeeze(input_data[:, :, :, :, it, :]),
                            xi=(egres, xgres, ygres, zgres, cgres),
                            method=interp_method,
                        )
                    )
                # if more than 1 echo and 1 channel:
                elif e_num > 1 and c_num == 0:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(e, x, y, z),
                            values=np.squeeze(input_data[:, :, :, :, it, 0]),
                            xi=(egres, xgres, ygres, zgres),
                            method=interp_method,
                        )
                    )
                # if 1 echo and more than 1 channel:
                elif e_num == 1 and c_num > 1:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(x, y, z, c),
                            values=np.squeeze(input_data[0, :, :, :, it, :]),
                            xi=(xgres, ygres, zgres, cgres),
                            method=interp_method,
                        )
                    )
                # if 1 echo and 1 channel:
                elif e_num == 1 and c_num == 1:
                    interpolated_data_timepoint = np.squeeze(
                        interpn(
                            points=(x, y, z),
                            values=np.squeeze(input_data[0, :, :, :, it, 0]),
                            xi=(xgres, ygres, zgres),
                            method=interp_method,
                        )
                    )

                return interpolated_data_timepoint

            # interpolate multiple timesteps in parallel:
            interpolated_data_tuple = Parallel(n_jobs=number_of_cpu_cores)(
                delayed(interpolate_image)(it) for it in index_list
            )

            # if more than 1 echo and more than 1 channel:
            for it in range(len(t)):
                if e_num > 1 and c_num > 1:
                    interpolated_data[:, :, :, :, it, :] = interpolated_data_tuple[it]
                # if more than 1 echo and 1 channel:
                elif e_num > 1 and c_num == 0:
                    interpolated_data[:, :, :, :, it, 0] = interpolated_data_tuple[it]
                # if 1 echo and more than 1 channel:
                elif e_num == 1 and c_num > 1:
                    interpolated_data[0, :, :, :, it, :] = interpolated_data_tuple[it]
                # if 1 echo and 1 channel:
                elif e_num == 1 and c_num == 1:
                    interpolated_data[0, :, :, :, it, 0] = interpolated_data_tuple[it]

        elif use_multiprocessing == False and (t_num > 1):
            # generate grids (to interpolate onto)
            egres, xgres, ygres, zgres, tgres, cgres = np.meshgrid(
                eres, xres, yres, zres, tres, cres, indexing="ij"
            )

            # init empty array:
            interpolated_data = np.zeros(
                (
                    len(eres),
                    len(xres),
                    len(yres),
                    len(zres),
                    len(tres),
                    len(cres),
                ),
                dtype=dtype,
            )
            # if more than 1 echo and more than 1 channel:
            if e_num > 1 and c_num > 1:
                interpolated_data = np.squeeze(
                    interpn(
                        points=(e, x, y, z, t, c),
                        values=input_data[:, :, :, :, :, :],
                        xi=(egres, xgres, ygres, zgres, tgres, cgres),
                        method=interp_method,
                    )
                )
            # if more than 1 echo and 1 channel:
            elif e_num > 1 and c_num == 0:
                interpolated_data[:, :, :, :, :, 0] = np.squeeze(
                    interpn(
                        points=(e, x, y, z, t),
                        values=input_data[:, :, :, :, :, 0],
                        xi=(egres, xgres, ygres, zgres, tgres),
                        method=interp_method,
                    )
                )

            # if 1 echo and more than 1 channel:
            elif e_num == 1 and c_num > 1:
                interpolated_data[0, :, :, :, :, :] = np.squeeze(
                    interpn(
                        points=(x, y, z, t, c),
                        values=np.squeeze(input_data[0, :, :, :, :, :]),
                        xi=(xgres, ygres, zgres, tgres, cgres),
                        method=interp_method,
                    )
                )
            # if 1 echo and 1 channel:
            elif e_num == 1 and c_num == 1:
                interpolated_data[0, :, :, :, :, 0] = np.squeeze(
                    interpn(
                        points=(x, y, z, t),
                        values=input_data[0, :, :, :, :, 0],
                        xi=(xgres, ygres, zgres, tgres),
                        method=interp_method,
                    )
                )

        elif use_multiprocessing == False and (t_num == 1):
            # generate grids (to interpolate onto)
            egres, xgres, ygres, zgres, cgres = np.meshgrid(
                eres, xres, yres, zres, cres, indexing="ij"
            )
            # init empty array:
            interpolated_data = np.zeros(
                (len(eres), len(xres), len(yres), len(zres), len(tres), len(cres)),
                dtype=dtype,
            )
            # if more than 1 echo and more than 1 channel:
            if e_num > 1 and c_num > 1:
                interpolated_data[:, :, :, :, 0, :] = np.squeeze(
                    interpn(
                        points=(e, x, y, z, c),
                        values=input_data[:, :, :, :, 0, :],
                        xi=(egres, xgres, ygres, zgres, cgres),
                        method=interp_method,
                    )
                )

            # if more than 1 echo and  1 channel:
            elif e_num > 1 and c_num == 1:
                interpolated_data[:, :, :, :, 0, 0] = np.squeeze(
                    interpn(
                        points=(
                            e,
                            x,
                            y,
                            z,
                        ),
                        values=input_data[:, :, :, :, 0, 0],
                        xi=(egres, xgres, ygres, zgres),
                        method=interp_method,
                    )
                )
            # if 1 echo and more than 1 channel:
            elif e_num == 1 and c_num > 1:
                for it in range(len(t)):
                    interpolated_data[0, :, :, :, 0, :] = np.squeeze(
                        interpn(
                            points=(x, y, z, c),
                            values=np.squeeze(input_data[0, :, :, :, 0, :]),
                            xi=(xgres, ygres, zgres, cgres),
                            method=interp_method,
                        )
                    )
            # if 1 echo and 1 channel:
            elif e_num == 1 and c_num == 1:
                for it in range(len(t)):
                    interpolated_data[0, :, :, :, 0, 0] = np.squeeze(
                        interpn(
                            points=(x, y, z),
                            values=input_data[0, :, :, :, 0, 0],
                            xi=(xgres, ygres, zgres),
                            method=interp_method,
                        )
                    )

        # get the execution time
        et = time.time()
        elapsed_time = et - st
        # print excution time:
        logger.debug("Execution time:", elapsed_time, "seconds")

        return interpolated_data

    def calc_timeaxis(self, start_with_0=True):
        """
        calculate the time range [in s] on which the bssfp data was acquired.

        ---------
        Parameters:

        start_with_0 (bool)
            wether the time axis should start with 0 or 1 image repetition time

        Return:
            Time axis [in seconds]
        """
        time_axis = None
        # number of repetitions
        nr = self.method["PVM_NRepetitions"]
        # scan time per image in  seconds
        image_repetition_time = self.method["PVM_ScanTime"] / nr / 1000.0

        # time range start with 0:
        if start_with_0:
            time_axis = np.linspace(0, (nr - 1) * image_repetition_time, nr)
            # time range start with 1 image reptition time:
        else:
            time_axis = np.linspace(
                image_repetition_time, nr * image_repetition_time, nr
            )

        return time_axis

    def calc_AUC(
        self,
        signal=None,
        time_axis=None,
        axlist=None,
        display_ui=True,
        pyr_ind=0,
        mask_id=None,
        mask_key=None,
        signal_range_input=[None, None],
        noise_range_input=[None, None],
        noise_signal=None,
        apply_filter=True,
        save_dir=None,
    ):
        """
        Opens a GUI that lets you set the range to calculate the signal range
        and the noise range of a signal vs. time curve. automatically calculates
        the area under the curve (AUC)

        ------------
        Parameters:

        signal: 1xN (1 metabolite) or 2xN (2 metabolites) array. Usually extracted
            from a masked bssfp dataset (1xN) or (2xN)

        timeaxis: Time axis of the metabolites. If no input given, integer 1:N
            are used. (1xN)

        axlist: list of 2 pyplot axis: can be generated via
            fig, (ax1, ax2) = plt.subplots(1, 2,tight_layout=True)
            and set via:
            axlist = [ax1, ax2]

        pyr_ind: either 0 or 1, depending on which metabolite (pyr or lac) has been acquired
            first

        mask_id: can be used to indicate from which mask the signal curves are.

        mask_key: will be used to store the auc in a auc_ratio_dict under this key

        signal_range_input: list, can store indices corresponding to starting/ending points for signal summation

        noise_range_input: list, can store indices corresponding to starting/ending points for noise subtraction

        noise_signal: noise signal time curve that is subtracted from the signal.

        appyly_filter: If true, a savitzky Golay filter is applied to smooth the noise data

        ------------
        Return:
            area under the curve of pyruvate, lactate and the ratio of lac/pyr
            auc_pyr, auc_lac, auc_ratio


        -----------
        Example Usage:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,tight_layout=True, figsize=(12,4))

        roi_key = "kidneyL"
        phip_bssfp.calc_AUC(signal=(np.abs(phip_signal_pyr[roi_key]),
                                    np.abs(phip_signal_lac[roi_key])),
                            axlist = [ax1, ax2, ax3, ax4],
                            mask_key=roi_key,
                            signal_range_input=[4,51],
                            noise_range_input=[53,73],
                            noise_signal=np.abs(phip_signal_pyr['outside_ref']),
                            apply_filter=True)

        """
        from decimal import Decimal

        # no signal --> exit
        if signal is None:
            return False

        # if 6D take average along spatial dimensions:
        if np.ndim(signal) == 6:
            signal = np.squeeze(np.nanmean(signal, axis=(1, 2, 3)))
        # if 7D take average along spatial dimensions (pyr, lac in first dimension):
        if np.ndim(signal) == 7:
            signal = np.squeeze(np.nanmean(signal, axis=(2, 3, 4)))

        # if 6D take average along spatial dimensions:
        if np.ndim(noise_signal) == 6:
            noise_signal = np.squeeze(np.nanmean(noise_signal, axis=(1, 2, 3)))

        if time_axis is None:
            print(np.ndim(signal))
            if np.ndim(signal) == 1:
                time_axis = np.linspace(0, signal.shape[0] - 1, signal.shape[0])
            elif np.ndim(signal) == 2:
                time_axis = np.linspace(0, len(signal[0]) - 1, len(signal[0]))
            else:
                Warning("ndims signal has to be 1D or 2D!")
                return False

        if axlist is None:
            return False
        global auc_pyr, auc_lac, auc_ratio
        auc_pyr = 0.0
        auc_lac = 0.0
        auc_ratio = 0.0

        # define function that save the calculated AUCs:
        def save_auc(
            self,
            auc_lac=np.nan,
            auc_pyr=np.nan,
            auc_ratio=np.nan,
            mask_id=None,
            mask_key=None,
            signal_range=None,
            noise_range=None,
            auc_lac_dyn_noise=np.nan,
            auc_pyr_dyn_noise=np.nan,
            auc_ratio_dyn_noise=np.nan,
        ):
            # if a mask_id has been passed, store the calculated AUC under this mask_id
            if mask_id is None:
                self.auc_ratio_nomask = auc_ratio
                self.auc_lac_nomask = auc_lac
                self.auc_pyr_nomask = auc_pyr

                # try to save the AUCs caluclated by subtracting the time curves:
                if noise_signal is not None:
                    self.auc_ratio_dyn_noise_nomask = auc_ratio_dyn_noise
                    self.auc_lac_dyn_noise_nomask = auc_lac_dyn_noise
                    self.auc_pyr_dyn_noise_nomask = auc_pyr_dyn_noise
            # try to save the AUCs under the mask IDs
            else:
                # try to save AUC with mask ID:
                try:
                    self.auc_ratio[mask_id] = auc_ratio
                    self.auc_lac[mask_id] = auc_lac
                    self.auc_pyr[mask_id] = auc_pyr
                except:
                    try:
                        # init empty array for AUCs
                        setattr(self, "auc_ratio", np.zeros((10, 1)))
                        setattr(self, "auc_lac", np.zeros((10, 1)))
                        setattr(self, "auc_pyr", np.zeros((10, 1)))
                        # try to save AUC with mask ID:
                        # try to save AUC with mask ID:
                        self.auc_ratio[mask_id] = auc_ratio
                        self.auc_lac[mask_id] = auc_lac
                        self.auc_pyr[mask_id] = auc_pyr
                    except:
                        # save AUC without mask ID:
                        self.auc_ratio_nomask = auc_ratio
                        self.auc_lac_nomask = auc_lac
                        self.auc_pyr_nomask = auc_pyr
                # try to save the AUCs caluclated by subtracting the time curves:
                if noise_signal is not None:
                    # try to save AUC with mask ID:
                    try:
                        self.auc_ratio_dyn_noise[mask_id] = auc_ratio_dyn_noise
                        self.auc_lac_dyn_noise[mask_id] = auc_lac_dyn_noise
                        self.auc_pyr_dyn_noise[mask_id] = auc_pyr_dyn_noise
                    except:
                        try:
                            # init empty array for AUCs
                            setattr(self, "auc_ratio_dyn_noise", np.zeros((10, 1)))
                            setattr(self, "auc_lac_dyn_noise", np.zeros((10, 1)))
                            setattr(self, "auc_pyr_dyn_noise", np.zeros((10, 1)))
                            # try to save AUC with mask ID:
                            # try to save AUC with mask ID:
                            self.auc_ratio_dyn_noise[mask_id] = auc_ratio_dyn_noise
                            self.auc_lac_dyn_noise[mask_id] = auc_lac_dyn_noise
                            self.auc_pyr_dyn_noise[mask_id] = auc_pyr_dyn_noise
                        except:
                            # save AUC without mask ID:
                            self.auc_ratio_dyn_noise_nomask = auc_ratio_dyn_noise
                            self.auc_lac_dyn_noise_nomask = auc_lac_dyn_noise
                            self.auc_pyr_dyn_noise_nomask = auc_pyr_dyn_noise

            if mask_key is not None:
                # try to save AUCs in dic via passed key:
                try:
                    self.auc_ratio_dic[mask_key] = auc_ratio
                    self.auc_lac_dic[mask_key] = auc_lac
                    self.auc_pyr_dic[mask_key] = auc_pyr
                except:
                    try:
                        # init empty dictonary:
                        setattr(self, "auc_ratio_dic", {})
                        setattr(self, "auc_lac_dic", {})
                        setattr(self, "auc_pyr_dic", {})
                        # save AUCs:
                        self.auc_ratio_dic[mask_key] = auc_ratio
                        self.auc_lac_dic[mask_key] = auc_lac
                        self.auc_pyr_dic[mask_key] = auc_pyr
                    except:
                        pass

                # try to save the AUCs caluclated by subtracting the time curves:
                if noise_signal is not None:
                    # try to save AUCs in dic via passed key:
                    try:
                        self.auc_ratio_dyn_noise_dic[mask_key] = auc_ratio_dyn_noise
                        self.auc_lac_dyn_noise_dic[mask_key] = auc_lac_dyn_noise
                        self.auc_pyr_dyn_noise_dic[mask_key] = auc_pyr_dyn_noise
                    except:
                        try:
                            # init empty dictonary:
                            setattr(self, "auc_ratio_dyn_noise_dic", {})
                            setattr(self, "auc_lac_dyn_noise_dic", {})
                            setattr(self, "auc_pyr_dyn_noise_dic", {})
                            # save AUCs:
                            self.auc_ratio__dyn_noisedic[mask_key] = auc_ratio_dyn_noise
                            self.auc_lac_dyn_noise_dic[mask_key] = auc_lac_dyn_noise
                            self.auc_pyr_dyn_noise_dic[mask_key] = auc_pyr_dyn_noise
                        except:
                            pass

                # try to save signal/noise range via passed key:
                try:
                    self.auc_range_dic = {
                        "mask_key": {
                            "signal_range": signal_range,
                            "noise_range": noise_range,
                        }
                    }
                except:
                    try:
                        # init empty dictonary:
                        setattr(self, "auc_range_dic", {})
                        # save AUC ranges:
                        self.auc_range_dic = {
                            "mask_key": {
                                "signal_range": signal_range,
                                "noise_range": noise_range,
                            }
                        }
                    except:
                        pass

            else:
                pass

        # --------------------------- define plot function -----------------------------
        def plot_auc(
            signal_range=[0, 1],
            noise_range=[0, 1],
            subtract_noise=False,
            save_csv=False,
        ):
            # clear axis:
            for ax in axlist:
                ax.clear()

            # turn selected range into indices:
            noise_ind = [0] * 2
            noise_ind[0] = np.argmin(np.abs(time_axis - noise_range[0]))
            noise_ind[1] = np.argmin(np.abs(time_axis - noise_range[1]))

            signal_ind = [0] * 2
            signal_ind[0] = np.argmin(np.abs(time_axis - signal_range[0]))
            signal_ind[1] = np.argmin(np.abs(time_axis - signal_range[1]))

            # calc mean over noise range
            mean_noise = [0.0] * np.ndim(signal)
            if subtract_noise:
                # if pyruvate and lactate:
                if np.ndim(signal) == 2:
                    for s in range(np.ndim(signal)):
                        mean_noise[s] = np.mean(signal[s][noise_ind[0] : noise_ind[1]])
                elif np.ndim(signal) == 1:
                    mean_noise = np.mean(signal[noise_ind[0] : noise_ind[1]])
                else:
                    logger.critical("input signal has to be 1D or 2D")
            # calc mean average, use all signa channels:
            mean_noise = np.mean(mean_noise)
            signal_no_const_noise = signal - mean_noise

            # integrate signal:
            integrated_signal = [0.0] * (np.ndim(signal))
            for s in range(np.ndim(signal)):
                if np.ndim(signal) == 2:
                    integrated_signal[s] = np.sum(
                        signal_no_const_noise[s][signal_ind[0] : signal_ind[1]]
                    )
                elif np.ndim(signal) == 1:
                    integrated_signal = np.sum(
                        signal_no_const_noise[signal_ind[0] : signal_ind[1]]
                    )
                else:
                    logger.critical("input signal has to be 1D or 2D")

            if np.ndim(signal) > 1:
                if pyr_ind == 0:
                    auc_lac = integrated_signal[pyr_ind + 1]
                    auc_pyr = integrated_signal[pyr_ind]
                    auc_ratio = auc_lac / auc_pyr
                elif pyr_ind == 1:
                    auc_lac = integrated_signal[pyr_ind - 1]
                    auc_pyr = integrated_signal[pyr_ind]
                    auc_ratio = auc_lac / auc_pyr
                else:
                    Warning("Don't understand the indexing")
                    auc_lac = integrated_signal
                    auc_pyr = integrated_signal
                    auc_ratio = 1.0
                    pass

            elif np.ndim(signal) == 1:
                auc_lac = 0.0
                auc_pyr = integrated_signal
                auc_ratio = 0.0
            else:
                logger.critical("input signal has to be 1D or 2D")

            logger.critical("const. noise AUC pyr : " + str(auc_pyr))
            logger.critical("const. noise AUC lac : " + str(auc_lac))
            logger.critical("const. noise AUC lac/pyr : " + str(auc_ratio))

            # subplot 1 -----------------------------------------------------------
            # plot signal curves:
            if np.ndim(signal) == 2:
                for s in range(np.ndim(signal)):
                    axlist[0].plot(time_axis, signal[s])
            elif np.ndim(signal) == 1:
                axlist[0].plot(time_axis, signal)
            else:
                logger.critical("input signal has to be 1D or 2D")

            for r in range(2):
                # plot signal range:
                axlist[0].axvline(
                    signal_range[r],
                    color="black",
                    linewidth=1,
                    linestyle="dashed",
                    label="noise range",
                )
                # plot noise range:
                axlist[0].axvline(
                    noise_range[r],
                    color="red",
                    linewidth=1,
                    linestyle="dashed",
                    label="signal range",
                )

            # plot mean noise:
            axlist[0].axhline(
                mean_noise,
                color="red",
                linewidth=1,
                linestyle="dashed",
                label="signal range",
            )

            logger.critical(np.ndim(signal))
            # subplot 2 -----------------------------------------------------------
            # plot integrated signal:
            if np.ndim(signal) == 2:
                for s in range(np.ndim(signal)):
                    axlist[1].plot(time_axis, np.cumsum(signal[s] - mean_noise))
                    axlist[1].axhline(
                        np.cumsum(signal[s] - mean_noise)[-1],
                        color="red",
                        linewidth=1,
                        linestyle="dashed",
                    )
            elif np.ndim(signal) == 1:
                axlist[1].plot(time_axis, np.cumsum(signal - mean_noise))
                axlist[1].axhline(
                    np.cumsum(signal - mean_noise)[-1],
                    color="red",
                    linewidth=1,
                    linestyle="dashed",
                )
            else:
                logger.critical("input signal has to be 1D or 2D")

            if save_button.value:
                filename = os.path.join(save_dir, "time_and_aucs.csv")
                if np.ndim(signal) == 2:
                    data_to_save = np.column_stack(
                        (time_axis, signal[0] - mean_noise, signal[1] - mean_noise)
                    )
                    np.savetxt(filename, data_to_save, fmt="%f", delimiter=",")
                else:
                    data_to_save = np.column_stack((time_axis, signal - mean_noise))
                    np.savetxt(filename, data_to_save, fmt="%f", delimiter=",")

            for r in range(2):
                # plot signal range:
                axlist[1].axvline(
                    signal_range[r],
                    color="black",
                    linewidth=1,
                    linestyle="dashed",
                )

            # axlist[0].legend("noise range", "signal range")  # ,
            # title="AUC ratio " + str(np.round(auc_ratio, 3)),
            # )
            # set labels signal over time:
            axlist[0].set_title("Signal over repetitions")
            axlist[0].set_ylabel("I [a.u.] ")
            axlist[0].set_xlabel("Time since start [s] ")
            # set labels integrated signal over time:
            axlist[1].set_xlabel("Time since start [s] ")
            axlist[1].legend()
            if mask_key is not None:
                axlist[1].set_title(mask_key + " - integr. signal")
            else:
                axlist[1].set_title("integrated signal")
            # draw grid:
            axlist[1].grid(color="r", linestyle="-", linewidth=0.25, which="major")
            axlist[1].grid(color="r", linestyle="--", linewidth=0.25, which="minor")

            # plot the signal with the loaded noise_range subtracted:
            if noise_signal is not None:
                # if filter should be applied
                if apply_filter is True:
                    # imort savitzky-golay filter:
                    from scipy.signal import savgol_filter

                    noise_signal_smooth = savgol_filter(noise_signal, 30, 3)
                else:
                    # don't apply filter:
                    noise_signal_smooth = noise_signal

                # subtract dymamic (loaded) noise:
                signal_no_dyn_noise = signal - noise_signal_smooth
                # init empty array:
                integrated_signal_dyn_noise = [0.0] * (np.ndim(signal))
                # integrate signal:
                if np.ndim(signal) == 2:
                    for s in range(np.ndim(signal)):
                        integrated_signal_dyn_noise[s] = np.sum(signal_no_dyn_noise[s])
                elif np.ndim(signal) == 1:
                    integrated_signal_dyn_noise = np.sum(signal_no_dyn_noise)
                else:
                    logger.critical("input signal has to be 1D or 2D")

                # plot signal with subtracted noise:
                if len(axlist) > 2:
                    if np.ndim(signal) == 2:
                        # plot signal subtracted noise:
                        for s in range(np.ndim(signal)):
                            axlist[2].plot(time_axis, signal_no_dyn_noise[s])
                            axlist[2].set_title("Signal (subtr. dyn noise)")
                            axlist[2].set_ylabel("I [a.u.]")
                            axlist[2].set_xlabel("Time [s]")
                    elif np.ndim(signal) == 1:
                        # plot signal subtracted noise:
                        axlist[2].plot(time_axis, signal_no_dyn_noise)
                        axlist[2].set_title("Signal (subtr. dyn noise)")
                        axlist[2].set_ylabel("I [a.u.]")
                        axlist[2].set_xlabel("Time [s]")
                    else:
                        logger.critical("input signal has to be 1D or 2D")

                # plot integrated signal with subtracted noise:
                if len(axlist) > 3:
                    # plot signal subtracted noise:
                    if np.ndim(signal) == 2:
                        for s in range(len(signal)):
                            axlist[3].plot(time_axis, np.cumsum(signal_no_dyn_noise[s]))
                    elif np.ndim(signal) == 1:
                        axlist[3].plot(time_axis, np.cumsum(signal_no_dyn_noise))
                    else:
                        logger.critical("input signal has to be 1D or 2D")

                    axlist[3].grid(
                        color="r", linestyle="-", linewidth=0.25, which="major"
                    )
                    axlist[3].grid(
                        color="r", linestyle="--", linewidth=0.25, which="minor"
                    )
                    axlist[3].set_title("Integrated Signal (subtr. dyn noise)")
                    axlist[3].set_xlabel("Time [s]")

                    if np.ndim(signal) == 2:
                        if pyr_ind == 0:
                            auc_lac_dyn_noise = integrated_signal_dyn_noise[pyr_ind + 1]
                            auc_pyr_dyn_noise = integrated_signal_dyn_noise[pyr_ind]
                            auc_ratio_dyn_noise = auc_lac_dyn_noise / auc_pyr_dyn_noise
                        elif pyr_ind == 1:
                            auc_lac_dyn_noise = integrated_signal_dyn_noise[pyr_ind - 1]
                            auc_pyr_dyn_noise = integrated_signal_dyn_noise[pyr_ind]
                            auc_ratio_dyn_noise = auc_lac_dyn_noise / auc_pyr_dyn_noise
                        else:
                            Warning("Don't understand the indexing")
                            auc_lac_dyn_noise = integrated_signal_dyn_noise
                            auc_pyr_dyn_noise = integrated_signal_dyn_noise
                            auc_ratio_dyn_noise = 1
                            pass
                    elif np.ndim(signal) == 1:
                        auc_lac_dyn_noise = 0
                        auc_pyr_dyn_noise = integrated_signal_dyn_noise
                        auc_ratio_dyn_noise = 0

                    logger.critical("dyn. noise AUC pyr : " + str(auc_pyr_dyn_noise))
                    logger.critical("dyn. noise AUC lac : " + str(auc_lac_dyn_noise))
                    logger.critical(
                        "dyn. noise AUC lac/pyr : " + str(auc_ratio_dyn_noise)
                    )

                    # axlist[3].legend(
                    #     np.round(integrated_signal_dyn_noise, 3),
                    #     title="AUC ratio "
                    #     + str(
                    #         np.round(auc_ratio_dyn_noise, 3),
                    #     ),
                    # )

            # save the AUCs
            save_auc(
                self,
                auc_lac=auc_lac,
                auc_pyr=auc_pyr,
                auc_ratio=auc_ratio,
                mask_id=mask_id,
                mask_key=mask_key,
                signal_range=signal_range,
                noise_range=noise_range,
                auc_lac_dyn_noise=auc_lac_dyn_noise,
                auc_pyr_dyn_noise=auc_pyr_dyn_noise,
                auc_ratio_dyn_noise=auc_ratio_dyn_noise,
            )

            return auc_pyr, auc_lac, auc_ratio

        # -------------------------------define widgets -----------------------------
        if signal_range_input[0] is None:
            # we have no inputs therefore take the entire signal range
            slider_signal_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[0, np.max(time_axis)],
                max=np.max(time_axis),
                description="Signal range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            slider_noise_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[0, np.max(time_axis)],
                max=np.max(time_axis),
                description="Noise range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
        else:
            slider_signal_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[signal_range_input[0], signal_range_input[1]],
                max=np.max(time_axis),
                description="Signal range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            slider_noise_range_reps = widgets.IntRangeSlider(
                min=0.0,
                value=[noise_range_input[0], noise_range_input[1]],
                max=np.max(time_axis),
                description="Noise range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
        subtract_noise_checkbox = widgets.Checkbox(
            description="subtract noise", value=1
        )

        save_button = widgets.ToggleButton(description="Save time curves", value=0)
        # -----------------------------------------------------------------------
        out = widgets.interactive_output(
            plot_auc,
            {
                "signal_range": slider_signal_range_reps,
                "noise_range": slider_noise_range_reps,
                "subtract_noise": subtract_noise_checkbox,
                "save_csv": save_button,
            },
        )
        main_ui = widgets.VBox(
            [
                slider_signal_range_reps,
                slider_noise_range_reps,
                subtract_noise_checkbox,
                save_button,
            ]
        )

        if display_ui:
            display(main_ui, out)
        else:
            return main_ui

    def split_into_pyr_lac(
        self, input_data=None, pyr_ind=0, metab_axis=4, thirdMetab=False
    ):
        """
        splits the data into num_metab metabolites

        input_parameters:
        input_data: data that should be split
        pyr_ind: index of the pyruvate acquisition (either 0 or 1)
        metab_axis: axis on which the dataset should be split into pyruvate and
            lactate

        thirdMetab : in case data also contains data of 3rd metabolite (expected dimensions: pyr, 3rd metabolite, lac)
        returns num_metab datasets (pyr_data and lac_data)
        """
        # dont continue if there is no data:
        if input_data is None:
            return False

        # set pyruvate and lactate index:
        if thirdMetab:
            pyr_ind = 0
            thirdMetab_ind = 1
            lac_ind = 2
            dim_metab = 3
        elif pyr_ind == 0:
            lac_ind = 1
            dim_metab = 2
        elif pyr_ind == 1:
            lac_ind = 0
            dim_metab = 2
        else:
            Warning("Pyruvate index has to be 0 or 1!")
            return False

        # split the data into pyruvate and lactate data:
        # try:
        if metab_axis == 0:
            pyr_data = input_data[pyr_ind::dim_metab, :, :, :, :, :]
            lac_data = input_data[lac_ind::dim_metab, :, :, :, :, :]
            if thirdMetab:
                thirdMetab_data = input_data[thirdMetab_ind::dim_metab, :, :, :, :, :]
        if metab_axis == 1:
            pyr_data = input_data[:, pyr_ind::dim_metab, :, :, :, :]
            lac_data = input_data[:, lac_ind::dim_metab, :, :, :, :]
            if thirdMetab:
                thirdMetab_data = input_data[:, thirdMetab_ind::dim_metab, :, :, :, :]
        if metab_axis == 2:
            pyr_data = input_data[:, :, pyr_ind::dim_metab, :, :, :]
            lac_data = input_data[:, :, lac_ind::dim_metab, :, :, :]
            if thirdMetab:
                thirdMetab_data = input_data[:, :, thirdMetab_ind::dim_metab, :, :, :]
        if metab_axis == 3:
            pyr_data = input_data[:, :, :, pyr_ind::dim_metab, :, :]
            lac_data = input_data[:, :, :, lac_ind::dim_metab, :, :]
            if thirdMetab:
                thirdMetab_data = input_data[:, :, :, thirdMetab_ind::dim_metab, :, :]
        # default, consistent with used dimension convention:
        if metab_axis == 4:
            pyr_data = input_data[:, :, :, :, pyr_ind::dim_metab, :]
            lac_data = input_data[:, :, :, :, lac_ind::dim_metab, :]
            if thirdMetab:
                thirdMetab_data = input_data[:, :, :, :, thirdMetab_ind::dim_metab, :]
        if metab_axis == 5:
            pyr_data = input_data[:, :, :, :, :, pyr_ind::dim_metab]
            lac_data = input_data[:, :, :, :, :, lac_ind::dim_metab]
            if thirdMetab:
                thirdMetab_data = input_data[:, :, :, :, :, thirdMetab_ind::dim_metab]
        # except:
        #     pyr_data = None
        #     lac_data = None

        if thirdMetab:
            return pyr_data, lac_data, thirdMetab_data
        else:
            return pyr_data, lac_data

    def roi_signal(
        self,
        input_data=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        start_ind=None,
        end_ind=None,
        mean_per_slice=False,
    ):
        """
        Exctracts signal from ROIs from the input_data. ROIs are either defined
        in the passed mask_dict and can be adressed with the mask_key argument or can be passed directly
        as masks.
        The complex data is summed along the time dimension and then a maksk is applied. Then the magntiude
        of the masked complex data is taken and summed up.

        # Usage:
        dnp_signal_pyr_tumor_AUC =  dnp_bssfp.roi_signal(input_data=dnp_bssfp_itp_pyr,
                                                    mask_dict=mask_dict,
                                                    mask_key='tumor',
                                                    start_ind=0,
                                                    end_ind=dnp_bssfp_itp_pyr.shape[4],
                                                    mean_per_slice=False)

        """
        # no need to contiune if there is no data:
        if input_data is None:
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                masks = mask_dict[mask_key]

        # if mask is not defined:
        if masks is None:
            return False

        # empty ROI:
        if np.sum(masks) == 0:
            return np.nan

        # set range over which to average:
        if start_ind is None:
            start_ind = 0

        if end_ind is None:
            end_ind = input_data.shape[4]

        # Generate copy, otherwise mask_key in mask_dict will be overwritten!
        masks_copy = np.copy(masks)

        # Replace 0s with NaN:
        masks_copy[masks_copy == 0] = np.nan

        # average data along time dimension:
        # no need for nanmean because mean along time means either
        # signal or no signal (NaNs) arrays are averaged.
        data = np.abs(
            np.mean(input_data[:, :, :, :, start_ind:end_ind, :], axis=4, keepdims=True)
        )

        # to keep old method for now, does per slice averaging
        # before averaging per over mean of slices (wrong):
        if mean_per_slice is True:
            if np.nanmean(masks_copy) > 0:
                # Init empty masked data
                data_masked = np.empty(
                    (
                        input_data.shape[0],
                        input_data.shape[3],
                        input_data.shape[5],
                    )
                )

                # Fill the empty arrays
                # loop over echoes:
                for echo in range(0, input_data.shape[0], 1):
                    # loop over receive channels
                    for chan in range(0, input_data.shape[5], 1):
                        # loop over slices
                        for n_slice in range(0, input_data.shape[3], 1):
                            data_masked[echo, n_slice, chan] = np.nanmean(
                                data[echo, :, :, n_slice, 0, chan]
                                * masks_copy[:, :, n_slice]
                            )
                # sum over slices (Nan does not contribute)
                data_masked = np.nanmean(np.abs(data_masked))
            else:
                data_masked = np.nan
        else:
            if np.nanmean(masks_copy) > 0:
                # Init empty masked data
                data_masked = np.empty(
                    (
                        input_data.shape[0],
                        input_data.shape[5],
                    )
                )

                # Fill the empty arrays
                # loop over echoes:
                for echo in range(0, input_data.shape[0], 1):
                    # loop over receive channels
                    for chan in range(0, input_data.shape[5], 1):
                        data_masked[echo, chan] = np.nanmean(
                            data[echo, :, :, :, 0, chan] * masks_copy
                        )
                # sum over slices (Nan does not contribute)
                data_masked = np.nanmean(data_masked)
            else:
                data_masked = np.nan

        return data_masked

    def get_AUCR_and_std(
        self,
        input_data1=None,
        input_data2=None,
        input_data_orig=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        start_ind=None,
        end_ind=None,
        std=None,
    ):
        """
        Exctracts signal from ROIs from the input_data. ROIs are either defined
        in the passed mask_dict and can be adressed with the mask_key argument or can be passed directly
        as masks.
        The complex data is summed along the time dimension and then a maksk is applied. Then the magntiude
        of the masked complex data is taken and summed up.

        # Usage:
        dnp_signal_pyr_tumor_AUC =  dnp_bssfp.roi_signal(input_data=dnp_bssfp_itp_pyr,
                                                    mask_dict=mask_dict,
                                                    mask_key='tumor',
                                                    start_ind=0,
                                                    end_ind=dnp_bssfp_itp_pyr.shape[4],
                                                    mean_per_slice=False)

        """
        # no need to contiune if there is no data:
        if input_data1 is None:
            return False

        if input_data2 is None:
            return False

        if std is None:
            means, stds = self.get_mean_std_of_voxel(
                input_data=input_data_orig, db=True
            )
            std = np.mean(stds)

        # if no original datasetset given:
        if input_data_orig is None:
            image_shape_noninterp = input_data1.shape[1:4]
        else:
            image_shape_noninterp = input_data_orig.shape[1:4]

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                masks = mask_dict[mask_key]

        # if mask is not defined:
        if masks is None:
            return False

        # empty ROI:
        if np.sum(masks) == 0:
            return np.nan

        # set range over which to average:
        if start_ind is None:
            start_ind = 0

        if end_ind is None:
            end_ind = input_data1.shape[4]

        # Generate copy, otherwise mask_key in mask_dict will be overwritten!
        masks_copy = np.copy(masks)

        # Replace 0s with NaN:
        masks_copy[masks_copy == 0] = np.nan

        # average data along time dimension:
        # no need for nanmean because mean along time means either
        # signal or no signal (NaNs) arrays are averaged.
        data1 = np.abs(
            np.mean(
                input_data1[:, :, :, :, start_ind:end_ind, :], axis=4, keepdims=True
            )
        )
        data2 = np.abs(
            np.mean(
                input_data2[:, :, :, :, start_ind:end_ind, :], axis=4, keepdims=True
            )
        )

        AUCR = data2 / data1
        if np.nanmean(masks_copy) > 0:
            # Init empty masked data
            data_masked = np.empty(
                (
                    input_data1.shape[0],
                    input_data1.shape[1],
                    input_data1.shape[2],
                    input_data1.shape[3],
                    input_data1.shape[5],
                )
            )

            # Calculate the AUCR for each original voxel:
            # loop over echoes:
            for echo in range(0, input_data1.shape[0], 1):
                # loop over receive channels
                for chan in range(0, input_data1.shape[5], 1):
                    data_masked[echo, :, :, :, chan] = (
                        AUCR[echo, :, :, :, 0, chan] * masks_copy
                    )

            # calculate the AUCR mean:
            AUCR_mean = np.nanmean(data_masked)
            logger.critical(AUCR_mean)
            AUCR_all = data_masked.ravel()
            logger.critical(np.nanmean(AUCR_all))
            # remove NaNs
            mask = ~np.isnan(AUCR_all)
            AUCR_all = AUCR_all[mask]
            logger.critical(AUCR_all)
            # Calculate the variance of the AUCR:
            var_AUCR = (std / AUCR_all) ** 2

            # get number of subvoxels (mask resolution) per
            # bSSFP voxel. Sorted.
            nvoxels = self.get_number_mask_voxels_in_bssfp_voxels(
                image_shape=image_shape_noninterp,
                masks=None,
                mask_dict=mask_dict,
                mask_key=mask_key,
                normalize=False,
                make_long=True,
            )

            total_var_AUCR = np.sqrt(np.sum(var_AUCR) / np.sum(nvoxels))
            AUCR = np.nanmean(data_masked)
            return AUCR, total_var_AUCR

        else:
            data_masked = np.nan

        return False

    def roi_signal_curve(
        self,
        input_data=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        mean_per_slice=False,
    ):
        """
        (Note: recommended to use apply_mask in utils_general)
        Exctracts time curves from ROIs from the input_data. ROIs are either defined
        in the passed mask_dict and can be adressed with the mask_key argument orcan be passed directly
        as masks. The masks are applied onto the input data and the resulting masked data is summed up along
        all dimension besides the repetition/time dimension, returning a 1xT array
        """

        # no need to continue if there is no data:
        if input_data is None:
            logger.critical("no input data!")
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                try:
                    masks = mask_dict[mask_key]
                except:
                    logger.critical("can't extract mask!")
                    return False

        # masks = np.flip(masks, axis=2)

        # if mask is not defined:
        if masks is None:
            logger.critical("no mask defined!")
            return False

        # if ROI is empty:
        if np.sum(masks) == 0:
            logger.critical(f"ROI {mask_key} is empty!")
            return np.zeros((input_data.shape[4]))

        # Generate copy, otherwise mask_key in mask_dict will be overwritten!
        masks_copy = np.copy(masks)

        # replace 0 with NaN
        masks_copy[masks_copy == 0] = np.nan

        try:
            # to keep old method for now, does per slice averaging
            # before averaging per over mean of slices (wrong):
            if mean_per_slice is True:
                if input_data.dtype == "complex":
                    # covert mask to complex mask
                    masks_copy = masks_copy.astype(complex)
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],
                            input_data.shape[3],
                            input_data.shape[4],
                            input_data.shape[5],
                        ),
                        dtype=input_data.dtype,
                    )
                else:
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],
                            input_data.shape[3],
                            input_data.shape[4],
                            input_data.shape[5],
                        )
                    )

                # Fill the empty arrays
                # loop over repetitions:
                for rep in range(0, input_data.shape[4], 1):
                    # loop over echoes:
                    for echo in range(0, input_data.shape[0], 1):
                        # loop over receive channels
                        for chan in range(0, input_data.shape[5], 1):
                            # loop over slices
                            for n_slice in range(0, input_data.shape[3], 1):
                                data_masked[echo, n_slice, rep, chan] = np.nanmean(
                                    input_data[echo, :, :, n_slice, rep, chan]
                                    * masks_copy[:, :, n_slice]
                                )
                # sum over slices (Nan does not contribute)
                data_masked = np.squeeze(
                    np.nanmean(
                        np.nanmean(
                            np.nanmean(data_masked, axis=0, keepdims=True),
                            axis=1,
                            keepdims=True,
                        ),
                        axis=3,
                    )
                )
            # use new way of masking:
            else:
                if input_data.dtype == "complex":
                    # covert mask to complex mask
                    masks_copy = masks_copy.astype(complex)
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],  # N echoes
                            input_data.shape[4],  # N repetitions
                            input_data.shape[5],  # N receive channels
                        ),
                        dtype=input_data.dtype,  # complex dataset
                    )
                else:
                    # Init empty masked data
                    data_masked = np.empty(
                        (
                            input_data.shape[0],  # N echoes
                            input_data.shape[4],  # N repetitions
                            input_data.shape[5],  # N receive channels
                        )
                    )
                # loop over slices:
                for rep in range(0, input_data.shape[4], 1):
                    # loop over echoes:
                    for echo in range(0, input_data.shape[0], 1):
                        # loop over receive channels
                        for chan in range(0, input_data.shape[5], 1):
                            # apply mask:
                            data_masked[echo, rep, chan] = np.nanmean(
                                input_data[echo, :, :, :, rep, chan] * masks_copy
                            )
                # take average over echoes and receive channels:
                data_masked = np.squeeze(
                    np.nanmean(
                        np.nanmean(data_masked, axis=0, keepdims=True),
                        axis=2,
                    )
                )
        # return zeros (of length number of repetitions):
        except Exception as e:
            logger.critical("masking did not work --> returning 0s")
            data_masked = np.zeros((input_data.shape[4]))

        return data_masked

    def get_number_mask_voxels_in_bssfp_voxels(
        self,
        image_shape=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        normalize=False,
        make_long=False,
    ):
        """
        Input:

        image_shape: [z,x,y] of the noninterpolated dataset

        This function calculates the number of mask voxels per bSSFP voxel.
        This can be used to calculate the std. of a ROI. Can be inaccurate in case of
        matrix size differences like eg. masks 21 slices and bSSFP 12.
        (close to 2 but still rounded to 1)
        """
        # no need to contiune if there is no image_shape:
        if image_shape is None:
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                try:
                    masks = mask_dict[mask_key]
                except:
                    return False

        # masks = np.flip(masks, axis=2)

        # if mask is not defined:
        if masks is None:
            return False

        # empty ROI:
        if np.sum(masks) == 0:
            return np.nan

        # init empty:
        coverage = np.nan

        # replace 0 with NaN
        # masks[masks == 0] = np.nan
        mask_depth, mask_height, mask_width = masks.shape
        image_depth, image_height, image_width = image_shape

        block_depth = mask_depth / image_depth
        block_height = mask_height / image_height
        block_width = mask_width / image_width

        coverage = np.zeros(image_shape)
        for i in range(image_depth):
            for j in range(image_height):
                for k in range(image_width):
                    # Compute the block in the original mask corresponding to this voxel
                    start_i = round(i * block_depth)
                    end_i = round((i + 1) * block_depth)
                    start_j = round(j * block_height)
                    end_j = round((j + 1) * block_height)
                    start_k = round(k * block_width)
                    end_k = round((k + 1) * block_width)

                    # Extract this block from the mask
                    mask_block = masks[start_i:end_i, start_j:end_j, start_k:end_k]

                    # Compute the coverage for this voxel
                    if normalize is True:
                        coverage[i, j, k] = np.mean(mask_block)
                    else:
                        coverage[i, j, k] = np.sum(mask_block)

        # calculate ratio of mask to bssfp data resolution:
        # z_ratio = int(masks.shape[0] / input_data.shape[1])
        # x_ratio = int(masks.shape[1] / input_data.shape[2])
        # y_ratio = int(masks.shape[2] / input_data.shape[3])

        # # init empty array:
        # mask_voxels_per_bssfp_voxel = np.zeros(input_data.shape[1:4])
        # # loop over bssfp voxels:
        # for y in range(input_data.shape[3]):
        #     for x in range(input_data.shape[2]):
        #         for z in range(input_data.shape[1]):
        #             mask_voxels_per_bssfp_voxel[z, x, y] = np.sum(
        #                 masks[
        #                     (z - 1) * z_ratio : z * z_ratio,
        #                     (x - 1) * x_ratio : x * x_ratio,
        #                     (y - 1) * y_ratio : y * y_ratio,
        #                 ]
        #             )

        # if signal should be normalized to a range from min/Nvoxels to 1.
        if make_long is True:
            # make 1D array:
            mask_voxels_per_bssfp_voxel_allvals = np.ravel(coverage)

            # remove 0s:
            mask_voxels_per_bssfp_voxel_allvals = mask_voxels_per_bssfp_voxel_allvals[
                mask_voxels_per_bssfp_voxel_allvals != 0
            ]

            # normalize to the maximum amount of mask voxels that a bSSFP voxel can contain:
            nvoxels = np.sort(mask_voxels_per_bssfp_voxel_allvals)
            return nvoxels
        else:
            return coverage

    def get_mean_std_of_voxel(
        self,
        input_data=None,
        voxel_pos=None,
        echo=0,
        chan=0,
        start_ind=None,
        end_ind=None,
        db=False,
    ):
        """
        This function calculates the mean and standard deviation of
        the bssfp data (input_data) in the voxel at position voxel_pos

        Input:

        input_data: bSSFP data, expected order: echoes-read-phase-slice-repetitions-channels

        voxel_pos: voxel position in index read-phase-slice.
            If None is passed, 4 corner voxel are taken.

        echo: echo number, default = 0

        chan: acquisition channel number, default = 0

        Returns:

        mean and standarad deviation
        """

        # dont continue if no input_data is passed:
        if input_data is None:
            return False

        # default voxel_pos:
        if voxel_pos is None:
            # take 4 corner voxels (on opposite of read start (can be affected from
            # small read matrixces (~16 points)))):
            voxel_pos = [[-1, 0, 0], [-1, 0, -1], [-1, -1, 0], [-1, -1, -1]]

        # use full range if no index was passed:
        if start_ind is None:
            start_ind = 0

        # use full range if no index was passed:
        if end_ind is None:
            end_ind = input_data.shape[4]

        # if 1 voxel position was passed:
        if np.ndim(voxel_pos) == 1:
            stds = np.std(
                np.real(
                    input_data[
                        echo,
                        voxel_pos[0],
                        voxel_pos[1],
                        voxel_pos[2],
                        start_ind:end_ind,
                        chan,
                    ]
                )
            )
            # np.std cant work with complex data:
            if input_data.dtype == complex:
                stds = stds + 1j * np.std(
                    np.imag(
                        input_data[
                            echo,
                            voxel_pos[0],
                            voxel_pos[1],
                            voxel_pos[2],
                            start_ind:end_ind,
                            chan,
                        ]
                    )
                )
            # get mean of voxel signa:
            means = np.mean(
                input_data[
                    echo,
                    voxel_pos[0],
                    voxel_pos[1],
                    voxel_pos[2],
                    start_ind:end_ind,
                    chan,
                ]
            )

        # if more than 1 voxel position was passed:
        elif np.ndim(voxel_pos) == 2:
            # init empty array
            stds = np.zeros((np.shape(voxel_pos)[0], 1), dtype=input_data.dtype)
            means = np.zeros((np.shape(voxel_pos)[0], 1), dtype=input_data.dtype)
            for p in range(np.shape(voxel_pos)[0]):
                stds[p] = np.std(
                    np.real(
                        input_data[
                            echo,
                            voxel_pos[p][0],
                            voxel_pos[p][1],
                            voxel_pos[p][2],
                            start_ind:end_ind,
                            chan,
                        ]
                    )
                )
                # np.std cant work with complex data:
                if input_data.dtype == complex:
                    stds[p] = stds[p] + 1j * np.std(
                        np.imag(
                            input_data[
                                echo,
                                voxel_pos[p][0],
                                voxel_pos[p][1],
                                voxel_pos[p][2],
                                start_ind:end_ind,
                                chan,
                            ]
                        )
                    )
                means[p] = np.mean(
                    input_data[
                        echo,
                        voxel_pos[p][0],
                        voxel_pos[p][1],
                        voxel_pos[p][2],
                        start_ind:end_ind,
                        chan,
                    ]
                )

        else:
            Warning("voxel_pos has to be a 1D or 2D array:")
            return False

        if db is True:
            logger.critical("calculated mean from edge voxels is: " + str(means))
            logger.critical("calculated std from edge voxels is: " + str(stds))

        return means, stds

    def get_std_of_ROI(
        self,
        input_data=None,
        masks=None,
        mask_dict=None,
        mask_key=None,
        std=None,
        pyr_lac=True,
        db=False,
    ):
        """ """

        # no need to contiune if there is no data:
        if input_data is None:
            return False

        # load mask from dict if possible
        if mask_dict is not None:
            if mask_key is not None:
                try:
                    masks = mask_dict[mask_key]
                except:
                    return False

        # empty ROI:
        if np.sum(masks) == 0:
            Warning("Empty mask!")
            return np.nan

        # get std from data if no std is passed:
        if std is None:
            means, stds = self.get_mean_std_of_voxel(input_data=input_data, db=db)
            std = np.mean(stds)

        # get normalized_mask_voxels_in_bssfp_voxels. This sorted array
        # goes from 0 to 1. 0 means that no mask voxel was in the bSSFP voxel,
        # 1 means that the bSSFP voxel was completly filled by the mask voxel.
        # This acts a weight:
        normalized_mask_voxels_in_bssfp_voxels = (
            self.get_number_mask_voxels_in_bssfp_voxels(
                image_shape=input_data.shape[1:4],
                masks=None,
                mask_dict=mask_dict,
                mask_key=mask_key,
                normalize=True,
                make_long=True,
            )
        )

        normed_weights = normalized_mask_voxels_in_bssfp_voxels
        # Error propagation:
        # std_ROI = (
        #     1
        #     / (np.sum(normed_weights) / len(normed_weights))
        #     * np.sqrt(
        #         np.sum(
        #             (
        #                 (
        #                     normed_weights
        #                     / len(normed_weights)
        #                     * std
        #                     * 1
        #                     / np.sqrt(input_data.shape[4])
        #                 )
        #                 ** 2
        #             )
        #         )
        #     )
        # )

        # take into account that the AUC values was averaged:
        # divide by 2 because only have the repetitions is averaged (each)
        # if pyr_lac is True:
        #     std = std / np.sqrt(input_data.shape[4] / 2)
        # else:
        #     std = std / np.sqrt(input_data.shape[4])

        std_ROI = std * np.sqrt(np.sum(normed_weights**2) / len(normed_weights) ** 2)

        return std_ROI

    def get_AUCR_std(
        self,
        std=None,
        std_lac=None,
        AUC_pyr=None,
        AUC_lac=None,
    ):
        # if no AUC for pyruvate was passed, return NaN
        if AUC_pyr is None:
            return np.nan

        # if no AUC for lactate was passed, return NaN
        if AUC_lac is None:
            return np.nan

        # if no std was passed:
        if std is None:
            return np.nan
        # set standard dev. of pyruvate to std:
        else:
            std_pyr = std

        # use std for lactate if no specifig lactate std was passed:
        if std_lac is None:
            std_lac = std

        # calculate AUC ratio:
        AUCR = AUC_lac / AUC_pyr
        # Error propagation division:
        std_AUCR = AUCR * np.sqrt((std_lac / AUC_lac) ** 2 + (std_pyr / AUC_pyr) ** 2)

        return std_AUCR

    def reorient_reco(
        self,
        bssfp_seq2d=None,
        bssfp_custom=None,
        anatomical_seq2d_ax=None,
        anatomical_seq2d_sag=None,
        anatomical_seq2d_cor=None,
        fieldmap_seq2d_cor=None,
        anatomical_obj_ax=None,
        anatomical_obj_sag=None,
        anatomical_obj_cor=None,
        fieldmap_obj_cor=None,
        db=False,
    ):  # taken from MATLAB pvtools - readBrukerRaw.m
        """
        The bSSFP data and the anatomical images are rotated to match each other. So
        far only implemented for ...

        Parameters:
        ----------
        bssfp_seq2d ND-array
            PararVision reconstructed bSSFP dataset

        bssfp_custom ND-array
            Custom reconstructed bSSFP dataset

        anatomical_seq2d_ax ND-array
            Dataset that has the axial slices in it

        anatomical_seq2d_sag ND-array
            Dataset that has the sagittal slices in it

        anatomical_seq2d_cor ND-array
            Dataset that has the coronal slices in it

        Returns
        ----------
        bssfp_custom_data,
        bssfp_pv_data,
        anatomical_ax,
        anatomical_seq2d,
        anatomical_cor,

        """
        # read_orient = img_instance.method["PVM_SPackArrReadOrient"]
        # patient_pos = img_instance.acqp["ACQ_patient_pos"]

        # init:
        bssfp_pv_data = None

        # if seq2d was not specfied, use seq2d from bssfp object:
        if bssfp_seq2d is None:
            # use seq2d:
            bssfp_pv_data = self.seq2d
            # if object was input, use object info:
            # how many times the reorient function has been
            # performed on this object:
            num_permute_counts = get_counter(
                data_obj=self,
                counter_name="reorient_counter",
            )

        else:
            # if seq2d was specified, perform reorientation
            bssfp_pv_data = bssfp_seq2d
            num_permute_counts = 0

        # only perform reorientetation if the number of reorientations
        # num_permute_counts is 0:
        if num_permute_counts > 0:
            pass
        else:
            ## experimental step:
            bssfp_pv_data = np.transpose(bssfp_pv_data, (0, 2, 1, 3, 4, 5))
            # bssfp_pv_data = np.flip(bssfp_pv_data, 3)

            # if seq2d was not specfied:
            if bssfp_seq2d is None:
                # try to add one to the reorientation counter
                # (only works if object was passed):
                # add 1 to shift counter:
                add_counter(
                    data_obj=self,
                    counter_name="reorient_counter",
                    n_counts=1,
                )
                # overwrite the old seq2d:
                self.seq2d = bssfp_pv_data

        # bsffp data reconstructed from .fid file:
        bssfp_custom_data = None
        if bssfp_custom is None:
            pass
        else:
            logger.debug("Shape of bSSFP reco from .fid file:")
            bssfp_custom_data = bssfp_custom
            bssfp_custom_data = np.transpose(bssfp_custom_data, (0, 2, 1, 3, 4, 5))
            bssfp_custom_data = np.transpose(bssfp_custom_data, (2, 1, 0, 3, 4, 5))
            bssfp_custom_data = np.flip(bssfp_custom_data, axis=0)
            bssfp_custom_data = np.transpose(bssfp_custom_data, (2, 1, 0, 3, 4, 5))
            bssfp_custom_data = np.rot90(bssfp_custom_data, k=2, axes=(1, 0))
            bssfp_custom_data = np.flip(bssfp_custom_data, axis=3)
            ## experimental step:
            bssfp_custom_data = np.transpose(bssfp_custom_data, (0, 2, 1, 3, 4, 5))
            bssfp_custom_data = np.flip(bssfp_custom_data, axis=3)

        # anatomical axial image (2dseq file)
        anatomical_ax = None
        if anatomical_seq2d_ax is None:
            pass
        else:
            anatomical_ax = anatomical_seq2d_ax

        # if object was input, use object info:
        if anatomical_obj_ax is not None:
            # how many times the reorient function has been
            # performed on this object:
            num_permute_counts = get_counter(
                data_obj=anatomical_obj_ax,
                counter_name="reorient_counter",
            )
            # use anatomical image from object:
            anatomical_ax = anatomical_obj_ax.seq2d
        else:
            # if no object info available, set number of
            # permutation operations so far performed to 0:
            num_permute_counts = 0
            pass

        if anatomical_ax is None:
            pass
        else:
            # squeeze:
            if np.ndim(anatomical_ax) == 4:
                anatomical_ax = np.squeeze(anatomical_ax)
            # dont continue if permutation has already been applied to
            # axial anatomical image:
            if num_permute_counts > 0:
                logger.critical("axial anatomical image was already reshaped, skipping")
            else:
                anatomical_ax = np.transpose(anatomical_ax, (2, 0, 1))
                anatomical_ax = np.flip(anatomical_ax, axis=0)
                anatomical_ax = np.transpose(anatomical_ax, (1, 2, 0))
                anatomical_ax = np.flip(anatomical_ax, axis=1)

                # experimtental step:
                anatomical_ax = np.transpose(anatomical_ax, (2, 1, 0))
                # anatomical_ax = np.flip(anatomical_ax, axis=1)
                # logger.debug(anatomical_ax.shape)

                # try to add one to the reorientation counter (only works if object was passed)
                if anatomical_obj_ax is not None:
                    # add 1 to shift counter:
                    add_counter(
                        data_obj=anatomical_obj_ax,
                        counter_name="reorient_counter",
                        n_counts=1,
                    )
                    # overwrite old seq2d:
                    anatomical_obj_ax.seq2d = anatomical_ax

        # anatomical sagittal image (2dseq file)
        anatomical_sag = None
        if anatomical_seq2d_sag:
            # nothing implemented yet
            anatomical_sag = anatomical_seq2d_sag
            pass

        # if object was input, use object info:
        if anatomical_obj_sag is not None:
            # how many times the reorient function has been
            # performed on this object:
            num_permute_counts = get_counter(
                data_obj=anatomical_obj_sag,
                counter_name="reorient_counter",
            )
            # use anatomical image from object:
            anatomical_sag = anatomical_obj_sag.seq2d
        else:
            # if no object info available, set number of
            # permutation operations so far performed to 0:
            num_permute_counts = 0
            pass

        if anatomical_sag is None:
            pass
        else:
            # squeeze:
            if np.ndim(anatomical_sag) == 4:
                anatomical_sag = np.squeeze(anatomical_sag)
            # dont continue if permutation has already been applied to
            # axial anatomical image:
            if num_permute_counts > 0:
                logger.critical("axial anatomical image was already reshaped, skipping")
            else:
                anatomical_sag = np.transpose(anatomical_sag, (2, 0, 1))
                anatomical_sag = np.flip(anatomical_sag, axis=0)
                anatomical_sag = np.transpose(anatomical_sag, (1, 2, 0))
                anatomical_sag = np.flip(anatomical_sag, axis=1)

                # experimtental step:
                anatomical_sag = np.transpose(anatomical_ax, (2, 1, 0))
                # anatomical_ax = np.flip(anatomical_ax, axis=1)
                # logger.debug(anatomical_ax.shape)

                # try to add one to the reorientation counter (only works if object was passed)
                if anatomical_obj_sag is not None:
                    # add 1 to shift counter:
                    add_counter(
                        data_obj=anatomical_obj_sag,
                        counter_name="reorient_counter",
                        n_counts=1,
                    )
                    # overwrite old seq2d:
                    anatomical_obj_sag.seq2d = anatomical_sag

        # anatomical coronal image (2dseq file)
        anatomical_cor = None
        if anatomical_seq2d_cor is None:
            pass
        else:
            anatomical_cor = anatomical_seq2d_cor

        # if object was input, use object info:
        if anatomical_obj_cor is not None:
            # how many times the reorient function has been
            # performed on this object:
            num_permute_counts = get_counter(
                data_obj=anatomical_obj_cor,
                counter_name="reorient_counter",
            )
            # use anatomical image from object:
            anatomical_cor = anatomical_obj_cor.seq2d
        else:
            # if no object info available, set number of
            # permutation operations so far performed to 0:
            num_permute_counts = 0
            pass

        if anatomical_cor is None:
            # nothing to do:
            pass
        else:
            # dont continue if permutation has already been applied to
            # coronal anatomical image:
            if num_permute_counts > 0:
                logger.critical(
                    "coronal anatomical image was already reshaped, skipping"
                )
            else:
                anatomical_cor = np.transpose(anatomical_cor, (1, 0, 2))
                anatomical_cor = np.flip(anatomical_cor, axis=2)

                # try to add one to the reorientation counter (only works if object was passed)
                if anatomical_obj_cor is not None:
                    # add 1 to shift counter:
                    add_counter(
                        data_obj=anatomical_obj_cor,
                        counter_name="reorient_counter",
                        n_counts=1,
                    )
                    # overwrite the seq2d in the object:
                    anatomical_obj_cor.seq2d = anatomical_cor

        # fieldmap coronal image (2dseq file)
        fieldmap_cor = None
        if fieldmap_seq2d_cor is None:
            pass
        else:
            fieldmap_cor = fieldmap_seq2d_cor

        # if object was input, use object info:
        if fieldmap_obj_cor is not None:
            # how many times the reorient function has been
            # performed on this object:
            num_permute_counts = get_counter(
                data_obj=fieldmap_obj_cor,
                counter_name="reorient_counter",
            )
            # use fieldmap image from object:
            fieldmap_cor = fieldmap_obj_cor.seq2d
        else:
            # if no object info available, set number of
            # permutation operations so far performed to 0:
            num_permute_counts = 0
            pass

        if fieldmap_cor is None:
            # nothing to do:
            pass
        else:
            # dont continue if permutation has already been applied to
            # coronal fieldmap image:
            if num_permute_counts > 0:
                logger.critical("coronal fieldmap image was already reshaped, skipping")
            else:
                fieldmap_cor = np.squeeze(fieldmap_cor)
                fieldmap_cor = np.transpose(fieldmap_cor, (1, 0, 2))
                fieldmap_cor = np.flip(fieldmap_cor, axis=1)
                fieldmap_cor = np.flip(fieldmap_cor, axis=2)

                # try to add one to the reorientation counter (only works if object was passed)
                if fieldmap_obj_cor is not None:
                    # add 1 to shift counter:
                    add_counter(
                        data_obj=fieldmap_obj_cor,
                        counter_name="reorient_counter",
                        n_counts=1,
                    )
                    # overwrite the seq2d in the object:
                    fieldmap_obj_cor.seq2d = fieldmap_cor

        # plot overlay images:
        if 1 == 0:
            import matplotlib.pyplot as plt

            y = np.arange(-anatomical_ax.shape[1], anatomical_ax.shape[1] / 2, 1)
            x = np.arange(-anatomical_ax.shape[2], anatomical_ax.shape[2] / 2, 1)
            extent_ax = np.min(x), np.max(x), np.min(y), np.max(y)

            plt.figure()
            for ax in range(anatomical_ax.shape[0]):
                plt.subplot(7, 5, ax + 1).imshow(
                    anatomical_ax[ax, :, :], cmap="bone", extent=extent_ax
                )
                plt.subplot(7, 5, ax + 1).imshow(
                    np.abs(bssfp_pv_data[0, ax, :, :, 1, 0]),
                    alpha=0.6,
                    extent=extent_ax,
                    interpolation="sinc",
                )

                plt.subplot(7, 5, ax + 1).set_xticklabels([])
                plt.subplot(7, 5, ax + 1).set_yticklabels([])

            y = np.arange(-anatomical_cor.shape[0], anatomical_cor.shape[0] / 2, 1)
            x = np.arange(-anatomical_cor.shape[1], anatomical_cor.shape[1] / 2, 1)

            extent_cor = np.min(x), np.max(x), np.min(y), np.max(y)
            plt.figure()
            for cor in range(anatomical_cor.shape[2]):
                plt.subplot(7, 5, cor + 1).imshow(
                    anatomical_cor[:, :, cor], cmap="bone", extent=extent_cor
                )
                plt.subplot(7, 5, cor + 1).imshow(
                    np.abs(bssfp_pv_data[0, :, :, cor, 1, 0]),
                    alpha=0.6,
                    extent=extent_cor,
                    interpolation="sinc",
                )

                plt.subplot(7, 5, cor + 1).set_xticklabels([])
                plt.subplot(7, 5, cor + 1).set_yticklabels([])

        if fieldmap_obj_cor is None:
            return (
                bssfp_custom_data,
                bssfp_pv_data,
                anatomical_ax,
                anatomical_sag,
                anatomical_cor,
            )
        else:
            return (
                bssfp_custom_data,
                bssfp_pv_data,
                anatomical_ax,
                anatomical_sag,
                anatomical_cor,
                fieldmap_cor,
            )

    def plot_fieldmap_overlay(
        self,
        fieldmap=None,
        fig=None,
        axlist=None,
        anatomical=None,
        axial_coronal="axial",
        output_nuc="1h",
    ):
        """
        Plot overlay of fieldmap onto anatomical image.
        Example:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(12,5))
            bssfp.fieldmap_overlay(
            fieldmap=fieldmap,
            fig=fig,
            axlist=[ax1, ax2, ax3],
            anatomical=coronal,
            axial_coronal="coronal",

        Args:
            fieldmap (fieldmap object): the fieldmap object. Defaults to None.
            fig (matplotlib figure, optional): If colorbar ([Hz]) desired (recommended). Defaults to None.
            axlist (list of matplotlib axes): Has to be passed (see example). Defaults to None.
            anatomical (anatomical object): Either coronal or axial (specify which one in axial_coronal parameter). Defaults to None.
            axial_coronal (str, optional): wether anatomical object is "coronal" or "axial"." Defaults to "axial".
            output_nuc (str, optional): If plotted fieldmap should be in '1h' or '13c'. Defaults to "1h".
        """
        from hypermri.utils.utils_general import format_y_tick_labels, get_gmr

        # define gyromagnetic ratios:
        gmr_1h = get_gmr(nucleus="1h")  # MHz/T
        gmr_13c = get_gmr(nucleus="13c")  # MHz/T

        if fieldmap is not None:
            fm = shift_anat(
                csi_obj=self,
                anat_obj=fieldmap,
                use_scipy_shift=True,
            )
        else:
            return None

        if anatomical is not None:
            anat = shift_anat(
                csi_obj=self,
                anat_obj=anatomical,
                use_scipy_shift=True,
            )
        else:
            return None

        if output_nuc == "1h":
            pass
        elif output_nuc == "13c":
            fm = fm * gmr_13c / gmr_1h
        else:
            raise Exception("wrong nucleus, has to be 1h or 13c")
        if anat is None:
            return False

        from hypermri.utils.utils_general import define_grid as ut_define_grid
        import ipywidgets as widgets

        # define grids:
        self.get_extent_bssfp()
        ax_extent = self.ax_ext
        cor_extent = self.cor_ext

        # plot axial slice + overlay:
        if axial_coronal == "axial":
            logger.critical(f"anat.shape {anat.shape}")
            logger.critical(f"fm.shape {fm.shape}")

            slice_slider = widgets.IntSlider(
                value=0, min=0, max=anat.shape[1] - 1, description="Slice: "
            )
            frequency_range_slider = widgets.IntRangeSlider(
                value=[np.min(fm), np.max(fm)],
                min=1.5 * np.min(fm),
                max=1.5 * np.max(fm),
                description="frequency range: ",
            )

            # rearrange anatomical axial image:
            anat_im = anat.copy()
            anat_im = np.flip(anat_im, axis=1)

            # rearrange fieldmap image:
            # fm = np.flip(fm, axis=0)
            # fm = np.array([np.rot90(slice) for slice in fm])
            # fm = np.flip(fm, axis=2)

            # define grid of fieldmap:
            fieldmap_grid = ut_define_grid(data_obj=fieldmap)

            # function to find closest slice:
            def get_closest_slice(slice=0):
                (
                    ax_ind,
                    sag_ind,
                    cor_ind,
                ) = self.find_closest_slice2(
                    grid2=fieldmap_grid, pos1=[slice, 0, 0], grid1_obj=anatomical
                )
                return (
                    ax_ind,
                    sag_ind,
                    cor_ind,
                )

            def plot_img_fieldmap(slice, crange):
                print(slice)
                ax_ind, _, _ = get_closest_slice(slice)
                print(ax_ind)
                # anatomical
                axlist[0].imshow(
                    np.squeeze(anat_im[0, slice, :, :, 0, 0]),
                    extent=ax_extent,
                    cmap="bone",
                )
                axlist[0].set_title("slice # " + str(slice))

                # B0 Map
                im_ax = axlist[1].imshow(
                    np.rot90(np.squeeze(fm[0, ax_ind, :, :, 0, 0])),
                    extent=ax_extent,
                    vmin=crange[0],
                    vmax=crange[1],
                )
                axlist[1].set_title("slice # " + str(ax_ind))

                # anatpmical B0 Map
                axlist[2].imshow(
                    np.squeeze(anat_im[0, slice, :, :, 0, 0]),
                    extent=ax_extent,
                    cmap="bone",
                )
                im_ax_2 = axlist[2].imshow(
                    np.rot90(np.squeeze(fm[0, ax_ind, :, :, 0, 0])),
                    extent=ax_extent,
                    alpha=0.5,
                    cmap="bwr",
                    vmin=crange[0],
                    vmax=crange[1],
                )

                # Apply custom formatting to each subplot
                for ax in axlist[1:-1]:
                    format_y_tick_labels(ax)

                # generate colorbar if fig object was passed:
                if fig is None:
                    # colorbar needs figure:
                    pass
                else:
                    # nasty but otherwise plots new colorbar each time a paramter is changed :/
                    global cbar
                    global cbar_2
                    try:
                        cbar.remove()
                    except:
                        pass
                    try:
                        cbar_2.remove()
                    except:
                        pass
                    cbar = fig.colorbar(
                        im_ax,
                        ax=axlist[1],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )
                    cbar.remove()
                    cbar = fig.colorbar(
                        im_ax,
                        ax=axlist[1],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )

                    # for ax2
                    cbar_2 = fig.colorbar(
                        im_ax_2,
                        ax=axlist[2],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )
                    cbar_2.remove()
                    cbar_2 = fig.colorbar(
                        im_ax_2,
                        ax=axlist[2],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )
                    # set colorbar range, use 3 ticks (min, mean, max)
                    # md = (metab_clim[1] + metab_clim[0]) / 2
                    # cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
                    # cbar.set_ticklabels(
                    #     np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
                    # )

            out = widgets.interactive_output(
                plot_img_fieldmap,
                {"slice": slice_slider, "crange": frequency_range_slider},
            )

            main_ui = widgets.VBox(
                [
                    slice_slider,
                    frequency_range_slider,
                ]
            )

            display(main_ui, out)

        # plot coronal slice + overlay:
        elif axial_coronal == "coronal":
            # fm = np.flip(fm, axis=1)
            # slice_range = max(anat.shape[3] - 1, fieldmap.seq2d.shape[3] - 1)
            max_slice_slider = anat.shape[3] - 1
            slice_slider = widgets.IntSlider(
                value=0, min=0, max=max_slice_slider, description="Slice: "
            )

            frequency_range_slider = widgets.IntRangeSlider(
                value=[np.min(fm), np.max(fm)],
                min=1.5 * np.min(fm),
                max=1.5 * np.max(fm),
                description="frequency range: ",
            )

            # define grid of fieldmap:
            fieldmap_grid = ut_define_grid(data_obj=fieldmap)

            def get_closest_slice(slice=0):
                (
                    ax_ind,
                    sag_ind,
                    cor_ind,
                ) = self.find_closest_slice2(
                    grid2=fieldmap_grid, pos1=[0, 0, slice], grid1_obj=anatomical
                )
                return (
                    ax_ind,
                    sag_ind,
                    cor_ind,
                )

            def plot_img_fieldmap(slice, crange):
                _, _, cor_ind = get_closest_slice(slice)
                axlist[0].imshow(
                    np.squeeze(anat[0, :, :, slice, 0, 0]),
                    extent=cor_extent,
                    cmap="bone",
                )
                im_ax = axlist[1].imshow(
                    np.squeeze(fm[0, :, :, cor_ind, 0, 0]),
                    extent=cor_extent,
                    vmin=crange[0],
                    vmax=crange[1],
                )
                axlist[1].set_title("slice # " + str(slice))
                axlist[2].imshow(
                    np.squeeze(anat[0, :, :, slice, 0, 0]),
                    extent=cor_extent,
                    cmap="bone",
                )
                axlist[2].set_title("slice # " + str(cor_ind))
                im_ax_2 = axlist[2].imshow(
                    np.squeeze(fm[0, :, :, cor_ind, 0, 0]),
                    extent=cor_extent,
                    alpha=0.5,
                    cmap="bwr",
                    vmin=crange[0],
                    vmax=crange[1],
                )

                # Apply custom formatting to each subplot
                for ax in axlist[1:-1]:
                    format_y_tick_labels(ax)

                # generate colorbar if fig object was passed:
                if fig is None:
                    # colorbar needs figure:
                    pass
                else:
                    # nasty but otherwise plots new colorbar each time a paramter is changed :/
                    global cbar
                    global cbar_2
                    try:
                        cbar.remove()
                    except:
                        pass
                    try:
                        cbar_2.remove()
                    except:
                        pass
                    # for ax1
                    cbar = fig.colorbar(
                        im_ax,
                        ax=axlist[1],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )
                    cbar.remove()
                    cbar = fig.colorbar(
                        im_ax,
                        ax=axlist[1],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )

                    # for ax2
                    cbar_2 = fig.colorbar(
                        im_ax_2,
                        ax=axlist[2],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )
                    cbar_2.remove()
                    cbar_2 = fig.colorbar(
                        im_ax_2,
                        ax=axlist[2],
                        fraction=0.033,
                        pad=0.04,
                        label="f[Hz]",
                    )
                    # set colorbar range, use 3 ticks (min, mean, max)
                    # md = (metab_clim[1] + metab_clim[0]) / 2
                    # cbar.set_ticks([metab_clim[0], md, metab_clim[1]])
                    # cbar.set_ticklabels(
                    #     np.round([metab_clim[0], md, metab_clim[1]]).astype(int)
                    # )

            out = widgets.interactive_output(
                plot_img_fieldmap,
                {"slice": slice_slider, "crange": frequency_range_slider},
            )

            main_ui = widgets.VBox(
                [
                    slice_slider,
                    frequency_range_slider,
                ]
            )

            display(main_ui, out)

        else:
            raise Exception("wrong argument for axial_coronal")

    def postprocess_fid_file(
        self, input_data=None
    ):  # taken from MATLAB pvtools - readBrukerRaw.m
        """
        Complex data provided in fid file is padded with zeros to reach 128bits in read direction.
        Function deletes padded zeros from data depending on actual number of sampling steps in read direction.


        Returns
        ----------
        complex_file : np.ndarray
            Complex data from fid file without additional zeros.
            Reshaped to dimensions [ACQ_size[1]*ACQ_size[2]*NR*NI, PVM_Matrix[0]]

        """
        if input_data is None:
            complex_file = self.fid
        else:
            complex_file = input_data

        # delete zero lines if ACQ-experiment_mode == "Single Experiment"
        if self.acqp["ACQ_experiment_mode"] == "SingleExperiment":
            if self.method["PVM_EncNReceivers"] == 1:
                numSelectedReceivers = 1

                if self.acqp["GO_raw_data_format"] == "GO_32BIT_SGN_INT":
                    bits = 32
                elif self.acqp["GO_raw_data_format"] == "GO_16BIT_SGN_INT":
                    bits = 16
                elif self.acqp["GO_raw_data_format"] == "GO_32BIT_FLOAT":
                    bits = 32
                else:
                    bits = 32

                if self.acqp["GO_block_size"] == "Standard_KBlock_Format":
                    blockSize = int(
                        (
                            (
                                np.ceil(
                                    self.acqp["ACQ_size"][0]
                                    * numSelectedReceivers
                                    * (bits / 8)
                                    / 1024
                                )
                                * 1024
                                / (bits / 8)
                            )
                            / 2
                        )
                    )
                else:
                    blockSize = int(
                        (self.acqp["ACQ_size"][0] * numSelectedReceivers) / 2
                    )

                other_dimensions = int(
                    np.prod(self.acqp["ACQ_size"][1:])
                    * self.acqp["NI"]
                    * self.acqp["NR"]
                )

                complex_file = np.reshape(
                    complex_file,
                    (other_dimensions, blockSize),
                )
                if (
                    blockSize != self.acqp["ACQ_size"][0] * numSelectedReceivers / 2
                ):  # remove zero-lines
                    complex_file = complex_file[
                        :,
                        0 : int(self.acqp["ACQ_size"][0] * numSelectedReceivers / 2),
                    ]
        else:
            try:
                numSelectedReceivers = self.method["PVM_EncNReceivers"]

                if self.acqp["GO_raw_data_format"] == "GO_32BIT_SGN_INT":
                    bits = 32
                elif self.acqp["GO_raw_data_format"] == "GO_16BIT_SGN_INT":
                    bits = 16
                elif self.acqp["GO_raw_data_format"] == "GO_32BIT_FLOAT":
                    bits = 32
                else:
                    bits = 32

                if self.acqp["GO_block_size"] == "Standard_KBlock_Format":
                    blockSize = int(
                        (
                            (
                                np.ceil(
                                    self.acqp["ACQ_size"][0]
                                    * numSelectedReceivers
                                    * (bits / 8)
                                    / 1024
                                )
                                * 1024
                                / (bits / 8)
                            )
                            / 2
                        )
                    )
                else:
                    blockSize = int(
                        (self.acqp["ACQ_size"][0] * numSelectedReceivers) / 2
                    )

                other_dimensions = int(
                    np.prod(self.acqp["ACQ_size"][1:])
                    * self.acqp["NI"]
                    * self.acqp["NR"]
                )

                complex_file = np.reshape(
                    complex_file,
                    (other_dimensions, blockSize),
                )
                if (
                    blockSize != self.acqp["ACQ_size"][0] * numSelectedReceivers / 2
                ):  # remove zero-lines
                    complex_file = complex_file[
                        :,
                        0 : int(self.acqp["ACQ_size"][0] * numSelectedReceivers / 2),
                    ]

            except:
                Warning(
                    "Could not postprocess bssfp fid with "
                    + str(numSelectedReceivers)
                    + "Channels"
                )

        return complex_file

    def bssfp_fft(self, input_data=None, axes=(0, 1, 2), ift=False):
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
        if input_data.any():
            fid = input_data
            logger.debug("using input data")
        else:
            fid = self.fid
            logger.debug("using self.fid")

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

    def shift_anat(
        self,
        input_data=None,
        mat_bssfp=None,
        mat_anat=None,
        bssfp_obj=None,
        anat_obj=None,
        fov_bssfp=None,
        fov_anat=None,
        apply_fft=True,
        shift_vox=None,
        use_scipy_shift=False,
        shift_dims=2,
    ):
        """
        bSSFP as CSI has the issue that small matrix acquisitions are shifted.
        This function takes information about the matrix size of bSSFP and anatomical and
        shifts the bssfp acquisition accordingly. This makes more sense than shifting
        the bSSFP because coronal and axial anatomical require different shifts of the
        bSSFP data.

        Parameters:
        ----------
        input_data: reconstructed bssfp data. can be image space or k-space. If inpu_data
        is in image space, apply_fft has to be set true.

        mat_bssfp: matrix size of bssfp (in phase and slice encode directions)

        mat_anat: matrix size of axial anatomical image (in phase and slice encode directions)

        fov_bssfp: FOV of bssfp. If anatomical and bssfp have same FOV, this can be left empty

        fov_anat: FOV of anatomical. If anatomical and bssfp have same FOV, this can be left empty

        apply_fft: if input_data is in image space, this has to be set True, otherwise False. If use_cv is True, this should
        be false for input_data in image space.

        shift_vox: Can be used to manualy input the number of voxel in up/down (x dim) left/right direction (y dim)

        keep_intensity: if True, every repetition of the shifted data is normalized to the corresponding repetition
        of the input_data

        return shifted image
        ----------
        """

        # get parameters from object if passed:
        if bssfp_obj is not None:
            from hypermri.utils.utils_general import (
                define_imageFOV_parameters,
                define_imagematrix_parameters,
            )

            fov_bssfp = define_imageFOV_parameters(data_obj=bssfp_obj)
            mat_bssfp = define_imagematrix_parameters(data_obj=bssfp_obj)
        else:
            pass

        # get parameters from object if passed:
        if anat_obj is not None:
            from hypermri.utils.utils_general import (
                define_imageFOV_parameters,
                define_imagematrix_parameters,
                reorient_anat,
            )

            fov_anat = define_imageFOV_parameters(data_obj=anat_obj)
            mat_anat = define_imagematrix_parameters(data_obj=anat_obj)

            # use 2dseq from anat_obj if no input_data was passed:
            if input_data is None:
                try:
                    # check if data was already reoriented to match bssfp orientation:
                    reorient_anat_counter = get_counter(
                        data_obj=anat_obj, counter_name="reorient_anat"
                    )
                    # if not --> reorient:
                    if reorient_anat_counter == 0:
                        reorient_anat(data_obj=anat_obj)
                        input_data = anat_obj.seq2d_oriented
                    else:
                        input_data = anat_obj.seq2d_oriented
                except:
                    raise Exception("No input")
        else:
            pass

        # check if data was already shifted:
        num_shift_counts = get_counter(
            data_obj=anat_obj,
            counter_name="shift_counter",
        )

        if num_shift_counts == 0:
            pass
        else:
            return input_data

        # if no input dat was given, stop
        if input_data is None:
            return
        else:
            anat_data = input_data

        # transform to k-space:
        if apply_fft:
            anat_data = self.bssfp_fft(anat_data, axes=(1, 2, 3), ift=False)

        # check if matrix size was passed:
        if mat_bssfp is None:
            raise Exception("bssfp matrix size (mat_bssfp) is a necessary parameter")

        if np.size(mat_bssfp) < 2:
            raise Exception(
                "bssfp matrix has to have at least 2 elements, has "
                + str(np.size(mat_bssfp))
            )

        if mat_anat is None:
            raise Exception(
                "anatomical matrix size (mat_anat) is a necessary parameter"
            )
        if np.size(mat_anat) < 2:
            raise Exception(
                "anat matrix has to have at least 2 elements, has "
                + str(np.size(mat_bssfp))
            )

        # 2D Shift:
        if shift_dims == 2:
            raise Exception("2D implementation is wonky!")
            if fov_bssfp is None and fov_anat is None:
                # can use any value, doesnt matter as
                # long as they have the same FOV:
                fov_bssfp = fov_anat = [1.0, 1.0]
            elif fov_bssfp is None and fov_anat is not None:
                # use same FOV
                fov_bssfp = fov_anat
            elif fov_bssfp is not None and fov_anat is None:
                # use same FOV
                fov_anat = fov_bssfp
            else:
                pass

            # calculate the resultions ():
            res_bssfp = [a / b for a, b in zip(fov_bssfp, mat_bssfp)]
            res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]

            # calc necessary shift in both directions:
            shift_vox_list = [(a - b) / 2 / a for a, b in zip(res_bssfp, res_anat)]

            # shift_vox = [4 / f for f in mat_bssfp]
            # calculate necessary shift if none was passed:
            if shift_vox is None:
                shift_vox = [0, 0]
                shift_vox[0] = -shift_vox_list[
                    0
                ]  # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[1] = shift_vox_list[
                    1
                ]  # y-dim of array echoes-z-x-y-rep-chans
                # shift_vox = [0, 0]

            # use opencv to shift (reduces ringing artifacts)
            if use_cv is True:
                import cv2

                # translation matrix (first entries are for rotation, last for translation)
                # https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
                Mtrans = np.float32(
                    [[1, 0, -shift_vox[1]], [0, 1, shift_vox[0]]]
                )  # 1. up/down (y-axis) (positive value moves animal up)

                # init empty array:
                anat_data_shifted_real = np.zeros_like(anat_data)
                anat_data_shifted_imag = np.zeros_like(anat_data)

                # Choose interpolation method:
                if cv_interpolator == "INTER_CUBIC":
                    flag = cv2.INTER_CUBIC
                    logger.critical("INTER_CUBIC")
                elif cv_interpolator == "INTER_LANCZOS4":
                    flag = cv2.INTER_LANCZOS4
                    logger.critical("INTER_LANCZOS4")
                elif cv_interpolator == "INTER_NEAREST":
                    flag = cv2.INTER_NEAREST
                    logger.critical("INTER_NEAREST")
                elif cv_interpolator == "INTER_LINEAR":
                    flag = cv2.INTER_LINEAR
                    logger.critical("INTER_LINEAR")
                elif cv_interpolator == "INTER_AREA":
                    flag = cv2.INTER_AREA
                    logger.critical("INTER_AREA")
                # default setting:
                else:
                    flag = cv2.INTER_CUBIC
                    logger.critical("DEFAULT INTER_CUBIC")

                # loop over echoes:
                for e in range(anat_data.shape[0]):
                    # loop over repetitions:
                    for r in range(anat_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(anat_data.shape[1]):
                            # loop over channels
                            for c in range(anat_data.shape[5]):
                                image = np.squeeze(anat_data[e, k, :, :, r, c])
                                anat_data_shifted_real[
                                    e, k, :, :, r, c
                                ] = cv2.warpAffine(
                                    np.real(image),
                                    Mtrans,
                                    image.shape[::-1],
                                    flags=flag,
                                )
                                anat_data_shifted_imag[
                                    e, k, :, :, r, c
                                ] = cv2.warpAffine(
                                    np.imag(image),
                                    Mtrans,
                                    image.shape[::-1],
                                    flags=flag,
                                )
                anat_data_shifted = anat_data_shifted_real + 1j * anat_data_shifted_imag
            elif use_scipy_shift:
                from scipy.ndimage import shift

                # init empty array:
                anat_data_shifted_real = np.zeros_like(anat_data)
                anat_data_shifted_imag = np.zeros_like(anat_data)

                shift_vox[1] = -shift_vox[1]

                # loop over echoes:
                for e in range(anat_data.shape[0]):
                    # loop over repetitions:
                    for r in range(anat_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(anat_data.shape[1]):
                            # loop over channels
                            for c in range(anat_data.shape[5]):
                                image = np.squeeze(anat_data[e, k, :, :, r, c])
                                anat_data_shifted_real[e, k, :, :, r, c] = shift(
                                    np.real(image), shift_vox, mode="wrap"
                                )
                                anat_data_shifted_imag[e, k, :, :, r, c] = shift(
                                    np.imag(image), shift_vox, mode="wrap"
                                )
                anat_data_shifted = anat_data_shifted_real + 1j * anat_data_shifted_imag
            else:
                # k-space frequency range assuming a spatial samplingrate of d=1
                freq_k = [
                    np.fft.fftfreq(mat_bssfp[0], d=1),
                    np.fft.fftfreq(mat_bssfp[1], d=1),
                ]

                # prepare Fourier Shift (theorem)
                shift_exp = [
                    np.exp(1j * 2 * np.pi * b * c) for b, c in zip(shift_vox, freq_k)
                ]

                # has to be fftshifted:
                shift_exp = [np.fft.fftshift(e) for e in shift_exp]

                # make matrix out of the 2 1D shift arrays:
                shift_matx = np.rot90(np.tile(shift_exp[0], (mat_bssfp[1], 1)))
                shift_maty = np.repeat(shift_exp[1], mat_bssfp[0])
                shift_maty = np.rot90(
                    np.reshape(shift_maty, (mat_bssfp[1], mat_bssfp[0]))
                )

                # init empty k-space:
                anat_data_shifted = np.zeros_like(anat_data)
                logger.debug("anat_data_shifted" + str(anat_data_shifted.shape))
                # loop over echoes:
                for e in range(anat_data.shape[0]):
                    # loop over repetitions:
                    for r in range(anat_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(anat_data.shape[1]):
                            # loop over channels
                            for c in range(anat_data.shape[5]):
                                anat_data_shifted[e, k, :, :, r, c] = (
                                    np.squeeze(anat_data[e, k, :, :, r, c])
                                    * shift_matx
                                    * shift_maty
                                )

                # transform to image space:
                if apply_fft:
                    anat_data_shifted = self.bssfp_fft(
                        anat_data_shifted, axes=(1, 2, 3), ift=True
                    )

            # "Normalize"
            if keep_intensity:
                for r in range(anat_data.shape[4]):
                    anat_data_shifted[:, :, :, :, r, :] = (
                        anat_data_shifted[:, :, :, :, r, :]
                        / np.sum(np.abs(anat_data_shifted[:, :, :, :, r, :]))
                        * np.sum(np.abs(input_data[:, :, :, :, r, :]))
                    )

            return anat_data_shifted
        elif shift_dims == 3:
            # Set FOVs:
            if fov_bssfp is None and fov_anat is None:
                # use any value, doesn't matter as long as they have the same FOV
                fov_bssfp = fov_anat = [1.0, 1.0, 1.0]
            elif fov_bssfp is None and fov_anat is not None:
                # use same FOV
                fov_bssfp = fov_anat
            elif fov_bssfp is not None and fov_anat is None:
                # use same FOV
                fov_anat = fov_bssfp
            else:
                pass

            # to avoid misalignement, add a checker that checks if the image as
            #  already been rotated:

            if (len(fov_bssfp) != 3) or (len(fov_anat) != 3):
                logger.critical(
                    "mat_bssfp, mat_anat, fov_bssfp and fov_anat have to be of length 3!"
                )
                pass

            # calculate the resultions ():
            res_bssfp = [a / b for a, b in zip(fov_bssfp, mat_bssfp)]
            res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]

            # calc necessary shift in both directions (outdated?):
            shift_vox_list = [(a - b) / 2 / a for a, b in zip(res_bssfp, res_anat)]

            # probably better:
            shift_vox_list = [
                d * (a - b) / 2 / c
                for a, b, c, d in zip(res_bssfp, res_anat, fov_anat, mat_anat)
            ]

            # shift_vox = [4 / f for f in mat_bssfp]
            # calculate necessary shift if none was passed:
            if shift_vox is None:
                shift_vox = [0, 0, 0, 0, 0, 0]
                # front/back:
                shift_vox[1] = -shift_vox_list[0]
                # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[2] = shift_vox_list[1]
                # y-dim of array echoes-z-x-y-rep-chans
                # shift_vox = [0, 0]
                shift_vox[3] = -shift_vox_list[2]

            if use_scipy_shift:
                from scipy.ndimage import shift

                # shift image:
                anat_data_shifted = shift(anat_data, shift_vox, mode="wrap")

                # add 1 to shift counter:
                add_counter(
                    data_obj=anat_obj,
                    counter_name="shift_counter",
                    n_counts=1,
                )
                return anat_data_shifted

            else:
                pass
        else:
            Warning("Unknown shift dimension: " + str(shift_dims))

        return False

    def shift_bssfp(
        self,
        input_data=None,
        mat_bssfp=None,
        mat_anat=None,
        fov_bssfp=None,
        fov_anat=None,
        apply_fft=True,
        shift_vox=None,
        keep_intensity=True,
        use_cv=False,
        cv_interpolator="INTER_CUBIC",
        use_scipy_shift=False,
        shift_dims=2,
        mat_anat_cor=None,
    ):
        """
        bSSFP as CSI has the issue that small matrix acquisitions are shifted.
        This function takes information about the matrix size and shifts the
        bssfp acquisition accordingly.

        Parameters:
        ----------
        input_data: reconstructed bssfp data. can be image space or k-space. If inpu_data
        is in image space, apply_fft has to be set true.

        mat_bssfp: matrix size of bssfp (in phase and slice encode directions)

        mat_anat: matrix size of axial anatomical image (in phase and slice encode directions)

        fov_bssfp: FOV of bssfp. If anatomical and bssfp have same FOV, this can be left empty

        fov_anat: FOV of anatomical. If anatomical and bssfp have same FOV, this can be left empty

        apply_fft: if input_data is in image space, this has to be set True, otherwise False. If use_cv is True, this should
        be false for input_data in image space.

        shift_vox: Can be used to manually input the number of voxel in up/down (x dim) left/right direction (y dim)

        keep_intensity: if True, every repetition of the shifted data is normalized to the corresponding repetition
        of the input_data

        return shifted image

        ----------
        # perform the shift after the reconstruction:

        dnp_bssfp = scans[24]
        # in case the complex reco was reco number 3:
        dnp_bssfp_img = dnp_bssfp.Load_2dseq_file(recon_num=3)

        [_, dnp_bssfp_pv_reco] = dnp_bssfp.reconstruction(seq2d = dnp_bssfp_img)

        [_, dnp_bssfp_pv_reco, axial.seq2d, _, coronal.seq2d] = dnp_bssfp.reorient_reco(
            bssfp_seq2d=dnp_bssfp_pv_reco,
            bssfp_custom=dnp_bssfp.Reconstructed_data,
            anatomical_seq2d_ax=axial.seq2d,
            anatomical_seq2d_cor=coronal.seq2d)

        # Either:
        dnp_bssfp_pv_reco_shift = dnp_bssfp.shift_bssfp(input_data=dnp_bssfp_pv_reco,
                         mat_bssfp=dnp_bssfp_pv_reco.shape[2:4],
                                 mat_anat=axial.seq2d.shape[0:2],
                                 fov_bssfp=dnp_bssfp.method['PVM_Fov'][1:3],
                                 fov_anat=axial.method['PVM_Fov'],
                                 apply_fft=True)

        # Or:
        dnp_bssfp_pv_reco_shift = dnp_bssfp.shift_bssfp(input_data=dnp_bssfp_pv_reco,                           # complex, reordered bSSFP data
                                 mat_bssfp=dnp_bssfp_pv_complex_reco_combined.shape[2:4],                       # bssfp "axial" matrix size (phase and slice dim)
                                 mat_anat=axial.seq2d.shape[1:3],                                               # axial matrix size (phase and slice dim)
                                 fov_bssfp=dnp_bssfp.method['PVM_Fov'][1:3],                                    # bSSFP "axial" FOV (can be left out if bssfp and axial have same FOV)
                                 fov_anat=axial.method['PVM_Fov'],
                                 use_scipy_shift=True)
        ----------
        """
        Warning("Should not be used anymore. Used shift_anat instead!")

        # if no input dat was given, use fid
        if input_data is None:
            return
        else:
            bssfp_data = input_data

        # transform to k-space:
        if apply_fft:
            bssfp_data = self.bssfp_fft(bssfp_data, axes=(1, 2, 3), ift=False)

        # check if matrix size was passed:
        if mat_bssfp is None:
            raise Exception("bssfp matrix size (mat_bssfp) is a necessary parameter")
        if np.size(mat_bssfp) < 2:
            raise Exception(
                "bssfp matrix has to have at least 2 elements, has "
                + str(np.size(mat_bssfp))
            )

        if mat_anat is None:
            raise Exception(
                "anatomical matrix size (mat_anat) is a necessary parameter"
            )
        if np.size(mat_anat) < 2:
            raise Exception(
                "anat matrix has to have at least 2 elements, has "
                + str(np.size(mat_bssfp))
            )
        # 2D Shift:
        if shift_dims == 2:
            if fov_bssfp is None and fov_anat is None:
                # use any value, doesnt matter as long as they have the same FOV
                fov_bssfp = fov_anat = [1.0, 1.0]
            elif fov_bssfp is None and fov_anat is not None:
                # use same FOV
                fov_bssfp = fov_anat
            elif fov_bssfp is not None and fov_anat is None:
                # use same FOV
                fov_anat = fov_bssfp
            else:
                pass

            # calculate the resultions ():
            res_bssfp = [a / b for a, b in zip(fov_bssfp, mat_bssfp)]
            res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]

            # calc necessary shift in both directions:
            shift_vox_list = [(a - b) / 2 / a for a, b in zip(res_bssfp, res_anat)]

            # shift_vox = [4 / f for f in mat_bssfp]
            # calculate necessary shift if none was passed:
            if shift_vox is None:
                shift_vox = [0, 0]
                shift_vox[0] = -shift_vox_list[
                    0
                ]  # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[1] = shift_vox_list[
                    1
                ]  # y-dim of array echoes-z-x-y-rep-chans
                # shift_vox = [0, 0]

            # use opencv to shift (reduces ringing artifacts)
            if use_cv is True:
                import cv2

                # translation matrix (first entries are for rotation, last for translation)
                # https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
                Mtrans = np.float32(
                    [[1, 0, -shift_vox[1]], [0, 1, shift_vox[0]]]
                )  # 1. up/down (y-axis) (positive value moves animal up)

                # init empty array:
                bssfp_data_shifted_real = np.zeros_like(bssfp_data)
                bssfp_data_shifted_imag = np.zeros_like(bssfp_data)

                # Choose interpolation method:
                if cv_interpolator == "INTER_CUBIC":
                    flag = cv2.INTER_CUBIC
                    logger.critical("INTER_CUBIC")
                elif cv_interpolator == "INTER_LANCZOS4":
                    flag = cv2.INTER_LANCZOS4
                    logger.critical("INTER_LANCZOS4")
                elif cv_interpolator == "INTER_NEAREST":
                    flag = cv2.INTER_NEAREST
                    logger.critical("INTER_NEAREST")
                elif cv_interpolator == "INTER_LINEAR":
                    flag = cv2.INTER_LINEAR
                    logger.critical("INTER_LINEAR")
                elif cv_interpolator == "INTER_AREA":
                    flag = cv2.INTER_AREA
                    logger.critical("INTER_AREA")
                # default setting:
                else:
                    flag = cv2.INTER_CUBIC
                    logger.critical("DEFAULT INTER_CUBIC")

                # loop over echoes:
                for e in range(bssfp_data.shape[0]):
                    # loop over repetitions:
                    for r in range(bssfp_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(bssfp_data.shape[1]):
                            # loop over channels
                            for c in range(bssfp_data.shape[5]):
                                image = np.squeeze(bssfp_data[e, k, :, :, r, c])
                                bssfp_data_shifted_real[
                                    e, k, :, :, r, c
                                ] = cv2.warpAffine(
                                    np.real(image),
                                    Mtrans,
                                    image.shape[::-1],
                                    flags=flag,
                                )
                                bssfp_data_shifted_imag[
                                    e, k, :, :, r, c
                                ] = cv2.warpAffine(
                                    np.imag(image),
                                    Mtrans,
                                    image.shape[::-1],
                                    flags=flag,
                                )
                bssfp_data_shifted = (
                    bssfp_data_shifted_real + 1j * bssfp_data_shifted_imag
                )
            elif use_scipy_shift:
                from scipy.ndimage import shift

                # init empty array:
                bssfp_data_shifted_real = np.zeros_like(bssfp_data)
                bssfp_data_shifted_imag = np.zeros_like(bssfp_data)

                shift_vox[1] = -shift_vox[1]

                # loop over echoes:
                for e in range(bssfp_data.shape[0]):
                    # loop over repetitions:
                    for r in range(bssfp_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(bssfp_data.shape[1]):
                            # loop over channels
                            for c in range(bssfp_data.shape[5]):
                                image = np.squeeze(bssfp_data[e, k, :, :, r, c])
                                bssfp_data_shifted_real[e, k, :, :, r, c] = shift(
                                    np.real(image), shift_vox, mode="wrap"
                                )
                                bssfp_data_shifted_imag[e, k, :, :, r, c] = shift(
                                    np.imag(image), shift_vox, mode="wrap"
                                )
                bssfp_data_shifted = (
                    bssfp_data_shifted_real + 1j * bssfp_data_shifted_imag
                )
            else:
                # k-space frequency range assuming a spatial samplingrate of d=1
                freq_k = [
                    np.fft.fftfreq(mat_bssfp[0], d=1),
                    np.fft.fftfreq(mat_bssfp[1], d=1),
                ]

                # prepare Fourier Shift (theorem)
                shift_exp = [
                    np.exp(1j * 2 * np.pi * b * c) for b, c in zip(shift_vox, freq_k)
                ]

                # has to be fftshifted:
                shift_exp = [np.fft.fftshift(e) for e in shift_exp]

                # make matrix out of the 2 1D shift arrays:
                shift_matx = np.rot90(np.tile(shift_exp[0], (mat_bssfp[1], 1)))
                shift_maty = np.repeat(shift_exp[1], mat_bssfp[0])
                shift_maty = np.rot90(
                    np.reshape(shift_maty, (mat_bssfp[1], mat_bssfp[0]))
                )

                # init empty k-space:
                bssfp_data_shifted = np.zeros_like(bssfp_data)
                logger.debug("bssfp_data_shifted" + str(bssfp_data_shifted.shape))
                # loop over echoes:
                for e in range(bssfp_data.shape[0]):
                    # loop over repetitions:
                    for r in range(bssfp_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(bssfp_data.shape[1]):
                            # loop over channels
                            for c in range(bssfp_data.shape[5]):
                                bssfp_data_shifted[e, k, :, :, r, c] = (
                                    np.squeeze(bssfp_data[e, k, :, :, r, c])
                                    * shift_matx
                                    * shift_maty
                                )

                # transform to image space:
                if apply_fft:
                    bssfp_data_shifted = self.bssfp_fft(
                        bssfp_data_shifted, axes=(1, 2, 3), ift=True
                    )

            # "Normalize"
            if keep_intensity:
                for r in range(bssfp_data.shape[4]):
                    bssfp_data_shifted[:, :, :, :, r, :] = (
                        bssfp_data_shifted[:, :, :, :, r, :]
                        / np.sum(np.abs(bssfp_data_shifted[:, :, :, :, r, :]))
                        * np.sum(np.abs(input_data[:, :, :, :, r, :]))
                    )

            return bssfp_data_shifted
        elif shift_dims == 3:
            # Set FOVs:
            if fov_bssfp is None and fov_anat is None:
                # use any value, doesnt matter as long as they have the same FOV
                fov_bssfp = fov_anat = [1.0, 1.0, 1.0]
            elif fov_bssfp is None and fov_anat is not None:
                # use same FOV
                fov_bssfp = fov_anat
            elif fov_bssfp is not None and fov_anat is None:
                # use same FOV
                fov_anat = fov_bssfp
            else:
                pass

            # calculate the resultions ():
            res_bssfp = [a / b for a, b in zip(fov_bssfp, mat_bssfp)]
            res_anat = [a / b for a, b in zip(fov_anat, mat_anat)]

            # calc necessary shift in both directions:
            shift_vox_list = [(a - b) / 2 / a for a, b in zip(res_bssfp, res_anat)]

            # shift_vox = [4 / f for f in mat_bssfp]
            # calculate necessary shift if none was passed:
            if shift_vox is None:
                shift_vox = [0, 0]
                shift_vox[0] = -shift_vox_list[
                    0
                ]  # left/right, x-dim of array echoes-z-x-y-rep-chans
                shift_vox[1] = shift_vox_list[
                    1
                ]  # y-dim of array echoes-z-x-y-rep-chans
                # shift_vox = [0, 0]

            # use opencv to shift (reduces ringing artifacts)
            if use_cv is True:
                import cv2

                # translation matrix (first entries are for rotation, last for translation)
                # https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
                Mtrans = np.float32(
                    [[1, 0, -shift_vox[1]], [0, 1, shift_vox[0]]]
                )  # 1. up/down (y-axis) (positive value moves animal up)

                # init empty array:
                bssfp_data_shifted_real = np.zeros_like(bssfp_data)
                bssfp_data_shifted_imag = np.zeros_like(bssfp_data)

                # Choose interpolation method:
                if cv_interpolator == "INTER_CUBIC":
                    flag = cv2.INTER_CUBIC
                    logger.critical("INTER_CUBIC")
                elif cv_interpolator == "INTER_LANCZOS4":
                    flag = cv2.INTER_LANCZOS4
                    logger.critical("INTER_LANCZOS4")
                elif cv_interpolator == "INTER_NEAREST":
                    flag = cv2.INTER_NEAREST
                    logger.critical("INTER_NEAREST")
                elif cv_interpolator == "INTER_LINEAR":
                    flag = cv2.INTER_LINEAR
                    logger.critical("INTER_LINEAR")
                elif cv_interpolator == "INTER_AREA":
                    flag = cv2.INTER_AREA
                    logger.critical("INTER_AREA")
                # default setting:
                else:
                    flag = cv2.INTER_CUBIC
                    logger.critical("DEFAULT INTER_CUBIC")

                # loop over echoes:
                for e in range(bssfp_data.shape[0]):
                    # loop over repetitions:
                    for r in range(bssfp_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(bssfp_data.shape[1]):
                            # loop over channels
                            for c in range(bssfp_data.shape[5]):
                                image = np.squeeze(bssfp_data[e, k, :, :, r, c])
                                bssfp_data_shifted_real[
                                    e, k, :, :, r, c
                                ] = cv2.warpAffine(
                                    np.real(image),
                                    Mtrans,
                                    image.shape[::-1],
                                    flags=flag,
                                )
                                bssfp_data_shifted_imag[
                                    e, k, :, :, r, c
                                ] = cv2.warpAffine(
                                    np.imag(image),
                                    Mtrans,
                                    image.shape[::-1],
                                    flags=flag,
                                )
                bssfp_data_shifted = (
                    bssfp_data_shifted_real + 1j * bssfp_data_shifted_imag
                )
            elif use_scipy_shift:
                from scipy.ndimage import shift

                # init empty array:
                bssfp_data_shifted_real = np.zeros_like(bssfp_data)
                bssfp_data_shifted_imag = np.zeros_like(bssfp_data)

                shift_vox[1] = -shift_vox[1]

                # loop over echoes:
                for e in range(bssfp_data.shape[0]):
                    # loop over repetitions:
                    for r in range(bssfp_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(bssfp_data.shape[1]):
                            # loop over channels
                            for c in range(bssfp_data.shape[5]):
                                image = np.squeeze(bssfp_data[e, k, :, :, r, c])
                                bssfp_data_shifted_real[e, k, :, :, r, c] = shift(
                                    np.real(image), shift_vox, mode="wrap"
                                )
                                bssfp_data_shifted_imag[e, k, :, :, r, c] = shift(
                                    np.imag(image), shift_vox, mode="wrap"
                                )
                bssfp_data_shifted = (
                    bssfp_data_shifted_real + 1j * bssfp_data_shifted_imag
                )
            else:
                # k-space frequency range assuming a spatial samplingrate of d=1
                freq_k = [
                    np.fft.fftfreq(mat_bssfp[0], d=1),
                    np.fft.fftfreq(mat_bssfp[1], d=1),
                ]

                # prepare Fourier Shift (theorem)
                shift_exp = [
                    np.exp(1j * 2 * np.pi * b * c) for b, c in zip(shift_vox, freq_k)
                ]

                # has to be fftshifted:
                shift_exp = [np.fft.fftshift(e) for e in shift_exp]

                # make matrix out of the 2 1D shift arrays:
                shift_matx = np.rot90(np.tile(shift_exp[0], (mat_bssfp[1], 1)))
                shift_maty = np.repeat(shift_exp[1], mat_bssfp[0])
                shift_maty = np.rot90(
                    np.reshape(shift_maty, (mat_bssfp[1], mat_bssfp[0]))
                )

                # init empty k-space:
                bssfp_data_shifted = np.zeros_like(bssfp_data)
                logger.debug("bssfp_data_shifted" + str(bssfp_data_shifted.shape))
                # loop over echoes:
                for e in range(bssfp_data.shape[0]):
                    # loop over repetitions:
                    for r in range(bssfp_data.shape[4]):
                        # loop over points in read dim:
                        for k in range(bssfp_data.shape[1]):
                            # loop over channels
                            for c in range(bssfp_data.shape[5]):
                                bssfp_data_shifted[e, k, :, :, r, c] = (
                                    np.squeeze(bssfp_data[e, k, :, :, r, c])
                                    * shift_matx
                                    * shift_maty
                                )

                # transform to image space:
                if apply_fft:
                    bssfp_data_shifted = self.bssfp_fft(
                        bssfp_data_shifted, axes=(1, 2, 3), ift=True
                    )

            # "Normalize"
            if keep_intensity:
                for r in range(bssfp_data.shape[4]):
                    bssfp_data_shifted[:, :, :, :, r, :] = (
                        bssfp_data_shifted[:, :, :, :, r, :]
                        / np.sum(np.abs(bssfp_data_shifted[:, :, :, :, r, :]))
                        * np.sum(np.abs(input_data[:, :, :, :, r, :]))
                    )

            return bssfp_data_shifted
        else:
            Warning("Unknown shift dimension: " + str())

    def combine_multichannel(
        self, input_data=None, phase=None, chan_dim=None, rssq=False, db=False
    ):
        """
        Quickly computes the fftshift(fft(fftshift(input_data))) along the input axis.
        If input_data is not set, returns the fft along axis of the fid file

        Parameters:
        ----------
        input_data : (complex) ndarray (None)
            data that should be fourier-transformed.
            If blank, self.fid is used

        phase: float None
            Phase [degree] that should be used to combine the 2 channels.
            If non is given the phase will be automatically determined

        axes : tuple (0,1,2)
            axes along the fourier transform should be performed

        rssq : root sum of squares

        db : bool (False)
            Prints debugging info if True
        """

        # if no input dat was given, use fid
        if input_data is None:
            fid = self.fid
            logger.debug("using self.fid")
        else:
            fid = input_data
            logger.debug("using input data")
        logger.debug("dimensions of input_data = " + str(fid.ndim))

        # do not continue if only 1 channel input:
        if input_data.shape[5] == 1:
            # logger.critical("only 1 channel put in...")
            return input_data

        # if no channel dimension was given use 6th dimension for channel:
        if chan_dim is None:
            chan_dim = 5
            if chan_dim <= fid.ndim:
                chan_dim = chan_dim
            else:
                chan_dim_before = chan_dim
                chan_dim = fid.ndim
                Warning(
                    "Unexpected array dimensions: "
                    + str(fid.ndim)
                    + ".Default channel dimension: "
                    + str(chan_dim_before)
                    + ". Using"
                    + " the last array dimension as channel dimension"
                )
        else:
            if chan_dim <= fid.ndim - 1:
                chan_dim = chan_dim
            else:
                Warning(
                    "Inconsistent array dimensions: "
                    + str(fid.ndim)
                    + "and desired channel dimension: "
                    + str(chan_dim)
                    + ". Using"
                    + " the last array dimension as channel dimension"
                )
                chan_dim = fid.ndim

        # test if the array has more than 1 channel:
        if fid.shape[chan_dim - 1] > 1:
            max_val = np.max(fid)
            logger.debug("Max val is " + str(max_val))
            max_index = np.unravel_index(np.argmax(fid), fid.shape)
            logger.debug("Max index is " + str(max_index))
        else:
            Warning("Data has only " + fid.ndim[chan_dim] + " in the channel dimension")

        logger.debug("chan_dim = " + str(chan_dim))

        # use root mean square:
        if rssq is True:
            fid_combined = np.sqrt(
                np.abs(fid[:, :, :, :, :, [0]]) ** 2
                + np.abs(fid[:, :, :, :, :, [1]]) ** 2
            )
        # combine both channels using a phase term applied on channel 2
        else:
            # no phase input, find phase that maximizes the signal
            if phase is None:
                # find max in array, compare signal from max for phaseing
                # assuming the order Echoes, Read, phase, slice, repetitions, channels in the data:
                ch1_max = np.sum(
                    np.sum(np.sum(fid[max_index[0], :, :, :, max_index[4], 0]))
                )
                ch2_max = np.sum(
                    np.sum(np.sum(fid[max_index[0], :, :, :, max_index[4], 1]))
                )

                # define phase array from -pi to pi
                phase_range_rad = np.linspace(-np.pi, np.pi, 1000)
                phase_range_deg = phase_range_rad / np.pi * 180
                # combine both channels:
                ch12_combined = ch1_max + ch2_max * np.exp(1j * phase_range_rad)
                max_phase_ind = np.argmax(np.abs(ch12_combined))
                max_phase_rad_val = phase_range_rad[max_phase_ind]
                max_phase_deg_val = phase_range_deg[max_phase_ind]

                # plot result
                if db:
                    from matplotlib import pyplot as plt

                    fig, ax = plt.subplots()
                    ax.plot(phase_range_deg, np.real(ch12_combined))
                    ax.plot(phase_range_deg, np.real(ch12_combined))
                    ax.plot(
                        phase_range_deg[max_phase_ind], max(np.abs(ch12_combined)), "rx"
                    )
                    ax.title.set_text("Max phase = " + str(max_phase_deg_val))

                # combine channels:
                fid_combined = np.zeros((fid.shape[:-1] + (1,)), dtype=fid.dtype)
                # use this [] indexing to keep dimension
                fid_combined = fid[:, :, :, :, :, [0]] + fid[
                    :, :, :, :, :, [1]
                ] * np.exp(1j * max_phase_rad_val)

            # use input phase:
            else:
                max_phase_rad_val = phase / 180 * np.pi
                # combine channels:
                fid_combined = np.zeros((fid.shape[:-1] + (1,)), dtype=fid.dtype)
                # use this [] indexing to keep dimension
                fid_combined = fid[:, :, :, :, :, [0]] + fid[
                    :, :, :, :, :, [1]
                ] * np.exp(1j * max_phase_rad_val)

        return fid_combined

    def get_extent_bssfp(self):
        """
        Calculates the extent of a bssfp for axial, coronal and sagittal displaying and
        adds them as attributes to the class:

        self.cor_ext
        self.ax_ext
        self.sag_ext

        """
        read_orient = self.method["PVM_SPackArrReadOrient"]
        read_offset = self.method["PVM_SPackArrReadOffset"]
        phase_offset = self.method["PVM_SPackArrPhase1Offset"]
        slice_offset = self.method["PVM_SPackArrSliceOffset"]
        dim_read = self.method["PVM_Matrix"][0]
        dim_phase = self.method["PVM_Matrix"][1]
        dim_slice = self.method["PVM_Matrix"][2]
        mm_read = self.method["PVM_Fov"][0]
        mm_phase = self.method["PVM_Fov"][1]
        mm_slice = self.method["PVM_Fov"][2]
        if self.method["PVM_SPackArrSliceOrient"] == "coronal":
            if read_orient == "H_F":
                self.cor_ext = [
                    -mm_phase / 2 + phase_offset,
                    mm_phase / 2 + phase_offset,
                    -mm_read / 2 + read_offset,
                    mm_read / 2 + read_offset,
                ]
                self.ax_ext = [
                    -mm_phase / 2 + phase_offset,
                    mm_phase / 2 + phase_offset,
                    -mm_slice / 2 + slice_offset,
                    mm_slice / 2 + slice_offset,
                ]
                self.sag_ext = [
                    -mm_read / 2 + read_offset,
                    mm_read / 2 + read_offset,
                    -mm_slice / 2 + slice_offset,
                    mm_slice / 2 + slice_offset,
                ]

        else:
            raise NotImplementedError(
                self.method["PVM_SPackArrSliceOrient"], "not implemented yet"
            )

    def find_noise_region(self):
        """This function is supposed to find a region containing no signal that can be used as a reference of noise"""
        return True

    def generate_figures(
        self,
        bssfp_data=None,
        coronal_image=None,
        axial_image=None,
        axlist=None,
        fig=None,
        path_to_params=None,
        path_to_save=None,
        param_dict=None,
        pyr_ind=0,
        fig_prefix="fig_",
    ):
        """

        Load figure defining parameter file, plot the data and save it.
        Use the parameters as you would use them in plot3d_new2

        Parameters
        ----------
        path_to_params: str
            parameter (json) file
        path_to_save: str
            folder where the figures should be saved
        param_dict: dict
            parameters

        Examples
        --------

        similat to plot3d_new2, but with additional parameters path_to_params, path_to_save
        and param_dict

        >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3,tight_layout=True)

        >>> phantom_bssfp.generate_figures(bssfp_data= dnp_bssfp_pv_complex_reco_combined_shift,
        >>>               coronal_image=coronal,
        >>>               axial_image=axial,
        >>>               axlist=[ax1, ax2, ax3],
        >>>               path_to_params="lac_sig.json",
        >>>               fig=fig,
        >>>               path_to_save="figures_looped_lac\\",
        >>>               param_dict={'rep1': range(1,200,2)},
        >>>               fig_prefix="lac_")

        """
        import matplotlib.pyplot as plt

        # load json parameter file:
        plot_params = load_plot_params(param_file=path_to_params, data_obj=self)

        if plot_params is None:
            return
        if path_to_save is None:
            return
        if param_dict is None:
            return

        # generate path if none was passed:
        if path_to_save is None:
            path_to_save = "looped_figures/"

        # make it path if not:
        if path_to_save[-1] != "/":
            path_to_save = path_to_save + "/"

        # loop over parameters that should be looped through
        figure_counter = 0
        for key in param_dict:
            # generate copy to not reload each time
            plot_params_plot = plot_params
            # loop over values of plot parameter:
            for value in param_dict[key]:
                figure_counter = figure_counter + 1
                plot_params_plot["fig_name"] = (
                    path_to_save + fig_prefix + "%0.3d" % figure_counter
                )
                plot_params_plot[key] = value
                # call plotting function with new parameters:
                self.plot3D_new2(
                    bssfp_data=bssfp_data,
                    coronal_image=coronal_image,
                    axial_image=axial_image,
                    axlist=axlist,
                    fig=fig,
                    pyr_ind=pyr_ind,
                    plot_params=plot_params_plot,
                    save_fig=True,
                )

    def norm_bssfp_to_background_and_divide_by_mean_for_SSI(
        self,
        bssfp_data,
        coronal_object,
        slice_num_cor,
        axlist=None,
        bg_pixels_x=[0, 3],
        bg_pixels_y=[1, 8],
        num_hist_bins=25,
    ):
        """
        Function to prepare bssfp data for Structural Similarity comparison.

        Parameters
        ----------

        bssfp_data: 3D array
            Already summed in time or just for one repetition prepared bssfp image data
            for one metabolite. Shape is (phase, read, slice)
        coronal_object: BrukerExp instance
            coronal reference data
        slice_num_cor: int
            slice number of desired bssfp slice
        axlist: list
            contains axis elements to plot into
        bg_pixels_x: list, two entries
            background region choice in x-dimension, to be plotted as a Rectangle on bssfp image in this function
        bg_pixels_y: list, two entries
            background region choice in y-dimension, to be plotted as a Rectangle on bssfp image in this function
        num_hist_bins : int
            number of histogram bins in histogram plot

        Returns
        -------

        background value
        """
        if axlist is None:
            logger.warning(
                "Need to pass axlist (use fig, (ax1, ax2, ax3) = plt.subplots(1, 3) \
                        to generate and call function with: \
                        axlist=[ax1, ax2, ax3])"
            )
            return False
        if coronal_object.__module__.startswith("hypermri."):
            pass
        else:
            logger.critical("coronal_object must be a type BrukerExp object")

        # assumes data of shape (read, phase,slice)

        def save_background_val(self, background_val, background_std):
            setattr(self, "ssi_background_value", {})
            self.ssi_background_value = background_val
            setattr(self, "ssi_background_std", {})
            self.ssi_background_std = background_std

        def plot(
            slice_num_cor, bg_pixels_x, bg_pixels_y, num_hist_bins, cor_image_window
        ):
            print(
                "N-pixels=",
                str(
                    (bg_pixels_y[1] - bg_pixels_y[0])
                    * (bg_pixels_x[1] - bg_pixels_x[0])
                ),
            )
            for ax in axlist:
                ax.clear()
            axlist[0].imshow(
                bssfp_data[:, :, slice_num_cor],
                extent=[
                    0,
                    bssfp_data[:, :, slice_num_cor].shape[1],
                    bssfp_data[:, :, slice_num_cor].shape[0],
                    0,
                ],
            )
            axlist[0].add_patch(
                Rectangle(
                    (np.min(bg_pixels_x[0:1]), np.min(bg_pixels_y[0:1])),
                    bg_pixels_x[1] - bg_pixels_x[0],
                    bg_pixels_y[1] - bg_pixels_y[0],
                    color="r",
                    fc="r",
                    alpha=0.4,
                )
            )

            coronal_slice_corresponding_to_bssfp_slice = self.find_closest_slice(
                coronal_obj=coronal_object, bssfp_pos=[0, 0, slice_num_cor]
            )[2]

            axlist[1].imshow(
                coronal_object.seq2d[:, :, coronal_slice_corresponding_to_bssfp_slice],
                cmap="bone",
                vmin=cor_image_window[0],
                vmax=cor_image_window[1],
            )

            x_data, y_data, bin_size = Get_Hist(
                bssfp_data[
                    bg_pixels_y[0] : bg_pixels_y[1], bg_pixels_x[0] : bg_pixels_x[1], 0
                ],
                num_hist_bins,
            )
            axlist[2].bar(x_data, y_data, width=bin_size)

            background_val = np.mean(
                bssfp_data[
                    bg_pixels_y[0] : bg_pixels_y[1],
                    bg_pixels_x[0] : bg_pixels_x[1],
                    slice_num_cor,
                ]
            )
            background_std = np.std(
                bssfp_data[
                    bg_pixels_y[0] : bg_pixels_y[1],
                    bg_pixels_x[0] : bg_pixels_x[1],
                    slice_num_cor,
                ]
            )

            axlist[2].vlines(background_val, 0, np.max(y_data), color="r", label="mean")
            axlist[2].legend()
            axlist[2].set_xlabel("Pixel value")
            axlist[2].set_ylabel("Num of Pix")
            axlist[2].set_title("Histogram")
            axlist[0].set_title("bssfp slice " + str(slice_num_cor))
            axlist[1].set_title(
                "Coronal ref slice " + str(coronal_slice_corresponding_to_bssfp_slice)
            )
            save_background_val(self, background_val, background_std)
            print(background_val, background_std)

        # UI components

        num_histogram_bins_slider = widgets.IntSlider(
            min=1,
            value=10,
            max=75,
            description="Histogram bins",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        slice_num_cor_slider = widgets.IntSlider(
            min=0.0,
            value=slice_num_cor,
            max=bssfp_data.shape[2] - 1,
            description="Coronal slice number",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        cor_image_windowing_slider = widgets.IntRangeSlider(
            min=0,
            value=[np.min(coronal_object.seq2d), np.max(coronal_object.seq2d)],
            max=np.max(coronal_object.seq2d),
            description="Coronal img windowing",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        bg_pixels_x_slider = widgets.IntRangeSlider(
            min=0.0,
            value=[bg_pixels_x[0], bg_pixels_x[1]],
            max=bssfp_data.shape[1] - 1,
            description="Bg pixels x",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        bg_pixels_y_slider = widgets.IntRangeSlider(
            min=0.0,
            value=[bg_pixels_y[0], bg_pixels_y[1]],
            max=bssfp_data.shape[0] - 1,
            description="Bg pixels y",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )

        out = widgets.interactive_output(
            plot,
            {
                "slice_num_cor": slice_num_cor_slider,
                "bg_pixels_x": bg_pixels_x_slider,
                "bg_pixels_y": bg_pixels_y_slider,
                "num_hist_bins": num_histogram_bins_slider,
                "cor_image_window": cor_image_windowing_slider,
            },
        )
        main_ui = widgets.VBox(
            [
                slice_num_cor_slider,
                bg_pixels_x_slider,
                bg_pixels_y_slider,
                num_histogram_bins_slider,
                cor_image_windowing_slider,
            ]
        )
        display(main_ui, out)

    def find_sig_range_reps(
        self, dnp_bssfp_data, phip_bssfp_data, signal_range_dict, axlist=None
    ):
        # TODO: In the future one could rewrite this function so that it only loads one bssfp dataset and then
        #  comparing is then done in a second function that calls two times this function
        """
        Finds the repetitions where signal starts and stops for two bssfp datasets.

        This function was developed for the 2023 paper on comparing PHIP and DNP
        for HP pyruvate.
        This function assumes data of the shape: [ Echoes, Read, phase, slice, repetitions, channels]
        i.e.: dnp_bssfp_pv_complex_reco_combined_shift.shape = (1, 16, 12, 10, 250, 1)

        Parameters
        ----------

        dnp_bssfp_data: array-like
            Contains multi rep, 3D bssfp data, usually this is only pyruvate or lactate channel
        phip_bssfp_data:array-like
            Contains multi rep, 3D bssfp data, usually this is only pyruvate or lactate channel
        axlist: list
            contains subplots to plot into, i.e. [ax1,ax2,ax3]
        signal_range_dict: dict
            has two keys: "phip_signal_range_reps", "dnp_signal_range_reps"
            each of them has a list with two entries specifying the indices of the repetitions which should be normed

        Returns
        -------

        dnp_images[:,:,:,dnp_max_sign_rep]: dnp bssfp image stack at the highest signal repetition
            Has shape (Read, phase, slice)

        phip_images[:,:,:,phip_max_sign_rep]: phip bssfp image stack at the highest signal repetition
            has shape (Read, phase, slice)

        Plot of the process and result
        Example
        ------

        >>> dnp_sig_reps,phip_sig_reps = find_sig_range_reps(dnp_bssfp_pv_complex_reco_combined_shift,phip_bssfp_pv_complex_reco_combined_shift)
        """
        if axlist is None:
            logger.critical(
                "Need to pass axlist (use fig, (ax1, ax2, ax3) = plt.subplots(1, 3) \
                        to generate and call function with: \
                        axlist=[ax1, ax2, ax3])"
            )
            return False
        if signal_range_dict is None:
            logger.critical("Need to pass a signal range dictionary")
            return False
        else:
            setattr(self, "original_dict", deepcopy(signal_range_dict))
            setattr(self, "signal_range_dict", signal_range_dict)

        # checking if we have echoes or more than one channel
        if phip_bssfp_data.shape[0] == 1:
            pass
        else:
            logger.critical(
                "phip_bssfp_data.shape[0] is larger than 1, this function only works for single echo data"
            )
        if phip_bssfp_data.shape[5] == 1:
            pass
        else:
            logger.critical(
                "phip_bssf_data.shape[5] is larger than 1, this function only works for single channel data"
            )

        if dnp_bssfp_data.shape[0] == 1:
            pass
        else:
            logger.critical(
                "dnp_bssfp_data.shape[0] is larger than 1, this function only works for single echo data"
            )
        if dnp_bssfp_data.shape[5] == 1:
            pass
        else:
            logger.critical(
                "dnp_bssfp_data.shape[5] is larger than 1, this function only works for single channel data"
            )

        # shorten the array to remove echoes and multi channel dimensions
        phip_images = np.squeeze(phip_bssfp_data)
        dnp_images = np.squeeze(dnp_bssfp_data)
        global itgl_dnp, itgl_phip
        itgl_dnp = []
        itgl_phip = []
        for rep in range(phip_images.shape[3]):
            itgl_phip.append(np.sum(np.abs(phip_images[:, :, :, rep])))
            itgl_dnp.append(np.sum(np.abs(dnp_images[:, :, :, rep])))
        # max sig rep for phip and dnp
        max_sig_rep_phip = np.squeeze(np.argmin(np.abs(itgl_phip - np.max(itgl_phip))))
        max_sig_rep_dnp = np.squeeze(np.argmin(np.abs(itgl_dnp - np.max(itgl_dnp))))

        # plot results
        def plot_sig_range(dnp_sig_range, phip_sig_range):
            for ax in axlist:
                ax.clear()

            axlist[0].plot(itgl_dnp, label="dnp")
            axlist[0].plot(itgl_phip, label="phip")

            axlist[0].axvline(
                dnp_sig_range[0], 0, np.max(itgl_dnp), color="C0", linestyle="dashed"
            )
            axlist[0].axvline(
                dnp_sig_range[1], 0, np.max(itgl_dnp), color="C0", linestyle="dashed"
            )

            axlist[0].axvline(
                phip_sig_range[0], 0, np.max(itgl_phip), color="C1", linestyle="dashed"
            )
            axlist[0].axvline(
                phip_sig_range[1], 0, np.max(itgl_phip), color="C1", linestyle="dashed"
            )

            axlist[0].legend()
            axlist[0].set_xlabel("Repetition")
            axlist[0].set_ylabel("Sum over all pixels and slices")

            axlist[1].imshow(
                np.sum(
                    np.abs(phip_images[:, :, :, phip_sig_range[0] : phip_sig_range[1]]),
                    axis=(2, 3),
                )
            )
            axlist[2].imshow(
                np.sum(
                    np.abs(dnp_images[:, :, :, dnp_sig_range[0] : dnp_sig_range[1]]),
                    axis=(2, 3),
                )
            )
            axlist[1].set_title("PHIP")
            axlist[2].set_title("DNP")
            axlist[0].grid(color="k", linestyle="-", linewidth=0.25, which="major")
            axlist[0].grid(color="k", linestyle="--", linewidth=0.25, which="minor")

            # return dnp_images[:,:,:,dnp_sig_range[0]:dnp_sig_range[1]],phip_images[:,:,:,phip_sig_range[0]:phip_sig_range[1]]

        # check if the dropdown value changes and then update all
        def dropdown_eventhandler(change):
            if str(change.new) == "20 reps around peak maxima":
                self.signal_range_dict.update(
                    {"phip": [max_sig_rep_phip - 10, max_sig_rep_phip + 10]}
                )
                self.signal_range_dict.update(
                    {"dnp": [max_sig_rep_dnp - 10, max_sig_rep_dnp + 10]}
                )
            elif str(change.new) == "2 before, 18 after peak maxima":
                self.signal_range_dict.update(
                    {"phip": [max_sig_rep_phip - 2, max_sig_rep_phip + 18]}
                )
                self.signal_range_dict.update(
                    {"dnp": [max_sig_rep_dnp - 2, max_sig_rep_dnp + 18]}
                )

            elif str(change.new) == "Manual setting":
                self.signal_range_dict.update(self.original_dict)

            else:
                pass
            phip_slider_signal_range_reps.value = self.signal_range_dict["phip"]
            dnp_slider_signal_range_reps.value = self.signal_range_dict["dnp"]
            plot_sig_range(
                [self.signal_range_dict["dnp"][0], self.signal_range_dict["dnp"][1]],
                [self.signal_range_dict["phip"][0], self.signal_range_dict["phip"][1]],
            )

        options_dropdown = widgets.Dropdown(
            options=[
                "Manual setting",
                "20 reps around peak maxima",
                "2 before, 18 after peak maxima",
            ],
            value="Manual setting",
        )
        options_dropdown.observe(dropdown_eventhandler)

        # check if slider value changes and then update the dictionary
        def slider_change_eventhandler(change):
            self.signal_range_dict.update({"phip": phip_slider_signal_range_reps.value})
            self.signal_range_dict.update({"dnp": dnp_slider_signal_range_reps.value})

        phip_slider_signal_range_reps = widgets.IntRangeSlider(
            min=0.0,
            value=[
                self.signal_range_dict["phip"][0],
                self.signal_range_dict["phip"][1],
            ],
            max=phip_images.shape[3] - 1,
            description="PHIP Signal range:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        dnp_slider_signal_range_reps = widgets.IntRangeSlider(
            min=0.0,
            value=[self.signal_range_dict["dnp"][0], self.signal_range_dict["dnp"][1]],
            max=dnp_images.shape[3] - 1,
            description="DNP Signal range:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        phip_slider_signal_range_reps.observe(slider_change_eventhandler)
        dnp_slider_signal_range_reps.observe(slider_change_eventhandler)

        out = widgets.interactive_output(
            plot_sig_range,
            {
                "dnp_sig_range": dnp_slider_signal_range_reps,
                "phip_sig_range": phip_slider_signal_range_reps,
            },
        )
        main_ui = widgets.VBox(
            [
                dnp_slider_signal_range_reps,
                phip_slider_signal_range_reps,
                options_dropdown,
            ]
        )
        display(main_ui, out)

    def find_high_snr_slices(
        self,
        image_data,
        snr_per_slice,
        axlist=None,
        slice_default_indices_cor=None,
        slice_default_indices_sag=None,
        slice_default_indices_ax=None,
    ):
        """
        Finds the slices where there is sufficient signal to compute a SSI for a bssfp dataset.

        This function was developed for the 2023 paper on comparing PHIP and DNP
        for HP pyruvate.

        Parameters
        ----------
        image_data: np.ndarray
            shape (read, phase, slice), contains already preprocessed bssfp magnitude data for displaying
        snr_per_slice: dict
            contains snr per slice for three keys: coronal, axial, sagittal
        axlist: list
            contains subplots to plot into, i.e. [ax1,ax2,ax3]
        slice_indices: list, default None.
            if this dataset was already processed, one can hard code the correct slices here

        Returns
        -------

        """
        if axlist is None:
            logger.critical(
                "Need to pass axlist (use fig, (ax1,ax2,ax3) = plt.subplots(1, 3) \
                        to generate and call function with: \
                        axlist=[ax1,ax2,ax3]])"
            )
            return False
        setattr(self, "slice_signal_coronal", np.zeros((2, 1)))
        setattr(self, "slice_signal_axial", np.zeros((2, 1)))
        setattr(self, "slice_signal_sagittal", np.zeros((2, 1)))

        def plot_slice_range(
            slice_indices_cor, slice_indices_ax, slice_indices_sag, orientation
        ):
            for ax in axlist:
                ax.clear()
            if orientation == "coronal":
                axlist[0].plot(snr_per_slice["coronal"])
                axlist[0].axvline(
                    slice_indices_cor[0],
                    0,
                    np.max(snr_per_slice["coronal"]),
                    color="C0",
                    linestyle="dashed",
                )
                axlist[0].axvline(
                    slice_indices_cor[1],
                    0,
                    np.max(snr_per_slice["coronal"]),
                    color="C0",
                    linestyle="dashed",
                )

                self.slice_signal_coronal[0] = slice_indices_cor[0]
                self.slice_signal_coronal[1] = slice_indices_cor[1]

                axlist[0].set_title("Coronal")
                axlist[1].imshow(image_data[:, :, slice_indices_cor[0]])
                axlist[2].imshow(image_data[:, :, slice_indices_cor[1]])

            elif orientation == "sagittal":
                axlist[0].plot(snr_per_slice["sagittal"])
                axlist[0].axvline(
                    slice_indices_sag[0],
                    0,
                    np.max(snr_per_slice["sagittal"]),
                    color="C0",
                    linestyle="dashed",
                )
                axlist[0].axvline(
                    slice_indices_sag[1],
                    0,
                    np.max(snr_per_slice["sagittal"]),
                    color="C0",
                    linestyle="dashed",
                )

                self.slice_signal_sagittal[0] = slice_indices_sag[0]
                self.slice_signal_sagittal[1] = slice_indices_sag[1]

                axlist[0].set_title("sagittal")
                axlist[1].imshow(image_data[:, slice_indices_sag[0], :])
                axlist[2].imshow(image_data[:, slice_indices_sag[1], :])

            elif orientation == "axial":
                axlist[0].plot(snr_per_slice["axial"])
                axlist[0].axvline(
                    slice_indices_ax[0],
                    0,
                    np.max(snr_per_slice["axial"]),
                    color="C0",
                    linestyle="dashed",
                )
                axlist[0].axvline(
                    slice_indices_ax[1],
                    0,
                    np.max(snr_per_slice["axial"]),
                    color="C0",
                    linestyle="dashed",
                )

                self.slice_signal_axial[0] = slice_indices_ax[0]
                self.slice_signal_axial[1] = slice_indices_ax[1]

                axlist[0].set_title("axial")
                axlist[1].imshow(image_data[slice_indices_ax[0], :, :])
                axlist[2].imshow(image_data[slice_indices_ax[1], :, :])

            axlist[1].set_title("First slice ")
            axlist[2].set_title("Last slice")

            axlist[0].set_ylabel("Sum over all pixels and slices")
            axlist[0].set_xlabel("Slice")
            axlist[0].grid(color="k", linestyle="-", linewidth=0.25, which="major")
            axlist[0].grid(color="k", linestyle="--", linewidth=0.25, which="minor")

        # interactive input UI functions
        if slice_default_indices_cor:
            signal_range_slices_slider_cor = widgets.IntRangeSlider(
                min=0.0,
                value=[slice_default_indices_cor[0], slice_default_indices_cor[1]],
                max=image_data.shape[2] - 1,
                description="Coronal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_coronal = slice_default_indices_cor
        else:
            signal_range_slices_slider_cor = widgets.IntRangeSlider(
                min=0.0,
                value=[0, image_data.shape[2] - 1],
                max=image_data.shape[2] - 1,
                description="Coronal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_coronal = [0, image_data.shape[2] - 1]
        if slice_default_indices_ax:
            signal_range_slices_slider_ax = widgets.IntRangeSlider(
                min=0.0,
                value=[slice_default_indices_ax[0], slice_default_indices_ax[1]],
                max=image_data.shape[0] - 1,
                description="Axial slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_axial = slice_default_indices_ax
        else:
            signal_range_slices_slider_ax = widgets.IntRangeSlider(
                min=0.0,
                value=[0, image_data.shape[0] - 1],
                max=image_data.shape[0] - 1,
                description="Axial slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_axial = [0, image_data.shape[0] - 1]
        if slice_default_indices_sag:
            signal_range_slices_slider_sag = widgets.IntRangeSlider(
                min=0.0,
                value=[slice_default_indices_sag[0], slice_default_indices_sag[1]],
                max=image_data.shape[1] - 1,
                description="Sagittal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_sagittal = slice_default_indices_sag
        else:
            signal_range_slices_slider_sag = widgets.IntRangeSlider(
                min=0.0,
                value=[0, image_data.shape[1] - 1],
                max=image_data.shape[1] - 1,
                description="Sagittal slice range:",
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
            )
            self.slice_signal_sagittal = [0, image_data.shape[1] - 1]
        orientation_rb = widgets.RadioButtons(
            options=["coronal", "axial", "sagittal"],
            value="coronal",
            description="Orientation",
            disabled=False,
        )

        out = widgets.interactive_output(
            plot_slice_range,
            {
                "orientation": orientation_rb,
                "slice_indices_cor": signal_range_slices_slider_cor,
                "slice_indices_ax": signal_range_slices_slider_ax,
                "slice_indices_sag": signal_range_slices_slider_sag,
            },
        )
        first_ui = widgets.VBox([orientation_rb])
        second_ui = widgets.VBox(
            [
                signal_range_slices_slider_cor,
                signal_range_slices_slider_ax,
                signal_range_slices_slider_sag,
            ]
        )
        main_ui = widgets.HBox([first_ui, second_ui])
        display(main_ui, out)

    def find_noise_region(self):
        """This function is supposed to find a region containing no signal that can be used as a reference of noise"""
        return True

    # def load_plot_params(self, path_to_params=None):
    #     """
    #     this function lets you load in the plotting parameters for the plot3_new2 function
    #     """
    #     import json
    #
    #     # Check if path to file was passed:
    #     if path_to_params is None:
    #         Warning("path_to_text has to be defined")
    #         return
    #
    #     # get path to folder on above acquisition data::
    #     path_parentfolder = str(self.path)[0 : str(self.path).rfind("\\")]
    #     # try to load file:
    #     try:
    #         with open(os.path.join(path_parentfolder, path_to_params)) as f:
    #             plot_params = json.load(f)
    #         self.plot_params = plot_params
    #     except:
    #         Warning("loading plot parameters did not work")
    #         return
    #
    #     # check if bssfp has attribute plot_params, else create:
    #     if hasattr(self, "plot_params"):
    #         pass
    #     else:
    #         setattr(self, "plot_params", "")
    #     try:
    #         self.plot_params = plot_params
    #     except:
    #         Warning("Saving plot parameters did not work")
    #
    #     return plot_params

    # @deprecated("return_corresponding_anatomical_slice not usable anymore") # will be possible to use in 3.13
    def return_corresponding_anatomical_slice(
        self, coronal_image, slice_nr_cor, bssfp_data, plot_res=False
    ):
        """
        Finds the corresponding coronal slice of an anatomical reference to a specific bssfp scan.

        Parameters
        ----------
        coronal_image: BrukerExp instance
        slice_nr_cor: int
        bssfp_data: np.ndarray

        Returns
        -------
        cor_slice_ind: int
            slice index for anatomical reference image that corresponds to the slice_nr_cor input from a bssfp slice
        """
        dim_read_cor = coronal_image.method["PVM_Matrix"][0]  # was z
        dim_phase_cor = coronal_image.method["PVM_Matrix"][1]  # was y
        if coronal_image.method["PVM_SpatDimEnum"] == "<3D>":
            dim_slice_cor = coronal_image.method["PVM_Matrix"][2]  # was x
        else:
            dim_slice_cor = coronal_image.method["PVM_SPackArrNSlices"]  # was x

        # FOV:
        mm_read_cor = coronal_image.method["PVM_Fov"][0]
        mm_phase_cor = coronal_image.method["PVM_Fov"][1]
        mm_slice_gap_cor = coronal_image.method["PVM_SPackArrSliceGap"]

        if coronal_image.method["PVM_SpatDimEnum"] == "<3D>":
            mm_slice_cor = coronal_image.method["PVM_Fov"][2]
        else:
            mm_slice_cor = coronal_image.method["PVM_SliceThick"]  # was x
            mm_slice_cor = mm_slice_cor * dim_slice_cor + mm_slice_gap_cor * (
                dim_slice_cor - 1
            )

        dim_read_bssfp = bssfp_data.shape[1]  # was z
        dim_phase_bssfp = bssfp_data.shape[2]  # was y
        dim_slice_bssfp = bssfp_data.shape[3]  # was x

        mm_read_bssfp = self.method["PVM_Fov"][0]
        mm_phase_bssfp = self.method["PVM_Fov"][1]
        mm_slice_bssfp = self.method["PVM_Fov"][2]

        cor_grid = self.define_grid(
            mat=np.array((dim_read_cor, dim_phase_cor, dim_slice_cor)),
            fov=np.array((mm_read_cor, mm_phase_cor, mm_slice_cor)),
        )

        bssfp_grid = self.define_grid(
            mat=np.array((dim_read_bssfp, dim_phase_bssfp, dim_slice_bssfp)),
            fov=np.array((mm_read_bssfp, mm_phase_bssfp, mm_slice_bssfp)),
        )
        coronal_image = coronal_image.seq2d

        if slice_nr_cor >= dim_slice_bssfp:
            logger.critical(
                "Input coronal bssfp slice number cannot exceed " + str(dim_slice_bssfp)
            )
            return None
        else:
            # input slice nr from bssfp slices
            ax_slice_ind, cor_slice_ind = self.find_closest_slice(
                axial_anat_grid=None,
                coronal_anat_grid=cor_grid,
                bssfp_grid=bssfp_grid,
                bssfp_pos=[
                    None,
                    None,
                    slice_nr_cor,
                ],
            )
            if plot_res:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(
                    np.sum(np.abs(bssfp_data[0, :, :, slice_nr_cor, :, 0]), axis=2)
                )
                ax[1].imshow(coronal_image[:, :, cor_slice_ind], cmap="bone")
                ax[0].set_title("Bssfp, coronal direction, slice " + str(slice_nr_cor))
                ax[1].set_title("Coronal image, slice " + str(cor_slice_ind))
            return cor_slice_ind

    def plot3D(self, axlist, fig, parameter_dict=None, display_ui=True):
        """A helper function for an interactive image plot.

        Recommended usage:
            import matplotlib.pyplot as plt
            from hypermri.sequences import BSSFP
            from IPython.display import display

            # create  BSSFP object
            bSSFP1 = BSSFP(...)

            # create figure and axes objects to plot into
            fig, (ax1,ax2,ax3) = plt.subplots(ncols=3)

            # plot mge data into given axes object, provide custom parameter
            plot_params = {'cmap': 'viridis'}
            bSSFP1.plot3D([ax1,ax2,ax3], plot_params)

            # note: you can still add stuff to the axes object afterwards
            ax1.set_title("Axial")
            ax2.set_title("Coronal")
            ax3.set_title("sagittal")

            # render plot
            plt.show()

        Parameters
        ----------
        axlist : list of Axes
            The axes to draw to. Recommended to generate using the command:
            'fig, ax = plt.subplots()'.
            3D plot requires 3 axes. 1st axial, 2nd coronal, 3rd sagittal

        param_dict : dict
            Dictionary of keyword arguments to pass to ax.imshow.

        display_ui: bool
            Default is True. Directly calls IPython.display on the interactive
            sliders and renders them. If False, returns the ui object instead of
            displaying it.

        Returns
        ----------
        ui : Hbox, optional
            The ipywidgets HBox containing the interactive sliders is returned
            only if the option 'display_ui' is set to False.
        """
        # check if there is data to plot
        if not isinstance(self.Reconstructed_data, np.ndarray):
            logger.debug(
                "There is no reconstructed data available. Please call the function BSSFP.reconstruction first!"
            )

        else:
            if self.method["PVM_SpatDimEnum"] != "<3D>":
                logger.debug("3D print function but no 3D data available!")
            else:
                # read image dimensions
                dim_read = self.method["PVM_Matrix"][0]
                dim_phase = self.method["PVM_Matrix"][1]
                dim_slice = self.method["PVM_Matrix"][2]

                # use default paramter dictionary if none is provided
                if not parameter_dict:
                    parameter_dict = {"cmap": "bone"}

                # define image plotting function
                def plot_img(slice_nr_ax, slice_nr_cor, slice_nr_sag, echo_nr):
                    if len(axlist) != 3:
                        logger.debug(
                            "Not enough ax objects in axlist for 3D plot, please give 3 axes!"
                        )
                    else:
                        for ax in axlist:
                            ax.clear()

                        axlist[0].imshow(
                            np.rot90(
                                np.abs(
                                    self.Reconstructed_data[
                                        echo_nr, slice_nr_ax, :, :, 0, 0
                                    ]
                                ),
                                3,
                            ),
                            extent=[
                                (dim_phase // 2),
                                -(dim_phase // 2),
                                (dim_slice // 2),
                                -(dim_slice // 2),
                            ],
                            **parameter_dict,
                        )
                        axlist[0].set_aspect(dim_slice / dim_phase)
                        axlist[0].axvline(
                            slice_nr_sag - (dim_slice // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[0].axhline(
                            slice_nr_cor - (dim_phase // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[0].set_xlabel("phase")
                        axlist[0].set_ylabel("slice")
                        axlist[0].set_title("Axial")

                        axlist[1].imshow(
                            np.rot90(
                                np.abs(
                                    self.Reconstructed_data[
                                        echo_nr, :, :, slice_nr_cor, 0, 0
                                    ]
                                ),
                                2,
                            ),
                            extent=[
                                dim_phase // 2,
                                -dim_phase // 2,
                                -dim_read // 2,
                                dim_read // 2,
                            ],
                            **parameter_dict,
                        )
                        axlist[1].set_aspect(dim_phase / dim_read)
                        axlist[1].axhline(
                            slice_nr_ax - (dim_read // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[1].axvline(
                            slice_nr_sag - (dim_phase // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[1].set_xlabel("phase")
                        axlist[1].set_ylabel("read")
                        axlist[1].set_title("Coronal")

                        axlist[2].imshow(
                            np.rot90(
                                np.abs(
                                    self.Reconstructed_data[
                                        echo_nr, :, slice_nr_sag, :, 0, 0
                                    ]
                                ),
                                2,
                            ),
                            extent=[
                                dim_slice // 2,
                                -dim_read // 2,
                                -dim_read // 2,
                                dim_read // 2,
                            ],
                            **parameter_dict,
                        )
                        axlist[2].set_aspect(dim_slice / dim_read)
                        axlist[2].axhline(
                            slice_nr_ax - (dim_read // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[2].axvline(
                            slice_nr_cor - (dim_slice // 2),
                            color="white",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        axlist[2].set_xlabel("slice")
                        axlist[2].set_ylabel("read")
                        axlist[2].set_title("sagittal")

                # create interactive sliders for  slices in all dimensions
                slice_slider_ax = widgets.IntSlider(
                    value=dim_read // 2,
                    min=0,
                    max=dim_read - 1,
                    description="Slice Axial: ",
                )
                slice_slider_cor = widgets.IntSlider(
                    value=dim_phase // 2,
                    min=0,
                    max=dim_phase - 1,
                    description="Slice Coronal: ",
                )
                slice_slider_sag = widgets.IntSlider(
                    value=dim_slice // 2,
                    min=0,
                    max=dim_slice - 1,
                    description="Slice sagittal: ",
                )

                # create interactive slider for echoes
                echo_slider = widgets.IntSlider(
                    value=0, min=0, max=self.Nechoes - 1, description="Echo: "
                )

                # put both sliders inside a HBox for nice alignment  etc.
                ui_ax = widgets.VBox(
                    [slice_slider_ax, echo_slider],
                    layout=widgets.Layout(display="flex"),
                )

                ui_cor = widgets.VBox(
                    [slice_slider_cor],
                    layout=widgets.Layout(display="flex"),
                )

                ui_sag = widgets.VBox(
                    [slice_slider_sag],
                    layout=widgets.Layout(display="flex"),
                )

                main_ui = widgets.HBox((ui_ax, ui_cor, ui_sag))

                # connect plotting function with sliders
                # Note: The output is unused here because it is sufficient to display
                #       the HBox (stored in ui variable). Without the ui display you
                #       would have to call the display() function on the output of
                #       widgets.interactive_output().
                out = widgets.interactive_output(
                    plot_img,
                    {
                        "slice_nr_ax": slice_slider_ax,
                        "slice_nr_cor": slice_slider_cor,
                        "slice_nr_sag": slice_slider_sag,
                        "echo_nr": echo_slider,
                    },
                )

                # This displays the Hbox containing the slider widgets.
                if display_ui:
                    display(main_ui, out)
                else:
                    return main_ui

    def plot3D_new(
        self,
        axlist,
        anat_object=None,
        input_data=None,
        parameter_dict=None,
        display_ui=True,
        pyr_ind=0,
    ):
        """A helper function for an interactive image plot.

        Recommended usage:
            import matplotlib.pyplot as plt
            from hypermri.sequences import BSSFP
            from IPython.display import display

            # create  BSSFP object
            bSSFP1 = BSSFP(...)

            # create figure and axes objects to plot into
            fig, (ax1,ax2,ax3) = plt.subplots(ncols=3)

            # plot mge data into given axes object, provide custom parameter
            plot_params = {'cmap': 'viridis'}
            bSSFP1.plot3D([ax1,ax2,ax3], plot_params)

            # note: you can still add stuff to the axes object afterwards
            ax1.set_title("Axial")
            ax2.set_title("Coronal")
            ax3.set_title("sagittal")

            # render plot
            plt.show()

        Parameters
        ----------
        axlist : list of Axes
            The axes to draw to. Recommended to generate using the command:
            'fig, ax = plt.subplots()'.
            3D plot requires 3 axes. 1st axial, 2nd coronal, 3rd sagittal
        anat_object : BrukerExp instance
            contains anatomical reference images, usually coronal.
        input_data: nd-array (None)
            nd-array to plot, data is supposed to sorted in the order
            Echoes, Read, phase, slice, repetitions, channels

        param_dict : dict
            Dictionary of keyword arguments to pass to ax.imshow.

        display_ui: bool
            Default is True. Directly calls IPython.display on the interactive
            sliders and renders them. If False, returns the ui object instead of
            displaying it.

        Returns
        ----------
        ui : Hbox, optional
            The ipywidgets HBox containing the interactive sliders is returned
            only if the option 'display_ui' is set to False.
        """
        from IPython.display import clear_output

        # check if there is data to plot
        if input_data is None:
            if not isinstance(self.Reconstructed_data, np.ndarray):
                logger.debug(
                    "There is no reconstructed data available. "
                    + "Please call the function BSSFP.reconstruction first!"
                )
            else:
                data = self.Reconstructed_data
        else:
            data = input_data

        if self.method["PVM_SpatDimEnum"] != "<3D>":
            logger.debug("3D print function but no 3D data available!")
        else:
            self.data_to_plot = data
            # read image dimensions
            dim_read = self.method["PVM_Matrix"][0]  # was z
            dim_phase = self.method["PVM_Matrix"][1]  # was y
            dim_slice = self.method["PVM_Matrix"][2]  # was x
            mm_read = self.method["PVM_Fov"][0]
            mm_phase = self.method["PVM_Fov"][1]
            mm_slice = self.method["PVM_Fov"][2]
            reps = self.method["PVM_NRepetitions"]
            chans = self.method["PVM_EncNReceivers"]
            slice_thick_cor = mm_slice / dim_slice
            slice_thick_ax = mm_read / dim_read
            slice_thick_sag = mm_phase / dim_phase
            read_orient = self.method["PVM_SPackArrReadOrient"]
            read_offset = self.method["PVM_SPackArrReadOffset"]
            phase_offset = self.method["PVM_SPackArrPhase1Offset"]
            slice_offset = self.method["PVM_SPackArrSliceOffset"]
            if not read_orient == "H_F":
                # checking if we have the right read orientation
                # if this error comes up you have to implement different orientations possibly
                logger.debug(
                    "Careful this function is not evaluated for this read orientation"
                )
            else:
                pass

            # if anat_object is None:
            #     # transform the input data
            #     (
            #         transformed_cor,
            #         self.data_to_plot,
            #         anat_cor_ext,
            #         bssfp_ext,
            #     ) = orient_coronal_and_bssfp_for_plot(anat_object, self, input_data)
            # else:
            #     pass
            anat_cor_ext = None

            # hardcode for now:
            transformed_cor = anat_object

            # use default paramter dictionary if none is provided
            if not parameter_dict:
                parameter_dict = {"cmap": "viridis"}

            # define image plotting function
            def plot_img(
                slice_nr_ax,
                slice_nr_cor,
                slice_nr_sag,
                echo_nr,
                rep1,
                chan,
                rep2=None,
                plot_style="Abs",
                experiment_style="Perfusion",
                pyr_lac="Pyr",
                anat_overlay="Metab",
            ):
                if len(axlist) < 3:
                    logger.debug(
                        "Not enough ax objects in axlist for 3D plot, please give 3 axes!"
                    )
                else:
                    # init range:
                    rep_range = rep1
                    # choose plotstyle:
                    if plot_style == "Abs":
                        bssfp_image_data = np.absolute(self.data_to_plot)
                    elif plot_style == "Real":
                        bssfp_image_data = np.real(self.data_to_plot)
                    elif plot_style == "Imag":
                        bssfp_image_data = np.imag(self.data_to_plot)
                    elif plot_style == "Phase":
                        bssfp_image_data = np.angle(self.data_to_plot)
                    else:
                        # default
                        bssfp_image_data = np.absolute(self.data_to_plot)

                    # init empty arrays:
                    global pyr_data, lac_data
                    # deterMine experiment style:
                    if experiment_style == "Perfusion":
                        pyr_data = bssfp_image_data[:, :, :, :, :, :]
                    elif experiment_style == "Pyr + Lac":
                        pyr_data = bssfp_image_data[:, :, :, :, pyr_ind::2, :]
                        lac_data = bssfp_image_data[:, :, :, :, pyr_ind + 1 :: 2, :]
                    else:
                        Warning(
                            "unknown measurement style "
                            + str(plot_opts_measurement.value)
                        )

                    # determine which contrast to use:
                    if pyr_lac == "Pyr":
                        bssfp_image_data = pyr_data
                    elif (pyr_lac == "Lac") and (experiment_style == "Pyr + Lac"):
                        bssfp_image_data = lac_data
                    elif (pyr_lac == "Lac/Pyr") and (experiment_style == "Pyr + Lac"):
                        bssfp_image_data = lac_data / pyr_data
                    else:
                        bssfp_image_data = pyr_data
                        Warning("unknown Contrast" + str(plot_opts_metabolite.value))

                    # takes mean over both repetitions:
                    if rep_avg_checkbox.value:
                        # valid range:
                        if rep2 > rep1:
                            rep_range = range(rep1, rep2)
                            # assuming the order Echoes, Read, phase, slice, repetitions, channels in the data:
                            bssfp_image_data = np.mean(
                                bssfp_image_data[:, :, :, :, rep_range, :],
                                axis=4,
                                keepdims=True,
                            )
                            rep_range = 0
                        # use start repetition
                        else:
                            rep_range = rep1
                    # take start repetition value:
                    else:
                        rep_range = rep1

                    # clear axis:
                    for ax in axlist:
                        ax.clear()

                    # axial view:
                    axlist[0].imshow(
                        np.squeeze(
                            (
                                bssfp_image_data[
                                    echo_nr, slice_nr_ax, :, :, rep_range, chan
                                ]
                            )
                        ),
                        extent=self.ax_ext,
                        **parameter_dict,
                    )
                    axlist[0].set_xlabel("mm (phase)")
                    axlist[0].set_ylabel("mm (slice)")

                    axlist[0].set_title("Axial")
                    # axlist[0].set_aspect(dim_y / dim_z)

                    axlist[0].axhline(
                        mm_slice / 2 - slice_nr_cor * slice_thick_cor,
                        color="white",
                        linewidth=1,
                        linestyle="solid",
                    )
                    axlist[0].axvline(
                        mm_phase / 2 - slice_nr_sag * slice_thick_sag,
                        color="white",
                        linewidth=1,
                        linestyle="dashed",
                    )

                    # sagittal view:
                    axlist[1].imshow(
                        np.squeeze(
                            (
                                bssfp_image_data[
                                    echo_nr, :, slice_nr_sag, :, rep_range, chan
                                ]
                            )
                        ),
                        extent=self.sag_ext,
                        **parameter_dict,
                    )
                    axlist[1].set_xlabel("mm (read)")
                    axlist[1].set_ylabel("mm (slice)")

                    axlist[1].set_title("Sagittal")
                    # axlist[0].set_aspect(dim_y / dim_z)

                    axlist[1].axhline(
                        mm_slice / 2 - slice_nr_cor * slice_thick_cor,
                        color="white",
                        linewidth=1,
                        linestyle="solid",
                    )
                    axlist[1].axvline(
                        mm_read / 2 - slice_nr_ax * slice_thick_ax,
                        color="white",
                        linewidth=1,
                        linestyle="dashed",
                    )

                    # coronal view:

                    if anat_overlay == "Metab + Anat":
                        axlist[2].imshow(
                            np.squeeze((transformed_cor[:, :, slice_nr_cor])),
                            extent=anat_cor_ext,
                            cmap="bone",
                        )
                        # coronal bssfp data
                        axlist[2].imshow(
                            np.squeeze(
                                (
                                    bssfp_image_data[
                                        echo_nr, :, :, slice_nr_cor, rep_range, chan
                                    ]
                                )
                            ),
                            extent=self.cor_ext,
                            alpha=0.3,
                            **parameter_dict,
                        )
                        axlist[2].set_title("Coronal")
                        axlist[2].set_xlabel("mm")
                        axlist[2].set_ylabel("mm")
                    elif anat_overlay == "None":
                        axlist[2].imshow(
                            np.squeeze(
                                (
                                    bssfp_image_data[
                                        echo_nr, :, :, slice_nr_cor, rep_range, chan
                                    ]
                                )
                            ),
                            extent=self.cor_ext,
                            **parameter_dict,
                        )

                    elif anat_overlay == "Anat":
                        axlist[2].imshow(
                            np.squeeze((transformed_cor[:, :, slice_nr_cor])),
                            extent=anat_cor_ext,
                            cmap="bone",
                        )

                    # axlist[0].set_aspect(dim_y / dim_z)
                    axlist[2].set_xlabel("mm (phase)")
                    axlist[2].set_ylabel("mm (read)")
                    axlist[2].set_title("Coronal")
                    axlist[2].axhline(
                        mm_read / 2 - slice_nr_ax * slice_thick_ax,
                        color="white",
                        linewidth=1,
                        linestyle="solid",
                    )
                    axlist[2].axvline(
                        mm_phase / 2 - slice_nr_sag * slice_thick_sag,
                        color="white",
                        linewidth=1,
                        linestyle="dashed",
                    )

                    # plot time curve
                    if len(axlist) > 3:
                        if pyr_lac == "Pyr":
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=2,
                            )
                            # only possible if pyruvate and lactate were measured
                            if experiment_style == "Pyr + Lac":
                                axlist[3].plot(
                                    np.squeeze(
                                        (
                                            lac_data[
                                                echo_nr,
                                                slice_nr_ax,
                                                slice_nr_sag,
                                                slice_nr_cor,
                                                :,
                                                chan,
                                            ]
                                        )
                                    ),
                                    linewidth=1,
                                )
                        # only possible if pyruvate and lactate were measured
                        elif (experiment_style == "Pyr + Lac") and (pyr_lac == "Lac"):
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=1,
                            )
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        lac_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=2,
                            )
                        elif (experiment_style == "Pyr + Lac") and (
                            pyr_lac == "Lac/Pyr"
                        ):
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        lac_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                )
                                / np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=1,
                            )
                        else:
                            Warning("dont know what to do")
                            axlist[3].plot(
                                np.squeeze(
                                    (
                                        pyr_data[
                                            echo_nr,
                                            slice_nr_ax,
                                            slice_nr_sag,
                                            slice_nr_cor,
                                            :,
                                            chan,
                                        ]
                                    )
                                ),
                                linewidth=2,
                            )

                        axlist[3].axvline(
                            rep1,
                            color="black",
                            linewidth=1,
                            linestyle="dashed",
                        )
                        # draw 2nd line:
                        if (rep2 > rep1) and (rep_avg_checkbox.value):
                            axlist[3].axvline(
                                rep2,
                                color="black",
                                linewidth=1,
                                linestyle="dashed",
                            )

            global rep_avg_checkbox_option
            # create interactive sliders for  slices in all dimensions
            rep_avg_checkbox = widgets.Checkbox(description="Mean Reps")
            rep_avg_checkbox_option = widgets.VBox(
                layout=widgets.Layout(display="flex")
            )

            plot_opts_anat_overlay = widgets.RadioButtons(
                options=["Metab", "Metab + Anat", "Anat"],
                value="Metab",
                description="Anat overlay",
                disabled=False,
                layout=widgets.Layout(width="15%"),
            )

            # create interactive slider for echoes
            echo_slider = widgets.IntSlider(
                value=0, min=0, max=self.Nechoes - 1, description="Echo: "
            )

            slice_slider_rep1 = widgets.IntSlider(
                value=0,
                min=0,
                max=reps - 1,
                description="Rep. Start: ",
            )
            slice_slider_rep2 = widgets.IntSlider(
                value=0,
                min=0,
                max=reps - 1,
                description="Rep. End: ",
            )
            # set the minimum of the end reptition to the value of the start reptition
            widgetLink = widgets.jslink(
                (slice_slider_rep1, "value"), (slice_slider_rep2, "min")
            )

            plot_opts_part = widgets.RadioButtons(
                options=["Abs", "Real", "Imag", "Phase"],
                value="Abs",
                description="Plot style:",
                disabled=False,
                layout=widgets.Layout(width="10%"),
            )

            plot_opts_metabolite = widgets.RadioButtons(
                options=["Pyr", "Lac", "Lac/Pyr"],
                value="Pyr",
                description="Metabolite:",
                disabled=True,  # activate if plot_opts_measurement is set to Pyr+Lac
                layout=widgets.Layout(width="10%"),
            )

            plot_opts_measurement = widgets.RadioButtons(
                options=["Perfusion", "Pyr + Lac"],
                value="Perfusion",  # Defaults to 'pineapple'
                description="Experiment:",
                disabled=False,
                layout=widgets.Layout(width="10%"),
            )

            # init sliders:
            slice_slider_ax = widgets.IntSlider(
                value=mm_read // 2,
                min=0,
                max=dim_read - 1,
                description="Slice Axial: ",
            )
            slice_slider_cor = widgets.IntSlider(
                value=mm_slice // 2,
                min=0,
                max=dim_slice - 1,
                description="Slice Coronal: ",
            )
            slice_slider_sag = widgets.IntSlider(
                value=mm_phase // 2,
                min=0,
                max=dim_phase - 1,
                description="Slice sagittal: ",
            )
            slice_slider_chan = widgets.IntSlider(
                value=0,
                min=0,
                max=chans - 1,
                description="Channel: ",
            )

            def set_rep_range(args):
                """
                Sets repetition range and contrast options (Pyr, Lac, ...) depending
                on the experiment type
                """
                # changed from perfusion to pyr+lac
                if args["new"] == "Pyr + Lac":
                    slice_slider_rep1.max = np.ceil(reps / 2) - 1
                    slice_slider_rep2.max = np.ceil(reps / 2) - 1

                    slice_slider_rep1.value = np.ceil(slice_slider_rep1.value / 2)
                    slice_slider_rep2.value = np.ceil(slice_slider_rep2.value / 2)

                    plot_opts_metabolite.disabled = False
                elif args["new"] == "Perfusion":
                    slice_slider_rep1.max = reps - 1
                    slice_slider_rep2.max = reps - 1

                    slice_slider_rep1.value = slice_slider_rep1.value * 2
                    slice_slider_rep2.value = slice_slider_rep2.value * 2

                    plot_opts_metabolite.value = "Pyr"
                    plot_opts_metabolite.disabled = True
                else:
                    Warning("Unkown experiment type" + str(args["new"]))

            # change options depending on experiment type:
            plot_opts_measurement.observe(set_rep_range, names="value")

            # generate boxes:
            ui_sag = widgets.VBox(
                [
                    slice_slider_sag,
                    slice_slider_rep1,
                    slice_slider_rep2,
                    rep_avg_checkbox,
                    plot_opts_anat_overlay,
                    plot_opts_measurement,
                ],
                layout=widgets.Layout(width="33%"),
            )
            ui_ax = widgets.VBox(
                [slice_slider_ax, echo_slider, plot_opts_part],
                layout=widgets.Layout(width="33%"),
            )
            ui_cor = widgets.VBox(
                [slice_slider_cor, slice_slider_chan, plot_opts_metabolite],
                layout=widgets.Layout(width="33%"),
            )

            # def plot_ui(args):
            clear_output()
            # connect plotting function with sliders
            # Note: The output is unused here because it is sufficient to display
            #       the HBox (stored in ui variable). Without the ui display you
            #       would have to call the display() function on the output of
            #       widgets.interactive_output().def update_ui
            out = widgets.interactive_output(
                plot_img,
                {
                    "slice_nr_ax": slice_slider_ax,
                    "slice_nr_cor": slice_slider_cor,
                    "slice_nr_sag": slice_slider_sag,
                    "echo_nr": echo_slider,
                    "rep1": slice_slider_rep1,
                    "chan": slice_slider_chan,
                    "rep2": slice_slider_rep2,
                    "plot_style": plot_opts_part,
                    "experiment_style": plot_opts_measurement,
                    "pyr_lac": plot_opts_metabolite,
                    "anat_overlay": plot_opts_anat_overlay,
                },
            )
            main_ui = widgets.HBox([ui_ax, ui_cor, ui_sag])

            # This displays the Hbox containing the slider widgets.
            if display_ui:
                display(main_ui, out)
            else:
                return main_ui
