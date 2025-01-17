from ..brukerexp import BrukerExp
from ..utils.utils_logging import LOG_MODES, init_default_logger
import numpy as np

logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")
from .base_anatomy import BaseAnatomy



class FLASH(BaseAnatomy):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""

        super().__init__(path_or_BrukerExp)
        self.TE = self.get_TE()
        self.data = self.get_data()

    def get_data(self):
        return np.flipud(np.rot90(np.squeeze(self.seq2d)))

    # These functions can be removed in future releases as they are already implemented in the BaseAnatomy class
    def get_TE(self):
        return self.method["PVM_EchoTime"]

    def get_Nechoes(self) -> int:
        return self.method["PVM_NEchoImages"]

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

        # globally change log level, default is critical
        logger.setLevel(LOG_MODES[log_mode])

    # def remove_0s_from_fid(
    #     self,
    #     input_data=None,
    #     force_remove=False,
    # ):
    #     """
    #     fid-file contains 0s, should probably be in fixed BrukerExp but
    #     for now is taken care of here
    #     Returns
    #     -------
    #
    #     """
    #     if input_data is None:
    #         fid = self.fid
    #
    #     else:
    #         fid = input_data
    #
    #     if self.method["PVM_SpatDimEnum"] == "<2D>":
    #         # only tested for symmetric matrix, so [0] and [1] could be switched
    #         fid = np.reshape(
    #             fid, [self.method["PVM_Matrix"][0], -1, self.method["PVM_Matrix"][1]]
    #         )
    #         # use only non-zero entries:
    #         fid_without_0s = fid[:, ::2, :]
    #         fid_without_0s = np.reshape(
    #             fid_without_0s,
    #             [
    #                 self.method["PVM_NRepetitions"],
    #                 self.method["PVM_Matrix"][0],
    #                 self.method["PVM_Matrix"][1],
    #             ],
    #         )
    #     else:
    #         logger.critical("removing 0s only implemented for 2D datasets yet :(")
    #         fid_without_0s = None
    #
    #     return fid_without_0s

    def reorder_kspace(self, input_data=None):
        """reorders and shapes the kspace data (from fid-file) to match the order echoes - z - x - y - reps - chans"""
        if input_data is None:
            input_data = self.remove_0s_from_fid()
        else:
            pass
        # data is expected to have shape [reps - x - y]
        if self.method["PVM_SpatDimEnum"] == "<2D>":
            k_space_array = np.reshape(
                input_data,
                [
                    1,
                    self.method["PVM_NRepetitions"],
                    self.method["PVM_Matrix"][0],
                    self.method["PVM_Matrix"][1],
                    1,
                    self.method["PVM_EncNReceivers"],
                ],
            )
            k_space_array = np.transpose(k_space_array, [0, 4, 3, 2, 1, 5])
            k_space_array = np.flip(np.flip(k_space_array, axis=2), axis=3)
        else:
            logger.critical("removing 0s only implemented for 2D datasets yet :(")

        self.kspace_array = k_space_array
        return k_space_array

    def reco_image(self, input_data=None):
        """
        Reconstructs an image from the raw data
        Parameters
        ----------
        input_data = k-space data (expected shape [echoes - z - x - y - reps - rxchannels])

        Returns
        -------
        flash_image: image-space data

        """
        from ..utils.utils_general import mri_fft

        # get fid data:
        if input_data is None:
            input_data = self.reorder_kspace()

        # perform 3D FT along z-x-y:
        flash_image = mri_fft(
            input_data=input_data,
            axes=(
                1,
                2,
                3,
            ),
        )
        self.flash_image_non_shift = flash_image
        return flash_image

    def shift_flash_image(self, input_data=None):
        """
        Shifts the from k-space data reconstructed image to it actual position
        Returns
        -------

        """
        from ..utils.utils_general import (
            get_extent,
            define_imageFOV_parameters,
            define_imagematrix_parameters,
        )

        if input_data is None:
            # get the image that was reconstructed from k-space:
            input_data = self.reco_image()

        # define resolution:
        res = [
            a / b
            for a, b in zip(
                define_imageFOV_parameters(data_obj=self),
                define_imagematrix_parameters(data_obj=self),
            )
        ]
        ext = get_extent(data_obj=self)

        # define offsets in [mm]
        off_x_mm = (ext[0][0] + ext[0][1]) / 2
        off_y_mm = (ext[0][2] + ext[0][3]) / 2
        off_z_mm = (ext[1][0] + ext[1][1]) / 2

        # define offsets in vox
        off_x = int(off_x_mm / res[1])
        off_y = int(off_y_mm / res[2])
        off_z = int(off_z_mm / res[0])

        for r in range(self.method["PVM_NRepetitions"]):
            for c in range(self.method["PVM_EncNReceivers"]):
                input_data[
                    0,
                    :,
                    :,
                    :,
                    r,
                    c,
                ] = np.roll(
                    np.roll(
                        np.roll(
                            input_data[
                                0,
                                :,
                                :,
                                :,
                                [r],
                                [c],
                            ],
                            off_z,
                            axis=1,
                        ),
                        off_x,
                        axis=2,
                    ),
                    off_y,
                    axis=3,
                )

        self.flash_image = input_data
        return input_data

    def get_extent(self, view):
        """Calculate the image extent for a given view.

              [:, :, x]         [x, :, :]         [:, x, :]
               b-----.           d-----.           b-----.
        main : |     |   proj1 : |     |   proj2 : |     |
               a,c---d           c,e---f           a,e---f

        Parameter
        ----------
        view : str {'main', 'proj1', 'proj2'}
            Returns extent of the image itself or one of its two projections.
        """
        if len(self.fov) == 2:
            read_ext, phase_ext = self.fov
            slice_ext = self.Nslices * self.method["PVM_SliceThick"]
        else:
            read_ext, phase_ext, slice_ext = self.fov

        a = -read_ext / 2  # - 0.5
        b = read_ext / 2  # - 0.5
        c = -phase_ext / 2  # - 0.5
        d = phase_ext / 2  # - 0.5
        e = -slice_ext / 2  # - 0.5
        f = slice_ext / 2  # - 0.5

        # extent is:
        # left, right, bottom, top
        if view == "main":
            return (c, d, a, b)
        if view == "proj1":
            return (e, f, c, d)
        if view == "proj2":
            return (e, f, a, b)
