from ..brukerexp import BrukerExp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_interactions import hyperslicer
from ..utils.utils_general import reorient_anat, get_extent
import warnings as warning


class BaseAnatomy(BrukerExp):
    """Experimental base class for anatomy classes."""

    def __init__(self, path_or_BrukerExpObj, *args, **kwargs):
        """Accepts directory path or BrukerExp object as input."""
        super().__init__(path_or_BrukerExpObj, *args, **kwargs)

        self.Nslices = self.get_Nslices()
        self.Nechoes = self.get_Nechoes()

        # write header file:
        from ..utils.utils_general import init_header

        self.header = init_header(data_object=self)
        self.fov = self.get_fov()
        self.matrix = self.get_matrix()
        self.resolution = self.get_resolution()
        self.reps = self.get_reps()
        self.chans = self.get_chans()
        self.extent = get_extent(data_obj=self)

        # reoriented seq2ds will be stored as seq2d_oriented in the data_objs to avoid repeated rotations:
        try:
            self.seq2d_oriented = reorient_anat(data_obj=self, overwrite_seq2d=True)
        except:
            warning.warn("Could not reorient seq2d.")
        # self.kspace = self.read_kspace(input_data=self.fid);

    def remove_0s_from_fid(
        self,
        input_data=None,
        force_remove=False,
    ):
        """
        fid-file contains 0s, should probably be fixed in BrukerExp but
        for now is taken care of here
        Returns
        -------
        fid_without_0s : np.ndarray
            FID without 0s.
        """
        if input_data is None:
            fid = self.fid
        else:
            fid = input_data
        mat = self.matrix
        fid_without_0s = np.reshape(fid, (-1, mat[0] * 2))[:, : mat[0]]

        if self.Nslices == 1:
            if self.reps > 1:
                fid_without_0s = np.reshape(
                    fid_without_0s, (self.reps, mat[1], self.chans, mat[0])
                )
                from ..utils.utils_spectroscopy import make_NDspec_6Dspec

                fid_without_0s = make_NDspec_6Dspec(
                    input_data=fid_without_0s, provided_dims=["reps", "y", "chans", "x"]
                )
        return fid_without_0s

    def read_kspace(self, input_data=None):
        fid = self.remove_0s_from_fid(input_data=input_data)
        return fid

    def get_Nslices(self) -> int:
        """Get number of slices from both 2D-multislice and 3D acquisitions."""
        SpatialDim = self.method["PVM_SpatDimEnum"]

        if SpatialDim == "<2D>":
            out = self.method["PVM_SPackArrNSlices"]
        elif SpatialDim == "<3D>":
            out = self.method["PVM_Matrix"][-1]
        else:
            raise NotImplementedError(f"PVM_SpatDimEnum = '{SpatialDim}'")

        return out

    def get_Nechoes(self) -> int:
        """Get number of acquired echoes (for MGE e.g.)."""
        return self.method.get("PVM_NEchoImages", 1)

    def get_matrix(self):
        """Get acquisition matrix."""
        mat = self.method["PVM_Matrix"]
        if self.method["PVM_SpatDimEnum"] == "<2D>":
            return [*mat, self.Nslices]
        else:
            return mat

    def get_fov(self):
        """Get acquisition field-of-view."""
        fov = self.method["PVM_Fov"]
        if self.method["PVM_SpatDimEnum"] == "<2D>":
            slice_thick = self.method["PVM_SliceThick"]
            return [*fov, slice_thick * self.Nslices]
        else:
            return fov

    def get_reps(self):
        "Get number of repetitions."
        reps = self.method["PVM_NRepetitions"]
        return reps

    def get_resolution(self):
        res = self.method["PVM_SpatResol"]
        if self.method["PVM_SpatDimEnum"] == "<2D>":
            slice_thick = self.method["PVM_SliceThick"]
            return [*res, slice_thick]
        else:
            return res

    def get_chans(self):
        """Get number of receive channels."""
        chans = self.method["PVM_EncNReceivers"]
        return chans

    def plot(self, img_axes=(0, 1), **kwargs):
        """Plot scan using mpl_interactive.hyperslicer.

        img_axes : tuple
            The two axes containing a single 2D image for imshow(). All other
            axes are given a slider widget.

        **kwargs :
            All kwargs will be passed to the hyperslicer first and then to the
            imshow command for the image.
        """
        fig, ax = plt.subplots()

        data = self.seq2d
        data = np.moveaxis(data, img_axes[1], -1)
        data = np.moveaxis(data, img_axes[0], -1)

        _ = hyperslicer(data, ax=ax, **kwargs)

        fig.tight_layout()

        return fig, ax

    def summary(self):
        """Return summary table of frequency related parameter as pd.dataframe."""

        # Frequency Table
        df = pd.DataFrame()

        paramlist = {
            "Reference Frequency [MHz]": "PVM_FrqRef",
            "Reference Frequency [ppm]": "PVM_FrqRefPpm",
            "Working Frequency [MHz]": "PVM_FrqWork",
            "Working Frequency [ppm]": "PVM_FrqWorkPpm",
            "Working Frequency Offset [Hz]": "PVM_FrqWorkOffset",
            "Working Frequency Offset [ppm]": "PVM_FrqWorkOffsetPpm",
            "-------------------------------" "Matrix size": "PVM_Matrix",
            "FOV [cm]": "PVM_Fov",
        }

        for description, key in paramlist.items():
            try:
                df[description] = list(self.method[key])
            except KeyError:
                pass

        cols = ["Reference Frequency [MHz]"]
        df[cols] = df[df[cols] > 0][cols]
        df = df.dropna().T

        df.columns = ["test"]
        styles = [
            {"selector": "th", "props": [("font-size", "109%")]},  # tabular headers
            {"selector": "td", "props": [("font-size", "107%")]},  # tabular data
        ]

        df_styler = df.style.set_table_styles(styles)

        from IPython.display import display

        display(df_styler, df_styler)
        # return df_styler, df_styler
