from .base_anatomy import BaseAnatomy

import numpy as np


class RARE(BaseAnatomy):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""

        super().__init__(path_or_BrukerExp)

        # save some handy method parameters for easier access
        self.pvm_fov = self.method["PVM_Fov"]
        self.pvm_res = self.method["PVM_SpatResol"]
        self.pvm_matrix = self.method["PVM_Matrix"]
        self.pvm_orientation = self.method["PVM_SPackArrSliceOrient"]
        self.pvm_readorientation = self.method["PVM_SPackArrReadOrient"]

        self.data = self.get_data()

    def get_data(self):
        # if self.pvm_orientation == "coronal":
        #     out = np.squeeze(self.seq2d)
        #     out = out[..., 2:-2]
        #     print(out.shape)
        #     out = np.swapaxes(out, 1, 2)
        #     print(out.shape)
        #     return np.flipud(np.rot90(out))
        return np.flipud(np.rot90(np.squeeze(self.seq2d)))

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
