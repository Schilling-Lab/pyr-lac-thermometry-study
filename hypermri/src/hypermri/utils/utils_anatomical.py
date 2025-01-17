import warnings

from .utils_logging import init_default_logger
from IPython.display import display

import numpy as np
import ipywidgets as widgets
from skimage import measure
import matplotlib.pyplot as plt
from skimage.transform import resize
from mpl_interactions import image_segmenter
import os
from matplotlib_scalebar.scalebar import ScaleBar

logger = init_default_logger(__name__)

# see https://github.com/ianhi/mpl-interactions/pull/266
try:
    from mpl_interactions import image_segmenter_overlayed
except ImportError:
    logger.critical("Image_segmenter_overlayed not found")


def bruker2complex(arr):
    """Turns bruker semi-complex arrays (2xfloat) into numpy.complex (1xcomplex)"""
    assert arr.shape[-1] == 2, "Last dimension must be complex dimension."
    return arr[..., 0] + 1j * arr[..., 1]


def orient_coronal_and_bssfp_for_plot(coronal_object, bssfp_object, bssfp_reco):
    """
    Transforms coronal 2dseq and bssfp reco images to be plotted.
    Parameters
    ----------
    coronal_image
    bssfp_data

    Returns
    -------
    coronal_transformed:
    bssfp_transformed:
    coronal_extent:
    bssfp_extent:
    """
    read_orient = coronal_object.method["PVM_SPackArrReadOrient"]
    # checking if this function does actually work for the data provided
    read_offset = coronal_object.method["PVM_SPackArrReadOffset"]
    phase_offset = coronal_object.method["PVM_SPackArrPhase1Offset"]
    slice_offset = coronal_object.method["PVM_SPackArrSliceOffset"]
    # bssfp offsets
    read_offset_bssfp = bssfp_object.method["PVM_SPackArrReadOffset"]
    phase_offset_bssfp = bssfp_object.method["PVM_SPackArrPhase1Offset"]
    slice_offset_bssfp = bssfp_object.method["PVM_SPackArrSliceOffset"]

    if read_offset != 0:
        raise NotImplementedError("Read offset not implemented yet")
    else:
        pass
    if phase_offset != 0:
        raise NotImplementedError("Phase offset not implemented yet")
    else:
        pass
    if slice_offset != 0:
        raise NotImplementedError("Slice offset not implemented yet")
    else:
        pass

    if read_offset_bssfp != 0:
        raise NotImplementedError("Read offset not implemented yet")
    else:
        pass
    if phase_offset_bssfp != 0:
        raise NotImplementedError("Phase offset not implemented yet")
    else:
        pass
    if slice_offset_bssfp != 0:
        raise NotImplementedError("Slice offset not implemented yet")
    else:
        pass

    if coronal_object.method["PVM_SPackArrSliceOrient"] == "coronal":
        if read_orient == "H_F":
            coronal_transformed = np.transpose(coronal_object.seq2d, (1, 0, 2))
            bssfp_transformed = np.flip(
                np.flip(np.flip(bssfp_reco, axis=1), axis=2), axis=3
            )
            cor_fov = coronal_object.method["PVM_Fov"]
            bssfp_fov = bssfp_object.method["PVM_Fov"]

            coronal_extent = [
                -cor_fov[1] / 2,
                cor_fov[1] / 2,
                cor_fov[0] / 2,
                -cor_fov[0] / 2,
            ]
            bssfp_extent = [
                -bssfp_fov[1] / 2,
                bssfp_fov[1] / 2,
                bssfp_fov[0] / 2,
                -bssfp_fov[0] / 2,
            ]
        else:
            raise NotImplementedError(
                read_orient, " - This read orientation is not implemented"
            )
    else:
        raise NotImplementedError(
            coronal_object.method["PVM_SPackArrSliceOrient"],
            " - This anatomical image orientation is not implemented",
        )

    return coronal_transformed, bssfp_transformed, coronal_extent, bssfp_extent


def get_segmenter_list_overlayed(
    anatomical_data,
    bssfp_data,
    anat_ext=None,
    bssfp_ext=None,
    overlay=0.25,
    bssfp_cmap="viridis",
    n_rois=1,
    figsize=(4, 4),
    vmin=None,
    vmax=None,
    mask_alpha=0.7,
    rot_images_deg=0,
    flip_images=False,
    masks_drawn_on=None,
):
    """
    Returns a list of image_segmenter_overlayed type entries (coronal images). TO be read by draw_masks_on_anatomical
    Parameters
    ----------
    anatomical_data: np.array containing anatomical data
        shape: (read, phase, slice)
        Assumes that dim[2] is slice.
    bssfp_data: np.array containing bssfp data
        shape : (echos,read,phase,slice,reps,channels)
        Gets averaged over all reps per default. Can be changed later below.
    anat_ext : list, optional
        If bssfp and anatomical dont have the same extent you need to give this value.
        gives the extent of the anatomical image in mm for example: [-15,15,10,10]
    bssfp_ext : list, optional
        If bssfp and anatomical dont have the same extent you need to give this value.
        gives the extent of the bssfp image in mm for example: [-15,15,10,10]
    overlay : float
        alpha value of the secondary image, per default 0.25
    bssfp_cmap : str, optional
        bssfp colormap
    n_rois: int, optional
        number of rois to be segemented, so far limited to 1.
    rot_images_deg: float, optional
        by how many degree the plotted images should be rotated to match other plotting modalities
    Returns
    -------
    seg_list: list
        Contains image_segmenter_overlayed type objects
    """
    # print(anatomical_data.shape)
    # print(bssfp_data.shape)
    if np.ndim(anatomical_data) > 3:
        anatomical_data = np.squeeze(anatomical_data)

    if masks_drawn_on is None:
        pass
    elif masks_drawn_on == "coronal":
        # assume that there is only one coronal slice
        if np.ndim(anatomical_data) == 2:
            anatomical_data = anatomical_data[..., np.newaxis]
        pass
    elif masks_drawn_on == "axial":
        if np.ndim(anatomical_data) == 2:
            # assume that there is only one axial slice
            anatomical_data = anatomical_data[np.newaxis, ...]
        anatomical_data = np.rot90(np.transpose(anatomical_data, (1, 2, 0)))
    else:
        raise Exception(f"masks_drawn_on = {masks_drawn_on} not yet implemented!")
    # print(anatomical_data.shape)
    # print(bssfp_data.shape)
    rot_images_deg = np.floor(rot_images_deg / 90)
    line_properties = {"color": "red", "linewidth": 1}
    seg_list = [
        image_segmenter_overlayed(
            np.flip(np.rot90(anatomical_data[:, :, s], rot_images_deg), 0)
            if flip_images
            else np.rot90(anatomical_data[:, :, s], rot_images_deg),
            second_img=np.flip(
                np.rot90(np.mean(bssfp_data[0, :, :, s, :, 0], axis=2), rot_images_deg),
                0,
            )
            if flip_images
            else np.rot90(
                np.mean(bssfp_data[0, :, :, s, :, 0], axis=2), rot_images_deg
            ),
            img_extent=anat_ext,
            second_img_extent=bssfp_ext,
            second_img_alpha=overlay,
            second_img_cmap=bssfp_cmap,
            figsize=figsize,
            nclasses=n_rois,
            props=line_properties,
            lineprops=None,
            mask_alpha=mask_alpha,
            cmap="bone",
            vmin=vmin,
            vmax=vmax,
        )
        for s in range(anatomical_data.shape[2])
    ]
    return seg_list


def get_segmenter_list_overlayed_standardized(
    anatomical_data,
    secondary_data,
    anat_ext=None,
    sec_ext=None,
    overlay=0.25,
    sec_cmap="viridis",
    n_rois=1,
    figsize=(4, 4),
    second_img_vmin=None,
    second_img_vmax=None,
    vmin=None,
    vmax=None,
    mask_alpha=0.7,
    **kwargs,
):
    """
    Returns a list of image_segmenter_overlayed type entries. To be read by draw_masks_on_anatomical.
    This is an updated version of get_segmetner_list_overlayed, which was BSSFP exclusive
    Parameters
    ----------
    anatomical_data: np.array containing anatomical data
        shape:  read, phase,slice
        Assumes that dim[2] is slice.
    bssfp_data: np.array containing bssfp data
        shape :read, phase,slice
        Gets averaged over all reps per default. Can be changed later below.
    anat_ext : list, optional
        If bssfp and anatomical dont have the same extent you need to give this value.
        gives the extent of the anatomical image in mm for example: [-15,15,10,10]
    bssfp_ext : list, optional
        If bssfp and anatomical dont have the same extent you need to give this value.
        gives the extent of the bssfp image in mm for example: [-15,15,10,10]
    overlay : float
        alpha value of the secondary image, per default 0.25
    bssfp_cmap : str, optional
        bssfp colormap
    n_rois: int, optional
        number of rois to be segemented, so far limited to 1.
    Returns
    -------
    seg_list: list
        Contains image_segmenter_overlayed type objects
    """
    line_properties = {"color": "red", "linewidth": 1}
    seg_list = [
        image_segmenter_overlayed(
            anatomical_data[:, :, s],
            second_img=secondary_data[:, :, s],
            img_extent=anat_ext,
            second_img_extent=sec_ext,
            second_img_alpha=overlay,
            second_img_cmap=sec_cmap,
            second_img_vmin=second_img_vmin,
            second_img_vmax=second_img_vmax,
            figsize=figsize,
            nclasses=n_rois,
            props=line_properties,
            lineprops=None,
            mask_alpha=mask_alpha,
            cmap="bone",
            vmin=vmin,
            vmax=vmax,
        )
        for s in range(anatomical_data.shape[2])
    ]
    return seg_list


def get_segmenter_list(
    anatomical_data,
    n_rois=1,
    figsize=(4, 4),
    rot_images_deg=0,
    flip_images=False,
    mask_drawn_on=None,
):
    """
    Returns a list of image_segmenter type entries. TO be read by draw_masks_on_anatomical
    Parameters
    ----------
    anatomical_data: np.array containing anatomical data
        shape: (read, phase, slice)
        Assumes that dim[2] is slice.
    n_rois: int, optional
        number of rois to be segemented, so far limited to 1.
    Returns
    -------
    seg_list: list
        Contains image_segementer type objects
    """

    if mask_drawn_on is None:
        pass
    elif mask_drawn_on == "axial":
        if np.ndim(anatomical_data) > 3:
            anatomical_data = np.squeeze(anatomical_data)
        anatomical_data = np.transpose(anatomical_data, axes=(1, 2, 0))
    else:
        raise Exception(f"mask_drawn_on ={mask_drawn_on} is not yet implemented!")

    if np.ndim(anatomical_data) > 3:
        anatomical_data = np.squeeze(anatomical_data)
    rot_images_deg = np.floor(rot_images_deg / 90)
    line_properties = {"color": "red", "linewidth": 1}
    seg_list = [
        image_segmenter(
            np.flip(np.rot90(anatomical_data[:, :, s], rot_images_deg), 0)
            if flip_images
            else np.rot90(anatomical_data[:, :, s], rot_images_deg),
            figsize=figsize,
            nclasses=n_rois,
            props=line_properties,
            mask_alpha=0.76,
            cmap="gray",
        )
        for s in range(anatomical_data.shape[2])
    ]
    return seg_list


def draw_masks_on_anatomical_single_slice(segmenter, roi_names=None):
    """
    Loads a segmenter_list and then allows the user to draw ROIs which are saved in the segmenter_list
    """

    # define image plotting function
    def plot_imgs(eraser_mode, roi_key):
        temp_seg = segmenter
        temp_seg.erasing = eraser_mode
        if roi_names:
            # names instead of numbers
            roi_number = roi_names.index(roi_key) + 1
        else:
            # default numbering
            roi_number = roi_key
        temp_seg.current_class = roi_number
        display(temp_seg)

    n_rois = segmenter.nclasses

    # Making the UI
    if roi_names:
        class_selector = widgets.Dropdown(options=roi_names, description="ROI name")
    else:
        class_selector = widgets.Dropdown(
            options=list(range(1, n_rois + 1)), description="ROI number"
        )

    erasing_button = widgets.Checkbox(value=False, description="Erasing")

    # put both sliders inside a HBox for nice alignment  etc.
    ui = widgets.HBox(
        [erasing_button, class_selector],
        layout=widgets.Layout(display="flex"),
    )

    sliders = widgets.interactive_output(
        plot_imgs,
        {
            "eraser_mode": erasing_button,
            "roi_key": class_selector,
        },
    )

    display(ui, sliders)


def mask_parameter_map(
    param_map,
    masks,
    weight_result=False,
    plot=False,
    anat_ref_data=None,
    anat_ref_extent=None,
):
    """
    Applies a mask
    Parameters
    ----------
    param_map: 2D, np.array
    masks: dictionary containing 2D masks
    plot: bool, True if QA plots wanted
    anat_ref_data: Optional, if plot activated, need to pass an anatomical reference image
    anat_ref_extent: Optional, if plot activated, need to pass the extent of that anatomical ref image
    Returns
    -------
    meaned_params: dict, keys are same as masks.keys()
    """
    from ..utils.utils_general import calc_coverage
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if plot is True:
        if anat_ref_extent is None:
            logger.critical(
                "anat_ref_extent needs to be passed to function to enable plotting"
            )
            anat_ref_extent = [0, 1, 0, 1]
            # plot=False
        elif anat_ref_data is None:
            logger.critical(
                "anat_ref_data needs to be passed to function to enable plotting"
            )
            plot = False
    else:
        pass
    if np.ndim(param_map) == 3:
        param_map_reshaped = np.reshape(
            param_map,
            newshape=(
                1,
                1,
                param_map.shape[0],
                param_map.shape[1],
                param_map.shape[2],
                1,
            ),
        )
    else:
        param_map_reshaped = np.reshape(
            param_map, newshape=(1, 1, param_map.shape[0], param_map.shape[1], 1, 1)
        )
    meaned_params = {}
    param_stds = {}
    keys = list(masks.keys())
    covs = {}

    for key in keys:
        mask = masks[key]
        if np.ndim(mask) == 6:
            mask_reshaped = mask
        elif np.ndim(mask) == 3:
            mask_reshaped = np.reshape(
                mask, newshape=(1, 1, mask.shape[0], mask.shape[1], mask.shape[2], 1)
            )
        else:
            mask_reshaped = np.reshape(
                mask, newshape=(1, 1, mask.shape[0], mask.shape[1], 1, 1)
            )

        coverage = calc_coverage(param_map_reshaped, mask_reshaped)

        if weight_result is True:
            from ..utils.utils_general import compute_weighted_mean_and_std

            # compute weighted average according to coverage mask
            weighted_coverage = np.squeeze(coverage)
            # for better display replace 0 with nan in the displayed version of the coverage mask
            weighted_coverage_display = np.squeeze(
                np.where(coverage > 0, coverage, np.nan)
            )
            not_nan_indices = np.where(
                np.logical_not(np.isnan(np.squeeze(param_map_reshaped)))
            )
            try:
                # use weighted statistics function
                weighted_average, weighted_std = compute_weighted_mean_and_std(
                    np.squeeze(param_map_reshaped)[not_nan_indices],
                    weighted_coverage[not_nan_indices],
                )
                # weighted_statistics = DescrStatsW(np.squeeze(param_map_reshaped)[not_nan_indices],
                #                              weights=weighted_coverage[not_nan_indices])
                # weighted_average = weighted_statistics.mean
                # weighted_std = weighted_statistics.std

            except ZeroDivisionError:
                logger.critical(
                    "No values except nan in "
                    + key
                    + " mask, returning nan for the mean of that ROI"
                )
                weighted_average = np.nan
                weighted_std = np.nan
            meaned_params.update({key: weighted_average})
            param_stds.update({key: weighted_std})

            covs.update({key: weighted_coverage_display})

        else:
            logger.info(
                "Computing np.nanmean with coverage set so that a pixel is included if it is inside the mask"
            )
            # need to change the weighting from cov, which has a percent value
            # depending on how much the pixel in the ROI is in the image selected
            # we change this to binary, so a pixel is either in our out
            coverage_mask = np.squeeze(np.where(coverage > 0, 1, np.nan))
            meaned_params.update(
                {key: np.nanmean(coverage_mask * np.squeeze(param_map_reshaped))}
            )
            param_stds.update(
                {key: np.nanstd(coverage_mask * np.squeeze(param_map_reshaped))}
            )

            covs.update({key: coverage_mask})

    plt.close("all")
    if plot is True:
        if np.ndim(anat_ref_data) == 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)

            @widgets.interact(ROI=keys, nslice=(0, anat_ref_data.shape[2] - 1, 1))
            def update(ROI=keys[0], nslice=0):
                ax1.cla()
                ax2.cla()

                ax1.imshow(
                    anat_ref_data[:, :, nslice], extent=anat_ref_extent, cmap="bone"
                )
                ax1.imshow(
                    np.squeeze(covs[ROI])[:, :, nslice],
                    extent=anat_ref_extent,
                    alpha=0.8,
                    cmap="summer",
                )
                ax2.imshow(
                    anat_ref_data[:, :, nslice], extent=anat_ref_extent, cmap="bone"
                )

                ax2.imshow(
                    np.squeeze(
                        covs[ROI][:, :, nslice]
                        * np.squeeze(param_map_reshaped)[:, :, nslice]
                    ),
                    extent=anat_ref_extent,
                    alpha=0.8,
                )

                ax2.set_title(
                    "Mean masked val = " + str(np.round(meaned_params[ROI], 2))
                )
                ax1.set_title("Anatomical ref + coverage")

        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)

            @widgets.interact(ROI=keys)
            def update(ROI=keys[0]):
                ax1.cla()
                ax2.cla()

                ax1.imshow(anat_ref_data, extent=anat_ref_extent, cmap="bone")
                ax1.imshow(
                    np.squeeze(covs[ROI]),
                    extent=anat_ref_extent,
                    alpha=0.8,
                    cmap="summer",
                )
                ax2.imshow(anat_ref_data, extent=anat_ref_extent, cmap="bone")

                ax2.imshow(
                    np.squeeze(covs[ROI] * np.squeeze(param_map_reshaped)),
                    extent=anat_ref_extent,
                    alpha=0.8,
                )

                ax2.set_title(
                    "Mean masked val = " + str(np.round(meaned_params[ROI], 2))
                )
                ax1.set_title("Anatomical ref + coverage")

    return meaned_params, param_stds


def draw_masks_on_anatomical(anatomical_segmenter_list, roi_names=None):
    """
    Loads a segmenter_list and then allows the user to draw ROIs which are saved in the segmenter_list
    """

    # define image plotting function
    def plot_imgs(n_slice, eraser_mode, roi_key):
        temp_seg = anatomical_segmenter_list[n_slice]
        temp_seg.erasing = eraser_mode
        if roi_names:
            # names instead of numbers
            roi_number = roi_names.index(roi_key) + 1
        else:
            # default numbering
            roi_number = roi_key
        temp_seg.current_class = roi_number
        display(temp_seg)

    n_rois = anatomical_segmenter_list[0].nclasses
    n_slices = len(anatomical_segmenter_list)

    # Making the UI
    if roi_names:
        class_selector = widgets.Dropdown(options=roi_names, description="ROI name")
    else:
        class_selector = widgets.Dropdown(
            options=list(range(1, n_rois + 1)), description="ROI number"
        )

    erasing_button = widgets.Checkbox(value=False, description="Erasing")
    # create interactive slider for echoes

    slice_slider = widgets.IntSlider(
        value=n_slices // 2, min=0, max=n_slices - 1, description="Slice: "
    )

    # put both sliders inside a HBox for nice alignment  etc.
    ui = widgets.HBox(
        [erasing_button, slice_slider, class_selector],
        layout=widgets.Layout(display="flex"),
    )

    sliders = widgets.interactive_output(
        plot_imgs,
        {
            "n_slice": slice_slider,
            "eraser_mode": erasing_button,
            "roi_key": class_selector,
        },
    )

    display(ui, sliders)


def get_masks_standardized(segmenter_list, roi_keys=None, plot_res=False):
    """
    Extract the masks for a given segmenter list.
    Standardized for image dimensions
    Parameters
    ---------
    segmenter_list: list of image_segmenter_overlayed objects

    roi_keys: list of str, optional
        Default is none, then just strings of numbers from 0-number_of_rois are the keys
        Suggested Roi key names: bloodvessel, tumor, kidneyL, kidneyR, tumor2, phantom, outside_ref

    plot_res: bool, optional.
        if one wants the result to be plotted for QA, default is False.
    Returns
    --------
    mask_per_slice: dict
        entries can be called via the selected keys and have
        shape: (read, phase, slice, number_of_rois)

    Examples
    --------
    If we give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list,['Tumor','Kidney','Vessel'],True)
    masked_bssfp = masks['Tumor'] * bssfp_image

    If we dont give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list)
    masks['1'].shape --> (read,phase,slice)
    masked_bssfp = masks['1'] * bssfp_image


    """
    n_slices = len(segmenter_list)
    n_rois = segmenter_list[0].nclasses
    if not roi_keys:
        # set default names
        roi_keys = [str(n) for n in range(1, n_rois + 1)]
    else:
        # use given keys
        pass
    mask_per_slice = np.zeros(
        (
            n_slices,
            segmenter_list[0].mask.shape[0],
            segmenter_list[0].mask.shape[1],
            n_rois,
        )
    )
    for slic in range(0, n_slices):
        for roi in range(0, n_rois):
            test_mask = segmenter_list[slic].mask == roi + 1
            mask_per_slice[slic, :, :, roi] = test_mask

    mask_dict = dict()
    # convert all zeros into nan

    mask_per_slice = np.where(mask_per_slice == 0, np.nan, mask_per_slice)
    for idx, roi_key in enumerate(roi_keys):
        mask_dict.update({roi_key: mask_per_slice[:, :, :, idx]})

    if plot_res:
        fig, ax = plt.subplots(1, n_rois)

        @widgets.interact(slices=(0, n_slices - 1, 1))
        def update(slices=0):
            if n_rois > 1:
                [ax[n].imshow(mask_per_slice[slices, :, :, n]) for n in range(n_rois)]
                [ax[n].set_title("ROI " + str(roi_keys[n])) for n in range(n_rois)]
            else:
                ax.imshow(mask_per_slice[slices, :, :, 0])
                ax.set_title("ROI " + str(roi_keys[0]))

        first_key = next(iter(mask_dict))
        for roi_num, key in enumerate(list(mask_dict.keys())):
            test = mask_dict[key]
            all_entries = []
            for slic in range(mask_dict[first_key].shape[0]):
                mask_entries = len(np.where(test[slic, :, :] > 0)[0])
                if mask_entries > 0:
                    all_entries.append(slic)
                else:
                    pass
            print(
                key,
                " is segmented in slices: ",
                all_entries,
                " of ",
                mask_dict[first_key].shape[0] - 1,
            )

    return mask_dict


def get_masks_standardized_single_slice(segmenter, roi_keys=None, plot_res=False):
    """
    Extract the masks for a given segmenter list.
    Standardized for image dimensions
    Parameters
    ---------
    segmenter_list: list of image_segmenter_overlayed objects

    roi_keys: list of str, optional
        Default is none, then just strings of numbers from 0-number_of_rois are the keys
        Suggested Roi key names: bloodvessel, tumor, kidneyL, kidneyR, tumor2, phantom, outside_ref

    plot_res: bool, optional.
        if one wants the result to be plotted for QA, default is False.
    Returns
    --------
    mask_per_slice: dict
        entries can be called via the selected keys and have
        shape: (read, phase, slice, number_of_rois)

    Examples
    --------
    If we give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list,['Tumor','Kidney','Vessel'],True)
    masked_bssfp = masks['Tumor'] * bssfp_image

    If we dont give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list)
    masks['1'].shape --> (read,phase,slice)
    masked_bssfp = masks['1'] * bssfp_image


    """

    n_rois = segmenter.nclasses
    if not roi_keys:
        # set default names
        roi_keys = [str(n) for n in range(1, n_rois + 1)]
    else:
        # use given keys
        pass
    mask_per_roi = np.zeros(
        (
            segmenter.mask.shape[0],
            segmenter.mask.shape[1],
            n_rois,
        )
    )

    for roi in range(0, n_rois):
        test_mask = segmenter.mask == roi + 1
        mask_per_roi[:, :, roi] = test_mask

    mask_dict = dict()
    # convert all zeros into nan

    mask_per_slice = np.where(mask_per_roi == 0, np.nan, mask_per_roi)
    for idx, roi_key in enumerate(roi_keys):
        mask_dict.update({roi_key: mask_per_slice[:, :, idx]})

    if plot_res:
        fig, ax = plt.subplots(1, n_rois - 1)

        if n_rois > 1:
            [ax[n].imshow(mask_per_roi[:, :, n]) for n in range(n_rois - 1)]
            [ax[n].set_title("ROI " + str(roi_keys[n])) for n in range(n_rois - 1)]
        else:
            ax.imshow(mask_per_roi[:, :, 0])
            ax.set_title("ROI " + str(roi_keys[0]))

    return mask_dict


def get_masks(segmenter_list, roi_keys=None, plot_res=False, masks_drawn_on=None):
    """
    Extract the masks for a given segmenter list.

    Parameters
    ---------
    segmenter_list: list of image_segmenter_overlayed objects

    roi_keys: list of str, optional
        Default is none, then just strings of numbers from 0-number_of_rois are the keys
        Suggested Roi key names: bloodvessel, tumor, kidneyL, kidneyR, tumor2, phantom, outside_ref

    plot_res: bool, optional.
        if one wants the result to be plotted for QA, default is False.
    Returns
    --------
    mask_per_slice: dict
        entries can be called via the selected keys and have
        shape: (read, phase, slice, number_of_rois)

    Examples
    --------
    If we give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list,['Tumor','Kidney','Vessel'],True)
    masked_bssfp = masks['Tumor'] * bssfp_image

    If we dont give keys:
    masks = ut_anat.get_masks_multi_rois(segmenter_list)
    masks['1'].shape --> (read,phase,slice)
    masked_bssfp = masks['1'] * bssfp_image


    """
    n_slices = len(segmenter_list)
    n_rois = segmenter_list[0].nclasses
    if not roi_keys:
        # set default names
        roi_keys = [str(n) for n in range(1, n_rois + 1)]
    else:
        # use given keys
        pass
    mask_per_slice = np.zeros(
        (
            segmenter_list[0].mask.shape[0],
            segmenter_list[0].mask.shape[1],
            n_slices,
            n_rois,
        )
    )
    for slic in range(0, n_slices):
        for roi in range(0, n_rois):
            test_mask = segmenter_list[slic].mask == roi + 1
            mask_per_slice[:, :, slic, roi] = test_mask

    mask_dict = dict()

    for idx, roi_key in enumerate(roi_keys):
        mask_dict.update({roi_key: mask_per_slice[:, :, :, idx]})

    if plot_res:
        fig, ax = plt.subplots(1, n_rois, figsize=(12, 5), tight_layout=True)

        @widgets.interact(slices=(0, n_slices - 1, 1))
        def update(slices=0):
            if n_rois > 1:
                [ax[n].imshow(mask_per_slice[:, :, slices, n]) for n in range(n_rois)]
                [ax[n].set_title("ROI " + str(roi_keys[n])) for n in range(n_rois)]
            else:
                ax.imshow(mask_per_slice[:, :, slices, 0])
                ax.set_title("ROI " + str(roi_keys[0]))

        first_key = next(iter(mask_dict))
        for roi_num, key in enumerate(list(mask_dict.keys())):
            test = mask_dict[key]
            all_entries = []
            for slic in range(mask_dict[first_key].shape[2]):
                mask_entries = len(np.where(test[:, :, slic] > 0)[0])
                if mask_entries > 0:
                    all_entries.append(slic)
                else:
                    pass
            print(
                key,
                " is segmented in slices: ",
                all_entries,
                " of ",
                mask_dict[first_key].shape[2] - 1,
            )

    if masks_drawn_on is None:
        pass
    elif masks_drawn_on == "coronal":
        from ..utils.utils_spectroscopy import make_NDspec_6Dspec

        for mask in mask_dict:
            mask_dict[mask] = make_NDspec_6Dspec(
                input_data=mask_dict[mask], provided_dims=(1, 2, 3)
            )

    elif masks_drawn_on == "axial":
        from ..utils.utils_spectroscopy import make_NDspec_6Dspec

        for mask in mask_dict:
            mask_dict[mask] = make_NDspec_6Dspec(
                input_data=np.rot90(mask_dict[mask], k=-1),
                provided_dims=(2, 3, 1),
            )
    else:
        raise Exception("masks_drawn_on has to be None, coronal or axial!")
        pass

    return mask_dict


def list_masks(dirpath,custom_file_ending=False):
    """
    Lists all files that end with masks.npz in a given directory.
    Parameters
    ----------
    dirpath: str
        Folder where we expect the mask files.

    Returns
    -------
    files: list of str
        contains names of all mask files in dirpath.
    """
    print("Found files in " + str(dirpath) + " :")
    files = []
    for file in os.listdir(dirpath):
        if custom_file_ending is not False:
            if file.endswith(custom_file_ending):
                files.append(file)
                print(file)
        else:
            if file.endswith("masks.npz"):
                files.append(file)
                print(file)
            elif file.endswith("masks.pkl"):
                files.append(file)
                print(file)
            else:
                pass
    return files


def load_mask(dirpath, mask_name, plot_res=False):
    """
    Loads .npz file and retrieves the mask dictionary.
    Parameters
    ----------
    filepath: path to mask file

    Returns
    -------

    """
    data_loaded = np.load(os.path.join(dirpath, mask_name), allow_pickle=True)
    mask_dict = data_loaded["arr_0"][()]
    keys = list(mask_dict.keys())
    n_rois = len(keys)
    mask_dim = np.ndim(mask_dict.get(list(keys)[0]))
    if mask_dim > 2:
        n_slices = mask_dict.get(list(keys)[0]).shape[2]
    else:
        n_slices = 1

    plt.close("all")
    if plot_res:
        if n_slices > 1:
            fig, ax = plt.subplots(1)

            @widgets.interact(key=keys, slices=(0, n_slices - 1, 1))
            def update(slices=0, key=keys[0]):
                ax.cla()
                ax.imshow(mask_dict[key][:, :, slices])
                ax.set_title("ROI " + str(key))

        else:
            fig, ax = plt.subplots(1)

            @widgets.interact(key=keys)
            def update(key=keys[0]):
                ax.cla()
                ax.imshow(mask_dict[key])
                ax.set_title("ROI " + str(key))

    return mask_dict


def make_mask_6D(mask_dict=None, mask_key=None, mask=None, mask_drawn_on="axial"):
    """

    Parameters
    ----------
    mask_dict
    mask_key
    mask
    mask_drawn_on

    Returns
    -------

    """

    def reshape_mask(mask=None):
        if np.ndim(mask) < 6:
            from ..utils.utils_spectroscopy import make_NDspec_6Dspec

            if mask_drawn_on == "axial":
                provided_dims = ["z", "y", "x"]
                mask_copy = make_NDspec_6Dspec(
                    input_data=mask, provided_dims=provided_dims
                )
                mask_copy = np.flip(mask_copy, axis=3)
            elif mask_drawn_on == "coronal":
                provided_dims = ["y", "z", "x"]
                mask_copy = make_NDspec_6Dspec(
                    input_data=mask, provided_dims=provided_dims
                )
            else:
                # assume axial:
                provided_dims = ["z", "x", "y"]
                mask_copy = make_NDspec_6Dspec(
                    input_data=mask, provided_dims=provided_dims
                )

        else:
            mask_copy = mask
        return mask_copy

    if mask_dict is None and mask_key is None and mask is None:
        return None

    if mask_dict is not None and mask_key is None and mask is None:
        for mask in mask_dict:
            mask_dict[mask] = reshape_mask(mask=mask)
            return mask_dict
    elif mask_dict is not None and mask_key is not None and mask is None:
        mask = reshape_mask(mask=mask_dict[mask_key])
        return mask
    elif mask_dict is None and mask_key is None and mask is not None:
        mask = reshape_mask(mask=mask)
        return mask
    else:
        logger.critical("Either pass mask or mask_dict (and key)")
        return None


def get_roi_coords(mask_dict=None, mask_drawn_on="axial"):
    """
    Extract the contours of a mask segmented with mpl_interactions image_segmenter_overlayed

    Parameters
    ---------
    segmenter_list: list

    Returns
    --------
    contours: list(np.arrays)
    """

    # do this to make old masks still readable

    contours = list()

    first_key = next(iter(mask_dict))
    if len(mask_dict[first_key].shape) <= 2:
        for roi in mask_dict.keys():
            contours.append(measure.find_contours(np.nan_to_num(mask_dict[roi])))
        return contours
    else:
        n_slices = mask_dict[first_key].shape[2]
        for roi in mask_dict.keys():
            for slic in range(n_slices):
                contours.append(measure.find_contours(mask_dict[roi][:, :, slic]))
        contours_reshaped = [
            contours[n : n + n_slices] for n in range(0, len(contours), n_slices)
        ]
        return contours_reshaped


def check_segmentation(mask_dict):
    """
    Prints which ROI was segmented on which slice for checking.
    Parameters
    ----------
    mask_dict: dict
        Contains masks that were obtained using get_masks

    Returns
    -------

    """
    first_key = next(iter(mask_dict))
    for roi_num, key in enumerate(list(mask_dict.keys())):
        test = mask_dict[key]
        all_entries = []
        for slic in range(mask_dict[first_key].shape[2]):
            mask_entries = len(np.where(test[:, :, slic] > 0)[0])
            if mask_entries > 0:
                all_entries.append(slic)
            else:
                pass
        print(
            key,
            " is segmented in slice ",
            all_entries,
            " of ",
            mask_dict[first_key].shape[2],
        )


def plot_preloaded_mask(mask_dict, anatomical):
    """
    Plots a preloaded mask on a anatomical image.
    Parameters
    ----------
    mask_dict: dictionary
    anatomical: 3D-array
    e.g. coronal.sed2d
    """
    fig, ax = plt.subplots(1)

    @widgets.interact(n=(0, anatomical.shape[2] - 1, 1))
    def update(n=0):
        [l.remove() for l in ax.lines]
        plot_segmented_roi_on_anat(ax, anatomical, mask_dict, n)


def plot_segmented_roi_on_anat(
    axis, anatomical, mask_dict, slice_number=0, vmin_anat=False, vmax_anat=False,plot_legend=False,linewidth=2
):
    """
    Plots the contours of a segmented mask on an anatomical image
    Parameters
    ----------
    axis: subplot object into which we plot.
    anatomical: np.array
    mask_dict: dict
        contains masks from segmentation
    slice: int
        Number of slice to plot.
    vmin_anat: float
        Windowing of anatomical image lower bound
    vmax_anat: float
        Windowing of anatomical image upper bound
    Returns
    -------

    """
    first_key = next(iter(mask_dict))
    if len(mask_dict[first_key].shape) <= 2:
        roi_coords = get_roi_coords(mask_dict)
        if vmin_anat is not False:
            axis.imshow(
                anatomical,
                cmap="gray",
                vmin=vmin_anat,
                vmax=vmax_anat,
            )
        else:
            axis.imshow(anatomical, cmap="gray")
        for roi_num in range(len(roi_coords)):
            if len(roi_coords[roi_num][slice_number]) > 0:
                if len(roi_coords[roi_num]) == 1:
                    axis.plot(
                        np.squeeze(roi_coords[roi_num])[:, 1],
                        np.squeeze(roi_coords[roi_num])[:, 0],
                        linewidth=linewidth,
                        color="C" + str(roi_num),
                        label=list(mask_dict.keys())[roi_num],
                    )
                else:
                    for elem in range(len(roi_coords[roi_num])):
                        axis.plot(
                            roi_coords[roi_num][elem][:, 1],
                            roi_coords[roi_num][elem][:, 0],
                            linewidth=linewidth,
                            color="C" + str(roi_num),
                            alpha=1 / (elem + 1),
                            label=list(mask_dict.keys())[roi_num],
                        )

                # axis.set_title('Slice ' + str(slice_number))
        if plot_legend is not False:
            if len(axis.get_legend_handles_labels()[0]) > 0:
                axis.legend()
            else:
                axis.legend_ = None
        else:
            pass
    else:
        n_slices = anatomical.shape[2]

        try:
            roi_coords = get_roi_coords(mask_dict)
            if vmin_anat is not False:
                axis.imshow(
                    anatomical[:, :, slice_number],
                    cmap="gray",
                    vmin=vmin_anat,
                    vmax=vmax_anat,
                )
            else:
                axis.imshow(anatomical[:, :, slice_number], cmap="gray")
            for roi_num in range(len(roi_coords)):
                if len(roi_coords[roi_num][slice_number]) > 0:
                    if len(roi_coords[roi_num][slice_number]) == 1:
                        axis.plot(
                            np.squeeze(roi_coords[roi_num][slice_number])[:, 1],
                            np.squeeze(roi_coords[roi_num][slice_number])[:, 0],
                            linewidth=linewidth,
                            color="C" + str(roi_num),
                            label=list(mask_dict.keys())[roi_num],
                        )
                    else:
                        for elem in range(len(roi_coords[roi_num][slice_number])):
                            axis.plot(
                                roi_coords[roi_num][slice_number][elem][:, 1],
                                roi_coords[roi_num][slice_number][elem][:, 0],
                                linewidth=linewidth,
                                color="C" + str(roi_num),
                                alpha=1 / (elem + 1),
                                label=list(mask_dict.keys())[roi_num],
                            )

                    # axis.set_title('Slice ' + str(slice_number))
            if plot_legend is not False:
                if len(axis.get_legend_handles_labels()[0]) > 0:
                    axis.legend()
                else:
                    axis.legend_ = None
            else:
                pass
        except IndexError:
            logger.critical("Slice number cannot be larger than " + str(n_slices))


def interpolate_array(
    input_data,
    input_data_object,
    interp_size=None,
    interp_method="linear",
    use_multiprocessing=False,
    dtype=None,
    number_of_cpu_cores=None,
):
    """ "
    This function interpolates the input data onto a new grid in the same range as
    input_data but points in each dimension are described by interp_size. Uses scipy's RegularGridInterpolator
    method. Multithreading can be activated.

    Parameters:
    ----------
    input data: ND array
        Expected to have the order Echoes, Read, phase, slice, repetitions, channels

    interp_size: tuple
        Describes the interpolated data size.
        Has to be 6D: number of echhoes, points in  Read, phase, slice; number of repetitions, channels

    interp_method: ""
        interpolation method that is used. Supported are "linear", "nearest",
        # if scipy==1.10.0: "slinear", "cubic", "quintic" and "pchip".

    use_multiprocessing:
        toggles multiprocessing

    dtype:
        should give you the option to change the dtype of the to be interpolated array to reduce the size

    number_of_cpu_cores:
        Number of CPU cores to if multi processing is desired

    Examples:
    --------
    ------------------------------------------------------------------------
    # interpolate 1st and 2nd dimension onto a 10x higher grid:

    # get extend in x-y-z:
    x, y, z = dnp_bssfp.define_grid()
    interp_factor = 10

    xres = np.linspace(x[0], x[-1], interp_factor*len(x))
    yres = np.linspace(y[0], y[-1], interp_factor*len(y))

    zres = np.linspace(z[0], z[-1], len(z))
    t = np.arange(0,dnp_bssfp_pv_reco.shape[4],1)

    test_nearest = dnp_bssfp.interpolate_bssfp(bssfp_data=dnp_bssfp_pv_reco,
                        interp_size=(1,len(xres),len(yres),len(zres),len(t),2),
                        interp_method="nearest")


    plt.figure()
    for k in range(10):
    ax = plt.subplot(3,4,k+1)
    ax.imshow(np.squeeze(np.mean(test_linear[:,:,:,k,:,:], axis=3)))


    ------------------------------------------------------------------------
    # interpolate every 2nd image (lactate) onto a 4x higher time resoltion:

    # get extend in x-y-z:
    x, y, z = dnp_bssfp.define_grid()
    interp_factor = 4

    xres = np.linspace(x[0], x[-1], len(x))
    yres = np.linspace(y[0], y[-1], len(y))
    zres = np.linspace(z[0], z[-1], len(z))

    t = np.arange(0,dnp_bssfp_pv_reco.shape[4],1)
    tres = np.linspace(z[0], t[-1], interp_factor*len(t))


    test_lac_interp = dnp_bssfp.interpolate_bssfp(bssfp_data=dnp_bssfp_pv_reco[:,:,:,:,::2,:],
                        interp_size=(1,len(xres),len(yres),len(zres),len(tres),2),
                        interp_method="linear")
    plt.figure()
    reps = range(100)
    for k in reps:
    ax = plt.subplot(10,10,k+1)
    ax.imshow(np.squeeze(test_lac_interp[0,:,:,5,k+150,0]))
    """

    from scipy.interpolate import interpn
    import numpy as np
    import time
    from ..utils.utils_general import define_grid

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

    if interp_size is None:
        logger.error("You have to enter an interpolation size")
        interp_size = input_data.shape

    # check method:
    allowed_interp_methods = ["linear", "nearest", "splinef2d", "cubic"]
    if interp_method not in allowed_interp_methods:
        logger.critical(
            "uknown interpolation method: " + interp_method + ", using linear instead."
        )
        interp_method = "linear"

    if dtype is None:
        dtype = input_data.dtype

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
    x, y, z = define_grid(input_data_object)
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


def interpolate_arrays_skiimage(array_1, array_2):
    """
    Interpolates array 1 to array 2 dimensions. Formerly known as interpolate_bssf_to_anatomical
    Limitation: can only do spline interpolation, this is often not wanted
    Parameters
    ----------
    array_1: np.array
    array_2: np.array

    Returns
    -------
    array_1_new: np.array
        Has desired shape
    """
    # interpolate input image up to desired resolution
    new_shape = array_2.shape
    array_1_new = resize(array_1, new_shape)
    return array_1_new


def define_imagematrix_parameters(data_obj):
    """
    Warning: Does not take into account the orientation and offsets of the
    object (yet)
    Define the imaging matrix in voxel.
    Returns imaging matrix dimensions  as dim_read, dim_phase, dim_slice
    Input
    -----
    data_obj: Sequence object
    """
    if data_obj is None:
        return None, None, None

    dim_read = data_obj.method["PVM_Matrix"][0]  # was z
    dim_phase = data_obj.method["PVM_Matrix"][1]  # was y
    if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
        dim_slice = data_obj.method["PVM_Matrix"][2]  # was x
    else:
        dim_slice = data_obj.method["PVM_SPackArrNSlices"]  # was x
    return dim_read, dim_phase, dim_slice


def define_imageFOV_parameters(data_obj):
    """
    Warning: Does not take into account the orientation and offsets of the
    object (yet)
    Calculates the FOV in mm.
    Returns FOV in as mm_read, mm_phase, mm_slice.
    Input
    -----
    data_obj: Sequence object
    """
    if data_obj is None:
        return None, None, None

    # FOV:
    mm_read = data_obj.method["PVM_Fov"][0]
    mm_phase = data_obj.method["PVM_Fov"][1]
    mm_slice_gap = data_obj.method["PVM_SPackArrSliceGap"]

    if data_obj.method["PVM_SpatDimEnum"] == "<3D>":
        mm_slice = data_obj.method["PVM_Fov"][2]
    else:
        _, _, dim_slice = define_imagematrix_parameters(data_obj=data_obj)
        mm_slice = data_obj.method["PVM_SliceThick"]  # was x
        mm_slice = mm_slice * dim_slice + mm_slice_gap * (dim_slice - 1)

    return mm_read, mm_phase, mm_slice


def define_grid(data_obj):
    """
    Warning: Does not take into account the orientation and offsets yet
    Defines a 2D/3D grid of the image.
    """

    if data_obj is None:
        return None

    try:
        mat = np.array(define_imagematrix_parameters(data_obj=data_obj))
        fov = np.array(define_imageFOV_parameters(data_obj=data_obj))
    except:
        mat = data_obj.method["PVM_Matrix"]
        fov = data_obj.method["PVM_Fov"]

    # calculate resolution:
    res = fov / mat

    # init:
    ext_1 = ext_2 = ext_3 = None
    if (len(fov) > 0) and (len(mat) > 0):
        ext_1 = np.linspace(-fov[0] / 2 + res[0] / 2, fov[0] / 2 - res[0] / 2, mat[0])
    if (len(fov) > 1) and (len(mat) > 1):
        ext_2 = np.linspace(-fov[1] / 2 + res[1] / 2, fov[1] / 2 - res[1] / 2, mat[1])
    if (len(fov) > 2) and (len(mat) > 2):
        ext_3 = np.linspace(-fov[2] / 2 + res[2] / 2, fov[2] / 2 - res[2] / 2, mat[2])

    return ext_1, ext_2, ext_3


def plot_anat_images(
    anatomical=None,
    plot_params={},
    slice_orient="axial",
):
    """
    Plots anatomical images as a grid of 2D slices.

    Parameters
    ----------
    anatomical : object (optional)
        Object containing the anatomical image data. Default is None.
        This object is assumed to have an attribute `seq2d_oriented` that
        stores the image data in the format (_, slice, row, col, repetitions, channels).
    plot_params : dict (optional)
        Dictionary containing plotting parameters. Keys can include:
            - "colormap" (str): Colormap for the images (default: "bone").
            - "showticks" (bool): Whether to show axis ticks (default: True).
            - "showtitle" (bool): Whether to show slice number titles (default: True).
            - "figsize" (tuple): Figure size in inches (default: (5, 5)).
            - "savepath" (str): Path to save the figure (default: None).
            - "flip_rows_columns" (bool): Whether to flip rows and columns when creating the subplot grid.
            Default is False.
    slice_orient : str (optional)
        Orientation of the slices to plot ("axial", "sagittal", or "coronal").
        Default is "axial".

    Returns
    -------
    None
        The function displays the plot and optionally saves it.
    """
    from ..utils.utils_general import get_extent

    # Determine slice dimension based on orientation
    if slice_orient == "axial":
        slice_dim = 1
    elif slice_orient == "sagittal":
        slice_dim = 2
    elif slice_orient == "coronal":
        slice_dim = 3
    else:
        slice_orient = "axial"  # Fallback to axial if invalid orientation
        slice_dim = 1

    # Calculate subplot grid dimensions
    num_slices = anatomical.seq2d_oriented.shape[slice_dim]
    grid_size = int(np.ceil(np.sqrt(num_slices)))
    grid_cols = (
        grid_size - 1 if grid_size * (grid_size - 1) >= num_slices else grid_size
    )
    grid_rows = grid_size

    # Get extent values
    ax_ext, sag_ext, cor_ext = get_extent(data_obj=anatomical)

    # Unpack plotting parameters with defaults
    colormap = plot_params.get("colormap", "bone")
    showticks = plot_params.get("showticks", True)
    showtitle = plot_params.get("showtitle", True)
    figsize = plot_params.get("figsize", (5, 5))
    savepath = plot_params.get("savepath", None)
    savename = plot_params.get("savename", "overview_anat_images" + slice_orient)
    flip_rows_columns = plot_params.get("flip_rows_columns", False)

    # Create subplots, flipping rows and columns if specified.
    if flip_rows_columns:
        fig, axs = plt.subplots(grid_cols, grid_rows, figsize=figsize)
    else:
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    axs = axs.flatten()

    # Plot slices based on orientation
    for k in range(num_slices):
        if slice_orient == "axial":
            img = np.rot90(np.squeeze(anatomical.seq2d_oriented[0, k, :, :, 0, 0]))
        elif slice_orient == "sagittal":
            img = np.rot90(np.squeeze(anatomical.seq2d_oriented[0, :, k, :, 0, 0]))
        elif slice_orient == "coronal":
            img = np.rot90(np.squeeze(anatomical.seq2d_oriented[0, :, :, k, 0, 0]))

        axs[k].imshow(img, extent=ax_ext, cmap=colormap)

        if not showticks:
            axs[k].axis("off")
        if showtitle:
            axs[k].set_title(k)

    # Turn off unused subplots
    for i in range(num_slices, len(axs)):
        axs[i].set_axis_off()

    plt.tight_layout()
    plt.show()

    # Save the plot if a path is provided
    if savepath is not None:
        plt.savefig(
            os.path.join(savepath, savename + ".svg"), dpi=600, transparent=True
        )
        plt.savefig(
            os.path.join(savepath, savename + ".png"), dpi=600, transparent=True
        )


def plot_voxel_grids(image1=None, image2=None, plot_params={}):
    """
    Plots two images side by side, each with an overlaid grid to highlight their respective voxel (pixel) sizes.
    The function supports rotation and custom color mapping for the visualization of each image. If only one image is provided,
    it will be displayed with its grid. The function allows customization of the slice index, line color, rotation,
    and colormap for each image through the `plot_params` dictionary.

    Parameters
    ----------
    image1 : ndarray, optional
        A 3D or higher-dimensional array representing the first image. The function expects the image to be in a specific
        format where the relevant slice is indexed as `image1[0, slice_ind_image1, :, :, 0, 0]`. If `None`, the function
        will not display this image.
    image2 : ndarray, optional
        A 3D or higher-dimensional array representing the second image. Similar to `image1`, it is expected to be in a
        format where the relevant slice is indexed as `image2[0, slice_ind_image2, :, :, 0, 0]`. If `None`, only `image1`
        will be displayed if it is provided.
    plot_params : dict, optional
        A dictionary containing parameters for plotting. Supported parameters include:
        - `slice_ind_image1` (int): The slice index for `image1`. Defaults to 0.
        - `slice_ind_image2` (int): The slice index for `image2`. Defaults to 0.
        - `line_color_image1` (str): Line color for the grid overlaid on `image1`. Defaults to "yellow".
        - `line_color_image2` (str): Line color for the grid overlaid on `image2`. Defaults to "white".
        - `rot_image` (int): Number of times the images should be rotated by 90 degrees. Defaults to 1.
        - `cmap` (str): The colormap used for both images. Defaults to "jet".

    Returns
    -------
    None
        The function does not return a value but displays a matplotlib plot with the images and their grids.

    Raises
    ------
    ValueError
        If `image1` is not provided, a critical log message is displayed indicating that `image1` must be passed.

    Notes
    -----
    The function is designed to visually compare two images of potentially different resolutions by overlaying a grid on each.
    This can be particularly useful for examining the alignment of voxel data in medical imaging or comparing model predictions
    against ground truth in machine learning tasks.
    Use utils_general.calculate_overlap to calculate the overlap on a voxel by voxel basis.

    Examples
    --------
    >>> plot_voxel_grids(image1=my_image_array, plot_params={'slice_ind_image1': 10, 'linecolor_image1': 'blue'})
    This will display `my_image_array` with a slice index of 10 and a blue grid overlaid.
    """
    figsize = plot_params.get("figsize", (12,6))
    slice_ind_image1 = plot_params.get("slice_ind_image1", 0)
    slice_ind_image2 = plot_params.get("slice_ind_image2", 0)
    linecolor_image1 = plot_params.get("linecolor_image1", "yellow")
    linecolor_image2 = plot_params.get("linecolor_image2", "white")
    linewidth_image1 = plot_params.get("linewidth_image1", 0.5)
    linewidth_image2 = plot_params.get("linewidth_image2", 0.5)
    linestyle_image1 = plot_params.get("linestyle_image1", "-")
    linestyle_image2 = plot_params.get("linestyle_image2", "-")
    colorbar_image1 = plot_params.get("colorbar_image1", False)
    colorbar_image2 = plot_params.get("colorbar_image2", False)
    colorbar_label_image1 = plot_params.get("colorbar_label_image1", "signal [a.u]")
    colorbar_label_image2 = plot_params.get("colorbar_label_image2", "signal [a.u]")
    vmin_image1 = plot_params.get("vmin_image1", None)
    vmax_image1 = plot_params.get("vmax_image1", None)
    vmin_image2 = plot_params.get("vmin_image2", None)
    vmax_image2 = plot_params.get("vmax_image2", None)
    rot_image = plot_params.get("rot_image", 1)
    cmap_image1 = plot_params.get("cmap_image1", "jet")
    cmap_image2 = plot_params.get("cmap_image2", "jet")
    title_image1 = plot_params.get("title_image1", "")
    title_image2 = plot_params.get("title_image2", "")
    savepath = plot_params.get("savepath", None)

    extent_image1 = plot_params.get("extent_image1", None)
    extent_image2 = plot_params.get("extent_image2", None)

    if extent_image1 is not None and extent_image2 is None:
        extent_image2 = extent_image1

    if extent_image2 is not None and extent_image1 is None:
        extent_image1 = extent_image2

    if extent_image1 is None and extent_image2 is None:
        extent_image1 = [0, 1, 0, 1]
        extent_image2 = [0, 1, 0, 1]

    startx_image1 = extent_image1[0]
    endx_image1 = extent_image1[1]
    starty_image1 = extent_image1[2]
    endy_image1 = extent_image1[3]

    startx_image2 = extent_image2[0]
    endx_image2 = extent_image2[1]
    starty_image2 = extent_image2[2]
    endy_image2 = extent_image2[3]

    provided_dims = plot_params.get("provided_dims", ["x", "y"])

    if image1 is not None and image2 is not None:
        if np.ndim(image1) < 6:
            from ..utils.utils_spectroscopy import make_NDspec_6Dspec

            if np.ndim(image1) == len(provided_dims):
                image1 = make_NDspec_6Dspec(
                    input_data=image1, provided_dims=provided_dims
                )  # have to guess dimensions
            else:
                image1 = make_NDspec_6Dspec(input_data=image1)
        if np.ndim(image2) < 6:
            from ..utils.utils_spectroscopy import make_NDspec_6Dspec

            if np.ndim(image2) == len(provided_dims):
                image2 = make_NDspec_6Dspec(
                    input_data=image2, provided_dims=provided_dims
                )  # have to guess dimensions
            else:
                image2 = make_NDspec_6Dspec(input_data=image2)

        image1 = np.squeeze(image1[0, slice_ind_image1, :, :, 0, 0])
        image2 = np.squeeze(image2[0, slice_ind_image2, :, :, 0, 0])
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Overlay the low-resolution image
        im2 = ax[0].imshow(
            np.rot90(image1, rot_image),
            extent=extent_image1,
            alpha=1,
            cmap=cmap_image1,
            vmin=vmin_image1,
            vmax=vmax_image1,
        )
        ax[0].set_title(title_image1)

        # Add grid for high-resolution image
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax[0].axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax[0].axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Add grid for low-resolution image
        for x in np.linspace(startx_image2, endx_image2, image2.shape[0] + 1):
            ax[0].axvline(
                x,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )
        for y in np.linspace(starty_image2, endy_image2, image2.shape[1] + 1):
            ax[0].axhline(
                y,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )

        # Set the limits and labels
        ax[0].set_xlim(startx_image1, endx_image1)
        ax[0].set_ylim(starty_image1, endy_image1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # # Display the high-resolution image
        im1 = ax[1].imshow(
            np.rot90(image2, rot_image),
            extent=extent_image2,
            alpha=1,
            cmap=cmap_image2,
            vmin=vmin_image2,
            vmax=vmax_image2,
        )

        # Add grid for high-resolution image
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax[1].axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax[1].axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Add grid for low-resolution image
        for x in np.linspace(startx_image2, endx_image2, image2.shape[0] + 1):
            ax[1].axvline(
                x,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )
        for y in np.linspace(starty_image2, endy_image2, image2.shape[1] + 1):
            ax[1].axhline(
                y,
                color=linecolor_image2,
                linestyle=linestyle_image2,
                linewidth=linewidth_image2,
            )

        # Set the limits and labels
        ax[1].set_xlim(startx_image2, endx_image2)
        ax[1].set_ylim(starty_image2, endy_image2)
        ax[1].set_title(title_image2)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.tight_layout()
        plt.show()

        if colorbar_image2:
            fig.colorbar(im2, ax=ax[0], label=colorbar_label_image2, shrink=0.8)
        if colorbar_image1:
            fig.colorbar(im1, ax=ax[1], label=colorbar_label_image1, shrink=0.8)

        plt.tight_layout()

    elif image1 is not None and image2 is None:
        if np.ndim(image1) < 6:
            from ..utils.utils_spectroscopy import make_NDspec_6Dspec

            if np.ndim(image1) == len(provided_dims):
                image1 = make_NDspec_6Dspec(
                    input_data=image1, provided_dims=provided_dims
                )  # have to guess dimensions
            else:
                image1 = make_NDspec_6Dspec(input_data=image1)

        image1 = np.squeeze(image1[0, slice_ind_image1, :, :, 0, 0])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Overlay the low-resolution image
        im1 = ax.imshow(
            np.rot90(image1, rot_image),
            extent=extent_image1,
            alpha=1,
            cmap=cmap_image1,
            vmin=vmin_image1,
            vmax=vmax_image1,
        )

        # Add grid
        for x in np.linspace(startx_image1, endx_image1, image1.shape[0] + 1):
            ax.axvline(
                x,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )
        for y in np.linspace(starty_image1, endy_image1, image1.shape[1] + 1):
            ax.axhline(
                y,
                color=linecolor_image1,
                linestyle=linestyle_image1,
                linewidth=linewidth_image1,
            )

        # Set the limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Set the limits and labels
        ax.set_xlim(startx_image1, endx_image1)
        ax.set_ylim(starty_image1, endy_image1)
        ax.set_title(title_image1)
        if colorbar_image1:
            fig.colorbar(im1, ax=ax, label=colorbar_label_image1, shrink=0.8)
        plt.show()
        plt.tight_layout()
    else:
        logger.critical("image1 has to be passed!")
    if savepath is not None:
        try:
            # plt.savefig(savepath + ".svg", format="svg", dpi=600, transparent=True)
            plt.savefig(savepath + ".png", format="png", dpi=600, transparent=True)
        except:
            raise Warning(f"image could not be save as {savepath}.png/.svg")


def calc_mask_area_volume(
    mask=None,
    mask_dict=None,
    mask_key=None,
    data_obj=None,
    grid=None,
    masks_drawn_on=None,
):
    """
    Calculate the area and volume of a specified region of interest (ROI) within a mask.
    The mask should be a 3D (or higher) array with binary values (0s and 1s), where 1s
    represent the ROI. The function computes the area and volume using the resolution
    defined by the `grid`. If `grid` is not provided and `data_obj` is provided, the grid
    will be defined based on `data_obj`.

    Parameters
    ----------
    mask : ndarray, optional
        A multi-dimensional numpy array where the ROI is defined with ones. If not provided,
        `mask_dict` and `mask_key` must be used to retrieve the mask.
    mask_dict : dict, optional
        A dictionary of masks from which to retrieve the mask using `mask_key` if `mask`
        is not provided.
    mask_key : any, optional
        The key to retrieve the mask from `mask_dict`. Required if `mask_dict` is provided.
    data_obj : object, optional
        An object from which to derive the grid dimensions if `grid` is not provided.
        The function `define_grid` should be able to use this object to define the grid.
    grid : list of ndarray, optional
        A list containing arrays that define the grid in each dimension. Each array in the list
        should contain coordinates for the grid points in that dimension.
    masks_drawn_on : str, optional
        A string indicating the orientation of the slices in the mask. Valid options are
        "axial", "coronal". If not provided, defaults to calculating area and volume
        assuming an axial orientation.

    Raises
    ------
    ValueError
        If neither `mask` nor both `mask_dict` and `mask_key` are provided.
        If `mask_key` cannot be found in `mask_dict`.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - An array of areas of the ROI for each slice according to `masks_drawn_on`.
        - The total volume of the ROI across all slices.

    Warns
    -----
    UserWarning
        If neither `grid` nor `data_obj` is provided, warns that the area/volume
        will be computed in arbitrary units (a.u.).
    """
    if mask is None:
        if mask_dict is None or mask_key is None:
            raise ValueError(
                "Either a mask or a mask_dict with a mask_key must be provided!"
            )
        try:
            mask = mask_dict[mask_key]
        except KeyError:
            raise ValueError(f"Cannot load mask_key {mask_key} from mask_dict!")

    if data_obj is None:
        if grid is None:
            from warnings import warn

            warn("Area/Volume will be in [a.u.]!")
            grid = []
            if mask.shape[0] == 1:
                grid.append(np.arange(0, 2))
            else:
                grid.append(np.arange(mask.shape[0]))
            grid.append(np.arange(mask.shape[1]))
            grid.append(np.arange(mask.shape[2]))
        else:
            pass
    else:
        grid = define_grid(data_obj=data_obj)
        slice_gap = data_obj.method.get("PVM_SPackArrSliceGap", 0.0)
        if slice_gap == 0:
            pass
        else:
            warnings.warn(
                f"Volume estimation for slice gap = {slice_gap}mm (=/= 0mm) is not yet implemented!"
            )

    # define grid:
    resolution_grid = [
        grid[0][1] - grid[0][0],
        grid[1][1] - grid[1][0],
        grid[2][1] - grid[2][0],
    ]

    if masks_drawn_on == "coronal":
        from warnings import warn

        warn(f"Not yet tested with slice orientation {masks_drawn_on}")

    if masks_drawn_on == "axial":
        unit_area = resolution_grid[0] * resolution_grid[1]
        unit_volume = resolution_grid[0] * resolution_grid[1] * resolution_grid[2]
    elif masks_drawn_on == "coronal":
        unit_area = resolution_grid[0] * resolution_grid[2]
        unit_volume = resolution_grid[0] * resolution_grid[1] * resolution_grid[2]
    else:
        unit_area = resolution_grid[0] * resolution_grid[1]
        unit_volume = resolution_grid[0] * resolution_grid[1] * resolution_grid[2]

    if masks_drawn_on == "axial":
        area = np.zeros((mask.shape[1], 1))
        for slice in range(mask.shape[1]):
            area[slice] = np.sum(mask[:, slice, :, :, ::]) * unit_area
    elif masks_drawn_on == "coronal":
        area = np.zeros((mask.shape[3], 1))
        for slice in range(mask.shape[3]):
            area[slice] = np.sum(mask[:, :, :, slice, ::]) * unit_area
    else:
        area = np.zeros((mask.shape[1], 1))
        for slice in range(mask.shape[1]):
            area[slice] = np.sum(mask[:, slice, :, :, ::]) * unit_area

    volume = np.sum(mask) * unit_volume
    return area, volume


def add_scalebar(
    px,
    ax=None,
    units="mm",
    fixed_value=None,
    color="k",
    box_alpha=0.0,
    box_color="w",
    location="lower right",
    frameon=False,
    pad=0.075,
    length_fraction=None,
    border_pad=0.075,
    fontsize=12,
):
    """
    Add scalebar to image, given the pixel size 'px'.
    """

    if not ax:
        ax = plt.gca()
    ax.add_artist(
        ScaleBar(
            px,
            units=units,
            fixed_value=fixed_value,
            color=color,
            box_alpha=box_alpha,
            box_color=box_color,
            location=location,
            frameon=frameon,
            pad=pad,
            length_fraction=length_fraction,
            border_pad=border_pad,
            font_properties={"size": fontsize},
        )
    )

def calc_trajectory(data_obj=None, db=False):

    # get trajectory:
    fid = data_obj.fid

    # length of trajetory:
    traj_length = data_obj.method["PVM_Matrix"][0] * data_obj.method["PVM_Matrix"][1]

    # number of slices
    nslices = data_obj.method["PVM_SPackArrNSlices"]

    # number of repetitions:
    nr = data_obj.method["PVM_NRepetitions"]

    # reshape trajectory
    fid = np.reshape(fid, (nr, nslices, traj_length))


    if db is True:
        fig, ax = plt.subplots(1,3, figsize=(12,3))
        for r in range(nr):
            ax[0].plot(np.real(fid[r,0,:]))
            ax[1].plot(np.real(fid[r, 1, :]))

    # average along the repetitions:
    fid = np.squeeze(np.mean(fid, axis=0))

    # define slice distance:
    dx = data_obj.method["PVM_SPackArrSliceDistance"] * 1e-3


    traj = (np.unwrap(np.angle(fid[0,:])) - np.unwrap(np.angle(fid[1,:]))) / (2 * np.pi * dx)

    return traj


def Get_Voxel_patch_params_old(voxel_instance, img_instance, voxel_num):
    """
    FIXME: Apparently this function returns wrong coordinates for a measuremnt on PV7
    FIXME it seems that the z axis is flipped
    FIXME so we have introduced a case Andre for the orientation
    FIXME This needs to be checked


    Returns patch parameters for a given voxel instance (e.g. MV PRESS)
    Parameters
    ----------
    voxel_instance
    img_instance
    voxel_num
    Returns
    -------
    dict: voxel_patch_params
    ----
    Example:
    mvpress=hp.BrukerExp('/user/Documents/Measurement/',5,'PyratID')
    sagittal = hp.BrukerExp('/user/Documents/Measurement/',6,'PyratID')

    patch_params_sagittal = Get_Voxel_patch_params(mvpress,sagittal,0)
    fig,ax=plt.subplots(1)
    ax.imshow(sagittal.seq2d[:,:,0]) # slice 0
    ax.add_patch(Rectangle(**patch_params_sagittal))
    This plots now a patch with the desired size onto the image
    """
    read_orient = img_instance.method['PVM_SPackArrReadOrient']
    vox_pos = voxel_instance.method["PVM_VoxArrPosition"]
    vox_size = voxel_instance.method["PVM_VoxArrSize"]
    n = voxel_num
    if img_instance.method['PVM_SPackArrSliceOrient'] == 'coronal':
        if read_orient == 'H_F':
            # normal case
            patch_params_coronal = {
                "xy": (
                    vox_pos[n][0] - vox_size[n][0] / 2.0,
                    vox_pos[n][2] - vox_size[n][2] / 2.0,
                ),
                "width": vox_size[n][0],
                "height": vox_size[n][2],
            }
            return patch_params_coronal
        elif read_orient == 'ANDRE':
            # nasty press voxel 09 Nov 2022
            patch_params_coronal = {
                "xy": (
                    vox_pos[n][0] - vox_size[n][0] / 2.0,
                    -vox_pos[n][2] - vox_size[n][2] / 2.0,
                ),
                "width": vox_size[n][0],
                "height": vox_size[n][2],
            }
            return patch_params_coronal
        else:
            # FIXME this is not correct
            # normal case
            patch_params_coronal = {
                "xy": (
                    vox_pos[n][0] - vox_size[n][0] / 2.0,
                    vox_pos[n][2] - vox_size[n][2] / 2.0,
                ),
                "width": vox_size[n][0],
                "height": vox_size[n][2],
            }
            return patch_params_coronal
    elif img_instance.method['PVM_SPackArrSliceOrient'] == 'axial':
        if read_orient == 'L_R':
            patch_params_axial = {
                "xy": (
                    vox_pos[n][0] - vox_size[n][0] / 2.0,
                    vox_pos[n][1] - vox_size[n][1] / 2.0,
                ),
                "width": vox_size[n][0],
                "height": vox_size[n][1],
            }
            return patch_params_axial
        elif read_orient == 'A_P':
            patch_params_axial = {
                "xy": (
                    vox_pos[n][0] - vox_size[n][0] / 2.0,
                    vox_pos[n][1] - vox_size[n][1] / 2.0,
                ),
                "width": vox_size[n][0],
                "height": vox_size[n][1],
            }
            return patch_params_axial
    elif img_instance.method['PVM_SPackArrSliceOrient'] == 'sagittal':
        if read_orient == 'H_F':
            patch_params_sagittal = {
                "xy": (
                    vox_pos[n][1] - vox_size[n][1] / 2.0,
                    vox_pos[n][2] - vox_size[n][2] / 2.0,
                ),
                "width": vox_size[n][1],
                "height": vox_size[n][2],
            }
            return patch_params_sagittal
        elif read_orient == 'A_P':
            # somehow this is the same as above
            patch_params_sagittal = {
                "xy": (
                    vox_pos[n][1] - vox_size[n][1] / 2.0,
                    vox_pos[n][2] - vox_size[n][2] / 2.0,
                ),
                "width": vox_size[n][1],
                "height": vox_size[n][2],
            }
            return patch_params_sagittal

    else:
        return None


def to_be_done_orient_anatomical_2dseq_to_pv(image_object):
    """
    Reorients seq2d matrix to a consistent coordinate system
    Goal of this function is to reorient seq2d data such that it is correct

    Parameters
    ----------
    image_object: BrukerExp instance
        Contains the meta data and also 2dseq files

    Returns
    -------
    oriented_image: np.array
        This now has the same orientation as the scans in Paravision.
    """
    ACQ_patient_pos = image_object.acqp["ACQ_patient_pos"]
    image_data = image_object.seq2d
    if ACQ_patient_pos == "Head_Prone":
        if image_object.method["PVM_SPackArrSliceOrient"] == "axial":
            oriented_image = np.fliplr(np.rot90(image_data, 1))
        elif image_object.method["PVM_SPackArrSliceOrient"] == "coronal":
            oriented_image = np.flip(np.rot90(image_data, 3), 2)
        elif image_object.method["PVM_SPackArrSliceOrient"] == "sagittal":
            oriented_image = np.rot90(image_data, 2)
    elif ACQ_patient_pos == "Head_Supine":
        # Note This orientation is not tested
        if image_object.method["PVM_SPackArrSliceOrient"] == "axial":
            oriented_image = np.fliplr(np.rot90(image_data, 3))
        elif image_object.method["PVM_SPackArrSliceOrient"] == "coronal":
            oriented_image = np.flip(np.rot90(image_data, 3))
        elif image_object.method["PVM_SPackArrSliceOrient"] == "sagittal":
            oriented_image = np.flip(np.fliplr(image_data, 2))

    return oriented_image


def to_be_done_orient_bssfp(bssfp_scan, bssfp_reco):
    """
    Reorients reconstructed bssfp matrix to a consistent coordinate system

    Parameters
    ----------
    bssfp_scan: BrukerExp instance
        Contains the meta data and also 2dseq files
    bssfp_reco : nd.array
        shaped: (Echoes, Read, phase, slice, repetitions, channels)
                ( 0,      1,      2,      3,       4,         5)
        reconstructed bssfp data
    Returns
    -------
    oriented_bssfp: np.array
        This now has the same orientation as the scans in Paravision.
    """

    ACQ_patient_pos = bssfp_scan.acqp["ACQ_patient_pos"]
    if ACQ_patient_pos == "Head_Prone":
        if bssfp_scan.method["PVM_SPackArrSliceOrient"] == "axial":
            print(
                bssfp_scan.method["PVM_SPackArrSliceOrient"],
                ACQ_patient_pos,
                " - Caution this is not tested yet",
            )
            oriented_image = np.fliplr(np.rot90(bssfp_reco, 1))
        elif bssfp_scan.method["PVM_SPackArrSliceOrient"] == "coronal":
            # this is the tested variant
            oriented_image = np.flip(np.rot90(bssfp_reco, 3), 1)
        elif bssfp_scan.method["PVM_SPackArrSliceOrient"] == "sagittal":
            print(
                bssfp_scan.method["PVM_SPackArrSliceOrient"],
                ACQ_patient_pos,
                " - Caution this is not tested yet",
            )
            oriented_image = np.rot90(bssfp_reco, 2)
    elif ACQ_patient_pos == "Head_Supine":
        print(ACQ_patient_pos, " - Caution this is not tested yet")
        # Note This orientation is not tested
        if bssfp_scan.method["PVM_SPackArrSliceOrient"] == "axial":
            oriented_image = np.fliplr(np.rot90(bssfp_reco, 3))
        elif bssfp_scan.method["PVM_SPackArrSliceOrient"] == "coronal":
            oriented_image = np.flip(np.rot90(bssfp_reco, 3))
        elif bssfp_scan.method["PVM_SPackArrSliceOrient"] == "sagittal":
            oriented_image = np.flip(np.fliplr(bssfp_reco, 2))

    return oriented_image


def Get_Voxel_patch_params(voxel_instance, img_instance, voxel_num):
    """
    FIXME: Apparently this function returns wrong coordinates for a measuremnt on PV7
    FIXME it seems that the z axis is flipped
    FIXME so we have introduced a case Andre for the orientation
    FIXME This needs to be checked


    Returns patch parameters for a given voxel instance (e.g. MV PRESS)
    Parameters
    ----------
    voxel_instance
    img_instance
    voxel_num
    Returns
    -------
    dict: voxel_patch_params
    ----
    Example:
    mvpress=hp.BrukerExp('/user/Documents/Measurement/',5,'PyratID')
    sagittal = hp.BrukerExp('/user/Documents/Measurement/',6,'PyratID')

    patch_params_sagittal = Get_Voxel_patch_params(mvpress,sagittal,0)
    fig,ax=plt.subplots(1)
    ax.imshow(sagittal.seq2d[:,:,0]) # slice 0
    ax.add_patch(Rectangle(**patch_params_sagittal))
    This plots now a patch with the desired size onto the image
    """
    read_orient = img_instance.method['PVM_SPackArrReadOrient']
    vox_pos = voxel_instance.method["PVM_VoxArrPosition"]
    vox_size = voxel_instance.method["PVM_VoxArrSize"]
    pat_pos=img_instance.acqp['ACQ_patient_pos']
    n = voxel_num
    if pat_pos=='Head_Supine':
        if img_instance.method['PVM_SPackArrSliceOrient'] == 'coronal':
            if read_orient == 'H_F':
                # normal case
                patch_params_coronal = {
                    "xy": (
                        vox_pos[n][0] - vox_size[n][0] / 2.0,
                        vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][0],
                    "height": vox_size[n][2],
                }
                return patch_params_coronal
            elif read_orient == 'ANDRE':
                # nasty press voxel 09 Nov 2022
                patch_params_coronal = {
                    "xy": (
                        vox_pos[n][0] - vox_size[n][0] / 2.0,
                        -vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][0],
                    "height": vox_size[n][2],
                }
                return patch_params_coronal
            else:
                # FIXME this is not correct
                # normal case
                patch_params_coronal = {
                    "xy": (
                        vox_pos[n][0] - vox_size[n][0] / 2.0,
                        vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][0],
                    "height": vox_size[n][2],
                }
                return patch_params_coronal
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'axial':
            if read_orient == 'L_R':
                patch_params_axial = {
                    "xy": (
                        vox_pos[n][0] - vox_size[n][0] / 2.0,
                        -vox_pos[n][1] - vox_size[n][1] / 2.0,
                    ),
                    "width": vox_size[n][0],
                    "height": vox_size[n][1],
                }
                return patch_params_axial
            elif read_orient == 'A_P':
                patch_params_axial = {
                    "xy": (
                        vox_pos[n][0] - vox_size[n][0] / 2.0,
                        vox_pos[n][1] - vox_size[n][1] / 2.0,
                    ),
                    "width": vox_size[n][0],
                    "height": vox_size[n][1],
                }
                return patch_params_axial
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'sagittal':
            if read_orient == 'H_F':
                patch_params_sagittal = {
                    "xy": (
                        vox_pos[n][1] - vox_size[n][1] / 2.0,
                        vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][1],
                    "height": vox_size[n][2],
                }
                return patch_params_sagittal
            elif read_orient == 'A_P':
                # somehow this is the same as above
                patch_params_sagittal = {
                    "xy": (
                        vox_pos[n][1] - vox_size[n][1] / 2.0,
                        vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][1],
                    "height": vox_size[n][2],
                }
                return patch_params_sagittal
        else:
            return None
    elif pat_pos=='Head_Prone':
        if img_instance.method['PVM_SPackArrSliceOrient'] == 'axial':
            if read_orient == 'L_R':
                patch_params_axial = {
                    "xy": (
                        vox_pos[n][0] - vox_size[n][0] / 2.0,
                        -vox_pos[n][1] - vox_size[n][1] / 2.0,
                    ),
                    "width": vox_size[n][0],
                    "height": vox_size[n][1],
                }
                return patch_params_axial
            elif read_orient == 'A_P':
                logger.error('Patient position'+str(pat_pos)+'not implemented for read orientation '+str(read_orient)+' in axial orientation')
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'coronal':
            if read_orient == 'H_F':
                patch_params_coronal = {
                    "xy": (
                        vox_pos[n][0] - vox_size[n][0] / 2.0,
                        vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][0],
                    "height": vox_size[n][2],
                }
                return patch_params_coronal
            else:
                logger.error('Patient position' + str(pat_pos) + 'not implemented for read orientation ' + str(
                    read_orient) + ' in coronal orientation')
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'sagittal':
            if read_orient == 'H_F':
                patch_params_sagittal = {
                    "xy": (
                        vox_pos[n][1] - vox_size[n][1] / 2.0,
                        vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][1],
                    "height": vox_size[n][2],
                }
                return patch_params_sagittal
            elif read_orient == 'A_P':
                # somehow this is the same as above
                patch_params_sagittal = {
                    "xy": (
                        vox_pos[n][1] - vox_size[n][1] / 2.0,
                        vox_pos[n][2] - vox_size[n][2] / 2.0,
                    ),
                    "width": vox_size[n][1],
                    "height": vox_size[n][2],
                }
                return patch_params_sagittal
        else:
            return None


def Define_Extent_press(img_instance):
    """
    Calculates the extent of a anatomical image from the method file
    To be used on data of type: axial, coronal or sagittal

    Returns
    -------
    extent, np.array
    e.g.:
    """
    if img_instance.method['PVM_Fov'].shape == (2,):
        PVM_Fov = img_instance.method['PVM_Fov']
        slice_offset = img_instance.method['PVM_SPackArrSliceOffset']
        read_offset = img_instance.method['PVM_SPackArrReadOffset']
        phase_offset = img_instance.method['PVM_SPackArrPhase1Offset']
        read_orient = img_instance.method['PVM_SPackArrReadOrient']
        if img_instance.method['PVM_SPackArrSliceOrient'] == 'coronal':
            if read_orient == 'H_F':
                # coronal is xz plane
                # z is read, x is phase and y (into the plane) is slice
                # FOV(0) is in z dimension (up down on image in pv)
                # FOV(1) is in x dimension (left right on image in pv)
                return [
                    PVM_Fov[1] / 2 + phase_offset,  # [1] is x direction, i.e. phase
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,  # [0] is y direction,i.e. read offset in coronal
                    PVM_Fov[0] / 2 + read_offset,
                ]
            elif read_orient == 'L_R':
                # x is read, z is phase and y (into the plane) is slice
                # FOV(0) is in x dimension (left right on image in pv)
                # FOV(1) is in z dimension (up down on image in pv)
                # return array should be flipped, i.e. in case we had (-5,5,-3,3) before
                # we now go from (-3,3,-5,5)
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
            else:
                raise KeyError('This read orientation is not known for coronal')
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'axial':
            if read_orient == 'L_R':
                # axial is xy plane
                # y is phase
                # x is read
                # z is slice
                # FOV[0] is x dimension, left right on image
                # FOV[1] is y dimension, into the image
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                ]
            elif read_orient == 'A_P':
                # phase encoding flipped
                # axial is xy plane
                # y is read
                # x is phase
                # z is slice
                # FOV[0] is y dimension, left right on image
                # FOV[1] is x dimension, into the image
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                raise KeyError('This read orientation is not know for axial')
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'sagittal':
            # sagittal is yz plane
            if read_orient == 'H_F':
                # %FIXME THIS IS NOT RIGHT YET; NEEDS TO BE TESTED IN PHANTOM
                # phase is in y direction
                # read in z direction
                # FOV[0] is in z direction
                # FOV[1] in y direction
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset
                ]

    elif img_instance.method['PVM_Fov'].shape == (3,):
        # TODO adapt this in case it does not work properly for 3D FLASH acqusitions
        PVM_Fov = img_instance.method['PVM_Fov']
        slice_offset = img_instance.method['PVM_SPackArrSliceOffset']
        read_offset = img_instance.method['PVM_SPackArrReadOffset']
        phase_offset = img_instance.method['PVM_SPackArrPhase1Offset']
        read_orient = img_instance.method['PVM_SPackArrReadOrient']
        if img_instance.method['PVM_SPackArrSliceOrient'] == 'coronal':
            if read_orient == 'H_F':
                # coronal is xz plane
                # z is read, x is phase and y (into the plane) is slice
                # FOV(0) is in z dimension (up down on image in pv)
                # FOV(1) is in x dimension (left right on image in pv)

                return [
                    PVM_Fov[1] / 2 + phase_offset,  # [1] is x direction, i.e. phase
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,  # [0] is y direction,i.e. read offset in coronal
                    PVM_Fov[0] / 2 + read_offset,
                ]
            elif read_orient == 'L_R':
                # x is read, z is phase and y (into the plane) is slice
                # FOV(0) is in x dimension (left right on image in pv)
                # FOV(1) is in z dimension (up down on image in pv)
                # return array should be flipped, i.e. in case we had (-5,5,-3,3) before
                # we now go from (-3,3,-5,5)
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
            else:
                raise KeyError('This read orientation is not known for coronal')
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'axial':
            if read_orient == 'L_R':
                # axial is xy plane
                # y is phase
                # x is read
                # z is slice
                # FOV[0] is x dimension, left right on image
                # FOV[1] is y dimension, into the image
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset,
                ]
            elif read_orient == 'A_P':
                # phase encoding flipped
                # axial is xy plane
                # y is read
                # x is phase
                # z is slice
                # FOV[0] is y dimension, left right on image
                # FOV[1] is x dimension, into the image
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                raise KeyError('This read orientation is not know for axial')
        elif img_instance.method['PVM_SPackArrSliceOrient'] == 'sagittal':
            # sagittal is yz plane
            if read_orient == 'H_F':
                # %FIXME THIS IS NOT RIGHT YET; NEEDS TO BE TESTED IN PHANTOM
                # phase is in y direction
                # read in z direction
                # FOV[0] is in z direction
                # FOV[1] in y direction
                return [
                    PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    PVM_Fov[0] / 2 + read_offset,
                ]
            else:
                return [
                    PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[0] / 2 + read_offset,
                    -PVM_Fov[1] / 2 + phase_offset,
                    PVM_Fov[1] / 2 + phase_offset
                ]
        else:
            pass
    else:
        return None