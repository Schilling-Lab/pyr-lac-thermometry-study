from .utils_logging import init_default_logger
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


logger = init_default_logger(__name__)


def find_max_sig_rep(dnp_bssfp_data, phip_bssfp_data):
    """
    Finds the repetitions with the highest signal for two bssfp datasets.
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
    Returns
    -------
    dnp_images[:,:,:,dnp_max_sign_rep]: dnp bssfp image stack at the highest signal repetition
        Has shape (Read, phase, slice)

    phip_images[:,:,:,phip_max_sign_rep]: phip bssfp image stack at the highest signal repetition
        has shape (Read, phase, slice)

    Plot of the process and result
    Example
    ------
    dnp_max_sig_rep,phip_max_sig_rep = find_max_sig_rep(dnp_bssfp_pv_complex_reco_combined_shift,phip_bssfp_pv_complex_reco_combined_shift)
    """
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
    itgl_dnp = []
    itgl_phip = []
    for rep in range(phip_images.shape[3]):
        itgl_phip.append(np.sum(np.abs(phip_images[:, :, :, rep])))
        itgl_dnp.append(np.sum(np.abs(dnp_images[:, :, :, rep])))
    dnp_max_sign_rep = np.argmin(np.abs(itgl_dnp - np.max(itgl_dnp)))
    phip_max_sign_rep = np.argmin(np.abs(itgl_phip - np.max(itgl_phip)))
    plt.close("all")
    fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
    ax[0].plot(itgl_dnp, label="dnp")
    ax[0].plot(itgl_phip, label="phip")
    ax[0].scatter(dnp_max_sign_rep, itgl_dnp[dnp_max_sign_rep], color="r")
    ax[0].scatter(phip_max_sign_rep, itgl_phip[phip_max_sign_rep], color="r")
    ax[0].legend()
    ax[0].set_xlabel("Repetition")
    ax[0].set_ylabel("Sum over all pixels and slices")
    ax[0].set_title(
        "PHIP = " + str(phip_max_sign_rep) + " // DNP = " + str(dnp_max_sign_rep)
    )
    ax[1].imshow(np.sum(np.abs(phip_images[:, :, :, phip_max_sign_rep]), axis=2))
    ax[2].imshow(np.sum(np.abs(dnp_images[:, :, :, dnp_max_sign_rep]), axis=2))
    ax[1].set_title("PHIP")
    ax[2].set_title("DNP")
    fig.suptitle(
        "Summed over all slices and displayed for the rep with the highest signal"
    )
    return (
        dnp_images[:, :, :, dnp_max_sign_rep],
        phip_images[:, :, :, phip_max_sign_rep],
    )


def apply_sig_range(dnp_bssfp_data, phip_bssfp_data, signal_range_dict):
    """
    Cuts bssfp data in a range of repetitions chose before. Usually this range is given by find_sig_range_reps
    Parameters
    ----------
    dnp_bssfp_data: array-like
        Contains multi rep, 3D bssfp data, usually this is only pyruvate or lactate channel

    phip_bssfp_data:array-like
        Contains multi rep, 3D bssfp data, usually this is only pyruvate or lactate channel

    dnp_sig_reps: list of indices

    phip_sig_reps: list of indices

    Returns
    -------
    dnp_images[:,:,:,dnp_sig_reps[0]:dnp_sig_reps[1]] : bssfp-array for only chosen reptition range
    phip_images[:,:,:,phip_sig_reps[0]:phip_sig_reps[1]] : bssfp-array for only chosen reptition range
    """
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
    dnp_sig_reps = signal_range_dict["dnp"]
    phip_sig_reps = signal_range_dict["phip"]
    return np.abs(
        np.sum(dnp_images[:, :, :, dnp_sig_reps[0] : dnp_sig_reps[1]], axis=3)
    ), np.abs(np.sum(phip_images[:, :, :, phip_sig_reps[0] : phip_sig_reps[1]], axis=3))


def calculate_ssi(dnp_bssfp_data, dnp_bssfp, phip_bssfp_data, phip_bssfp):
    """

    Parameters
    ----------
    dnp_bssfp_data: array
        3D array containg magnitude bssfp image data where a background has been subtracted
    phip_bssfp_data: array
        3D array containg magnitude bssfp image data where a background has been subtracted
    Returns
    -------
    structural_sim_indices: dict
        contains structural similarity indices for different slice orientations
    mean_squared_errors: dict
        mean squared errors for different slice orientations
    snr_per_slice_dnp: dict
        SNR value for each slice in the different orientations for DNP bssfp
    snr_per_slice_phip: dict
        SNR value for each slice in the different orientations for PHIP bssfp
    """
    structural_sim_indices = dict()
    mean_squared_errors = dict()
    snr_per_slice_dnp = dict()
    snr_per_slice_phip = dict()
    dnp_std = dnp_bssfp.ssi_background_std
    phip_std = phip_bssfp.ssi_background_std

    keys = ["coronal", "axial", "sagittal", "3D"]
    # coronal slices
    ssi_cor = []
    mse_cor = []
    snr_per_sl_cor_phip = []
    snr_per_sl_cor_dnp = []

    for idx in range(dnp_bssfp_data.shape[2]):
        phip_slice = np.abs(
            (phip_bssfp_data[:, :, idx]) / np.mean(phip_bssfp_data[:, :, idx])
        )
        dnp_slice = np.abs(
            (dnp_bssfp_data[:, :, idx]) / np.mean(dnp_bssfp_data[:, :, idx])
        )
        max_vals = [np.max(dnp_slice), np.max(phip_slice)]
        min_vals = [np.min(dnp_slice), np.min(phip_slice)]
        snr_per_sl_cor_phip.append(np.sum(phip_bssfp_data[:, :, idx] / phip_std))
        snr_per_sl_cor_dnp.append(np.sum(dnp_bssfp_data[:, :, idx] / dnp_std))
        ssi_cor.append(
            ssim(phip_slice, dnp_slice, data_range=np.max(max_vals) - np.min(min_vals))
        )
        mse_cor.append(mean_squared_error(phip_slice, dnp_slice))

    snr_per_slice_dnp.update({"coronal": snr_per_sl_cor_dnp})
    snr_per_slice_phip.update({"coronal": snr_per_sl_cor_phip})

    structural_sim_indices.update({"coronal": ssi_cor})
    mean_squared_errors.update({"coronal": mse_cor})

    # sagittal slices
    ssi_sag = []
    mse_sag = []
    snr_per_sl_sag_phip = []
    snr_per_sl_sag_dnp = []

    for idx in range(dnp_bssfp_data.shape[1]):
        phip_slice = np.abs(
            (phip_bssfp_data[:, idx, :]) / np.mean(phip_bssfp_data[:, idx, :])
        )
        dnp_slice = np.abs(
            (dnp_bssfp_data[:, idx, :]) / np.mean(dnp_bssfp_data[:, idx, :])
        )
        max_vals = [np.max(dnp_slice), np.max(phip_slice)]
        min_vals = [np.min(dnp_slice), np.min(phip_slice)]
        snr_per_sl_sag_phip.append(np.sum(phip_bssfp_data[:, idx, :] / phip_std))
        snr_per_sl_sag_dnp.append(np.sum(dnp_bssfp_data[:, idx, :] / dnp_std))
        ssi_sag.append(
            ssim(phip_slice, dnp_slice, data_range=np.max(max_vals) - np.min(min_vals))
        )
        mse_sag.append(mean_squared_error(phip_slice, dnp_slice))

    snr_per_slice_dnp.update({"sagittal": snr_per_sl_sag_dnp})
    snr_per_slice_phip.update({"sagittal": snr_per_sl_sag_phip})

    structural_sim_indices.update({"sagittal": ssi_sag})
    mean_squared_errors.update({"sagittal": mse_sag})
    # axial slices
    ssi_ax = []
    mse_ax = []
    snr_per_sl_ax_phip = []
    snr_per_sl_ax_dnp = []
    for idx in range(dnp_bssfp_data.shape[0]):
        phip_slice = np.abs(
            (phip_bssfp_data[idx, :, :]) / np.mean(phip_bssfp_data[idx, :, :])
        )
        dnp_slice = np.abs(
            (dnp_bssfp_data[idx, :, :]) / np.mean(dnp_bssfp_data[idx, :, :])
        )
        max_vals = [np.max(dnp_slice), np.max(phip_slice)]
        min_vals = [np.min(dnp_slice), np.min(phip_slice)]
        snr_per_sl_ax_phip.append(np.sum(phip_bssfp_data[idx, :, :] / phip_std))
        snr_per_sl_ax_dnp.append(np.sum(dnp_bssfp_data[idx, :, :] / dnp_std))
        ssi_ax.append(
            ssim(phip_slice, dnp_slice, data_range=np.max(max_vals) - np.min(min_vals))
        )
        mse_ax.append(mean_squared_error(phip_slice, dnp_slice))

    snr_per_slice_dnp.update({"axial": snr_per_sl_ax_dnp})
    snr_per_slice_phip.update({"axial": snr_per_sl_ax_phip})

    structural_sim_indices.update({"axial": ssi_ax})
    mean_squared_errors.update({"sagittal": mse_ax})

    # 3D, so whole images
    phip_3D_meaned = phip_bssfp_data / np.mean(phip_bssfp_data)
    dnp_3D_meaned = dnp_bssfp_data / np.mean(dnp_bssfp_data)

    structural_sim_indices.update(
        {
            "3D": ssim(
                phip_3D_meaned,
                dnp_3D_meaned,
                data_range=np.max([dnp_3D_meaned, phip_3D_meaned])
                - np.min([dnp_3D_meaned, phip_3D_meaned]),
            )
        }
    )

    max_snr_slice_cor_dnp = np.argmin(
        np.abs(snr_per_sl_cor_dnp - np.max(snr_per_sl_cor_dnp))
    )
    max_snr_slice_ax_dnp = np.argmin(
        np.abs(snr_per_sl_ax_dnp - np.max(snr_per_sl_ax_dnp))
    )
    max_snr_slice_sag_dnp = np.argmin(
        np.abs(snr_per_sl_sag_dnp - np.max(snr_per_sl_sag_dnp))
    )

    max_snr_slice_cor_phip = np.argmin(
        np.abs(snr_per_sl_cor_phip - np.max(snr_per_sl_cor_phip))
    )
    max_snr_slice_ax_phip = np.argmin(
        np.abs(snr_per_sl_ax_phip - np.max(snr_per_sl_ax_phip))
    )
    max_snr_slice_sag_phip = np.argmin(
        np.abs(snr_per_sl_sag_phip - np.max(snr_per_sl_sag_phip))
    )

    structural_sim_indices.update(
        {"coronal_best_slice_dnp": ssi_cor[max_snr_slice_cor_dnp]}
    )
    structural_sim_indices.update(
        {"sagittal_best_slice_dnp": ssi_sag[max_snr_slice_sag_dnp]}
    )
    structural_sim_indices.update(
        {"axial_best_slice_dnp": ssi_ax[max_snr_slice_ax_dnp]}
    )

    structural_sim_indices.update(
        {"coronal_best_slice_phip": ssi_cor[max_snr_slice_cor_phip]}
    )
    structural_sim_indices.update(
        {"sagittal_best_slice_phip": ssi_sag[max_snr_slice_sag_phip]}
    )
    structural_sim_indices.update(
        {"axial_best_slice_phip": ssi_ax[max_snr_slice_ax_phip]}
    )

    # plotting
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), tight_layout=True)

    ax[0, 0].plot(range(phip_bssfp_data.shape[2]), structural_sim_indices["coronal"])
    ax[0, 1].plot(range(phip_bssfp_data.shape[1]), structural_sim_indices["sagittal"])
    ax[0, 2].plot(range(phip_bssfp_data.shape[0]), structural_sim_indices["axial"])

    ax[0, 0].scatter(
        range(phip_bssfp_data.shape[2]),
        structural_sim_indices["coronal"],
        marker="x",
        color="r",
    )
    ax[0, 1].scatter(
        range(phip_bssfp_data.shape[1]),
        structural_sim_indices["sagittal"],
        marker="x",
        color="r",
    )
    ax[0, 2].scatter(
        range(phip_bssfp_data.shape[0]),
        structural_sim_indices["axial"],
        marker="x",
        color="r",
    )

    ax[0, 0].set_title("Coronal")
    ax[0, 1].set_title("Sagittal")
    ax[0, 2].set_title("Axial")

    ax[0, 0].set_xlabel("Slices")
    ax[0, 1].set_xlabel("Slices")
    ax[0, 2].set_xlabel("Slices")

    ax[0, 0].set_ylabel("SSI")

    ax[1, 0].plot(
        range(dnp_bssfp_data.shape[2]), snr_per_slice_dnp["coronal"], label="DNP"
    )
    ax[1, 1].plot(
        range(dnp_bssfp_data.shape[1]), snr_per_slice_dnp["sagittal"], label="DNP"
    )
    ax[1, 2].plot(
        range(dnp_bssfp_data.shape[0]), snr_per_slice_dnp["axial"], label="DNP"
    )

    ax[1, 0].plot(
        range(phip_bssfp_data.shape[2]), snr_per_slice_phip["coronal"], label="PHIP"
    )
    ax[1, 1].plot(
        range(phip_bssfp_data.shape[1]), snr_per_slice_phip["sagittal"], label="PHIP"
    )
    ax[1, 2].plot(
        range(phip_bssfp_data.shape[0]), snr_per_slice_phip["axial"], label="PHIP"
    )

    ax[0, 0].scatter(
        max_snr_slice_cor_dnp, structural_sim_indices["coronal_best_slice_dnp"]
    )
    ax[0, 0].scatter(
        max_snr_slice_cor_phip, structural_sim_indices["coronal_best_slice_phip"]
    )

    ax[0, 1].scatter(
        max_snr_slice_sag_dnp, structural_sim_indices["sagittal_best_slice_phip"]
    )
    ax[0, 1].scatter(
        max_snr_slice_sag_phip, structural_sim_indices["sagittal_best_slice_phip"]
    )

    ax[0, 2].scatter(
        max_snr_slice_ax_dnp, structural_sim_indices["axial_best_slice_phip"]
    )
    ax[0, 2].scatter(
        max_snr_slice_ax_phip,
        structural_sim_indices["axial_best_slice_phip"],
        label="SSI from slice with highest SNR",
    )

    ax[1, 0].vlines(
        max_snr_slice_cor_dnp,
        0,
        np.max(snr_per_slice_dnp["coronal"]),
        linestyle="dashed",
        color="k",
    )
    ax[1, 1].vlines(
        max_snr_slice_sag_dnp,
        0,
        np.max(snr_per_slice_dnp["sagittal"]),
        linestyle="dashed",
        color="k",
    )
    ax[1, 2].vlines(
        max_snr_slice_ax_dnp,
        0,
        np.max(snr_per_slice_dnp["axial"]),
        linestyle="dashed",
        color="k",
    )

    ax[1, 0].vlines(
        max_snr_slice_cor_phip,
        0,
        np.max(snr_per_slice_phip["coronal"]),
        linestyle="dashed",
        color="k",
    )
    ax[1, 1].vlines(
        max_snr_slice_sag_phip,
        0,
        np.max(snr_per_slice_phip["sagittal"]),
        linestyle="dashed",
        color="k",
    )
    ax[1, 2].vlines(
        max_snr_slice_ax_phip,
        0,
        np.max(snr_per_slice_phip["axial"]),
        linestyle="dashed",
        color="k",
        label="slice with highest snr",
    )

    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[1, 2].legend()

    ax[1, 0].set_ylabel("Signal in slice")
    return (
        structural_sim_indices,
        mean_squared_errors,
        snr_per_slice_dnp,
        snr_per_slice_phip,
    )
