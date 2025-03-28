import numpy as np
import pickle
import datetime
import pandas as pd
import os
import hypermri
from scipy.optimize import curve_fit
import scipy.io as sio
from tqdm.auto import tqdm
from matplotlib import cm
from astropy.modeling import models, fitting
import hypermri.utils.utils_general as utg
import hypermri.utils.utils_spectroscopy as uts
import hypermri.utils.utils_fitting as utf
import sys
# define paths:
sys.path.append('../')
import Template_Cambridge
basepath,_ = Template_Cambridge.import_all_packages(scan_is_csi=True)

from mpl_interactions import image_segmenter_overlayed

from hypermri.utils.utils_fitting import def_fit_params, fit_t2_pseudo_inv, fit_freq_pseudo_inv, fit_data_pseudo_inv, fit_func_pseudo_inv, plot_fitted_spectra, basefunc
from hypermri.utils.utils_general import get_gmr, calc_sampling_time_axis
from hypermri.utils.utils_spectroscopy import apply_lb, multi_dim_linebroadening, get_metab_cs_ppm, generate_fid, get_freq_axis, make_NDspec_6Dspec, freq_to_index, find_npeaks


for studyfoldernum in tqdm([122,127,128,129,130,131,132,133,134]):
  
    studyfolder='HV-'+str(studyfoldernum) 
    animal_id  = studyfolder
    file_name_prefix = 'hv_'+str(studyfoldernum)
    dataset = sio.loadmat(os.path.join(basepath, studyfolder, file_name_prefix+'.mat'))['spec']
    dataset_header_struct_arr = sio.loadmat(os.path.join(basepath, studyfolder, file_name_prefix+"_header.mat"))['header']

    # Get field names from the structured array
    field_names = dataset_header_struct_arr.dtype.names

    dataset_header = {}

    # Iterate over the field names and process the data
    for field in field_names:
        # Access the data for each field
        dataset_header[field] = np.squeeze(dataset_header_struct_arr[field][0][0][0])



    for key in dataset_header['image'].dtype.names:
        temp = np.array(dataset_header['image'][key]).item()

        # Check if the structure is a NumPy array
        if isinstance(temp, np.ndarray):
            if temp.ndim == 0:
                value = temp.item()  # Use .item() for 0-dimensional array
            else:
                try:
                    value = temp[0][0]  # Use indexing for higher-dimensional array
                except:
                    pass
            dataset_header['image'][key] = value



    patient_info = {}
    patient_info['ID'] = animal_id
    if dataset_header['exam']['patsex'] == 0:
        patient_info['sex'] = 'male'
    else:
        patient_info['sex'] = 'female'
    patient_info['weight'] = float(dataset_header['exam']['patweight'] / 1e3)
    patient_info['pyr_vol'] = patient_info['weight'] * 0.4
    patient_info['scan_date'] = str(dataset_header['rdb_hdr']['scan_date'])
    patient_info['scan_time'] = str(dataset_header['rdb_hdr']['scan_time'])
    print(patient_info)


    import numpy as np

    rdb_hdr = dataset_header['rdb_hdr']
    fields =  rdb_hdr.dtype.names
    for f in fields:
        # print(f"{f}: {rdb_hdr[f]}")
        pass


    # repetition time:
    tr = dataset_header['image']['tr'] / 1e6

    tr = 4e3


    # bandwidth
    bw = dataset_header['rdb_hdr']['spectral_width']


    # center frequency:
    freq_cent_hz = dataset_header['rdb_hdr']['ps_mps_freq'] / 10.0 

    # gyromagnetic ratio MHz/T
    gmr_mhz_t = get_gmr(nucleus="13c")

    # B0 in Tesla:
    b0_off_t = freq_cent_hz / (gmr_mhz_t * 1e6)


    freq0 = 3 *  (gmr_mhz_t)

    freq_off_hz = freq_cent_hz - freq0*1e6

    freq_off_ppm = (freq_cent_hz - freq0*1e6) / (freq0*1e6) * 1e6

    # sampling time:
    dt = 1./bw

    # flip angle:
    fa = dataset_header['image']['mr_flip']


    dyn_fid = dataset


    # spectrum has to be flipped (and complex conjugated to have proper FID) (see flip_spec_complex_
    dyn_spec = dyn_fid

    freq_range = np.squeeze(get_freq_axis(npoints=dyn_spec.shape[0], sampling_dt=dt, unit='Hz'))
    time_axis = calc_sampling_time_axis(npoints=dyn_spec.shape[0], sampling_dt=dt)

    input_data_raw= make_NDspec_6Dspec(input_data=dyn_spec, provided_dims=["spec", "x", "y","z"])

    input_data_raw = np.conj(np.flip(input_data_raw, axis=0))
    ## Interpolate CSI data
    interpolation_factor=1
    input_data = utg.interpolate_dataset(input_data=input_data_raw,
                                               interp_size=(input_data_raw.shape[0],
                                                            input_data_raw.shape[1],
                                                            interpolation_factor*input_data_raw.shape[2],
                                                            interpolation_factor*input_data_raw.shape[3],
                                                            input_data_raw.shape[4],
                                                            input_data_raw.shape[5]),
                                               interp_method="cubic")

    # metabs = ['pyruvate', 'lactate', 'pyruvatehydrate', 'alanine', 'urea']
    metabs = ['pyruvate', 'lactate', 'bicarbonate', 'pyruvatehydrate']

    niter = 4 # number of iterations:
    npoints  = 31# number of tested points per iteration:
    rep_fitting = 15

    # define fit parameters:
    fit_params = {}

    # define peak frequencies:
    fit_params["metabs"] = metabs

    fit_params["b0"] = 3
    fit_params["init_fit"] = True
    fit_params["coff"] = 0
    fit_params["zoomfactor"] = 1.5

    fit_params["metab_t2s"] = [0.03 for _ in metabs]
    fit_params["max_t2_s"] = [0.4,0.4,0.4,0.4]
    fit_params["min_t2_s"] = [0.01, 0.01, 0.01, 0.01]

    fit_params["range_t2s_s"] = 0.39

    gmr = get_gmr(nucleus="13c")

    fit_params["freq_range_Hz"] = freq_range
    fit_params["freq_range_ppm"] = fit_params["freq_range_Hz"]/(gmr*fit_params['b0']) + 175
    fit_params["range_freqs_Hz"] = 100

    fit_params["show_tqdm"] = True

    # define peak frequencies:
    fit_params["metabs_freqs_ppm"] = [get_metab_cs_ppm(metab=m) for m in metabs]

    max_timepoint_1D = np.argmax(input_data.flatten())
    max_timepoint_6D = np.unravel_index(max_timepoint_1D, input_data.shape)
    result = input_data[(slice(None),) + max_timepoint_6D[1:]]
    max_peakindex = uts.find_npeaks(input_data=result,
                                    npeaks=4,
                                    freq_range=fit_params['freq_range_ppm'],
                                    plot=False,
                                   sort_order=None)

    # compare found pyruvate position with defined pyruvate position:
    pyr_data =  fit_params["freq_range_ppm"][max_peakindex][0]
    pyr_lit = fit_params["metabs_freqs_ppm"][0]
    print('Pyr found at',pyr_data)
    print('Pyr literature at',pyr_lit)
    diff_pyr_data_lit = pyr_data - pyr_lit
    fit_params["metabs_freqs_ppm"] = [get_metab_cs_ppm(metab=m)+diff_pyr_data_lit for m in metabs]
    print('---------------------',fit_params["metabs_freqs_ppm"])


    # peak_indices = uts.find_npeaks(input_data=

    fit_params["niter"] = niter # number of iterations:
    fit_params["npoints"] = npoints # number of tested points per iteration:
    fit_params["rep_fitting"] = rep_fitting # number of tested points per iteration:

    fit_params["fit_range_repetitions"] = dyn_spec.shape[0]
    fit_params["use_all_cores"] = True
    fit_params = def_fit_params(fit_params=fit_params)

    #for k in fit_params.keys():
    #    print(f"{k} : {fit_params[k]}")

    fit_spectrums, fit_amps, fit_freqs, fit_t2s,  fit_stds = fit_data_pseudo_inv(input_data=input_data,
                                                                          fit_params=fit_params, 
                                                                          dbplot=False,
                                                                          use_multiprocessing=True)


    fit_results = {}
    fit_results["fit_spectrums"] = fit_spectrums
    fit_results["fit_freqs"] = fit_freqs
    fit_results["fit_amps"] = fit_amps
    fit_results["fit_t2s"] = fit_t2s
    fit_results["fit_params"] = fit_params
    fit_results["fit_stds"] = fit_stds

    savepath=os.path.join(basepath,'2025_refit_t2_400ms_freq_100Hz')

    utg.save_as_pkl(dir_path=savepath,
                    filename=animal_id + '_fit_spectra_redone',
                    file = fit_results,
                    file_keys=fit_results.keys(),)
