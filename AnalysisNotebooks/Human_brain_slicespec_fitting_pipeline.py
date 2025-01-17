import numpy as np
import os
import scipy.io as sio
import hypermri.utils.utils_general as utg
import hypermri.utils.utils_spectroscopy as uts

import sys
# define paths:
sys.path.append('../')
import Template_utsw
basepath,_ = Template_utsw.import_all_packages()
from hypermri.utils.utils_fitting import def_fit_params, fit_data_pseudo_inv
from hypermri.utils.utils_general import get_gmr
from hypermri.utils.utils_spectroscopy import get_metab_cs_ppm, make_NDspec_6Dspec




study_ids=['','','','']


fit_reps=range(40)

injections=[[1,],[1,2],[1,2],[1,2]]
for idx,study_ID in enumerate(study_ids):
    print('Fitting ',study_ID)
    for injection in injections[idx]:
        print('Fitting raw spectrum, injection:',injection)
        spectral_data = np.squeeze(
            sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_' + str(injection) + '.mat'))['data'])
        fid = np.squeeze(sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_' + str(injection) + '.mat'))['fid'])
        time = np.squeeze(sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_' + str(injection) + '.mat'))['time'])
        ppm = np.array(
            np.squeeze(sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_' + str(injection) + '.mat'))['ppm']))
        #print(spectral_data.shape, fid.shape)
        bw = 5000  # Hz
        dwelltime = 1 / bw
        nsample_points = fid.shape[0]
        freq_range_Hz = uts.get_freq_axis(unit="Hz", sampling_dt=dwelltime, npoints=nsample_points)
        gmr = get_gmr(nucleus="13c")
        freq_range_ppm = freq_range_Hz / gmr / 3 + ppm[4095]
        #print(dwelltime, nsample_points)

        input_fid = make_NDspec_6Dspec(input_data=fid, provided_dims=["spec", "reps", "chan", "z"])
        input_spec = np.fft.fftshift(np.fft.fft(input_fid, axis=0), axes=(0,))
        mod_fid = np.conj(np.flip(input_fid, axis=0))
        mod_spec = np.flip(np.fft.fftshift(np.fft.fft(mod_fid, axis=0), axes=(0,)), axis=0)
        fit_data=mod_spec
        metabs = ['pyruvate', 'lactate', 'bicarbonate', 'pyruvatehydrate']

        niter = 4  # number of iterations:
        npoints = 31  # number of tested points per iteration:
        rep_fitting = 15

        # define fit parameters:
        fit_params = {}
        fit_params['signal_domain'] = 'spectral'
        # define peak frequencies:
        fit_params["metabs"] = metabs

        fit_params["b0"] = 3
        fit_params["init_fit"] = True
        fit_params["coff"] = 0
        fit_params["zoomfactor"] = 1.5

        fit_params["metab_t2s"] = [0.03 for _ in metabs]
        fit_params["max_t2_s"] = [0.07, 0.07, 0.07, 0.07]
        fit_params["min_t2_s"] = [0.01, 0.01, 0.01, 0.01]

        fit_params["range_t2s_s"] = 0.1

        gmr = get_gmr(nucleus="13c")

        fit_params["freq_range_ppm"] = freq_range_ppm
        fit_params["freq_range_Hz"] = freq_range_Hz

        fit_params["range_freqs_Hz"] = 40.

        fit_params["show_tqdm"] = True

        # define peak frequencies:
        fit_params["metabs_freqs_ppm"] = [get_metab_cs_ppm(metab=m) for m in metabs]
        # define peak frequencies:
        max_timepoint = np.argmax(np.sum(np.abs(fit_data[:, 0, 0, 0, :, 0]), axis=0))
        max_peakindex = uts.find_npeaks(input_data=np.squeeze(np.abs(fit_data[:, 0, 0, 0, max_timepoint, 0])), npeaks=1,
                                        plot=False, freq_range=freq_range_ppm)

        pyr_ppm_lit = uts.get_metab_cs_ppm(metab="pyruvate")
        pyr_ppm_diff_lit_meas = pyr_ppm_lit - fit_params['freq_range_ppm'][max_peakindex]
        fit_params["metabs_freqs_ppm"] = [uts.get_metab_cs_ppm(metab=m) - pyr_ppm_diff_lit_meas for m in metabs]

        fit_params["niter"] = niter  # number of iterations:
        fit_params["npoints"] = npoints  # number of tested points per iteration:
        fit_params["rep_fitting"] = rep_fitting  # number of tested points per iteration:

        fit_params["fit_range_repetitions"] = fit_reps
        fit_params = def_fit_params(fit_params=fit_params)
        fit_spectrums, fit_amps, fit_freqs, fit_t2s, fit_stds = fit_data_pseudo_inv(input_data=fit_data,
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
        import pickle

        utg.save_as_pkl(dir_path=basepath + 'fit_results/',
                        filename=study_ID + '_fit_spectra_study_'+str(injection)+'_raw_spec',
                        file=fit_results,
                        file_keys=fit_results.keys(), )