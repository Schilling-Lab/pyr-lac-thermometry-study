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




study_ids=['3TC230','3TC718','3TC741','3TC755']


fit_reps=range(40)
fit_processed=False
t2max=0.4
freq_max=100

injections=[[1,],[1,2],[1,2],[1,2]]
for idx,study_ID in enumerate(study_ids):
    print('Fitting ',study_ID)
    if fit_processed==True:
        print('Fitting processed spectrum')
        # Fitting processed spectrum by performing an ifft first, then taking the first 1000 points of that fid
        # to fit
        for injection in injections[idx]:

            rawdata = np.squeeze(sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_'+str(injection)+'.mat'))['data'])
            ppm = np.array(np.squeeze(sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_'+str(injection)+'.mat'))['ppm']))


            input_data = make_NDspec_6Dspec(input_data=rawdata, provided_dims=["spec", "reps", "chan", "z"])
            input_fid = np.fft.ifft(np.fft.ifftshift(input_data, axes=0), axis=0)
            mod_fid = np.conj(np.flip(input_fid, axis=0))

            cut_off = 1000

            mod_fid_short = mod_fid[:cut_off, ...]

            mod_spec_short = np.fft.fftshift(np.fft.fft(mod_fid_short, axis=0), axes=(0,))
            mod_spec = np.fft.fftshift(np.fft.fft(mod_fid, axis=0), axes=(0,))


            nsample_points = cut_off
            gmr = get_gmr(nucleus="13c")
            hz_axis_raw = (ppm - ppm[4095]) * gmr * 3
            bw_hz = uts.get_bw_from_freq_axis(freq_axis=hz_axis_raw)
            sampling_dt = utg.get_sampling_dt(bw_hz=bw_hz)

            freq_range_Hz = uts.get_freq_axis(unit="Hz", sampling_dt=sampling_dt, npoints=nsample_points)
            hz_axis_short = freq_range_Hz
            ppm_axis_short = hz_axis_short / gmr / 3 + ppm[4095]

            metabs = ['pyruvate', 'lactate', 'bicarbonate', 'pyruvatehydrate']

            niter = 4  # number of iterations:
            npoints = 31  # number of tested points per iteration:
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
            fit_params["max_t2_s"] = [t2max, t2max, t2max, t2max]
            fit_params["min_t2_s"] = [0.01, 0.01, 0.01, 0.01]

            fit_params["range_t2s_s"] = t2max-0.01

            gmr = get_gmr(nucleus="13c")

            fit_params["freq_range_ppm"] = ppm
            fit_params["freq_range_Hz"] = hz_axis_raw

            fit_params["range_freqs_Hz"] = freq_max

            fit_params["show_tqdm"] = True

            # define peak frequencies:
            fit_params["metabs_freqs_ppm"] = [get_metab_cs_ppm(metab=m) for m in metabs]
            # define peak frequencies:
            max_timepoint = np.argmax(np.sum(np.abs(mod_spec[:, 0, 0, 0, :, 0]), axis=0))
            max_peakindex = uts.find_npeaks(input_data=np.squeeze(np.abs(mod_spec[:, 0, 0, 0, max_timepoint, 0])), npeaks=1,
                                            plot=False, freq_range=ppm)

            pyr_ppm_lit = uts.get_metab_cs_ppm(metab="pyruvate")

            pyr_ppm_diff_lit_meas = pyr_ppm_lit - fit_params['freq_range_ppm'][max_peakindex]
            fit_params["metabs_freqs_ppm"] = [uts.get_metab_cs_ppm(metab=m) - pyr_ppm_diff_lit_meas for m in metabs]
            print('diff',pyr_ppm_diff_lit_meas, 'lit',pyr_ppm_lit, 'measured',fit_params['freq_range_ppm'][max_peakindex])
            print(fit_params["metabs_freqs_ppm"])


            fit_params["niter"] = niter  # number of iterations:
            fit_params["npoints"] = npoints  # number of tested points per iteration:
            fit_params["rep_fitting"] = rep_fitting  # number of tested points per iteration:

            fit_params["fit_range_repetitions"] = fit_reps
            fit_params = def_fit_params(fit_params=fit_params)


            fit_params_short = fit_params.copy()
            fit_params_short['nsamplepoints'] = cut_off
            fit_params_short['time_axis'] = fit_params_short['time_axis'][:fit_params_short['nsamplepoints']]

            fit_params_short['freq_range_Hz'] = hz_axis_short
            fit_params_short['freq_range_ppm'] = ppm_axis_short

            fit_spectrums, fit_amps, fit_freqs, fit_t2s, fit_stds = fit_data_pseudo_inv(input_data=mod_spec_short,
                                                                                        fit_params=fit_params_short,
                                                                                        dbplot=False,
                                                                                        use_multiprocessing=True)
            fit_results = {}
            fit_results["fit_spectrums"] = fit_spectrums
            fit_results["fit_freqs"] = fit_freqs
            fit_results["fit_amps"] = fit_amps
            fit_results["fit_t2s"] = fit_t2s
            fit_results["fit_params"] = fit_params_short
            fit_results["fit_stds"] = fit_stds
            import pickle

            utg.save_as_pkl(dir_path=basepath + 'fit_results/',
                            filename=study_ID + '_fit_spectra_study_'+str(injection)+'_processed_spec__'+str(freq_max)+'Hz_'+str(t2max)+'ms_',
                            file=fit_results,
                            file_keys=fit_results.keys(), )
    else:
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
            fit_params["max_t2_s"] = [t2max, t2max, t2max, t2max]
            fit_params["min_t2_s"] = [0.01, 0.01, 0.01, 0.01]

            fit_params["range_t2s_s"] =t2max-0.01

            gmr = get_gmr(nucleus="13c")

            fit_params["freq_range_ppm"] = freq_range_ppm
            fit_params["freq_range_Hz"] = freq_range_Hz

            fit_params["range_freqs_Hz"] = freq_max

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
                            filename=study_ID + '_fit_spectra_study_'+str(injection)+'_raw_spec__'+str(freq_max)+'Hz_'+str(t2max)+'ms_',
                            file=fit_results,
                            file_keys=fit_results.keys(), )

            
            
            
#redoing with different t2max,freqmax
fit_reps=range(40)
fit_processed=True
t2max=0.4
freq_max=50

injections=[[1,],[1,2],[1,2],[1,2]]
for idx,study_ID in enumerate(study_ids):
    print('Fitting ',study_ID)
    if fit_processed==True:
        print('Fitting processed spectrum')
        # Fitting processed spectrum by performing an ifft first, then taking the first 1000 points of that fid
        # to fit
        for injection in injections[idx]:

            rawdata = np.squeeze(sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_'+str(injection)+'.mat'))['data'])
            ppm = np.array(np.squeeze(sio.loadmat(os.path.join(basepath, str(study_ID) + '_raw_'+str(injection)+'.mat'))['ppm']))


            input_data = make_NDspec_6Dspec(input_data=rawdata, provided_dims=["spec", "reps", "chan", "z"])
            input_fid = np.fft.ifft(np.fft.ifftshift(input_data, axes=0), axis=0)
            mod_fid = np.conj(np.flip(input_fid, axis=0))

            cut_off = 1000

            mod_fid_short = mod_fid[:cut_off, ...]

            mod_spec_short = np.fft.fftshift(np.fft.fft(mod_fid_short, axis=0), axes=(0,))
            mod_spec = np.fft.fftshift(np.fft.fft(mod_fid, axis=0), axes=(0,))


            nsample_points = cut_off
            gmr = get_gmr(nucleus="13c")
            hz_axis_raw = (ppm - ppm[4095]) * gmr * 3
            bw_hz = uts.get_bw_from_freq_axis(freq_axis=hz_axis_raw)
            sampling_dt = utg.get_sampling_dt(bw_hz=bw_hz)

            freq_range_Hz = uts.get_freq_axis(unit="Hz", sampling_dt=sampling_dt, npoints=nsample_points)
            hz_axis_short = freq_range_Hz
            ppm_axis_short = hz_axis_short / gmr / 3 + ppm[4095]

            metabs = ['pyruvate', 'lactate', 'bicarbonate', 'pyruvatehydrate']

            niter = 4  # number of iterations:
            npoints = 31  # number of tested points per iteration:
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
            fit_params["max_t2_s"] = [t2max, t2max, t2max, t2max]
            fit_params["min_t2_s"] = [0.01, 0.01, 0.01, 0.01]

            fit_params["range_t2s_s"] = t2max-0.01

            gmr = get_gmr(nucleus="13c")

            fit_params["freq_range_ppm"] = ppm
            fit_params["freq_range_Hz"] = hz_axis_raw

            fit_params["range_freqs_Hz"] = freq_max

            fit_params["show_tqdm"] = True

            # define peak frequencies:
            fit_params["metabs_freqs_ppm"] = [get_metab_cs_ppm(metab=m) for m in metabs]
            # define peak frequencies:
            max_timepoint = np.argmax(np.sum(np.abs(mod_spec[:, 0, 0, 0, :, 0]), axis=0))
            max_peakindex = uts.find_npeaks(input_data=np.squeeze(np.abs(mod_spec[:, 0, 0, 0, max_timepoint, 0])), npeaks=1,
                                            plot=False, freq_range=ppm)

            pyr_ppm_lit = uts.get_metab_cs_ppm(metab="pyruvate")

            pyr_ppm_diff_lit_meas = pyr_ppm_lit - fit_params['freq_range_ppm'][max_peakindex]
            fit_params["metabs_freqs_ppm"] = [uts.get_metab_cs_ppm(metab=m) - pyr_ppm_diff_lit_meas for m in metabs]
            print('diff',pyr_ppm_diff_lit_meas, 'lit',pyr_ppm_lit, 'measured',fit_params['freq_range_ppm'][max_peakindex])
            print(fit_params["metabs_freqs_ppm"])


            fit_params["niter"] = niter  # number of iterations:
            fit_params["npoints"] = npoints  # number of tested points per iteration:
            fit_params["rep_fitting"] = rep_fitting  # number of tested points per iteration:

            fit_params["fit_range_repetitions"] = fit_reps
            fit_params = def_fit_params(fit_params=fit_params)


            fit_params_short = fit_params.copy()
            fit_params_short['nsamplepoints'] = cut_off
            fit_params_short['time_axis'] = fit_params_short['time_axis'][:fit_params_short['nsamplepoints']]

            fit_params_short['freq_range_Hz'] = hz_axis_short
            fit_params_short['freq_range_ppm'] = ppm_axis_short

            fit_spectrums, fit_amps, fit_freqs, fit_t2s, fit_stds = fit_data_pseudo_inv(input_data=mod_spec_short,
                                                                                        fit_params=fit_params_short,
                                                                                        dbplot=False,
                                                                                        use_multiprocessing=True)
            fit_results = {}
            fit_results["fit_spectrums"] = fit_spectrums
            fit_results["fit_freqs"] = fit_freqs
            fit_results["fit_amps"] = fit_amps
            fit_results["fit_t2s"] = fit_t2s
            fit_results["fit_params"] = fit_params_short
            fit_results["fit_stds"] = fit_stds
            import pickle

            utg.save_as_pkl(dir_path=basepath + 'fit_results/',
                            filename=study_ID + '_fit_spectra_study_'+str(injection)+'_processed_spec__'+str(freq_max)+'Hz_'+str(t2max)+'ms_',
                            file=fit_results,
                            file_keys=fit_results.keys(), )
    else:
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
            fit_params["max_t2_s"] = [t2max, t2max, t2max, t2max]
            fit_params["min_t2_s"] = [0.01, 0.01, 0.01, 0.01]

            fit_params["range_t2s_s"] =t2max-0.01

            gmr = get_gmr(nucleus="13c")

            fit_params["freq_range_ppm"] = freq_range_ppm
            fit_params["freq_range_Hz"] = freq_range_Hz

            fit_params["range_freqs_Hz"] = freq_max

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
                            filename=study_ID + '_fit_spectra_study_'+str(injection)+'_raw_spec__'+str(freq_max)+'Hz_'+str(t2max)+'ms_',
                            file=fit_results,
                            file_keys=fit_results.keys(), )