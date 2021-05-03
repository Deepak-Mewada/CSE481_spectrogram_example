from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import mne

from mne.time_frequency import tfr_morlet

def diff_specs(sp, data_lp, ecog = False, roi_of_interest = 47, pad_val = 0.5,
               ecog_srate = 250, decim_spec = 50):
    """
    Computes difference spectrograms for ECoG data.
    Inputs:
            ecog : plotting ECoG (True) or EEG (False) spectrograms
            roi_of_interest : index of ROI that we want to analyze difference spectrograms for
            pad_val : used for removing edge effects in spectrogram calculation (sec)
            ecog_srate : sampling rate of ECoG data (Hz)
    """
    # change as needed for different data saves
    savename = 'diff_spec_eeg_tfr.h5'
    
    nROIs = 1
    # change this if you'd like to look at a different time window
    tlim = [-1, 1]
    tlim_orig = tlim.copy()
    tlim[0] -= pad_val
    tlim[1] += pad_val
    
    # change this for whatever appropriate subject names in your data
    pats_ids_in = ['EE'+str(val).zfill(2) for val in np.arange(1,16).tolist()]
    
    # Update with whatever classes are in your dataset
    event_dict = {'rest':1,'move':2}
    
    custom_rois = True
    n_chans_eeg = 61
    if custom_rois:
        custom_roi_inds = get_custom_motor_rois() # load custom roi's from precentral, postcentral, and inf parietal (AAL2)
    else:
        custom_roi_inds = None
    print("Determining ROIs")
    proj_mat_out,good_ROIs,chan_ind_vals_all = proj_mats_good_rois(['EE01_bH'],
                                                                   n_chans_all = n_chans_eeg,
                                                                   rem_bad_chans=False,
                                                                   dipole_dens_thresh=None,
                                                                   custom_roi_inds=custom_roi_inds,
                                                                   chan_cut_thres=n_chans_eeg,
                                                                   roi_proj_loadpath= data_lp+'proj_mat/')
    print("ROIs found")
    n_chans_all = n_chans_eeg
    proj_mat_out = proj_mat_out[:,roi_of_interest:(roi_of_interest+1),:]

    # load in the data
    X,y,_,_,sbj_order,_ = load_data(pats_ids_in, data_lp,
                                    n_chans_all=n_chans_all,
                                    test_day=None, tlim=tlim)
    X[np.isnan(X)] = 0 # set all NaN's to 0
    
    # Identify the number of unique labels (or classes) present
    labels_unique = np.unique(y)
    nb_classes = len(labels_unique)

    # Determine subjects in train/val/test sets for current fold
    n_subjs = len(pats_ids_in)
    train_sbj = np.arange(n_subjs)

    X_train = X.copy() # shape (n_epochs, n_channels, n_times)
    Y_train = y.copy()
    sbj_order_train = sbj_order.copy() # important for projection matrix input


    n_filts = 1
    per_sbj_power = []
    power_proj_diff = [[[] for j in range(n_filts)] for k in range(n_subjs)]
    for k,curr_train_sbj in tqdm(enumerate(train_sbj)):
        curr_ev_inds = np.nonzero(sbj_order_train==curr_train_sbj)[0]
        X_train_sbj = X_train[curr_ev_inds,...]
        Y_train_sbj = Y_train[curr_ev_inds]
        sbj_order_train_sbj = sbj_order_train[curr_ev_inds]

        # Create info for epochs array
        ch_names = ['eeg-'+str(ind) for ind in range(X_train_sbj.shape[1])]
        ch_types = ['eeg']*X_train_sbj.shape[1]
        info = mne.create_info(ch_names=ch_names, sfreq=ecog_srate, ch_types=ch_types)

        # Filter data using Conv1D
        X_train_sbj = np.expand_dims(X_train_sbj,1)

        # Create epoched data prior to time-frequency computation
        events = np.zeros([len(Y_train_sbj),3])
        events[:,0] = np.arange(events.shape[0])
        events[:,2] = Y_train_sbj
        events = events.astype('int')

        for j in range(n_filts):
            epochs = mne.EpochsArray(X_train_sbj[:,j,...], info=info, events=events, event_id=event_dict, tmin=tlim[0])

            # Compute and project power for move events
            power = compute_tfr(epochs,'move',tlim,freqs = np.arange(1, 124, 5),crop_val=pad_val,decim=decim_spec)
            power_move_proj = np.median(roi_proj_pow(power.data,sbj_order_train_sbj,nROIs,proj_mat_out,ecog),axis=0).squeeze()

            # Compute and project power for rest events
            power_2 = compute_tfr(epochs,'rest',tlim,freqs = np.arange(1, 124, 5),crop_val=pad_val,decim=decim_spec)
            power_rest_proj = np.median(roi_proj_pow(power_2.data,sbj_order_train_sbj,nROIs,proj_mat_out,ecog),axis=0).squeeze()

            # Take difference of rest and move average power
            power_proj_diff[k][j] = power_move_proj - power_rest_proj


    pow_sh = power_proj_diff[0][0].shape
    final_spec = np.asarray(power_proj_diff).reshape((-1,pow_sh[0],pow_sh[1])).mean(axis=0)
    # Create dummy power variable
    power_2.drop_channels(power_2.info['ch_names'][1:])
    power_2 = power_2.average()
    power_2.data[-2:] = final_spec # put data into dummy power variable

    # Save final spectrogram and time/frequencies
    power_2.save(sp+'sbj_avg_'+savename,overwrite=True)

    return power_2


def get_custom_motor_rois(regions=['precentral','postcentral','parietal_inf']):
    '''
    Returns ROI indices for those within the precentral, postcentral, and inferior parietal regions (accoring to AAL2)
    '''
    precentral_inds = [2263,2557,2558,2571,2587,2845,2846,2847,2858,2859,2873,2874,3113,3123,3124,3136,3137,3138,3151,3153,3154,3359,3360,3369,3370,3371,3383,3384,3559,3565,3566,3567,3568,3576,3577,3578,3579,3589,3590,3722,3723,3724,3729,3730,3731,3739,3740,3752,3837]
    postcentral_inds = [2236,2237,2238,2246,2247,2248,2545,2546,2547,2555,2556,2569,2570,2835,2836,2843,2844,2856,2857,2871,3110,3111,3112,3121,3122,3133,3134,3135,3149,3350,3351,3355,3356,3357,3358,3367,3368,3381,3382,3395,3555,3556,3557,3563,3564,3574,3575,3587,3588,3720,3721,3727,3728,3737,3738,3832,3834,3835,3836,3842,3843]
    parietal_inf_inds = [3106,3107,3108,3116,3117,3118,3119,3120,3131,3132,3143,3144,3145,3146,3147,3148,3161,3347,3348,3349,3352,3353,3354,3364,3365,3366,3376,3378,3379,3380,3553,3554,3561,3562]
    
    # Account for Matlab indexing starting at 1
    precentral_inds = [val-1 for val in precentral_inds]
    postcentral_inds = [val-1 for val in postcentral_inds]
    parietal_inf_inds = [val-1 for val in parietal_inf_inds]
    
#     custom_roi_inds = np.union1d(np.union1d(precentral_inds,postcentral_inds),parietal_inf_inds) #select for sensorimotor ROIs
    custom_roi_inds = []
    for val in regions:
        eval('custom_roi_inds.extend('+val+'_inds)')
    return custom_roi_inds


def proj_mats_good_rois(patient_ids,dipole_dens_thresh = .1, n_chans_all = 150,
                        roi_proj_loadpath = '.../',
                        atlas = 'none', rem_bad_chans = True, custom_roi_inds=None, chan_cut_thres = None):
    '''
    Loads projection matrix for each subject and determines good ROIs to use
    
    Parameters
    ----------
    dipole_dens_thresh : threshold to use when deciding good ROI's (based on average channel density for each ROI)
    n_chans_all : number of channels to output (should be >= to maximum number of channels across subjects)
    roi_proj_loadpath : where to load projection matrix CSV files
    atlas : ROI projection atlas to use (aal, loni, brodmann, or none)
    rem_bad_chans : whether to remove bad channels from projection step, defined from abnormal SD or IQR across entire day
    '''
    #Find good ROIs first
    df_all = []
    for s,patient in enumerate(patient_ids):
        df = pd.read_csv(roi_proj_loadpath+atlas+'_'+patient+'_elecs2ROI.csv')
        if s==0:
            dipole_densities = df.iloc[0]
        else:
            dipole_densities += df.iloc[0]
        df_all.append(df)

    dipole_densities = dipole_densities/len(patient_ids) 
    if custom_roi_inds is None:
        good_ROIs = np.nonzero(np.asarray(dipole_densities)>dipole_dens_thresh)[0]
    else:
        good_ROIs = custom_roi_inds.copy()
    
    #Now create projection matrix output (patients x roi x chans)
    n_pats = len(patient_ids)
    proj_mat_out = np.zeros([n_pats,len(good_ROIs),n_chans_all])
    chan_ind_vals_all = []
    for s,patient in enumerate(patient_ids):
        df_curr = df_all[s].copy()
        chan_ind_vals = np.nonzero(df_curr.transpose().mean().values!=0)[0][1:]
        chan_ind_vals_all.append(chan_ind_vals)
        if rem_bad_chans:
            # Load param file from pre-trained model
            file_pkl = open(roi_proj_loadpath+'bad_ecog_electrodes.pkl', 'rb')
            bad_elecs_ecog = pickle.load(file_pkl)
            file_pkl.close()
            inds2drop = bad_elecs_ecog[s]
            if chan_cut_thres is not None:
                all_inds = np.arange(df_curr.shape[0])
                inds2drop = np.union1d(inds2drop,all_inds[all_inds>chan_cut_thres])
            df_curr.iloc[inds2drop] = 0
            #Renormalize across ROIs
            sum_vals = df_curr.sum(axis=0).values
            for i in range(len(sum_vals)):
                df_curr.iloc[:,i] = df_curr.iloc[:,i]/sum_vals[i]
        n_chans_curr = len(chan_ind_vals) #df_curr.shape[0]
        tmp_mat = df_curr.values[chan_ind_vals,:]
        proj_mat_out[s,:,:n_chans_curr] = tmp_mat[:,good_ROIs].T
    return proj_mat_out,good_ROIs,chan_ind_vals_all


def load_data(pats_ids_in, lp, n_chans_all=64, test_day=None, tlim=[-1,1], event_types=['rest','move']):
    '''
    Load ECoG data from all subjects and combine (uses xarray variables)
    
    If len(pats_ids_in)>1, the number of electrodes will be padded or cut to match n_chans_all
    If test_day is not None, a variable with test data will be generated for the day specified
        If test_day = 'last', the last day will be set as the test day.
    '''
    if not isinstance(pats_ids_in, list):
        pats_ids_in = [pats_ids_in]
    sbj_order,sbj_order_test = [],[]
    X_test_subj,y_test_subj = [],[] #placeholder vals
    
    #Gather each subjects data, and concatenate all days
    for j in tqdm(range(len(pats_ids_in))):
        pat_curr = pats_ids_in[j]
        ep_data_in = xr.open_dataset(lp+pat_curr+'_ecog_data.nc')
        ep_times = np.asarray(ep_data_in.time)
        time_inds = np.nonzero(np.logical_and(ep_times>=tlim[0],ep_times<=tlim[1]))[0]
        n_ecog_chans = (len(ep_data_in.channels)-1)
        
        if test_day == 'last':
            test_day_curr = np.unique(ep_data_in.events)[-1] #select last day
        else:
            test_day_curr = test_day
        
        if n_chans_all < n_ecog_chans:
            n_chans_curr = n_chans_all
        else:
            n_chans_curr = n_ecog_chans
        
        days_all_in = np.asarray(ep_data_in.events)
        
        if test_day is None:
            #No test output here
            days_train = np.unique(days_all_in)
            test_day_curr = None
        else:
            days_train = np.unique(days_all_in)[:-1]
            day_test_curr = np.unique(days_all_in)[-1]
            days_test_inds = np.nonzero(days_all_in==day_test_curr)[0]
            
        #Compute indices of days_train in xarray dataset
        days_train_inds = []
        for day_tmp in list(days_train):
            days_train_inds.extend(np.nonzero(days_all_in==day_tmp)[0])
        
        #Extract data and labels
        dat_train = ep_data_in[dict(events=days_train_inds,channels=slice(0,n_chans_curr),
                                time=time_inds)].to_array().values.squeeze()
        labels_train = ep_data_in[dict(events=days_train_inds,channels=ep_data_in.channels[-1],
                                    time=0)].to_array().values.squeeze()
        
        sbj_order += [j]*dat_train.shape[0]
        
        if test_day is not None:
            dat_test = ep_data_in[dict(events=days_test_inds,channels=slice(0,n_chans_curr),
                                       time=time_inds)].to_array().values.squeeze()
            labels_test = ep_data_in[dict(events=days_test_inds,channels=ep_data_in.channels[-1],
                                      time=0)].to_array().values.squeeze()
            sbj_order_test += [j]*dat_test.shape[0]
            
        #Pad data in electrode dimension if necessary
        if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
            dat_sh = list(dat_train.shape)
            dat_sh[1] = n_chans_all
            #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
            X_pad = np.zeros(dat_sh)
            X_pad[:,:n_ecog_chans,...] = dat_train
            dat_train = X_pad.copy()
            
            if test_day is not None:
                dat_sh = list(dat_test.shape)
                dat_sh[1] = n_chans_all
                #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                X_pad = np.zeros(dat_sh)
                X_pad[:,:n_ecog_chans,...] = dat_test
                dat_test = X_pad.copy() 
        
        #Concatenate across subjects
        if j==0:
            X_subj = dat_train.copy()
            y_subj = labels_train.copy()
            if test_day is not None:
                X_test_subj = dat_test.copy()
                y_test_subj = labels_test.copy()
        else:
            X_subj = np.concatenate((X_subj,dat_train.copy()),axis=0)
            y_subj = np.concatenate((y_subj,labels_train.copy()),axis=0)
            if test_day is not None:
                X_test_subj = np.concatenate((X_test_subj,dat_test.copy()),axis=0)
                y_test_subj = np.concatenate((y_test_subj,labels_test.copy()),axis=0)
    
    sbj_order = np.asarray(sbj_order)
    sbj_order_test = np.asarray(sbj_order_test)
    print('Data loaded!')
    
    return X_subj,y_subj,X_test_subj,y_test_subj,sbj_order,sbj_order_test


def compute_tfr(epochsAllMove,eventType,epoch_times,freqs = np.arange(6, 123, 3),crop_val=0.5,decim=30):
    """
    Computes spectrogram using Morlet wavelets (log-scaled).
    """
    n_cycles = freqs / 4.  # different number of cycle per frequency

    #Compute power for move trials
    print('Computing power...')
    power = tfr_morlet(epochsAllMove[eventType], freqs=freqs, n_cycles=n_cycles, use_fft=False,
                       return_itc=False, decim=decim, n_jobs=1,average=False)
    print('Power computation complete!')
    power.crop(epoch_times[0]+crop_val, epoch_times[1]-crop_val) #trim epoch to avoid edge effects
    power.data = 10*np.log10(power.data+\
                             np.finfo(np.float32).eps) #convert to log scale
    power.data[np.isinf(power.data)]=0 #set infinite values to 0
    return power


def roi_proj_pow(X_in,sbj_order,nROIs,proj_mat_out,ecog=True):
    """
    Project spectral power from electrodes to ROI's prior for random forest classification
    """
    #Project to ROIs using matrix multiply
    X_in_sh = list(X_in.shape)
    X_in_sh[1] = nROIs
    X_in_proj = np.zeros(X_in_sh)
    for s in range(X_in.shape[0]):
        sh_orig = X_in_proj.shape
        X_in_ep = X_in[s,...].reshape(X_in.shape[1],-1)
        if ecog:
            X_in_ep_proj = proj_mat_out[sbj_order[s],...] @ X_in_ep
        else:
            X_in_ep_proj = proj_mat_out[0,...] @ X_in_ep # EEG data has same projection matrix
        X_in_proj[s,...] = X_in_ep_proj.reshape(sh_orig[1:])
    del X_in,X_in_ep_proj,X_in_ep
    
    return X_in_proj