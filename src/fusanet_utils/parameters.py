def default_params():
    params = {}
    params['features'] = {}
    params['features']['sampling_rate'] = 8000
    params['features']['number_of_channels'] = 1
    params['features']['combine_channels'] = 'mean'
    return params

def default_logmel_parameters(): 
    params = default_params()
    params['features']['use_logmel'] = True
    params['features']['overwrite'] = True
    params['features']['mel_transform'] = {}
    params['features']['mel_transform']['n_mels'] = 64
    params['features']['mel_transform']['n_fft'] = 512
    params['features']['mel_transform']['hop_length'] = 512
    params['features']['mel_transform']['normalized'] = False
    return params
