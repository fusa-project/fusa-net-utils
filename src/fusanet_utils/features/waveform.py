from typing import Dict
import torchaudio
import torch

def get_waveform(file_path: str, params: Dict) -> torch.Tensor:
    waveform, origin_sr = torchaudio.load(file_path)
    return waveform_preprocessing(waveform, origin_sr, params)

def waveform_preprocessing(waveform: torch.Tensor, origin_sr: int, params: Dict) -> torch.Tensor:
    origin_ch = waveform.size()[0]
    target_sr = params["sampling_rate"]
    target_ch = params["number_of_channels"]
    if not origin_sr == target_sr:
        waveform = torchaudio.transforms.Resample(origin_sr, target_sr)(waveform)
    # TODO: Separar las pistas como audios independientes (duplicar a nivel de dataset)
    if target_ch == 1 and origin_ch == 2:
        how_to = params['combine_channels']
        if how_to == 'mean':
            waveform =  torch.mean(waveform, dim=0, keepdim=True)
        elif how_to == 'left':
            waveform = waveform[0, :].view(1,-1)
        elif how_to == 'right':
            waveform = waveform[1, :].view(1,-1)
    
    if target_ch == 2 and origin_ch == 1:
        waveform = waveform.repeat(2, 1)
    
    if 'waveform_normalization' in params:
        if params['waveform_normalization']['scope'] == 'local':
            waveform = local_normalizer(waveform, params)   
    return waveform

def zscore(waveform):
    return torch.mean(waveform), torch.std(waveform)

def minmax(waveform):
    min = torch.min(waveform)
    max = torch.max(waveform)
    return min, max-min

def local_normalizer(waveform, params):
    how_to = params['waveform_normalization']['type']
    if how_to == 'zscore':
        center, scale = zscore(waveform)
    elif how_to == 'minmax':
        center, scale = minmax(waveform)
    elif how_to == 'rms':
        center, scale = 0, 1
    elif how_to == 'peak':
        center, scale = 0, 1
    return (waveform - center)/scale  

def rolling_zscore(dataset, params):
    mean = 0.0
    std = 0.0
    n_samples = 0
    for file_path, _ in dataset:
        waveform = get_waveform(file_path, params)
        mean += torch.sum(waveform)
        n_samples += len(waveform)
    mean = mean/n_samples

    for file_path, _ in dataset:
        waveform = get_waveform(file_path, params)
        std += torch.sum((waveform - mean)**2)
    std = torch.sqrt(std/n_samples)
    return mean, std

def rolling_minmax(dataset, params):
    min = float('inf')
    max = -float('inf')
    for file_path, _ in dataset:
        waveform = get_waveform(file_path, params)
        local_min = torch.min(waveform)
        local_max = torch.max(waveform)
        if local_min < min:
            min = local_min
        if local_max > max:
            max = local_max
    return min, max-min

class Global_normalizer():

    def __init__(self, params, dataset):
        
        how_to = params['waveform_normalization']['type']
        if how_to == 'zscore':
            self.center, self.scale = rolling_zscore(dataset, params)
        elif how_to == 'minmax':
            self.center, self.scale = rolling_minmax(dataset, params)        

    def __call__(self, waveform):
        return (waveform - self.center)/self.scale
        