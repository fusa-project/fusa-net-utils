import logging
import torch
from .waveform import get_waveform

logger = logging.getLogger(__name__)

def rolling_zscore(dataset, params):
    mean = 0.0
    std = 0.0
    n_samples = 0
    for file_path, _ in dataset:
        waveform = get_waveform(file_path, params)
        mean += torch.sum(waveform)
        n_samples += torch.numel(waveform)        
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
        logger.info("Computing stats for global normalization")                
        if how_to == 'zscore':
            self.center, self.scale = rolling_zscore(dataset, params)
        elif how_to == 'minmax':
            self.center, self.scale = rolling_minmax(dataset, params)        

    def __call__(self, waveform):
        return (waveform - self.center)/self.scale