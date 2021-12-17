import io
import logging
import pathlib
from typing import Dict, Union
import numpy as np
import torchaudio
import torch
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def get_waveform(file: Union[str, pathlib.Path, bytes], params: Dict, global_normalizer=None) -> torch.Tensor:
    if type(file) == str or type(file) == pathlib.PosixPath :
        logger.debug(f"Loading: {file}")
        asegment = AudioSegment.from_file(file)
    elif type(file) == bytes or  type(file) == bytearray:
        asegment = AudioSegment.from_file(io.BytesIO(file))
    origin_sr = asegment.frame_rate
    channel_sounds = asegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    # Convert to float32
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    # Convert to tensor
    waveform = torch.from_numpy(fp_arr).T    
    return waveform_preprocessing(waveform, origin_sr, params, global_normalizer)

def waveform_preprocessing(waveform: torch.Tensor, origin_sr: int, params: Dict, global_normalizer=None) -> torch.Tensor:
    origin_ch = waveform.size()[0]
    target_sr = params["sampling_rate"]
    target_ch = params["number_of_channels"]
    if not origin_sr == target_sr:
        logger.debug(f"Original shape: {waveform.shape} and sampling rate {origin_sr}")
        waveform = torchaudio.transforms.Resample(origin_sr, target_sr)(waveform)
        logger.debug(f"Resampled shape: {waveform.shape} and sampling rate {target_sr}")
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
        elif params['waveform_normalization']['scope'] == 'global' and global_normalizer is not None:
            waveform = global_normalizer(waveform)
    return waveform

def zscore(waveform):
    # TODO: NORMALIZE BY CHANNEL
    return torch.mean(waveform), torch.std(waveform)

def minmax(waveform):
    # TODO: NORMALIZE BY CHANNEL
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
    else:
        center, scale = 0, 1
    return (waveform - center)/scale  
        