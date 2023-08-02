import io
import logging
import pathlib
from typing import Dict, Union
import torchaudio
import torch
from .waveform_backends import read_pydub, read_soundfile

logger = logging.getLogger(__name__)

def get_waveform(file: Union[str, pathlib.Path, bytes, bytearray], params: Dict, global_normalizer=None) -> torch.Tensor:
    logger.info("get_waveform before IO")
    if isinstance(file, bytes) or  isinstance(file, bytearray):
        file = io.BytesIO(file)
    else:
        logger.debug(f"Loading: {file}")
    logger.info("get_waveform after IO")
    samples, origin_sr = read_pydub(file)
    logger.info("read_pydub finish")
    if samples is None:
        logger.error(f"Could not read {file} with pudub, defaulting to soundfile")
        samples, origin_sr = read_soundfile(file)
    if len(samples.shape) == 3:
        samples = samples[:, 0, :]
    waveform = torch.from_numpy(samples).T
    logger.info(f"waveform shape: {waveform.shape}")
    logger.debug("insider finish get_waveform")
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
        
