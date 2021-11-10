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
            how_to = params['waveform_normalization']['type']
            if how_to == 'zscore':
                waveform = (waveform - torch.mean(waveform))/torch.std(waveform)
            elif how_to == 'minmax':
                pass
            elif how_to == 'rms':
                pass
            elif how_to == 'peak':
                pass

    return waveform