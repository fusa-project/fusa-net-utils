from os.path import isfile, splitext
from typing import Dict
import torch
import torchaudio

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
            return torch.mean(waveform, dim=0, keepdim=True)
        elif how_to == 'left':
            return waveform[0, :].view(1,-1)
        elif how_to == 'right':
            return waveform[1, :].view(1,-1)
    elif target_ch == 2 and origin_ch == 1:
        return waveform.repeat(2, 1)
    else:
        return waveform


def compute_logmel(waveform, params):
    sample_rate = params["sampling_rate"]
    mel_params = params["mel_transform"]
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=mel_params['n_fft'], hop_length=mel_params['hop_length'], n_mels=mel_params['n_mels'], normalized=mel_params["normalized"])
    mel_spectrogram = mel_transform(waveform)
    eps = torch.min(mel_spectrogram[mel_spectrogram>0.0])
    return (mel_spectrogram + eps).log10()

def compute_mfcc(waveform, params):
    pass


class FeatureProcessor():

    def __init__(self, params: Dict={}):
        self.params = params
        self.processors = self.__find_processors()    

    def __find_processors(self) -> Dict:
        processors = {}
        if 'mel_transform' in self.params:
            processors['mel_transform'] = compute_logmel
        if 'mfcc_transform' in self.params:
            processors['mfcc_transform'] = compute_mfcc
        return processors   

    def __feature_path(self, waveform_path: str, feature_name: str) -> str:
        return splitext(waveform_path)[0]+"_"+feature_name+".pt"    

    def compute_features(self, waveform: torch.Tensor) -> Dict:
        sample = dict()
        for feature_name, processor in self.processors.items():
            sample[feature_name] = processor(waveform)
        return sample
        
    def write_features(self, waveform_path: str) -> None:
        for feature_name, processor in self.processors.items():
            feature_path = self.__feature_path(waveform_path, feature_name)
            if not isfile(feature_path) or self.params["overwrite"]: 
                waveform =  get_waveform(waveform_path, self.params)
                feature = processor(waveform, self.params)
                torch.save(feature, feature_path)
    
    def read_features(self, waveform_path: str) -> Dict:
        sample = dict()
        for feature_name, _ in self.processors.items():
            feature_path = self.__feature_path(waveform_path, feature_name)
            if isfile(feature_path):
                sample[feature_name] = torch.load(feature_path)
            else:
                raise FileNotFoundError("Feature file not found")
        return sample

    






    

        