from typing import Dict
import torch
import torchaudio

from .base import Feature
from .global_normalizer import Global_normalizer


class LogMel(Feature):
    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        sample_rate = self.params["sampling_rate"]
        mel_params = self.params["mel_transform"]
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=mel_params['n_fft'],
            hop_length=mel_params['hop_length'],
            n_mels=mel_params['n_mels'],
            normalized=mel_params["normalized"])
        mel_spectrogram = mel_transform(waveform)
        eps = torch.min(mel_spectrogram[mel_spectrogram > 0.0])
        logmel = (mel_spectrogram + eps).log10()
        return logmel


class FeatureProcessor():
    def __init__(self, params: Dict = {}, normalizer: Global_normalizer = None):
        self.params = params
        self.normalizer = normalizer
        self.processors = self.__find_processors()

    def __find_processors(self) -> Dict:
        # Update this to add new features
        processors = {}
        if 'mel_transform' in self.params:
            processors['mel_transform'] = LogMel(self.params)
        if 'mfcc_transform' in self.params:
            pass
        if 'rp_transform' in self.params:
            pass
        return processors

    def compute_features(self, waveform: torch.Tensor) -> Dict:
        """
        Used by torch-serve
        TODO: PASS GLOBAL NORMALIZER TO TORCH SERVE
        """
        sample = dict()
        for feature_name, processor in self.processors.items():
            sample[feature_name] = processor.compute(waveform)
        return sample

    def write_features(self, waveform_path: str) -> None:
        for processor in self.processors.values():
            processor.write_to_disk(waveform_path, self.normalizer)

    def read_features(self, waveform_path: str) -> Dict:
        sample = dict()
        for feature_name, processor in self.processors.items():
            sample[feature_name] = processor.read_from_disk(waveform_path)
        return sample
