from abc import ABC, abstractmethod
import numpy as np
import torch
import colorednoise as cn
import librosa
from typing import Dict

class Noise(ABC):
    def __init__(self, SNR_min: float=-1., SNR_max: float=1.):
        self.SNR_min = SNR_min
        self.SNR_max = SNR_max

    @abstractmethod
    def generate_noise(self, n_samples: int, random_state=None) -> np.array:
        pass

    def __call__(self, sample: Dict, snr=None, random_state=None):
        if snr is None:
            snr = self.SNR_min + torch.rand(1).item()*(self.SNR_max - self.SNR_min)
        signal = sample['waveform']
        assert signal.shape[0] == 1, "Only use for mono audio"
        signal_rms = np.mean(librosa.feature.rms(y=signal[0, :]))
        noise = self.generate_noise(signal.shape[-1], random_state)
        noise_rms = np.mean(librosa.feature.rms(y=noise[0, :]))
        factor = (signal_rms)/(np.exp(snr/20)*noise_rms)
        sample['waveform'] = signal + noise*factor
        return sample


class WhiteNoise(Noise):

    def generate_noise(self, n_samples: int, random_state=None) -> torch.Tensor:
        # Add generator to set random seed
        return torch.randn(1, n_samples)


class PinkNoise(Noise):

    def generate_noise(self, n_samples: int, random_state=None) -> torch.Tensor:
        noise = cn.powerlaw_psd_gaussian(beta=1, samples=n_samples, fmin=0, random_state=random_state)
        return torch.from_numpy(noise).reshape(1, -1)


class RedNoise(Noise):

    def generate_noise(self, n_samples: int, random_state=None) -> torch.Tensor:
        noise = cn.powerlaw_psd_gaussian(beta=2, samples=n_samples, fmin=0, random_state=random_state)
        return torch.from_numpy(noise).reshape(1, -1)

def test_white_noise():
    waveform = torch.randn(1, 1000)
    sample = {}
    sample['waveform'] = waveform
    sample['label'] = 'asd'
    transformed_sample = WhiteNoise()(sample)
    return transformed_sample


