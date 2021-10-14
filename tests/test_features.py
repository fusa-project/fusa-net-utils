import wave
import pytest
import numpy as np
import torch

from fusanet_utils.features.processor import LogMel
from fusanet_utils.features.waveform import get_waveform
from fusanet_utils.parameters import default_logmel_parameters


@pytest.fixture()
def mock_wav(tmp_path):
    wav_file_path = tmp_path / "demo.wav"
    sampling_rate = 44100
    t = np.linspace(0, 1, sampling_rate)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.array([audio, audio]).T
    audio = (audio * (2**15 - 1)).astype("<h")
    with wave.open(str(wav_file_path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sampling_rate)
        f.writeframes(audio.tobytes())

    return wav_file_path


@pytest.mark.parametrize("sampling_rate, channels", [(8000, 1), (44100, 2),
                                                     (22050, 1)])
def test_get_waveform(mock_wav, sampling_rate, channels):
    params = default_logmel_parameters()
    params['features']['sampling_rate'] = sampling_rate
    params['features']['number_of_channels'] = channels
    waveform = get_waveform(mock_wav, params['features'])
    assert waveform.shape[0] == channels
    assert waveform.shape[1] == sampling_rate
    assert type(waveform) == torch.Tensor


@pytest.mark.parametrize("n_mels, n_fft, expected_windows", [(64, 400, 81),
                                                             (32, 800, 41),
                                                             (16, 200, 161)])
def test_logmel(mock_wav, n_mels, n_fft, expected_windows):
    params = default_logmel_parameters()
    params['features']['mel_transform']['n_mels'] = n_mels
    params['features']['mel_transform']['n_fft'] = n_fft
    params['features']['mel_transform']['hop_length'] = n_fft // 4
    waveform = get_waveform(mock_wav, params['features'])
    logmel = LogMel(params['features']).compute(waveform)
    assert logmel.shape[0] == 1
    assert logmel.shape[1] == n_mels
    assert logmel.shape[2] == expected_windows
