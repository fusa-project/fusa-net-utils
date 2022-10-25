import pytest
import csv
import wave
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from fusanet_utils.datasets.external import ESC, UrbanSound8K, FolderDataset
from fusanet_utils.datasets.fusa import FUSA_dataset
from fusanet_utils.transforms import Collate_and_transform
from fusanet_utils.parameters import default_logmel_parameters


def create_mock_audio(path, sampling_rate, audio):
    audio = np.array([audio, audio]).T
    audio = (audio * (2**15 - 1)).astype("<h")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sampling_rate)
        f.writeframes(audio.tobytes())

@pytest.fixture(scope="session")
def mock_folder(tmp_path_factory):
    datasets_path = tmp_path_factory.mktemp("datasets")
    audio_folder = datasets_path / "datasets" / "test_folder"
    audio_folder.mkdir(parents=True)
    sampling_rate = 44100
    time = np.linspace(0, 1, sampling_rate)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * time)
    create_mock_audio(audio_folder / "1-100032-A-0.wav", sampling_rate, audio)
    np.random.seed(12345)
    create_mock_audio(audio_folder / "1-100032-A-1.wav", sampling_rate,
                      0.01 * np.random.randn(sampling_rate // 2))

    return audio_folder

def test_folder_dataset(mock_folder):
    dataset = FolderDataset(mock_folder)
    assert len(dataset.categories) == 1
    assert dataset.categories == ['test_folder']
    assert len(dataset) == 2

def test_fusa_mock(mock_folder):
    params = default_logmel_parameters()
    dataset = FUSA_dataset(ConcatDataset([FolderDataset(mock_folder)]),
                           params)
    assert len(dataset) == 2
    assert 'label' in dataset[0]
    assert dataset.label_int2string(dataset[0]['label'])[0] == "test_folder"
    assert 'waveform' in dataset[0]
    assert 'mel_transform' in dataset[0]
    assert 'waveform' in dataset[1]
    assert 'mel_transform' in dataset[1]
    assert dataset[0]["waveform"].shape == torch.Size([1, 8000])
    assert dataset[1]["waveform"].shape == torch.Size([1, 4000])
    assert dataset[0]["mel_transform"].shape == torch.Size([1, 64, 32])
    assert dataset[1]["mel_transform"].shape == torch.Size([1, 64, 16])

@pytest.fixture(scope="session")
def mock_esc50(tmp_path_factory):

    datasets_path = tmp_path_factory.mktemp("datasets")
    meta_folder = datasets_path / "datasets" / "ESC-50" / "meta"
    meta_folder.mkdir(parents=True)
    audio_folder = datasets_path / "datasets" / "ESC-50" / "audio"
    audio_folder.mkdir()
    feature_folder = datasets_path / "features"
    feature_folder.mkdir()

    # Create mock taxonomy
    mock_taxonomy = {
        "animal/dog": {
            "ESC": ["dog"]
        },
        "human/talk": {
            "ESC": [""]
        },
        "human/others": {
            "ESC": ["snoring"]
        }
    }
    with open(datasets_path / 'fusa_taxonomy.json', 'w') as f:
        json.dump(mock_taxonomy, f)

    # Create mock metadata
    with open(meta_folder / "esc50.csv", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([
            'filename', 'fold', 'target', 'category', 'esc10', 'src_file',
            'take'
        ])
        writer.writerow(
            ['1-100032-A-0.wav', '1', '0', 'dog', 'True', '100032', 'A'])
        writer.writerow(
            ['1-100032-A-1.wav', '1', '0', 'snoring', 'True', '100032', 'A'])

    # Create mock  audio
    sampling_rate = 44100
    time = np.linspace(0, 1, sampling_rate)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * time)
    create_mock_audio(audio_folder / "1-100032-A-0.wav", sampling_rate, audio)
    np.random.seed(12345)
    create_mock_audio(audio_folder / "1-100032-A-1.wav", sampling_rate,
                      0.01 * np.random.randn(sampling_rate // 2))

    return datasets_path


def test_esc(mock_esc50):
    dataset = ESC(mock_esc50)
    assert len(dataset.categories) == 2
    assert dataset.categories == ['animal/dog', 'human/others']
    assert len(dataset) == 2
    assert dataset.labels[0] == 'animal/dog'


def test_fusa_esc(mock_esc50):
    params = default_logmel_parameters()
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)
    assert len(dataset) == 2
    assert 'label' in dataset[0]
    assert dataset.label_int2string(dataset[0]['label'])[0] == "animal/dog"
    assert 'waveform' in dataset[0]
    assert 'mel_transform' in dataset[0]
    assert 'filename' in dataset[0]
    assert 'waveform' in dataset[1]
    assert 'mel_transform' in dataset[1]
    assert dataset[0]["filename"] == '1-100032-A-0.wav'
    assert dataset[0]["waveform"].shape == torch.Size([1, 8000])
    assert dataset[1]["waveform"].shape == torch.Size([1, 4000])
    assert dataset[0]["mel_transform"].shape == torch.Size([1, 64, 32])
    assert dataset[1]["mel_transform"].shape == torch.Size([1, 64, 16])


def test_fusa_esc_collate_pad(mock_esc50):
    params = default_logmel_parameters()
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)
    my_collate = Collate_and_transform(params['features'])
    loader = DataLoader(dataset,
                        shuffle=False,
                        batch_size=2,
                        collate_fn=my_collate)
    batch = next(iter(loader))
    # batches are padded to match longest sample
    assert batch["mel_transform"].shape[-1] == max([sample["mel_transform"].shape[-1] for sample in dataset])

def test_fusa_esc_collate_crop(mock_esc50):
    params = default_logmel_parameters()
    params['features']['collate_resize'] = 'crop'
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)
    my_collate = Collate_and_transform(params['features'])
    loader = DataLoader(dataset,
                        shuffle=False,
                        batch_size=2,
                        collate_fn=my_collate)
    batch = next(iter(loader))
    # batches are cropped to match shortest sample
    assert batch["mel_transform"].shape[-1] == min([sample["mel_transform"].shape[-1] for sample in dataset])


def test_local_zscore_normalizer(mock_esc50):
    params = default_logmel_parameters()

    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)

    for sample in dataset:
        assert torch.allclose(torch.mean(sample['waveform']),
                              torch.Tensor([0.0]),
                              atol=1e-5)
        assert torch.allclose(torch.std(sample['waveform']),
                              torch.Tensor([1.0]),
                              atol=1e-5)


def test_local_minmax_normalizer(mock_esc50):
    params = default_logmel_parameters()
    params['features']['waveform_normalization']['type'] = 'minmax'
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)

    for sample in dataset:
        assert torch.allclose(torch.min(sample['waveform']),
                              torch.Tensor([0.0]),
                              atol=1e-5)
        assert torch.allclose(torch.max(sample['waveform']),
                              torch.Tensor([1.0]),
                              atol=1e-5)


def test_global_zscore_normalizer(mock_esc50):
    params = default_logmel_parameters()
    params['features']['waveform_normalization']['scope'] = 'global'
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)

    assert torch.allclose(dataset.global_normalizer.center,
                          torch.Tensor([0.0]),
                          atol=1e-3)
    assert torch.allclose(dataset.global_normalizer.scale,
                          torch.Tensor([0.2886]),
                          atol=1e-3)


def test_global_minmax_normalizer(mock_esc50):
    params = default_logmel_parameters()
    params['features']['waveform_normalization']['scope'] = 'global'
    params['features']['waveform_normalization']['type'] = 'minmax'
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)

    assert torch.allclose(dataset.global_normalizer.center,
                          torch.Tensor([-0.5]),
                          atol=1e-3)
    assert torch.allclose(dataset.global_normalizer.scale,
                          torch.Tensor([1.0]),
                          atol=1e-3)


def test_no_normalizer(mock_esc50):
    params = default_logmel_parameters()
    #params['features']['waveform_normalization']['scope'] = None
    #params['features']['waveform_normalization']['type'] = 'none'
    params['features'].pop('waveform_normalization')
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)
    sine_wave = dataset[0]['waveform']
    assert torch.allclose(torch.min(sine_wave),
                          torch.Tensor([-0.5]),
                          atol=1e-3)
    assert torch.allclose(torch.max(sine_wave),
                          torch.Tensor([0.5]),
                          atol=1e-3)
    params = default_logmel_parameters()
    params['features']['waveform_normalization']['scope'] = 'none'
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           params)
    sine_wave = dataset[0]['waveform']
    assert torch.allclose(torch.min(sine_wave),
                          torch.Tensor([-0.5]),
                          atol=1e-3)
    assert torch.allclose(torch.max(sine_wave),
                          torch.Tensor([0.5]),
                          atol=1e-3)
