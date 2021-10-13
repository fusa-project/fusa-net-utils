import pytest
import csv
import wave
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from fusanet_utils.datasets.external import ESC, UrbanSound8K
from fusanet_utils.datasets.fusa import FUSA_dataset
from fusanet_utils.transforms import Collate_and_transform
from fusanet_utils.parameters import default_logmel_parameters


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
    mock_taxonomy = {"animal/dog": {"ESC": ["dog"]}}
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

    # Create mock  audio
    samplerate = 44100
    t = np.linspace(0, 1, samplerate)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    audio = np.array([audio, audio]).T
    audio = (audio * (2**15 - 1)).astype("<h")
    with wave.open(str(audio_folder / "1-100032-A-0.wav"), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(audio.tobytes())

    return datasets_path


def test_esc(mock_esc50):
    dataset = ESC(mock_esc50)
    assert dataset.categories[0] == 'animal/dog'
    assert dataset.labels[0] == 'animal/dog'


def test_fusa_esc(mock_esc50):
    params = default_logmel_parameters()
    dataset = FUSA_dataset(ConcatDataset([ESC(mock_esc50)]),
                           feature_params=params["features"])
    my_collate = Collate_and_transform(params['features'])
    loader = DataLoader(dataset,
                        shuffle=False,
                        batch_size=1,
                        collate_fn=my_collate)
    batch = next(iter(loader))
    assert batch["mel_transform"].shape == torch.Size([1, 1, 64, 16])
    assert dataset.label_int2string(batch['label'])[0] == "animal/dog"