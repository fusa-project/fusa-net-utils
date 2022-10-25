from typing import Dict, List
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, ConcatDataset

from ..features.processor import FeatureProcessor
from ..features.waveform import get_waveform
from ..features.global_normalizer import Global_normalizer
from .external import ESC, UrbanSound8K

class FUSA_dataset(Dataset):

    def __init__(self, dataset: ConcatDataset, params: Dict, waveform_transform=None):
        self.dataset = dataset
        self.categories = []
        for d in self.dataset.datasets:
            self.categories += d.categories
        self.categories = sorted(list(set(self.categories)))
        self.le = LabelEncoder().fit(self.categories)
        self.waveform_transform = waveform_transform
        self.params = params
        self.global_normalizer = None
        # Precompute waveforms 
        for file_path, _ in self.dataset:
            pass
        # Precompute global normalizer stats
        if 'waveform_normalization' in self.params['features']:
            if self.params['features']['waveform_normalization']['scope'] == 'global':
                self.global_normalizer = Global_normalizer(self.params, dataset)
        # Precompute features
        for file_path, _ in self.dataset:
            FeatureProcessor(self.params['features'], self.global_normalizer).write_features(file_path)        

    def __getitem__(self, idx: int) -> Dict:
        file_path, label = self.dataset[idx]
        waveform = get_waveform(file_path, self.params['features'], self.global_normalizer)
        
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)
        if isinstance(label, str): #TAG
            label_tag = torch.from_numpy(self.le.transform([label]))
            sample = {'filename': pathlib.Path(file_path).name, 'waveform': waveform, 'label': label_tag} 
        else: # SED
            label_sed, distance_sed = self.build_sed_labels(waveform.shape[1], label, self.params)
            sample = {'filename': pathlib.Path(file_path).name, 'waveform': waveform, 'label': label_sed, 'distance': distance_sed} 
        sample.update(FeatureProcessor(self.params['features']).read_features(file_path))             
        return sample

    def __len__(self) -> int:
        return len(self.dataset)

    def label_int2string(self, label_batch: torch.Tensor) -> List[int]:
        return list(self.le.inverse_transform(label_batch.numpy().ravel()))

    def label_dictionary(self) -> Dict:
        d = {}
        for key, value in enumerate(self.le.classes_):
            d[key] = value
        return d

    def build_sed_labels(self, n_samples: int, metadata: Dict, params: Dict) -> torch.Tensor:
        sample_rate = params['features']['sampling_rate']
        audio_seconds = n_samples // sample_rate
        try:
            hop_length = params['features']['mel_transform']['hop_length']
        except KeyError:
            hop_length = 320
            
        SED_MODEL_TYPE  = {
            'ADAVANNE-sed': n_samples // hop_length + 1,
            'HTS': 1024,
            'PANN-sed': n_samples // 320 + 1
        }
        audio_windows = SED_MODEL_TYPE[params['train']['model']]
        
        label = torch.zeros(audio_windows, len(self.categories))
        distance = torch.zeros(audio_windows, len(self.categories))
        label_idx = self.le.transform(list(metadata['class'])).astype('int')
        start_norm, end_norm = (audio_windows/audio_seconds)*metadata[['start (s)', 'end (s)']].values.T
        start_idx = start_norm.astype('int')
        end_idx = end_norm.astype('int')
        for k, entity in enumerate(label_idx): # TODO: Make this more efficient
            label[start_idx[k]:end_idx[k], entity] = 1.
            if 'distance' in metadata:
                distance[start_idx[k]:end_idx[k], entity] = metadata['distance'].values[k]
        
        return label, distance
