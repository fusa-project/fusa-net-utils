from typing import Dict, List
import pathlib
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, ConcatDataset

from ..features.processor import FeatureProcessor
from ..features.waveform import get_waveform
from ..features.global_normalizer import Global_normalizer
from .external import ESC, UrbanSound8K

class FUSA_dataset(Dataset):

    def __init__(self, dataset: ConcatDataset, feature_params: Dict, waveform_transform=None):
        self.dataset = dataset
        self.categories = []
        for d in self.dataset.datasets:
            self.categories += d.categories
        self.categories = sorted(list(set(self.categories)))
        self.le = LabelEncoder().fit(self.categories)
        self.waveform_transform = waveform_transform
        self.params = feature_params
        self.global_normalizer = None
        # Precompute waveforms 
        for file_path, _ in self.dataset:
            pass
        # Precompute global normalizer stats
        if 'waveform_normalization' in self.params:
            if self.params['waveform_normalization']['scope'] == 'global':
                self.global_normalizer = Global_normalizer(self.params, dataset)
        # Precompute features
        for file_path, _ in self.dataset:
            FeatureProcessor(self.params, self.global_normalizer).write_features(file_path)            

    def __getitem__(self, idx: int) -> Dict:
        file_path, label = self.dataset[idx]
        waveform = get_waveform(file_path, self.params, self.global_normalizer)
        
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)
        #sample = {'filename': pathlib.Path(file_path).name, 'waveform': waveform, 'label': torch.from_numpy(self.le.transform([label]))} #TAG
        sample = {'filename': pathlib.Path(file_path).name, 'waveform': waveform, 'label': label} #SED
        sample.update(FeatureProcessor(self.params).read_features(file_path))             
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


