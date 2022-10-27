import logging
import torch
from torch import Tensor
from torch.nn.functional import pad
from typing import Dict, List

logger = logging.getLogger(__name__)

def crop(data, target_length, collate_dim):
    if data.size(collate_dim) > target_length:
        if data.ndim == 2:
            if collate_dim == -1:  # audio
                data = data[:, :target_length]
            elif collate_dim == -2:  # sed label
                data = data[:target_length, :]
        elif data.ndim == 3:  # spectrogram
            data = data[:, :, :target_length]
        else:
            raise NotImplementedError("Only 3D o 4D tensors accepted")
    return data


def append(data, target_length, collate_dim):
    if data.size(collate_dim) < target_length:
        append_length = target_length-data.size(collate_dim)
        if data.ndim == 2:
            if collate_dim == -1:  # audio
                data = pad(data, (0, append_length, 0, 0), mode='constant', value=0)
            elif collate_dim == -2:  # sed label
                data = pad(data, (0, 0, 0, append_length), mode='constant', value=0)
        elif data.ndim == 3:  # spectrogram
            data = pad(data, (0, append_length, 0, 0, 0, 0), mode='constant', value=0)
        else:
            raise NotImplementedError("Only 3D o 4D tensors accepted")
    return data


class Collate_and_transform:

    def __init__(self, params: Dict, transforms: List = []):
        self.transforms = transforms
        self.resizer = params['collate_resize']

    def __call__(self, batch: List[Dict]) -> Dict:
        data_keys = list(batch[0].keys())
        if batch[0]['label'].ndim == 1:  # TAG
            data_keys.remove('label')
        data_keys.remove('filename')
        for key in data_keys:
            if not self.resizer == 'none':
                if key == 'label' or key == 'distance':
                    collate_dim = -2
                else:
                    collate_dim = -1
                lens = [sample[key].size(collate_dim) for sample in batch]
                logger.debug(f"{self.resizer} {lens}")
                if self.resizer == 'pad':
                    for sample in batch:
                        sample[key] = append(sample[key], max(lens), collate_dim)
                elif self.resizer == 'crop':
                    for sample in batch:
                        sample[key] = crop(sample[key], min(lens), collate_dim)
        # Data augmentation transforms
        transformed_batch = []
        for sample in batch:
            for transform in self.transforms:
                sample = transform(sample)
            transformed_batch.append(sample)
        mbatch = {}
        if sample['label'].ndim == 1:  # TAG
            mbatch['label'] = torch.LongTensor([sample['label'] for sample in transformed_batch])
        else:  # SED
            mbatch['label'] = torch.stack([sample['label'] for sample in transformed_batch], dim=0)
        mbatch['filename'] = [sample['filename'] for sample in transformed_batch]
        for key in data_keys:
            mbatch[key] = torch.stack([sample[key] for sample in transformed_batch], dim=0)
        return mbatch
