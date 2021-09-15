import torch
from torch import Tensor
from torch.nn.functional import pad
from typing import Dict, List

def resize(data, target_length):
    if data.size(-1) < target_length:
        append_length = target_length-data.size(-1)
        return pad(data, (0, append_length), mode='constant', value=0)
    elif data.size(-1) > target_length:
        if data.ndim == 2: # audio
            return data[:,:target_length]
        elif data.ndim == 3: # spectrogram
            return data[:,:,:target_length]
        else:
            raise NotImplementedError("Only 3D o 4D tensors accepted")
    else:
        return data

class Collate_and_transform:
    
    def __init__(self, params: Dict, transforms: List=[]):
        self.transforms = transforms
        self.resizer = params['collate_resize']
        
    def __call__(self, batch: List[Dict]) -> Tensor:
        data_keys = list(batch[0].keys())
        data_keys.remove('label')
        for key in data_keys:
            if not self.resizer == 'none':
                lens = [sample[key].size(-1) for sample in batch]  
                if self.resizer == 'pad':
                    target_length = max(lens)
                elif self.resizer == 'crop':
                    target_length = min(lens)
                for sample in batch:
                    sample[key] = resize(sample[key], target_length)
            
        # Data augmentation transforms
        transformed_batch = []
        for sample in batch:
            for transform in self.transforms:
                sample = transform(sample)
            transformed_batch.append(sample)
        mbatch = {}
        mbatch['label'] = torch.LongTensor([sample['label'] for sample in transformed_batch])
        for key in data_keys:
            mbatch[key] =  torch.stack([sample[key] for sample in transformed_batch], dim=0)
        return mbatch

