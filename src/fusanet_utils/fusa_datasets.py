from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, ConcatDataset

from .features import get_waveform, LogMelTransform
from .external_datasets import ESC, UrbanSound8K

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
        # Precompute logmel spectrogram if it does not exist or if overwrite is enabled     
        if feature_params["use_logmel"]:
            for file_path, _ in self.dataset:
                LogMelTransform(file_path, feature_params) 

    def __getitem__(self, idx: int) -> Dict:
        file_path, label = self.dataset[idx]
        waveform = get_waveform(file_path, self.params)
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)
        sample = {'waveform': waveform, 'label': torch.from_numpy(self.le.transform([label]))}
        if self.params["use_logmel"]:
            sample['logmel'] = LogMelTransform(file_path)()              
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

class FUSAv1(FUSA_dataset):
    def __init__(self, datasets_repo_path, feature_params, waveform_transform=None):
        dataset = ConcatDataset([ESC(datasets_repo_path), UrbanSound8K(datasets_repo_path)])
        super().__init__(dataset, feature_params, waveform_transform)

        
    
if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from .transforms import Collate_and_transform, RESIZER
    import yaml
    params = yaml.safe_load(open("experiments/ESC/params.yaml"))
    dataset = FUSA_dataset(ConcatDataset([ESC("./datasets")]), feature_params=params["features"])
    my_collate = Collate_and_transform(resizer=RESIZER.PAD)
    loader = DataLoader(dataset, shuffle=True, batch_size=5, collate_fn=my_collate)
    for batch in loader:
        break
    print(batch['waveform'].shape)
    print(batch['logmel'].shape)
    print(dataset.label_int2string(batch['label']))    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(batch['logmel'].detach().numpy()[0, 0])
    

