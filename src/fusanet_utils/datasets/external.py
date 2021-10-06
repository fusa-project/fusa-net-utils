from os.path import join, isfile
import warnings
from typing import Tuple, Dict
import pandas as pd
from torch.utils.data import Dataset

def get_label_transforms(repo_path: str, dataset_name: str) -> Dict:
    taxonomy_path = join(repo_path, "fusa_taxonomy.json")   
    a = pd.read_json(taxonomy_path).T[dataset_name].to_dict()
    transforms = {}
    for key, values in a.items():
        for value in values:
            transforms[value] = key
    return transforms

class ExternalDataset(Dataset):

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return (self._file_path(idx), self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)

class ESC(ExternalDataset):

    def __init__(self, repo_path: str):
        
        # TODO: Abstract parts of init
        label_transforms = get_label_transforms(repo_path, "ESC")
        datasets_path = join(repo_path, "datasets")        
        df = pd.read_csv(join(datasets_path, "ESC-50", "meta", "esc50.csv"))
        ESC_classes = df["category"].unique()
        # Verify that there are no typos in FUSA_taxonomy
        if not all([key in set(ESC_classes) for key in label_transforms.keys() if key != ""]):
            warnings.warn("Existen llaves de ESC que no calzan en fusa_taxonomy.json", UserWarning)
        
        self.audio_path = join(datasets_path, "ESC-50", "audio")
        # Verify that files exist
        file_exist = df["filename"].apply(lambda x: isfile(join(datasets_path, "ESC-50", "audio", x)))
        if not file_exist.all():
            warnings.warn("Existen rutas incorrectas o archivos perdidos", UserWarning)
            df = df.loc[file_exist]
        
        self.file_list, self.labels, self.categories = [], [], []
        for label in ESC_classes:
            if label in label_transforms:
                self.categories += [label_transforms[label]]
                mask = df.category == label
                self.file_list += list(df["filename"].loc[mask])
                self.labels += [label_transforms[label]]*sum(mask)

    def _file_path(self, idx: int) -> str:
        return join(self.audio_path, self.file_list[idx])


class UrbanSound8K(ExternalDataset):
    
    def __init__(self, repo_path: str):
        label_transforms = get_label_transforms(repo_path, "UrbanSound")
        datasets_path = join(repo_path, "datasets")
        df = pd.read_csv(join(datasets_path, "UrbanSound8K", "metadata", "UrbanSound8K.csv"))
        self.audio_path = join(datasets_path, "UrbanSound8K", "audio")
        self.file_list = []
        self.fold_list = []
        self.labels = []
        self.categories = []
        for label in df["class"].unique():
            if label in label_transforms:
                self.categories += [label_transforms[label]]
                mask = df["class"] == label
                self.file_list += list(df["slice_file_name"].loc[mask])
                self.fold_list += list(df["fold"].loc[mask])
                self.labels += [label_transforms[label]]*sum(mask)
        self.fold_list = ['fold' + str(fold) for fold in self.fold_list]


    def _file_path(self, idx: int) -> str:
        return join(self.audio_path, self.fold_list[idx], self.file_list[idx])
