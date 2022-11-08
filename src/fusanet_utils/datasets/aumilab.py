from curses import meta
from pathlib import Path
import logging
from typing import Tuple, Union, List
import pandas as pd
from torch.utils.data import Dataset
from .external import get_label_transforms

logger = logging.getLogger(__name__)

class AUMILAB(Dataset):
    
    def __init__(self, repo_path: Union[str, Path], categories: List=None):
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        label_transforms = get_label_transforms(repo_path, "SPASS")
        self.file_list, self.labels, self.categories = [], [], []
        dataset_path = repo_path / "datasets" / 'AUMILAB'
        df = pd.read_csv(dataset_path / 'metadata' / 'metadata.txt', delim_whitespace=True)
        for file_name, metadata in df.groupby('filename'):
            file_path = dataset_path / 'audios' / file_name
            if not file_path.exists():
                logger.warning(f"El archivo {file_name} no existe")
                continue
            self.file_list.append(file_path)
            metadata = metadata[["class", "start", "end"]]
            metadata = metadata.rename(columns={"end": "class", "start": "start (s)", "end": "end (s)"})
                        
            label_exists = metadata['class'].apply(lambda label: label in label_transforms)
            metadata_rows = metadata.loc[label_exists]
            metadata['class'] = metadata_rows['class'].loc[label_exists].apply(lambda label: label_transforms[label])
            self.labels.append(metadata)
            
            # Find number of classes
            if categories is None:
                for i in range(len(self.labels)):
                    self.categories += list(self.labels[i]['class'])
                self.categories = sorted(list(set(self.categories)))
            else:
                self.categories = categories
            self.categories = sorted(list(set(self.categories)))     
            
    def __getitem__(self, idx: int) -> Tuple[Path, pd.DataFrame]:
        return (self.file_list[idx], self.labels[idx])

    def __len__(self) -> int:
        return len(self.file_list)
