from pathlib import Path
import logging
from typing import Tuple, Union
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class AUMILAB(Dataset):
    
    def __init__(self, repo_path: Union[str, Path]):
        self.file_list, self.labels = [], []
        dataset_path = repo_path / "datasets" / 'AUMILAB'
        df = pd.read_csv(dataset_path / 'metadata' / 'metadata1_4.txt', delim_whitespace=True)
        for file_name, metadata in df.groupby('filename'):
            file_path = dataset_path / 'audios' / file_name
            if not file_path.exists():
                logger.warning(f"El archivo {file_name} no existe")
                continue
            self.file_list.append(file_path)
            metadata = metadata[["class", "start", "end"]]
            metadata = metadata.rename(columns={"end": "class", "start": "start (s)", "end": "end (s)"})
            self.labels.append(metadata)
            self.categories = [list(self.labels[i]['class']) for i in range(len(self.labels))]
            self.categories = sorted(list(set([c for category in self.categories for c in category])))
            
    def __getitem__(self, idx: int) -> Tuple[Path, pd.DataFrame]:
        return (self.file_list[idx], self.labels[idx])

    def __len__(self) -> int:
        return len(self.file_list)