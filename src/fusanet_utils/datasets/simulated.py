from pathlib import Path
import logging
from typing import Union, Tuple, Dict, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

from ..features.waveform import get_waveform

logger = logging.getLogger(__name__)

def create_sed_labels(labels: List[int], start_times: List[float], end_times: List[float], audio_length:int, sampling_frequency:int=32000, window_size:int=1000):
    pass


class SimulatedPoliphonic(Dataset):

    def __init__(self, repo_path: Union[str, Path], mini: bool=True):
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
 
        dataset_path = repo_path / "datasets" / "Poliphonic mono mini"
        features_path = repo_path / "features" / "Poliphonic mono mini"
        self.categories = []
        self.file_list = []
        self.label_list = []
        # Find number of classes
        for file in (dataset_path / "meta").glob("*.csv"):
            df = pd.read_csv(file, sep=',', header=0)
            self.categories.append(df["class"].unique)
        self.categories = sorted(list(set(self.categories)))
        n_classes = len(self.categories)
        self.le = LabelEncoder().fit(self.categories)
        
        for file in (dataset_path / "meta").glob("*.csv"):
            df = pd.read_csv(file, sep=',', header=0)
            for audio_file, metadata in df.groupby(by="audio_filename"):
                audio_path = dataset_path / "audio" / audio_file
                audio = get_waveform(audio_path, feature_params)
                self.file_list.append(audio_path)
                label = create_sed_labels()
                label_path = features_path / audio_file.stem + "_label.pt"
                with open(label_path, "wb") as f:
                    torch.save(label, f)
                self.label_list.append(label_path)
            break


    def __getitem__(self, idx: int) -> Tuple[Path, Path]:
        return (self.file_list[idx], self.label_list[idx])

    def __len__(self) -> int:
        return len(self.file_list)



