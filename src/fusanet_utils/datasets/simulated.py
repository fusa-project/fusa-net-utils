from pathlib import Path
import logging
from typing import Union, Tuple, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)


class SimulatedPoliphonic(Dataset):

    def __init__(self, repo_path: Union[str, Path], mini: bool=True):
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        if mini: 
            dataset_path = repo_path / "datasets" / "Poliphonic mono mini"
        else:
            dataset_path = repo_path / "datasets" / "Poliphonic mono"
        self.categories = []
        self.file_list = []
        self.label_list = []
        # Find number of classes
        for file in (dataset_path / "meta").glob("*.csv"):
            df = pd.read_csv(file, sep=',', header=0)
            self.categories += list(df["class"].unique())
        
        self.categories = sorted(list(set(self.categories)))
        self.le = LabelEncoder().fit(self.categories)
        
        for file in (dataset_path / "meta").glob("*.csv"):
            df = pd.read_csv(file, sep=',', header=0)
            for audio_file, metadata in df.groupby(by="audio_filename"):
                audio_path = dataset_path / "audios" / audio_file
                self.file_list.append(audio_path)
                audio_seconds = 10 # TODO: Read this from data
                # audio_samples = 44100*audio_seconds
                audio_windows = 1001 # audio_samples // 1000 # TODO: Calculate this from params (target sampling freq, number of windows)
                label = torch.zeros(audio_windows, len(self.categories)) 
                label_idx = self.le.transform(list(metadata['class'])).astype('int')
                start_norm, end_norm = (audio_windows/audio_seconds)*metadata[['start (s)', 'end (s)']].values.T
                start_idx = start_norm.astype('int')
                end_idx = end_norm.astype('int')
                for k, entity in enumerate(label_idx): # TODO: Make this more efficient
                    label[start_idx[k]:end_idx[k], entity] = 1.
                self.label_list.append(label)
            

    def __getitem__(self, idx: int) -> Tuple[Path, pd.DataFrame]:
        return (self.file_list[idx], self.label_list[idx])

    def __len__(self) -> int:
        return len(self.file_list)



