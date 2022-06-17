from pathlib import Path
import logging
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)

mic_location = {
    "park": np.array([40.0, 2.0, -20.0]),
    "street": np.array([40.0, 2.0, -4.5]),
    "square": np.array([40.0, 2.0, -40.0]),
    "waterfront": np.array([40.0, 2.0, -20.0]),
    "market": np.array([25.0, 2.0, -20.0]),
}


class SimulatedPoliphonic(Dataset):
    def __init__(
        self, repo_path: Union[str, Path], mini: bool = True, external: bool = False
    ):
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        if mini:
            dataset_path = repo_path / "datasets" / "Poliphonic mono mini"
        else:
            if external:
                dataset_path = repo_path / "datasets" / "Poliphonic mono external"
            else:
                dataset_path = repo_path / "datasets" / "Poliphonic mono"
        self.categories = []
        self.file_list = []
        self.label_list = []
        # Find number of classes
        for file in (dataset_path / "meta").glob("*.csv"):
            df = pd.read_csv(file, sep=",", header=0)
            self.categories += list(df["class"].unique())

        self.categories = sorted(list(set(self.categories)))

        for file in (dataset_path / "meta").glob("*.csv"):
            df = pd.read_csv(file, sep=",", header=0)
            for audio_file, metadata in df.groupby(by="audio_filename"):
                meta_label = metadata[["class", "start (s)", "end (s)"]]
                # Calculate distance to mic
                meta_dist = metadata[["x (m)", "y (m)", "z (m)"]].values
                this_key = [key for key in mic_location.keys() if key in audio_file][0]
                meta_dist -= mic_location[this_key]
                # Append to label medata and save
                dist_df = pd.DataFrame({'distance': np.sqrt(np.sum(meta_dist**2, axis=1))}, index=meta_label.index)
                self.label_list.append(pd.concat((meta_label, dist_df), axis=1))
                # save path to audio file
                audio_path = dataset_path / "audios" / audio_file
                self.file_list.append(audio_path)
                
    def __getitem__(self, idx: int) -> Tuple[Path, pd.DataFrame]:
        return (self.file_list[idx], self.label_list[idx])

    def __len__(self) -> int:
        return len(self.file_list)

