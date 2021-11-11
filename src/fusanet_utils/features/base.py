from abc import ABC, abstractmethod
from os.path import isfile, splitext
import pathlib
import torch

from .waveform import get_waveform

class Feature(ABC):
    
    def __init__(self, params):
        self.params = params
        super().__init__()
    
    @abstractmethod
    def compute(self, waveform: torch.Tensor):
        pass

    def create_path(self, waveform_path: pathlib.Path) -> pathlib.Path:
        feature_name = type(self).__name__
        file_name = waveform_path.stem + "_" + feature_name + ".pt"
        for k, part in enumerate(waveform_path.parts[::-1]):
            if part == 'datasets':
                break
        pre_path = pathlib.Path(*waveform_path.parts[:-(k+1)])
        pos_path = pathlib.Path(*waveform_path.parts[-k:-1])
        (pre_path / "features" / pos_path).mkdir(parents=True, exist_ok=True)
        return pre_path / "features" / pos_path / file_name
    
    def write_to_disk(self, waveform_path: str) -> None:
        feature_path = self.create_path(pathlib.Path(waveform_path))
        if not feature_path.exists() or self.params["overwrite"]: 
            waveform =  get_waveform(waveform_path, self.params)
            feature = self.compute(waveform)
            torch.save(feature, feature_path)

    def read_from_disk(self, waveform_path: str) -> torch.Tensor:
        feature_path = self.create_path(pathlib.Path(waveform_path))
        if feature_path.exists():
            return torch.load(feature_path)
        else:
            raise FileNotFoundError("Feature file not found")

