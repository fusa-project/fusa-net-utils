from abc import ABC, abstractmethod
from os.path import isfile, splitext
import torch

from .waveform_utils import get_waveform

class Feature(ABC):
    
    def __init__(self, params):
        self.params = params
        super().__init__()
    
    @abstractmethod
    def compute(self, waveform: torch.Tensor):
        pass

    def create_path(self, waveform_path: str) -> str:
        feature_name = type(self).__name__
        return splitext(waveform_path)[0]+"_"+feature_name+".pt" 
    
    def write_to_disk(self, waveform_path: str) -> None:
        feature_path = self.create_path(waveform_path)
        if not isfile(feature_path) or self.params["overwrite"]: 
            waveform =  get_waveform(waveform_path, self.params)
            feature = self.compute(waveform)
            torch.save(feature, feature_path)

    def read_from_disk(self, waveform_path: str) -> torch.Tensor:
        feature_path = self.create_path(waveform_path)
        if isfile(feature_path):
            return torch.load(feature_path)
        else:
            raise FileNotFoundError("Feature file not found")