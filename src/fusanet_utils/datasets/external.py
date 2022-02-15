from pathlib import Path
import logging
from typing import Tuple, Dict, Union
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from abc import abstractmethod

logger = logging.getLogger(__name__)

class FolderDataset(Dataset):

    def __init__(self, folder_path: Union[str, Path]):
        """
        Expects a path having subfolders containing audio files from the same class. 
        The label is taken from the subfolder's name
        """
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)        
        self.file_list = [f for f in folder_path.rglob("*") if f.is_file()]
        self.labels = [f.parent.stem for f in self.file_list]
        self.categories =  list(set(self.labels))
        
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return (self.file_list[idx], self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)

class DatasetWithCSVMetadata(Dataset):

    def __init__(self, repo_path: Union[str, Path], dataset_name: str, audio_rel_path: Path, metadata_rel_path: Path):
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        self.repo_path = repo_path
        # TODO: Abstract parts of init
        datasets_path = repo_path / "datasets"
        self.audio_prefix_path = datasets_path / audio_rel_path
        file_paths, labels = self.parse_metadata(datasets_path / metadata_rel_path)
        classes = labels.unique()
        label_transforms = self.get_label_transforms(dataset_name)
        # Verify that there are no typos in FUSA_taxonomy
        if not all([key in set(classes) for key in label_transforms.keys() if key != ""]):
            logger.warning(f"Existen llaves de {dataset_name} que no calzan en fusa_taxonomy.json")
        
        # Verify that all files exist
        file_exist = file_paths.apply(lambda file: file.exists())
        if not file_exist.all():
            logger.warning("Existen rutas incorrectas o archivos perdidos")
            file_paths = file_paths.loc[file_exist]
            labels = labels.loc[file_exist]
        
        self.file_list, self.labels, self.categories = [], [], []
        for label in classes:
            if label in label_transforms:
                self.categories += [label_transforms[label]]
                mask = labels == label
                self.file_list += list(file_paths.loc[mask])
                self.labels += [label_transforms[label]]*sum(mask)

    
    def get_label_transforms(self, dataset_name: str) -> Dict:
        taxonomy_path = self.repo_path / "fusa_taxonomy.json"
        a = pd.read_json(taxonomy_path).T[dataset_name].to_dict()
        transforms = {}
        for key, values in a.items():
            for value in values:
                transforms[value] = key
        return transforms

    @abstractmethod
    def parse_metadata(self, metadata_path: Path) -> Tuple:
        """
        Returns a tuple of a series with the paths and a series with the labels
        """
        pass

    def __getitem__(self, idx: int) -> Tuple[Path, int]:
        return (self.file_list[idx], self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)


class ESC(DatasetWithCSVMetadata):

    def __init__(self, repo_path: Union[str, Path]):
        super().__init__(repo_path, dataset_name="ESC", audio_rel_path=Path("ESC-50") / "audio", metadata_rel_path=Path("ESC-50") / "meta" / "esc50.csv")

    def parse_metadata(self, metadata_path: Path) -> Tuple:
        df = pd.read_csv(metadata_path)
        file_names = df["filename"]
        labels = df["category"]
        file_paths = file_names.apply(lambda file_name: self.audio_prefix_path / file_name)
        return file_paths, labels

class UrbanSound8K(DatasetWithCSVMetadata):

    def __init__(self, repo_path: Union[str, Path]):
        super().__init__(repo_path, dataset_name="UrbanSound", audio_rel_path=Path("UrbanSound8K") / "audio", metadata_rel_path=Path("UrbanSound8K") / "metadata" / "UrbanSound8K.csv")

    def parse_metadata(self, metadata_path: Path) -> Tuple:
        df = pd.read_csv(metadata_path)
        mask = df["end"] - df["start"] >= 0.7
        df = df.loc[mask]
        file_folds = df["fold"].apply(lambda x: Path(f'fold{x}')) 
        file_names = df["slice_file_name"].apply(lambda x: Path(x))
        labels = df["class"]
        file_paths = (file_folds / file_names).apply(lambda file_name: self.audio_prefix_path / file_name)
        return file_paths, labels

class VitGlobal(DatasetWithCSVMetadata):

    def __init__(self, repo_path: Union[str, Path]):
        super().__init__(repo_path, dataset_name="Vitglobal", audio_rel_path=Path("VitGlobal") / "audio" / "dataset", metadata_rel_path=Path("VitGlobal") / "meta" / "audios_eruido2022_20220121.csv")

    def parse_metadata(self, metadata_path: Path) -> Tuple:
        df = pd.read_csv(metadata_path)
        file_names = df["WAVE_URL"].apply(lambda x: Path(x).name)
        labels = df["WAVE_MEMO"]
        file_paths = file_names.apply(lambda file_name: self.audio_prefix_path / file_name)
        return file_paths, labels