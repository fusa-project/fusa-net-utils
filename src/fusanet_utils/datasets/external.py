from pathlib import Path
import logging
from typing import Tuple, Dict, Union, List
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from abc import abstractmethod

logger = logging.getLogger(__name__)

def get_label_transforms(repo_path: Union[Path, str], dataset_name: str) -> Dict:
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    inverse_transforms = pd.read_json(repo_path / "fusa_taxonomy.json").T[dataset_name].to_dict()
    transforms = {}
    for key, values in inverse_transforms.items():
        for value in values:
            transforms[value] = key
    return transforms

class FolderDataset(Dataset):

    def __init__(self, folder_path: Union[str, Path], label_transforms: Dict=None, allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg']):
        """
        Expects a path having subfolders containing audio files from the same class. 
        The label is taken from the subfolder's name
        """
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)        
        self.file_list = [f for f in folder_path.rglob("*") if f.is_file() and f.suffix in allowed_extensions]
        if label_transforms is not None:
            # This assummes that all keys are in transforms
            self.labels = [label_transforms[f.parent.stem] for f in self.file_list]
        else:
            self.labels = [f.parent.stem for f in self.file_list]
        self.categories =  list(set(self.labels))
        
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return (self.file_list[idx], self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)

class DatasetWithCSVMetadata(Dataset):

    def __init__(self, repo_path: Union[str, Path], audio_rel_path: Path, metadata_rel_path: Path, label_transforms: Dict=None):
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        self.repo_path = repo_path
        datasets_path = repo_path / "datasets"
        self.audio_prefix_path = datasets_path / audio_rel_path
        file_paths, labels = self.parse_metadata(datasets_path / metadata_rel_path)
        file_exists = file_paths.apply(lambda file: file.exists())
        if not file_exists.all():
            logger.warning("Existen rutas incorrectas o archivos perdidos")
            file_paths = file_paths.loc[file_exists]
            labels = labels.loc[file_exists]
        
        if label_transforms is not None:
            # Verify that there are no typos in FUSA_taxonomy
            if not all([key in set(labels.unique()) for key in label_transforms.keys() if key != ""]):
                logger.warning(f"{self.__class__}: Existen llaves que no calzan en fusa_taxonomy.json")
            label_exists = labels.apply(lambda label: label in label_transforms)
            file_paths = file_paths.loc[label_exists]
            labels = labels.loc[label_exists].apply(lambda label: label_transforms[label])
        
        self.file_list = file_paths.tolist()
        self.labels = labels.tolist()    
        self.categories = sorted(list(set(self.labels)))

    @abstractmethod
    def parse_metadata(self, metadata_path: Path) -> Tuple[pd.Series, pd.Series]:
        """
        Returns a tuple of series with the paths and labels
        """
        pass

    def __getitem__(self, idx: int) -> Tuple[Path, int]:
        return (self.file_list[idx], self.labels[idx])
        
    def __len__(self) -> int:        
        return len(self.file_list)


class ESC(DatasetWithCSVMetadata):

    def __init__(self, repo_path: Union[str, Path]):
        label_transforms = get_label_transforms(repo_path, "ESC")
        super().__init__(repo_path, audio_rel_path=Path("ESC-50") / "audio", metadata_rel_path=Path("ESC-50") / "meta" / "esc50.csv", label_transforms=label_transforms)

    def parse_metadata(self, metadata_path: Path) -> Tuple:
        df = pd.read_csv(metadata_path)
        file_names = df["filename"]
        labels = df["category"]
        file_paths = file_names.apply(lambda file_name: self.audio_prefix_path / file_name)
        return file_paths, labels

class UrbanSound8K(DatasetWithCSVMetadata):

    def __init__(self, repo_path: Union[str, Path]):
        label_transforms = get_label_transforms(repo_path, "UrbanSound")
        super().__init__(repo_path, audio_rel_path=Path("UrbanSound8K") / "audio", metadata_rel_path=Path("UrbanSound8K") / "metadata" / "UrbanSound8K.csv", label_transforms=label_transforms)

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
        label_transforms = get_label_transforms(repo_path, "Vitglobal")
        super().__init__(repo_path, audio_rel_path=Path("VitGlobal") / "audio" / "dataset", metadata_rel_path=Path("VitGlobal") / "meta" / "audios_eruido2022_20220121.csv", label_transforms=label_transforms)

    def parse_metadata(self, metadata_path: Path) -> Tuple:
        df = pd.read_csv(metadata_path)
        file_names = df["WAVE_URL"].apply(lambda x: Path(x).name)
        labels = df["WAVE_MEMO"]
        file_paths = file_names.apply(lambda file_name: self.audio_prefix_path / file_name)
        return file_paths, labels


class MMA_folder(DatasetWithCSVMetadata):

    def __init__(self, repo_path: Union[str, Path], folder_name: str, csv_name: str):
        super().__init__(repo_path, audio_rel_path=Path("MMA") / "Etiquetado 2" / folder_name, metadata_rel_path=Path("MMA") / "Etiquetado 2" / csv_name)

    def parse_metadata(self, metadata_path: Path) -> Tuple:
        df = pd.read_csv(metadata_path)
        file_names = df["audio"].apply(lambda x: Path(x).name)

        def select_first_label(label):
            if label[0] == '{':
                return pd.read_json(label).loc[0][0]
            return label

        labels = df["taxonomia"].apply(select_first_label)
        file_paths = file_names.apply(lambda file_name: self.audio_prefix_path / file_name)
        return file_paths, labels

class MMA(Dataset):
    def __init__(self, repo_path: Union[str, Path]):
        self.datasets = []
        self.datasets.append(MMA_folder(repo_path, "00629", "Autopista Central (00629) project-13-at-2022-02-04-17-40-cbaf7e83.csv"))
        self.datasets.append(MMA_folder(repo_path, "00849", "Ñuñoa (00849) project-4-at-2022-02-04-17-42-4f3ef0e5.csv"))
        self.datasets.append(MMA_folder(repo_path, "01006", "Las Condes (01006) project-5-at-2022-02-04-17-52-1013047c.csv"))
        self.datasets.append(MMA_folder(repo_path, "01008", "Viña del Mar (01008) project-6-at-2022-02-04-17-41-8a7bf886.csv"))
        self.datasets.append(MMA_folder(repo_path, "01009", "La Florida (01009) project-7-at-2022-02-04-17-41-6bbd85d2.csv"))
        self.dataset = ConcatDataset(self.datasets)  
        self.categories = []
        for dataset in self.datasets:
            self.categories += dataset.categories
        self.categories = sorted(list(set(self.categories)))      

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.dataset[idx]
        
    def __len__(self) -> int:        
        return len(self.dataset)

class SINGAPURA(Dataset):
    
    def __init__(self, repo_path: Union[str, Path], categories: List=None):
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        label_transforms = get_label_transforms(repo_path, "Singapura")
        dataset_path = repo_path / 'datasets' / 'SINGAPURA'
        self.singapura_classes = pd.read_json(dataset_path / "singapura_classes.json")
        self.file_list, self.labels, self.categories = [], [], []
        meta_folder = Path(dataset_path / 'labels_public')
        for file_path in list(meta_folder.rglob('*.csv')):
            df = pd.read_csv(file_path)
            for file_name, metadata in df.groupby('filename'):
                folder = (file_name.split('][')[1]).split('T')[0]
                file_path = dataset_path / 'labelled' / folder / file_name
            if not file_path.exists():
                logger.warning(f"El archivo {file_name} no existe")
                continue
            self.file_list.append(file_path)
            metadata = metadata[["event_label", "proximity", "onset", "offset"]]
            metadata = metadata.rename(columns={"event_label":"class", "onset": "start (s)", "offset": "end (s)"})
            metadata["class"] = metadata["class"].apply(lambda x: self.translate_classes(x))
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
    
    def translate_classes(self, singapura_class):
        category = int(singapura_class.split('-')[0])
        subcategory = int(singapura_class.split('-')[1])
        if category == 13 and subcategory == 2:
            subcategory = 0
        return self.singapura_classes[category][subcategory]