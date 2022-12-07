import torch
import pathlib
import logging
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchsampler import ImbalancedDatasetSampler

from ..augmentations.additive_noise import PinkNoise, WhiteNoise
from ..datasets.external import ESC, UrbanSound8K, VitGlobal, SINGAPURA
from ..datasets.fusa import FUSA_dataset
from ..datasets.simulated import SimulatedPoliphonic
from ..datasets.aumilab import AUMILAB
from ..models.naive import ConvolutionalNaive
from ..models.PANN_tag import Wavegram_Logmel_Cnn14
from ..models.PANN_sed import Cnn14_DecisionLevelAtt, AttBlock
from ..models.ADAVANNE_sed import SEDnet
from ..models.HTS.htsat import HTSAT_Swin_Transformer
from ..transforms import Collate_and_transform

logger = logging.getLogger(__name__)


def initialize_model(model_path: str, params: dict, n_classes: int):
    pretrained_cache = pathlib.Path("../../pretrained_models")
    if params['model'] == 'naive':
        model = ConvolutionalNaive(n_classes=n_classes)
    elif params['model'] == 'PANN-tag':
        model = Wavegram_Logmel_Cnn14(
            n_classes=527,
            sampling_rate=32000,
            n_fft=1024,
            hop_length=320,
            n_mels=64,
            fmin=50,
            fmax=14000
            )
        if params['pretrained']:
            if params['cuda']:
                checkpoint = torch.load(pretrained_cache / 'Wavegram_Logmel_Cnn14_mAP=0.439.pth')
            else:
                checkpoint = torch.load(pretrained_cache / 'Wavegram_Logmel_Cnn14_mAP=0.439.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            for param in model.parameters():
                param.requires_grad = False
        model.fc_audioset = torch.nn.Sequential(torch.nn.Linear(2048, 1024),
                                                torch.nn.Linear(1024, 512),
                                                torch.nn.Linear(512, n_classes))

    elif params['model'] == 'PANN-sed':
        model = Cnn14_DecisionLevelAtt(
            classes_num=527,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000
            )
        if params['pretrained']:
            if params['cuda']:
                checkpoint = torch.load(pretrained_cache / 'Cnn14_DecisionLevelAtt_mAP=0.425.pth')
            else:
                checkpoint = torch.load(pretrained_cache / 'Cnn14_DecisionLevelAtt_mAP=0.425.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            for param in model.parameters():
                param.requires_grad = False
        model.fc1 = torch.nn.Sequential(torch.nn.Linear(2048, 1024),
                                        torch.nn.Linear(1024, 512),
                                        torch.nn.Linear(512, 512))
        model.att_block = AttBlock(512, n_classes, activation='sigmoid')
    elif params['model'] == 'ADAVANNE-sed':
        model = SEDnet(n_classes=n_classes)
    elif params['model'] == 'HTS':
        model = HTSAT_Swin_Transformer(
            classes_num=n_classes,
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000
        )
    torch.save(model, model_path)


def create_dataset(root_path, params: dict,
                   stage: str = 'train'):
    dataset = []
    print(params[stage]['dataset'])
    if  'ESC' in params[stage]['dataset']:
        dataset.append(ESC(root_path))
    if  'US' in params[stage]['dataset']:
        dataset.append(UrbanSound8K(root_path))
    if 'VitGlobal' in params[stage]['dataset']:
        dataset.append(VitGlobal(root_path))
    if 'Poliphonic-mini' in params[stage]['dataset']:
        dataset.append(SimulatedPoliphonic(root_path, mini=True))
    if 'Poliphonic' in params[stage]['dataset']:
        dataset.append(SimulatedPoliphonic(root_path, mini=False))
    if 'Poliphonic-external' in params[stage]['dataset']:
        dataset.append(SimulatedPoliphonic(root_path, mini=False, external=True))
    if 'AUMILAB' in params[stage]['dataset']:
        dataset.append(AUMILAB(root_path, None, True))
    if 'SINGAPURA' in params[stage]['dataset']:
        dataset.append(SINGAPURA(root_path))
    # Create dataset for the experiment and save dictionary of classes index to names
    return FUSA_dataset(ConcatDataset(dataset), params)

def create_dataloaders(dataset, params: dict):
    train_size = int(params["train"]["train_percent"]*len(dataset))
    valid_size = len(dataset) - train_size
    train_subset, valid_subset = random_split(dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
    train_collate = Collate_and_transform(params['features'])
    if 'augmentation' in params['train']:
        if params['train']['augmentation'] is None:
            train_collate = Collate_and_transform(params['features'])
        elif params['train']['augmentation'] == 'pink':
            train_collate = Collate_and_transform(params['features'], transforms=[PinkNoise()])
        elif params['train']['augmentation'] == 'white':
            train_collate = Collate_and_transform(params['features'], transforms=[WhiteNoise()])
    valid_collate = Collate_and_transform(params['features'])
    train_loader = DataLoader(train_subset, shuffle=True,
                              batch_size=params["train"]["batch_size"],
                              collate_fn=train_collate, num_workers=4, pin_memory=True)
    if 'balanced' in params['train']:
        if params['train']['balanced']:
            train_loader = DataLoader(train_subset, 
                                    sampler=ImbalancedDatasetSampler(train_subset,
                                                                    labels=[sample['label'].item() for sample in train_subset]
                                                                    ),
                                    batch_size=params["train"]["batch_size"],
                                    collate_fn=train_collate, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=8, collate_fn=valid_collate, num_workers=4, pin_memory=True)
    return train_loader, valid_loader


