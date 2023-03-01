from typing import Dict, Tuple
import logging
import time
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from sklearn.metrics import classification_report
from dvclive import Live
import pandas as pd
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler

from .transforms import Collate_and_transform
from .augmentations.additive_noise import PinkNoise, WhiteNoise
from .datasets.external import ESC, UrbanSound8K, VitGlobal, SINGAPURA
from .datasets.fusa import FUSA_dataset
from .datasets.simulated import SimulatedPoliphonic
from .datasets.aumilab import AUMILAB
from .models.naive import ConvolutionalNaive
from .models.PANN_tag import Wavegram_Logmel_Cnn14
from .models.PANN_sed import Cnn14_DecisionLevelAtt, AttBlock
from .models.ADAVANNE_sed import SEDnet, TimeDistributed
from .models.HTS.htsat import HTSAT_Swin_Transformer
from .metrics import accuracy, f1_score, error_rate

logger = logging.getLogger(__name__)


def initialize_model(model_path: str, params: Dict, n_classes: int, cuda: bool):
    pretrained_cache = pathlib.Path("../../pretrained_models")
    if  params['model'] == 'naive':
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
        cls_in_shape = 2048
        if 'finetuning' in params:
            if params['finetuning'] == 'PANN-pretrained':
                if cuda:
                    model = torch.load(pretrained_cache / 'Wavegram_Logmel_Cnn14_mAP=0.439.pth')
                else:
                    model = torch.load(pretrained_cache / 'Wavegram_Logmel_Cnn14_mAP=0.439.pth', map_location=torch.device('cpu'))
            if params['freeze']:
                for param in model.parameters():
                    param.requires_grad = False
            if params['clf_head'] == '1L':
                model.fc1 = torch.nn.Linear(2048, 2048, bias=True)
            if params['clf_head'] == '3L':
                cls_in_shape = 512
                model.fc1 = torch.nn.Sequential(
                    torch.nn.Linear(2048, 1024, bias=True),
                    torch.nn.Linear(1024, 512, bias=True),
                    torch.nn.Linear(512, 512, bias=True)
                )
        model.fc_audioset = torch.nn.Linear(cls_in_shape, n_classes, bias=True)

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
        att_in_shape = 2048
        if 'finetuning' in params:
            if params['finetuning'] == 'PANN-pretrained':
                model_name = 'Cnn14_DecisionLevelAtt_mAP=0.425.pth'
            if params['finetuning'] == 'SPASS':
                model_name = 'Poliphonic-PANN-sed.pt'
            if cuda:
                model = torch.load(pretrained_cache / model_name)
            else:
                model = torch.load(pretrained_cache / model_name, map_location=torch.device('cpu'))
            if params['freeze']:
                for param in model.parameters():
                    param.requires_grad = False
            if params['clf_head'] == '1L':
                model.fc1 = torch.nn.Linear(2048, 2048, bias=True)
            if params['clf_head'] == '3L':
                att_in_shape = 512
                model.fc1 = torch.nn.Sequential(
                    torch.nn.Linear(2048, 1024, bias=True),
                    torch.nn.Linear(1024, 512, bias=True),
                    torch.nn.Linear(512, 512, bias=True)
                )
        model.att_block = AttBlock(att_in_shape, n_classes, activation='sigmoid')
        
    elif params['model'] == 'ADAVANNE-sed':
        model = SEDnet(n_classes=n_classes)
        if 'finetuning' in params:
            if params['finetuning'] == 'SPASS':
                model_name = 'Poliphonic-ADAVANNE-sed.pt'
            if cuda:
                model = torch.load(pretrained_cache / model_name)
            else:
                model = torch.load(pretrained_cache / model_name, map_location=torch.device('cpu'))
            if params['freeze']:
                for param in model.parameters():
                    param.requires_grad = False
            if params['clf_head'] == '1L':
                model.FC = torch.nn.Sequential(
                    TimeDistributed(torch.nn.Linear(64, out_features=32)),
                    torch.nn.Dropout(p=0.5))
        model.output = torch.nn.Sequential(
            TimeDistributed(torch.nn.Linear(32, out_features=n_classes)))
        
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
        if 'finetuning' in params:
            if params['finetuning'] == 'SPASS':
                model_name = 'Poliphonic-HTS.pt'
            if cuda:
                model = torch.load(pretrained_cache / model_name)
            else:
                model = torch.load(pretrained_cache / model_name, map_location=torch.device('cpu'))
            if params['freeze']:
                for param in model.parameters():
                    param.requires_grad = False
            if params['clf_head'] == '1L':
                SF = model.spec_size // (2 ** (len(model.depths) - 1)) // model.patch_stride[0] // model.freq_ratio
                model.tscam_conv = torch.nn.Conv2d(
                    in_channels = model.num_features,
                    out_channels = n_classes,
                    kernel_size = (SF, 3),
                    padding = (0,1)
                )
        model.head = torch.nn.Linear(n_classes, n_classes)

    torch.save(model, model_path)

def create_dataset(root_path, params: Dict, stage: str='train'):
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

def create_dataloaders(dataset, params: Dict):
    if 'ft_percent' in params['train']:
        train_size = int(params["train"]["train_percent"]*len(dataset) * params["train"]["ft_percent"])
        valid_size = int(len(dataset)* params["train"]["ft_percent"]) - train_size
        test_size = len(dataset) - train_size - valid_size
    if 'test_percent':
        train_size = int(params["train"]["train_percent"]*len(dataset))
        test_size = int(params["train"]["test_percent"]*len(dataset))
        valid_size = len(dataset) - train_size - test_size
    #train_subset, valid_subset = random_split(dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
    train_collate = Collate_and_transform(params['features'])
    train_subset, valid_subset, test_subset = random_split(dataset, (train_size, valid_size, test_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
    if 'augmentation' in params['train']:
        if params['train']['augmentation'] is None:
            train_collate = Collate_and_transform(params['features'])
        elif params['train']['augmentation'] == 'pink':
            train_collate = Collate_and_transform(params['features'], transforms=[PinkNoise()])
        elif params['train']['augmentation'] == 'white':
            train_collate = Collate_and_transform(params['features'], transforms=[WhiteNoise()])
    valid_collate = Collate_and_transform(params['features'])
    test_collate = Collate_and_transform(params['features'])
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
    test_loader = DataLoader(test_subset, batch_size=8, collate_fn=test_collate, num_workers=4, pin_memory=True)
    return train_loader, valid_loader, test_loader

def criterion(label):
    if label.ndim == 3: # SED
        return torch.nn.BCELoss()
    else:
        return torch.nn.CrossEntropyLoss()

class EarlyStopping():
    def __init__(self, metric='valid_loss', patience=5):

        self.metric = metric
        if self.metric=='valid_loss':
            self.best = np.Inf
            self.metric_operator = np.less
        elif self.metric == 'f1_score':
            self.best = -np.Inf
            self.metric_operator = np.greater
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch, epoch_value):
        if self.metric_operator(self.best, epoch_value):
            self.counter +=1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Training early stopped")
        else:
            self.best = epoch_value
            self.counter = 0
            logger.info(f"New best {self.metric} in epoch {epoch} ({self.best})")

def train(loaders: Tuple, params: Dict, model_path: str, cuda: bool) -> None:
    """
    Make more abstract to other models
    """
    live = Live()
    train_loader, valid_loader, test_loader = loaders
    torch.save(test_loader, 'test_dataloader.pt')

    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)
    model = torch.load(model_path)
    
    # TODO: Clean this function
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['learning_rate'])

    if cuda and torch.cuda.device_count() > 0:
        logger.info('GPU number: {}'.format(torch.cuda.device_count()))
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    logger.info(f'Using {device}')

    early_stopping = EarlyStopping(params["train"]["stopping_criteria"], params["train"]["patience"])
    clip = 10.0
    for epoch in range(params["train"]["nepochs"]):
        global_loss = 0.0
        global_accuracy = 0.0
        model.to(device)
        model.train()
        start_time = time.time()
        for batch in train_loader:
            marshalled_batch = {}
            for key in batch:
                if key == 'filename':
                    continue
                marshalled_batch[key] = batch[key].to(device, non_blocking=True)
                if key == 'waveform':
                    if 'SINGAPURA' in params["train"]["dataset"]:
                        amplifier = 1
                        marshalled_batch[key] = marshalled_batch[key] * amplifier
            optimizer.zero_grad()
            y = model.forward(marshalled_batch)
            loss = criterion(marshalled_batch['label'])(y, marshalled_batch['label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            global_loss += loss.item()
            global_accuracy += accuracy(y, marshalled_batch['label'])
        logger.info(f"{epoch}, train/loss {global_loss/n_train:0.4f}")
        logger.info(f"{epoch}, train/accuracy {global_accuracy/n_train:0.4f}")
        live.log('train/loss', global_loss/n_train)
        live.log('train/accuracy', global_accuracy/n_train)
        logger.info(f"train time: {time.time() - start_time:0.4f} [s]")

        global_loss = 0.0
        global_accuracy = 0.0
        global_f1_score = 0.0
        global_error_rate = 0.0
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for batch in valid_loader:
                marshalled_batch = {}
                for key in batch:
                    if key == 'filename':
                        continue
                    marshalled_batch[key] = batch[key].to(device, non_blocking=True)
                    if key == 'waveform':
                        if 'SINGAPURA' in params["train"]["dataset"]:
                            amplifier = 1
                            marshalled_batch[key] = marshalled_batch[key] * amplifier
                y = model.forward(marshalled_batch)
                loss = criterion(marshalled_batch['label'])(y, marshalled_batch['label'])
                global_loss += loss.item()
                global_accuracy += accuracy(y, marshalled_batch['label'])
                global_f1_score += f1_score(y, marshalled_batch['label'])
                global_error_rate += error_rate(y, marshalled_batch['label'])
        logger.info(f"{epoch}, valid/loss {global_loss/n_valid:0.4f}")
        logger.info(f"{epoch}, valid/accuracy {global_accuracy/n_valid:0.4f}")
        logger.info(f"{epoch}, f1_score_macro {global_f1_score/len(valid_loader):0.4f}")
        logger.info(f"{epoch}, error_rate {global_error_rate/len(valid_loader):0.4f}")
        if params["train"]["stopping_criteria"] == 'valid_loss':
            early_stopping(epoch, global_loss/n_valid)
        elif params["train"]["stopping_criteria"] == 'f1_score':
            early_stopping(epoch, global_f1_score/len(valid_loader))
        live.log('valid/loss', global_loss/n_valid)
        live.log('valid/accuracy', global_accuracy/n_valid)
        live.log('f1_score_macro', global_f1_score/len(valid_loader))
        live.log('error_rate', global_error_rate/len(valid_loader))
        logger.info(f"valid time: {time.time() - start_time:0.4f} [s]")
        live.next_step()

        #Saving best model
        if early_stopping.early_stop:
            break
        else:
            if device == 'cuda':
                model.cpu()
            torch.save(model, model_path)
            model.create_trace()
    
def evaluate_model(dataset, params: Dict, model_path: str, label_dictionary: Dict) -> None:
    model = torch.load(model_path)
    model.cpu()
    model.eval()
    names, predictions, labels = [], [], []
    preds_str, label_str = [], []

    my_collate = Collate_and_transform(params['features'])
    loader = DataLoader(dataset, batch_size=8, collate_fn=my_collate, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for batch in tqdm(loader):
            names.append(batch['filename'])
            predictions.append(model.forward(batch).argmax(dim=1).numpy())
            labels.append(batch['label'].numpy())
            preds_str += [label_dictionary[str(prediction)] for prediction in predictions[-1]]
            label_str += [label_dictionary[str(label)] for label in labels[-1]]
    names = np.concatenate(names)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    df = pd.DataFrame(list(zip(names, predictions, preds_str, labels, label_str)), columns=['filename', 'prediction_num', 'prediction_str', 'label_num', 'label_str'])
    df.to_csv('classification_table.csv')

    report = classification_report(y_true=labels, y_pred=predictions, labels=[int(label) for label in label_dictionary.keys()], target_names=label_dictionary.values(), output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.to_csv('classification_report.csv')
