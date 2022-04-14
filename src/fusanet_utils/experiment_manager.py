from typing import Dict, Tuple
import logging
import time
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.metrics import classification_report, f1_score
from dvclive import Live
import pandas as pd
from tqdm import tqdm

from .transforms import Collate_and_transform
from .datasets.external import ESC, UrbanSound8K, VitGlobal
from .datasets.fusa import FUSA_dataset
from .datasets.simulated import SimulatedPoliphonic
from .models.naive import ConvolutionalNaive
from .models.PANN_tag import Wavegram_Logmel_Cnn14
from .models.PANN_sed import Cnn14_DecisionLevelAtt, AttBlock

logger = logging.getLogger(__name__)


def initialize_model(model_path: str, params: Dict, n_classes: int, cuda: bool, pretrained: bool=True):
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
        if pretrained:
            if cuda:
                checkpoint = torch.load(pretrained_cache / 'Wavegram_Logmel_Cnn14_mAP=0.439.pth')
            else:
                checkpoint = torch.load(pretrained_cache / 'Wavegram_Logmel_Cnn14_mAP=0.439.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            for param in model.parameters():
                param.requires_grad = False
        model.fc_audioset = torch.nn.Sequential(torch.nn.Linear(2048, 1024), torch.nn.Linear(1024, 512), torch.nn.Linear(512, n_classes))

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
        if pretrained:
            if cuda:
                checkpoint = torch.load(pretrained_cache / 'Cnn14_DecisionLevelAtt_mAP=0.425.pth')
            else:
                checkpoint = torch.load(pretrained_cache / 'Cnn14_DecisionLevelAtt_mAP=0.425.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            for param in model.parameters():
                param.requires_grad = False
        model.att_block = AttBlock(2048, n_classes, activation='sigmoid')

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
    # Create dataset for the experiment and save dictionary of classes index to names
    return FUSA_dataset(ConcatDataset(dataset), feature_params=params["features"])

def create_dataloaders(dataset, params: Dict):
    train_size = int(params["train"]["train_percent"]*len(dataset))
    valid_size = len(dataset) - train_size
    train_subset, valid_subset = random_split(dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(params["train"]["random_seed"]))
    my_collate = Collate_and_transform(params['features'])
    train_loader = DataLoader(train_subset, shuffle=True, batch_size=params["train"]["batch_size"], collate_fn=my_collate, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=8, collate_fn=my_collate, num_workers=4, pin_memory=True)
    return train_loader, valid_loader

def train(loaders: Tuple, params: Dict, model_path: str, cuda: bool) -> None:
    """
    Make more abstract to other models
    """
    live = Live()
    train_loader, valid_loader = loaders

    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)    
    model = torch.load(model_path)
    #criterion = torch.nn.BCELoss()
    # TODO: Add param to select BCE/CrossEntropyLoss
    # TODO: Clean this function
    criterion=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['learning_rate'])

    if cuda and torch.cuda.device_count() > 0:
        logger.info('GPU number: {}'.format(torch.cuda.device_count()))
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    logger.info(f'Using {device}') 

    best_valid_loss = np.inf
    for epoch in range(params["train"]["nepochs"]):
        global_loss = 0.0
        global_accuracy = 0.0  
        model.to(device) 
        model.train()
        start_time = time.time()
        for batch in train_loader:
            marshalled_batch = {}
            for key in batch:
                #if key == 'label':
                #    batch[key] = torch.zeros(len(batch[key]), n_classes).scatter_(1, batch[key].unsqueeze(1), 1.)
                if key == 'filename':
                    continue
                marshalled_batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad()
            y = model.forward(marshalled_batch)
            loss = criterion(y, marshalled_batch['label'])
            loss.backward()
            optimizer.step()
            global_loss += loss.item()
            accuracy = torch.sum(y.argmax(dim=1) == marshalled_batch['label'])
            global_accuracy += accuracy.item()
        logger.info(f"{epoch}, train/loss {global_loss/n_train:0.4f}")
        logger.info(f"{epoch}, train/accuracy {global_accuracy/n_train:0.4f}")
        live.log('train/loss', global_loss/n_train)
        live.log('train/accuracy', global_accuracy/n_train)
        logger.info(f"train time: {time.time() - start_time:0.4f} [s]")
        
        global_loss = 0.0
        global_accuracy = 0.0 
        global_f1_score = 0.0 
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for batch in valid_loader:
                marshalled_batch = {}
                for key in batch:
                    #if key == 'label':
                    #    batch[key] = torch.zeros(len(batch[key]), n_classes).scatter_(1, batch[key].unsqueeze(1), 1.)
                    if key == 'filename':
                        continue
                    marshalled_batch[key] = batch[key].to(device, non_blocking=True)
                y = model.forward(marshalled_batch)
                loss = criterion(y, marshalled_batch['label'])
                global_loss += loss.item()                         
                accuracy = torch.sum(y.argmax(dim=1) == marshalled_batch['label'])
                global_accuracy += accuracy.item()
                global_f1_score += f1_score(marshalled_batch['label'].cpu(), y.cpu().argmax(dim=1), average='macro')
        logger.info(f"{epoch}, valid/loss {global_loss/n_valid:0.4f}")
        logger.info(f"{epoch}, valid/accuracy {global_accuracy/n_valid:0.4f}")
        logger.info(f"{epoch}, f1_score macro {global_f1_score/len(valid_loader):0.4f}")
        live.log('valid/loss', global_loss/n_valid)
        live.log('valid/accuracy', global_accuracy/n_valid)
        live.log('f1_score macro', global_f1_score/len(valid_loader))
        logger.info(f"valid time: {time.time() - start_time:0.4f} [s]")
        live.next_step()

        if global_loss < best_valid_loss:
            logger.info(f"new best valid loss in epoch {epoch}!")
            if device == 'cuda': 
                model.cpu()
            torch.save(model, model_path)
            model.create_trace()
            best_valid_loss = global_loss

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
