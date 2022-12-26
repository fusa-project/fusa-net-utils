import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_model(loaders: tuple, params: dict, model_path: str,
                   label_dictionary: dict) -> None:

    train_loader, valid_loader = loaders

    # n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)
    model = torch.load(model_path)
    cuda = params['evaluate']['cuda']
    if cuda and torch.cuda.device_count() > 0:
        logger.info('GPU number: {}'.format(torch.cuda.device_count()))
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    logger.info(f'Using {device}')
    model.to(device)

    model.eval()
    names, predictions, labels = [], [], []
    preds_str, label_str = [], []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            names.append(batch['filename'])
            labels.append(batch['label'].numpy())
            marshalled_batch = {}
            for key in batch:
                if key == 'filename':
                    continue
                marshalled_batch[key] = batch[key].to(device, non_blocking=True)
            predictions.append(model.forward(marshalled_batch).argmax(dim=1).cpu().numpy())
            preds_str += [label_dictionary[str(prediction)] for prediction in predictions[-1]]
            label_str += [label_dictionary[str(label)] for label in labels[-1]]
    names = np.concatenate(names)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    df = pd.DataFrame(list(zip(names,
                               predictions,
                               preds_str,
                               labels,
                               label_str)),
                      columns=['filename',
                               'prediction_num',
                               'prediction_str',
                               'label_num',
                               'label_str'])
    df.to_csv('classification_table.csv')

    labels_report = [int(label) for label in label_dictionary.keys()]
    report = classification_report(y_true=labels,
                                   y_pred=predictions,
                                   labels=labels_report,
                                   target_names=label_dictionary.values(),
                                   output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.to_csv('classification_report.csv')
