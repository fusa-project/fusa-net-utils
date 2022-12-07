import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..transforms import Collate_and_transform


def evaluate_model(dataset, params: dict, model_path: str,
                   label_dictionary: dict) -> None:
    model = torch.load(model_path)
    model.cpu()
    model.eval()
    names, predictions, labels = [], [], []
    preds_str, label_str = [], []

    my_collate = Collate_and_transform(params['features'])
    loader = DataLoader(dataset, batch_size=8, collate_fn=my_collate,
                        num_workers=4, pin_memory=True)
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
