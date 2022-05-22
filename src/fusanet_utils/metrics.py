import torch
import sklearn.metrics

def accuracy(y, label):
    if label.ndim == 3: # SED
        return torch.sum((y > 0.5) == label).item()/(label.shape[1]*label.shape[2])
    else:
        return torch.sum(y.argmax(1) == label).item()

def f1_score(y, label):
    if label.ndim == 3: # SED
        return 0.0
    else:
        return sklearn.metrics.f1_score(label.cpu().numpy(), y.cpu().argmax(dim=1).numpy(), average='macro')
