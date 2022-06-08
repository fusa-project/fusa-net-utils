import torch
import sklearn.metrics
import numpy as np

eps = torch.finfo(torch.float).eps

def accuracy(y, label):
    if label.ndim == 3: # SED
        return torch.sum((y > 0.5) == label).item()/(label.shape[1]*label.shape[2])
    else:
        return torch.sum(y.argmax(1) == label).item()

def f1_score(y, label):
    if label.ndim == 3: # SED
        frames_in_1_sec = 100
        y = torch.where(y > 0.5, 1., 0.)
        f1_score = np.round(f1_overall_1sec(y, label, frames_in_1_sec), 3)
        return f1_score
    else:
        return sklearn.metrics.f1_score(label.cpu().numpy(), y.cpu().argmax(dim=1).numpy(), average='macro')

def error_rate(y, label):
    if label.ndim == 3: # SED
        frames_in_1_sec = 100
        y = torch.where(y > 0.5, 1., 0.)
        error_rate = np.round(er_overall_1sec(y, label, frames_in_1_sec).item(), 3)
        return error_rate
    else:
        return eps

def reshape_3Dto2D(A):
    return A.view(A.shape[0] * A.shape[1], A.shape[2])

def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score

def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)

    FP = torch.logical_and(T == 0, O == 1).sum(1)
    FN = torch.logical_and(T == 1, O == 0).sum(1)
    S = torch.min(FP, FN).sum()
    D = torch.max(torch.tensor([0]), FN-FP).sum()
    I = torch.max(torch.tensor([0]), FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + torch.tensor([0]))
    return ER

def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = torch.zeros((new_size, O.shape[1]))
    T_block = torch.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block_i = O[int(i * block_size):int(i * block_size + block_size - 1),]
        T_block_i = T[int(i * block_size):int(i * block_size + block_size - 1),]
        O_block[i, :] = torch.max(O_block_i, axis=0)[0]
        T_block[i, :] = torch.max(T_block_i, axis=0)[0]
    return f1_overall_framewise(O_block, T_block)

def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / block_size)
    O_block = torch.zeros((new_size, O.shape[1]))
    T_block = torch.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block_i = O[int(i * block_size):int(i * block_size + block_size - 1),]
        T_block_i = T[int(i * block_size):int(i * block_size + block_size - 1),]
        O_block[i, :] = torch.max(O_block_i, axis=0)[0]
        T_block[i, :] = torch.max(T_block_i, axis=0)[0]
    return er_overall_framewise(O_block, T_block)
