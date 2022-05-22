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

    
##### CONTRIBUCION METRICAS SEDNET-ADAVANNE

##VERSION SEDNET-TORCH
#import torch
#import numpy as np
#from codes import utils
#from ignite.metrics import Metric


#class er_rate(Metric):
#
#    def __init__(self, output_transform=lambda x: x):
#        super(er_rate, self).__init__(output_transform=output_transform)
#
#    def reset(self):
#        self.FP = 0
#        self.FN = 0
#        self.N = 0
#
#    def update(self, output):
#        y_pred, y = output
#        if len(y_pred.shape) == 3:
#            O = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])
#            T = y.reshape(y.shape[0]*y.shape[1],y.shape[2])
#        else:
#            O = y_pred
#            T = y
#        FP = torch.logical_and(T == 0, O == 1).sum(0)
#        FN = torch.logical_and(T == 1, O == 0).sum(0)
#        self.FP += FP
#        self.FN += FN
#        self.N += T.sum()
#        # ... your custom implementation to update internal state on after a single iteration
#              
#    def compute(self):
#        # compute the metric using the internal variables
#        zero = torch.Tensor([0])
#        zero = zero.to("cuda:0")
#        S = torch.minimum(self.FP,self.FN).sum()
#        D = torch.maximum((self.FN-self.FP),zero).sum()
#        I = torch.maximum((self.FP-self.FN),zero).sum()
#        ER = (S+D+I)/(self.N+zero)
#        return ER.item()
#
#class f1_score_A(Metric):
#
#    def __init__(self, output_transform=lambda x: x):
#        super(f1_score, self).__init__(output_transform=output_transform)
#
#    def reset(self):
#        self.TP = 0
#        self.Nref = 0
#        self.Nsys = 0
#
#    def update(self, output):
#        y_pred, y = output
#        if len(y_pred.shape) == 3:
#            O = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])
#            T = y.reshape(y.shape[0]*y.shape[1],y.shape[2])
#        else:
#            O = y_pred
#            T = y
#        TP = ((2*T-O)==1).sum()
#        Nref = T.sum()
#        Nsys = O.sum()
#        self.TP += TP
#        self.Nref += Nref
#        self.Nsys += Nsys
#        # ... your custom implementation to update internal state on after a single iteration
#        
#              
#    def compute(self):
#        # compute the metric using the internal variables
#        prec = self.TP / (self.Nsys+torch.finfo(torch.float).eps)
#        recall = self.TP / (self.Nref+torch.finfo(torch.float).eps)
#        f1_score = 2 * prec * recall / (prec + recall + torch.finfo(torch.float).eps)
#        return f1_score
#
#def thresholded_output_transform(output):
#    y_pred, y = output
#    y_pred = torch.round(y_pred)
#    return y_pred, y

"""
METRICAS COPIADAS TEXTUAL DESDE SEDNET ADAVANNE

#####################
# Scoring functions
#
# Code blocks taken from Toni Heittola's repository: http://tut-arg.github.io/sed_eval/
#
# Implementation of the Metrics in the following paper:
# Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 'Metrics for polyphonic sound event detection',
# Applied Sciences, 6(6):162, 2016
#####################

eps = torch.finfo(torch.float).eps

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

    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)
    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER


def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block_i = O[int(i * block_size):int(i * block_size + block_size - 1),]
        T_block_i = T[int(i * block_size):int(i * block_size + block_size - 1),]
        O_block[i, :] = np.max(O_block_i, axis=0)
        T_block[i, :] = np.max(T_block_i, axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / block_size)
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block_i = O[int(i * block_size):int(i * block_size + block_size - 1),]
        T_block_i = T[int(i * block_size):int(i * block_size + block_size - 1),]
        O_block[i, :] = np.max(O_block_i, axis=0)
        T_block[i, :] = np.max(T_block_i, axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_scores_orig(pred, y, frames_in_1_sec=50):
    scores = dict()
    scores['f1_overall_1sec'] = np.around(f1_overall_1sec(pred, y, frames_in_1_sec),3)
    scores['er_overall_1sec'] = np.around(er_overall_1sec(pred, y, frames_in_1_sec),3)
    return scores
"""
#def compute_scores(pred, y):
#    scores = dict()
#    scores['f1'] = f1_overall_framewise(pred, y)
#    scores['er'] = er_overall_framewise(pred, y)
#    return scores
