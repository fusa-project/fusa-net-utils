import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ConvolutionalNaive(nn.Module):

    def __init__(self, n_classes, n_filters=32, n_hidden=32):
        super(type(self), self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(1, n_filters, (1, 3), (1, 2))
        self.conv2 = nn.Conv2d(n_filters, n_filters, (1, 3), (1, 2))
        self.global_time_pooling = nn.AdaptiveMaxPool2d((None, 1))
        self.linear_size = n_filters*64
        self.linear1 = nn.Linear(self.linear_size, n_hidden)    
        self.linear2 = nn.Linear(n_hidden, n_classes) 

    def forward(self, x):
        x = x['mel_transform']
        logger.debug(f"{x.shape}") 
        z = self.activation(self.conv1(x))
        logger.debug(f"{z.shape}") 
        z = self.activation(self.conv2(z))
        logger.debug(f"{z.shape}") 
        z = self.global_time_pooling(z)
        logger.debug(f"{z.shape}") 
        z = self.activation(self.linear1(z.view(-1, self.linear_size)))
        return self.linear2(z)

    def create_trace(self, path='traced_model.pt'):
        dummy_example = {'mel_transform': torch.randn(10, 1, 64, 500)}
        traced_model = torch.jit.trace(self, (dummy_example))
        traced_model.save(path)


    