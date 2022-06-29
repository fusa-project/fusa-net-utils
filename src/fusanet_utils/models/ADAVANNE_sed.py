import logging
from re import X
import torch
import torch.nn as nn
import pandas as pd

logger = logging.getLogger(__name__)

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


def create_set(elements, reference):
    df = pd.DataFrame(columns=reference.columns)
    for e in elements:
        data = reference[reference.filenames == e]
        df = df.append(data, ignore_index=True)
    return df


def num_2wet(df, num):
    for i in range(len(df)):
        if df.iloc[i, 2] <= num:
            if df.iloc[i, 3] >= num:
                return df.iloc[i, 0]


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


class SEDnet(nn.Module):

    def __init__(self, n_classes):

        super(type(self), self).__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 5)),
            nn.Dropout(p=0.5, inplace=True))
        self.CNN2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5, inplace=True))
        self.CNN3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5, inplace=True))
        self.RNN = nn.Sequential(
            nn.GRU(input_size=256, hidden_size=32, dropout=0.5,
                   bidirectional=True, batch_first=True),
            SelectItem(0),
            nn.Dropout(p=0.5),
            nn.GRU(input_size=64, hidden_size=32, dropout=0.5,
                   bidirectional=True, batch_first=True),
            SelectItem(0),
            nn.Dropout(p=0.5))
        self.FC = nn.Sequential(
            TimeDistributed(nn.Linear(64, out_features=32)),
            nn.Dropout(p=0.5))
        self.output = nn.Sequential(
            TimeDistributed(nn.Linear(32, out_features=n_classes)))

    def forward(self, x):
        logger.debug(f"Input: {x.keys()}")
        x = x['mel_transform']
        logger.debug(f"Input mel_transform: {x.shape}")
        x = x.permute(0, 1, 3, 2)
        z = self.CNN1(x)
        logger.debug(f"CNN1: {z.shape}")
        z = self.CNN2(z)
        logger.debug(f"CNN2: {z.shape}")
        z = self.CNN3(z)
        logger.debug(f"CNN3: {z.shape}")
        z = z.permute(0, 2, 1, 3)
        logger.debug(f"permute: {z.shape}")
        z = z.reshape((z.shape[0], z.shape[-3], -1))
        logger.debug(f"reshape: {z.shape}")
        z = self.RNN(z)
        logger.debug(f"RNN: {z.shape}")
        z = self.FC(z)
        logger.debug(f"FC: {z.shape}")
        z = self.output(z)
        logger.debug(f"output: {z.shape}")
        z = torch.sigmoid(z)
        logger.debug(f"sigmoid: {z.shape}")

        return z

    def create_trace(self, path='traced_model.pt'):
        dummy_example = {'mel_transform': torch.randn(4, 1, 40, 431)}
        traced_model = torch.jit.trace(self, (dummy_example), strict=False)
        traced_model.save(path)
        
        
if __name__ == '__main__':
    # Experiment to check the number of frames

    model = SEDnet(n_classes=527)
    
    windows = {}
    for n_samples in range(44100, 441000, 1000):
        windows_number = n_samples // 1024 + 1
        mel_transform = torch.randn([1, 1, 40, windows_number])
        pred = model({"mel_transform": mel_transform})
        windows[n_samples] = pred.shape[1]
        print(f"{windows_number} {pred.shape[1]} {pred.shape[1] == windows_number}")

    import scipy.stats
    print(scipy.stats.linregress(list(windows.keys()), list(windows.values())))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(facecolor='w')
    ax.plot(list(windows.keys()), list(windows.values()))
    ax.set_xlabel('Largo del audio')
    ax.set_ylabel('Cantidad de ventanas')
    plt.savefig('ventanas_sed.png')