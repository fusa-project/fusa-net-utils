import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

logger = logging.getLogger(__name__)

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out
    
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x

class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self, sampling_rate, n_fft, hop_length, n_mels, fmin, 
        fmax, n_classes):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, 
            win_length=n_fft, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sampling_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, n_classes, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        input = input["waveform"]
        input = torch.squeeze(input, 1)
        logger.debug(f"{input.shape}") 

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        logger.debug(f"{a1.shape}") 
        a1 = self.pre_block1(a1, pool_size=4)
        logger.debug(f"{a1.shape}") 
        a1 = self.pre_block2(a1, pool_size=4)
        logger.debug(f"{a1.shape}") 
        a1 = self.pre_block3(a1, pool_size=4)
        logger.debug(f"{a1.shape}") 
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        logger.debug(f"{a1.shape}") 
        a1 = self.pre_block4(a1, pool_size=(2, 1))
        logger.debug(f"{a1.shape}") 

        # Log mel spectrogram
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        logger.debug(f"{x.shape}") 
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, n_mels)
        logger.debug(f"{x.shape}")
        
        x = x.transpose(1, 3)
        logger.debug(f"{x.shape}") 
        x = self.bn0(x)
        logger.debug(f"{x.shape}") 
        x = x.transpose(1, 3)
        logger.debug(f"{x.shape}") 

        if self.training:
            x = self.spec_augmenter(x)
            logger.debug(f"{x.shape}")

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        logger.debug(f"{x.shape}")

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        min_len = torch.min(torch.tensor([x.shape[2], a1.shape[2]]))
        x = torch.cat((x[:, :, :min_len,:], a1[:, :, :min_len,:]), dim=1)
        logger.debug(f"{x.shape}")

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        logger.debug(f"{x.shape}")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        logger.debug(f"{x.shape}")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        logger.debug(f"{x.shape}")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        logger.debug(f"{x.shape}")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        logger.debug(f"{x.shape}")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        logger.debug(f"{x.shape}")
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        logger.debug(f"{x.shape}")
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        logger.debug(f"{x.shape}")
        embedding = F.dropout(x, p=0.5, training=self.training)
        logger.debug(f"{embedding.shape}")
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        logger.debug(f"{clipwise_output.shape}")
        
        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding
            }

        return self.fc_audioset(x)
        
    def create_trace(self, path='traced_model.pt'):
        dummy_example = torch.randn(1, 1, 160000)
        traced_model = torch.jit.trace(self, (dummy_example), strict=False)
        traced_model.save(path)