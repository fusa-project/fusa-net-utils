from torch.utils.data import DataLoader, ConcatDataset
from fusanet_utils.datasets.external import ESC
from fusanet_utils.datasets.fusa import FUSA_dataset
from fusanet_utils.transforms import Collate_and_transform
from fusanet_utils.parameters import default_logmel_parameters

    
params = default_logmel_parameters()
path = '/home/phuijse/WORK/FUSA/datasets'
dataset = FUSA_dataset(ConcatDataset([ESC(path)]), feature_params=params["features"])
my_collate = Collate_and_transform(params['features'])
loader = DataLoader(dataset, shuffle=False, batch_size=5, collate_fn=my_collate)
batch = next(iter(loader))
print(batch['waveform'].shape)
print(batch['mel_transform'].shape)
print(dataset.label_int2string(batch['label']))

#%matplotlib ipympl
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(figsize=(8, 4))
#ax.imshow(batch['logmel'].detach().numpy()[0, 0])