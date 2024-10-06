import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os 
import pickle as pkl

from densecls.model import DenseCLS_UNet
from utils.fixseed import fixseed
from utils.parser_util import pred_dense_cls_args

args = pred_dense_cls_args()
fixseed(args.seed)

data_dir = args.data_path
model_dir = args.model_path

n_frames = int(data_dir.split('/')[-1].split('_')[-2])
data_path = os.path.join(data_dir,'synthesis_samples.pkl')
out_path = os.path.join(data_dir, 'synthesis_dataset.pkl')

with open(data_path, 'rb') as f : 
    data = pkl.load(f)

synth_data = data['sample']

print('synth_data shape', synth_data.shape)

n_sample = synth_data.shape[0]
n_frames = synth_data.shape[1]
n_joints = synth_data.shape[2]
model = DenseCLS_UNet(in_channels=n_joints)    
model.load_state_dict(th.load(model_dir))
model.eval()

# Loop
model.to(args.device)
batch = th.Tensor(synth_data).permute(0,2,3,1).to(args.device)
output = model(batch) # [B x C x L]
_, predicted = th.max(output.data, 1)
predicted  = predicted.reshape(n_sample,n_frames)

outdata = {}
outdata['pred'] = predicted.cpu().numpy()
outdata['input'] = synth_data

with open(out_path, 'wb') as f : 
    pkl.dump(outdata, f)