import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import os 
import pickle as pkl
import sys
from densecls.model import DenseCLS_UNet
from data_utils.data_util import load_sin_motion
from utils.fixseed import fixseed
from utils.parser_util import train_dense_cls_args
import json
from Motion.AnimationStructure import get_kinematic_chain
from scipy.interpolate import interp1d
import sys

args = train_dense_cls_args()
fixseed(args.seed)

motion, _ = load_sin_motion(args)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)


motion = motion.permute(1,0,2)
n_joints, n_featues, n_frames  = motion.shape
accuracys = []
model = DenseCLS_UNet(in_channels=n_joints)    
        
# Configuration
max_iter    = int(3e4)
batch_size  = 128
print_every = 1e4
eval_every  = 1e4

# Loop
model.to(args.device)
model.train()
optm = th.optim.AdamW(params=model.parameters(),lr=1e-4,weight_decay=0.0)
schd = th.optim.lr_scheduler.ExponentialLR(optimizer=optm,gamma=0.99998)
criterion = th.nn.CrossEntropyLoss()
print(args.labels_str)
# Device Uniformly for Three Dense Labels 
if args.labels_str == '':
    labels_0 = int(n_frames*0.3)
    labels_1 = int(n_frames*0.3)
    labels_2 = n_frames - labels_0 - labels_1

else : 
    labels_str = args.labels_str
    labels_index = labels_str.split('s')
    labels_0 = int(labels_index[0]) 
    labels_1 = int(labels_index[1])
    labels_2 = int(labels_index[2])
        
labels_org = [0]*labels_0+[1]*labels_1+[2]*labels_2
labels_sublen = th.Tensor([labels_0,labels_1,labels_2])        
if motion.shape[-1] == len(labels_org):
    assert('wrong sequence length ')
    
    
def adjust_sequence_length(input_data, labels_index, new_labels_index):
    new_length = new_labels_index.cumsum()
    num_joints, num_dim, num_frames = motion.shape
    input_data = input_data.reshape(num_joints*num_dim, num_frames).transpose(1,0)
    new_data = np.zeros(input_data.shape)
    dims = input_data.shape[1]
    for dim in range(dims):
        data = input_data[:,dim]
        for i, length in enumerate(labels_index):
            start_idx = sum(labels_index[:i]) 
            end_idx = start_idx + length
            x = np.arange(start_idx, end_idx)
            f = interp1d(x, data[start_idx:end_idx], axis=0, kind='linear')
            new_length = new_labels_index[i]
            new_start_idx = sum(new_labels_index[:i])
            interpolated_sequence = f(np.linspace(start_idx, end_idx - 1, new_length))

            # print('new_start_idx', new_start_idx+new_length-new_start_idx)
            # print(interpolated_sequence.shape)
            # print(new_data.shape)
            # print(new_start_idx+new_length)
            # print(new_data[new_start_idx:new_start_idx+new_length, dim].shape)
            new_data[new_start_idx:new_start_idx+new_length, dim] = interpolated_sequence
    new_data = new_data.transpose(1,0)
    return new_data


# Adjust timing for motion and dense label
new_motion_data = []
new_labels_cond = []
for ratio_0 in [0.2,0.4,0.6,0.8,1.0] :
    for ratio_2 in [0.2,0.4,0.6,0.8,1.0] :
        labels_cond = [0]*round(labels_0*ratio_0)+[1]*(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2)))+[2]*round(labels_2*ratio_2)
        new_labels_index = np.array([round(labels_0*ratio_0),(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2))),round(labels_2*ratio_2)])
        motion_data = motion.cpu().numpy()
        labels_index = labels_sublen.numpy().astype(np.int32)
        output = adjust_sequence_length(motion_data, labels_index, new_labels_index)
        new_labels_mask = [round(labels_0*ratio_0),(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2))),round(labels_2*ratio_2)]
        new_motion_data.append(output)
        new_labels_cond.append(labels_cond)

motions = th.Tensor(np.array(new_motion_data)).to(args.device)
labels = F.one_hot(th.Tensor(np.array(new_labels_cond)).to(th.int64), num_classes=3).to(args.device).float()
# print(motions.shape)
motions = motions.unsqueeze(2)
for it in range(max_iter):
    # Zero gradient
    optm.zero_grad()
    idx = np.random.choice(motions.shape[0], batch_size)        
    batch = motions[idx]
    batch_label = labels[idx]
    # batch = batch.reshape(batch_size, -1, 167)
    output = model(batch) # [B x C x L]
    # Compute error
    loss = criterion(output, batch_label.reshape(-1,3))
    # Update
    loss.backward()
    optm.step()
    schd.step()
    
    # Print
    if (it%print_every) == 0 or it == (max_iter-1):
        print ("it:[%7d][%.1f]%% loss:[%.4f]"%(it,100*it/max_iter,loss.item()))

    if (it%eval_every) == 0 or it == (max_iter-1):
        correct = 0
        total = 0
        model.eval()

        with th.no_grad():
            ratio_0 = 0.5
            ratio_2 = 0.5            
            labels_cond = [0]*round(labels_0*ratio_0)+[1]*(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2)))+[2]*round(labels_2*ratio_2)
            if n_frames != len(labels_cond) : 
                diff_frames = (len(labels_cond)-n_frames)
                labels_cond = [0]*round(labels_0*ratio_0)+[1]*(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2)))+[2]*(round(labels_2*ratio_2)-diff_frames)
                new_labels_index = np.array([round(labels_0*ratio_0),(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2))),round(labels_2*ratio_2)-diff_frames])
            else :
                new_labels_index = np.array([round(labels_0*ratio_0),(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2))),round(labels_2*ratio_2)])    
            # new_labels_index = np.array([round(labels_0*ratio_0),(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2))),round(labels_2*ratio_2)])
            motion_data = motion.cpu().numpy()
            labels_index = labels_sublen.numpy().astype(np.int32)
            output = adjust_sequence_length(motion_data, labels_index, new_labels_index)
            batch = th.Tensor(output).unsqueeze(0).unsqueeze(2).to(args.device)
            label_cond = th.Tensor(labels_cond).to(args.device).repeat(1,1).reshape(-1).long()
            output = model(batch) # [B x C x L]
            _, predicted = th.max(output.data, 1)
            total += label_cond.size(0)
            correct += (predicted == label_cond).sum().item()
            accuracys.append(100 * correct // total)
            print(f'Half Accuracy : {100 * correct // total} %')
        th.save(model.state_dict(), args.output_path+'/model-'+str(it+1)+'.pt')         
        model.train()
        
with open(os.path.join(args.output_path,'dense_cls_accuracy.json'), 'w') as fw:
    json.dump(accuracys, fw, indent=4, sort_keys=True)
