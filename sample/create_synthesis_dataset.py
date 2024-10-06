# This code is based on https://github.com/openai/guided-diffusion
# This code is based on https://github.com/SinMDM/SinMDM
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import copy
import numpy as np
import torch as th 
import torch.nn.functional as F
from utils.parser_util import create_synthdatset_args
from utils.model_util import create_model_and_diffusion, load_model
from utils import dist_util
import shutil
from data_utils.data_util import load_sin_motion
from data_utils.tensors import collate
from Motion.transforms import repr6d2quat
from Motion import BVH
from Motion.Animation import positions_global as anim_pos
from Motion.Animation import Animation
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Quaternions import Quaternions
import pickle as pkl
import sys 

from densecls.model import DenseCLS_UNet
from utils.parser_util import pred_dense_cls_args


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = create_synthdatset_args()
    fixseed(args.seed)
    motion_data = None
    num_joints = None
    if args.dataset == 'humanml':
        motion = np.array(np.load(args.sin_path, allow_pickle=True)[None][0]['motion_raw'][0])  # benchmark npy
        motion = motion.transpose(1, 0, 2)  # n_feats x n_joints x n_frames   ==> n_joints x n_feats x n_frames
        n_frames = motion.shape[2]
        n_joints = motion.shape[1]    
    else : 
        motion, _ = load_sin_motion(args)
        n_joints = motion.shape[1]
        n_frames = motion.shape[2]
    n_frames *= 2
    dist_util.setup_dist(args.device)

    out_path = os.path.join('save', args.dataset, args.model_path.split('/')[-2])

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, motion_data, n_joints)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = th.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    diffusion.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # disable random masking
    
    model.requires_grad_(False)
    
    collate_args = [{'inp': th.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    _, model_kwargs = collate(collate_args)

    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    
    sample = sample.detach().cpu().numpy()
    th.cuda.empty_cache()

    sample = sample.transpose(0, 3, 1, 2)  # n_samples x n_features x n_joints x n_frames  ==>   n_samples x n_frames x n_joints x n_features
    joint_features_length = 9
    
    cls_model = DenseCLS_UNet(in_channels=n_joints)    
    cls_model.load_state_dict(th.load(args.cls_model_path))
    cls_model.eval()
    cls_model.to(args.device)
    
    synth_data = sample.copy()
    n_sample = synth_data.shape[0]
    n_frames = synth_data.shape[1]
    n_joints = synth_data.shape[2]


    batch = th.Tensor(synth_data).permute(0,2,3,1).to(args.device)
    output = cls_model(batch) # [B x C x L]
    _, predicted = th.max(output.data, 1)
    predicted  = predicted.reshape(n_sample,n_frames)


    data = {}
    data['input'] = sample
    data['pred'] = predicted.cpu().numpy()

    pkl_path = os.path.join(out_path, 'synthesis_dataset.pkl')
    pkl.dump(data, open(pkl_path, "wb"))
            

if __name__ == "__main__":
    main()
