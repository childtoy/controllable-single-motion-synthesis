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
import torch
from utils.parser_util import generate_sinmdm_args
from utils.model_util import create_model_and_diffusion, load_model
from utils import dist_util
import shutil
from data_utils.tensors import collate
from Motion.transforms import repr6d2quat
from Motion import BVH
from Motion.Animation import positions_global as anim_pos
from Motion.Animation import Animation
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Quaternions import Quaternions
import torch.nn.functional as F
import pickle as pkl
import sys 


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_sinmdm_args()
    fixseed(args.seed)
    motion_data = None
    num_joints = None
    print('args.sin_path', args.sin_path)
    if args.dataset == 'humanml':
        motion = np.array(np.load(args.sin_path, allow_pickle=True)[None][0]['motion_raw'][0])  # benchmark npy
        motion = motion.transpose(1, 0, 2)  # n_feats x n_joints x n_frames   ==> n_joints x n_feats x n_frames
        # motion = motion.to(torch.float32)  # align with network dtype
        n_frames = motion.shape[2]
        n_frames *= 2
        print(motion.shape)

           
    else : 
        sin_anim, joint_names, frametime = BVH.load(args.sin_path)
        num_joints = sin_anim.shape[1]
        n_frames = int(sin_anim.rotations.shape[0] * 2)
        
        
    dist_util.setup_dist(args.device)

    out_path = os.path.join('save', args.dataset, args.model_path.split('/')[-2],'synthesis_samples')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = None

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, motion_data, num_joints)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)

    model.to(dist_util.dev())
    diffusion.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # disable random masking
    
    model.requires_grad_(False)
    
    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
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
    
    sample = sample.cpu().numpy()
    sample = sample.transpose(0, 3, 1, 2)  # n_samples x n_features x n_joints x n_frames  ==>   n_samples x n_frames x n_joints x n_features
    joint_features_length = 9
    if args.dataset != 'humanml' :
        assert model.njoints % joint_features_length == 0
        xyz_samples = np.zeros((args.num_samples, n_frames, int(model.njoints / joint_features_length), 3))  # shape it to match the output of anim_pos
        
        for i, one_sample in enumerate(sample):
            bvh_path = os.path.join(out_path, f'sample{i:02d}.bvh')
            one_sample = one_sample.reshape(n_frames, -1, joint_features_length)
            quats = repr6d2quat(torch.tensor(one_sample[:, :, 3:])).numpy()
            anim = Animation(rotations=Quaternions(quats), positions=one_sample[:, :, :3],
                                orients=sin_anim.orients, offsets=sin_anim.offsets, parents=sin_anim.parents)
            BVH.save(os.path.expanduser(bvh_path), anim, joint_names, frametime, positions=True)  # "positions=True" is important for the dragon and does not harm the others
            xyz_samples[i] = anim_pos(anim)  # n_frames x n_joints x 3  =>
    data = {}
    data['sample'] = sample
    pkl_path = os.path.join(out_path, 'synthesis_samples.pkl')
    pkl.dump(data, open(pkl_path, "wb"))
            

if __name__ == "__main__":
    main()
