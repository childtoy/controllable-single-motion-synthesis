import numpy as np
import torch
from data_utils.mixamo.motion import MotionData
from Motion.transforms import quat2repr6d
from Motion import BVH
import pickle as pkl

def load_sin_motion(args):
    motion_data = None
    suffix = args.sin_path.lower()[-4:]
    assert suffix in ['.npy', '.bvh']
    if args.dataset == 'humanml':
        assert suffix == '.npy'
        try:
            motion = np.load(args.sin_path)  # only motion npy
            if len(motion.shape) == 2:
                motion = np.transpose(motion)
                motion = np.expand_dims(motion, axis=1)

        except:
            motion = np.array(np.load(args.sin_path, allow_pickle=True)[None][0]['motion_raw'][0])  # benchmark npy
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 0, 2)  # n_feats x n_joints x n_frames   ==> n_joints x n_feats x n_frames
        motion = motion.to(torch.float32)  # align with network dtype
    elif args.dataset == 'mixamo':  # bvh
        assert suffix == '.bvh'
        repr = 'repr6d' 
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr=repr, contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        _, raw_motion_joints, raw_motion_frames = motion_data.raw_motion.shape
        motion = motion_data.raw_motion

    else:
        assert args.dataset == 'bvh_general' and suffix == '.bvh'
        anim, _, _ = BVH.load(args.sin_path)
        repr_6d = quat2repr6d(torch.tensor(anim.rotations.qs))
        motion = np.concatenate([anim.positions, repr_6d], axis=2)
        motion = torch.from_numpy(motion)
        motion = motion.permute(1, 2, 0)  # n_frames x n_joints x n_feats  ==> n_joints x n_feats x n_frames
        motion = motion.reshape(-1, motion.shape[-1]).unsqueeze(0) # n_joints x n_feats x n_frames (n_joints = 1)
        motion = motion.to(torch.float32)  # align with network dtype

    motion = motion.to(args.device)
    return motion, motion_data

def load_seg_motion(args):
    motion_data = None
    with open(args.pkl_path, 'rb') as f:
        data = pkl.load(f)

    motion = data['input'] 
    motion = torch.from_numpy(motion).to(torch.float32) 
    motion = motion.to(args.device)
    
    labels = data['pred']
    labels = torch.from_numpy(labels).to(torch.float32)
    labels = labels.to(args.device)
    
    return motion, labels, motion_data

