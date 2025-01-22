# This code is based on https://github.com/openai/guided-diffusion
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
from utils.parser_util import generate_cosmos_args
from utils.model_util import create_model_and_diffusion, load_model
from utils import dist_util
from data_utils.get_data import get_dataset_loader
from data_utils.humanml.scripts.motion_process import recover_from_ric
import data_utils.humanml.utils.paramUtil as paramUtil
from data_utils.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_utils.tensors import collate
from data_utils.mixamo.motion import MotionData
from Motion.transforms import repr6d2quat
from Motion import BVH
from Motion.Animation import positions_global as anim_pos
from Motion.Animation import Animation
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Quaternions import Quaternions


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_cosmos_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    motion_data = None
    num_joints = None
    repr = 'repr6d' if args.repr == '6d' else 'quat'
    if args.dataset == 'mixamo':
        motion_data = MotionData(args.sin_path, padding=True, use_velo=True,
                                 repr=repr, contact=True, keep_y_pos=True,
                                 joint_reduction=True)
        fps = int(round(1 / motion_data.bvh_file.frametime))
        n_frames = motion_data.bvh_file.anim.shape[0]
        skeleton = get_kinematic_chain(motion_data.bvh_file.skeleton.parent)
        num_joints = motion_data.raw_motion.shape[1]
    elif args.dataset == 'bvh_general':
        sin_anim, joint_names, frametime = BVH.load(args.sin_path)
        fps = int(round(1 / frametime))
        skeleton = get_kinematic_chain(sin_anim.parents)
        n_frames = sin_anim.shape[0]
        num_joints = sin_anim.shape[1]
    else:
        assert args.dataset == 'humanml'
        fps = 20
        
    if args.labels_str != '' :
        labels_index = args.labels_str.split('s')
        print(args.labels_str)
        labels_0 = int(labels_index[0]) 
        labels_1 = int(labels_index[1]) 
        labels_2 = int(labels_index[2]) 
        data_len = labels_0 + labels_1 + labels_2
        ratio_0 = labels_0 / data_len
        ratio_1 = labels_1 / data_len
        ratio_2 = labels_2 / data_len        
        labels_org = [0]*round(ratio_0*data_len)+[1]*round(ratio_1*data_len)+[2]*round(ratio_2*data_len)+[0]*round((ratio_0*0.8)*data_len)+[1]*round((ratio_1*0.8)*data_len)+[2]*round((ratio_0*0.8)*data_len)
        
        new_labels_str = str(round(ratio_0*data_len))+'s'+str(round(ratio_1*data_len))+'s'+str(round(ratio_2*data_len))+'s'+str(round((ratio_0*0.8))*data_len)+'s'+str(round((ratio_1*0.8)*data_len))+'s'+str(round((ratio_0*0.8)*data_len))
        n_frames = len(labels_org) 

    dense_labels = labels_org
    num_densecond_dims = len(set(dense_labels))
    n_frames = len(labels_org)
    max_frames = n_frames
    
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'cosmos_{}_{}_seed{}'.format(name, niter, args.seed))

    # if os.path.exists(out_path):
    #     shutil.rmtree(out_path)
    # os.makedirs(out_path)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger than default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    if args.dataset in ['humanml']:
        print('Loading dataset...')
        data = load_dataset(args, max_frames, n_frames)
    else:
        data = None
    total_num_samples = args.num_samples

    print("Creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(args, motion_data, num_joints*9)

    labels = F.one_hot(th.Tensor(dense_labels).to(th.int64), num_classes=num_densecond_dims)
    labels = labels.to(args.device).repeat(args.batch_size,1).reshape(args.batch_size, n_frames,num_densecond_dims).permute(0,2,1).float()

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
    _, model_kwargs = collate(collate_args, labels)

    all_motions = []
    all_lengths = []
    all_text = []

    print(f'### Sampling')

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

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        n_joints = 22
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        skeleton = paramUtil.t2m_kinematic_chain
    # Recover XYZ *positions* from zoo vector representation
    elif model.data_rep in ['mixamo_vec', 'bvh_general_vec']:
        sample = sample.cpu().numpy()
        sample = sample.transpose(0, 3, 1, 2)  # n_samples x n_features x n_joints x n_frames  ==>   n_samples x n_frames x n_joints x n_features
        sample = sample[0:1]
        if args.dataset == 'mixamo':
            xyz_samples = np.zeros((args.num_samples, n_frames, 24, 3))  # shape it to match the output of anim_pos
        else:
            joint_features_length = 7 if args.repr=='quat' else 9
            assert model.njoints % joint_features_length == 0
            xyz_samples = np.zeros((args.num_samples, n_frames, int(model.njoints / joint_features_length), 3))  # shape it to match the output of anim_pos
        for i, one_sample in enumerate(sample):
            bvh_path = os.path.join(out_path, f'sample{i:02d}.bvh')
            if args.dataset == 'mixamo':
                motion_data.write(os.path.expanduser(bvh_path), th.tensor(one_sample.transpose((2, 1, 0))))
                generated_motion = MotionData(os.path.expanduser(bvh_path), padding=True,
                                    use_velo=True, repr=repr, contact=True, keep_y_pos=True, joint_reduction=True)
                anim = Animation(rotations=Quaternions(generated_motion.bvh_file.get_rotation().numpy()),
                                 positions=generated_motion.bvh_file.anim.positions,
                                 orients=generated_motion.bvh_file.anim.orients,
                                 offsets=generated_motion.bvh_file.skeleton.offsets,
                                 parents=generated_motion.bvh_file.skeleton.parent)
            else:
                if args.repr == '6d':
                    one_sample = one_sample.reshape(n_frames, -1, joint_features_length)
                    quats = repr6d2quat(th.tensor(one_sample[:, :, 3:])).numpy()
                else:
                    one_sample = one_sample.reshape(n_frames, -1, joint_features_length)
                    quats = one_sample[:, :, 3:]
                anim = Animation(rotations=Quaternions(quats), positions=one_sample[:, :, :3],
                                 orients=sin_anim.orients, offsets=sin_anim.offsets, parents=sin_anim.parents)
                # BVH.save(os.path.expanduser(bvh_path), anim, joint_names, frametime, positions=True)  # "positions=True" is important for the dragon and does not harm the others
            xyz_samples[i] = anim_pos(anim)  # n_frames x n_joints x 3  =>
            # print(xyz_samples.shape)
        # sample = xyz_samples.transpose(0, 2, 3, 1)  # n_samples x n_frames x n_joints x 3  =>  n_samples x n_joints x 3 x n_frames
        sample = xyz_samples
    print(sample.shape)
    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec', 'mixamo_vec', 'bvh_general_vec'] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
    assert rot2xyz_pose_rep == 'xyz'

    caption = new_labels_str
    length = n_frames
    motion = sample[0]
    motion_tmp = motion.copy()
    # motion[:,:,2] = motion_tmp[:,:,1] *-1
    # motion[:,:,1] = motion_tmp[:,:,2]
    # motion[:,0,:] = 0
    save_file = 'result_'+new_labels_str+'.mp4'
    animation_save_path = os.path.join(out_path, save_file)
    motion_to_plot = copy.deepcopy(motion)
    plot_3d_motion(animation_save_path, skeleton, motion_to_plot, dataset=args.dataset, title=caption, fps=fps)




def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
