# This code is based on https://github.com/openai/guided-diffusion
# This code is based on https://github.com/SinMDM/SinMDM
"""
Train a collable single motion synthesis
"""

import os
import json
import numpy as np
import torch

from data_utils.data_util import load_seg_motion
from utils.fixseed import fixseed
from utils.parser_util import train_cosmos_args
from utils import dist_util
from train.training_loop_cosmos_fm import TrainLoop
from utils.model_util import create_model
from train.train_platforms import  NoPlatform  

def main():
    args = train_cosmos_args()
    fixseed(args.seed)
    train_platform_type = eval('NoPlatform')
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("loading motion")
    motion, labels, motion_data = load_seg_motion(args)
    motion = motion.permute(0,2,3,1)
    
    print("creating model and diffusion...")
    args.unconstrained = True
    model = create_model(args, motion_data, motion.shape[1])
    model.to(dist_util.dev())
    # flow_matching.to(dist_util.dev())

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, data=None).run_loop(motion, labels)
    train_platform.close()

if __name__ == "__main__":
    main()
