import copy
import functools
import os
import random
import time
import sys

import blobfile as bf
import numpy as np
import torch as th 
import torch.nn.functional as F
from torch.optim import AdamW
from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler
from diffusion.resample import create_named_schedule_sampler
from utils import dist_util

from Motion.transforms import repr6d2quat, quat2repr6d
from Motion import BVH
from Motion.Animation import positions_global as anim_pos
from Motion.Animation import Animation
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Quaternions import Quaternions
from data_utils.humanml.utils.plot_script import plot_3d_motion

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.num_densecond_dim = args.num_densecond_dim
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.arch = args.arch
        self.lr_method = args.lr_method
        self.lr_step = args.lr_step
        self.lr_gamma = args.lr_gamma

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_method == 'StepLR':
            assert self.lr_step is not None
            self.lr_scheduler = th.optim.lr_scheduler.StepLR(self.opt, self.lr_step, gamma=self.lr_gamma)
        elif self.lr_method == 'ExponentialLR':
            self.lr_scheduler = th.optim.lr_scheduler.ExponentialLR(self.opt, gamma=self.lr_gamma)
        else:
            assert self.lr_method is None, f'lr scheduling {self.lr_method} is not supported.'

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = th.device("cpu")
        if th.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = th.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper = None
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def adjust_learning_rate(self, optimizer, step, args):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if hasattr(self, 'lr_scheduler'):
            return
        if step < args.warmup_steps:
            lr = args.lr * step / args.warmup_steps
        else:
            lr = args.lr
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)


    def run_loop(self, motion, labels):
        n_sample, n_joints, n_features, n_frames = motion.shape
        self.motion_shape = motion.shape
        labels = F.one_hot(labels.to(th.int64), num_classes=self.num_densecond_dim).to(self.args.device)
        labels = labels.permute(0,2,1).float()
        start_time_measure = time.time()
        time_measure = []
        for self.step in range(self.num_steps-self.resume_step):
            idx = np.random.choice(motion.shape[0], self.batch_size)        
            batch = motion[idx] # [B x C x L]
            cond = {'y': {'mask': None, 'frame_labels': labels[idx]}}
            self.run_step(batch, cond)
            self.visualize()  
            start_time_measure, time_measure = self.apply_logging(start_time_measure, time_measure)
            if self.total_step() % self.save_interval == 0 and self.total_step() != 0 or self.total_step() == self.num_steps - 1:
                self.save()
                self.visualize()                       
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                    return
            self.adjust_learning_rate(self.opt, self.total_step(), self.args)

        if len(time_measure) > 0:
            mean_times = sum(time_measure) / len(time_measure)
            print(f'Average time for {self.log_interval} iterations: {mean_times} seconds.')

    def apply_logging(self, start_time_measure, time_measure):
        if self.total_step() % self.log_interval == 0:
            for k, v in logger.get_current().name2val.items():
                if k == 'loss':
                    print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                if k in ['step', 'samples'] or '_q' in k:
                    continue
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')
            if self.total_step() > 0:
                end_time_measure = time.time()
                elapsed = end_time_measure - start_time_measure
                time_measure.append(elapsed)
                print(f'Time of last {self.log_interval} iterations: {int(elapsed)} seconds.')
                start_time_measure = time.time()
            self.train_platform.report_scalar(name='Learning Rate', value=self.opt.param_groups[0]['lr'],
                                              iteration=self.total_step(), group_name='LR')
        return start_time_measure, time_measure

    def visualize(self):
        self.ddp_model.eval()
        bvh_path = os.path.join(self.save_dir, f'eval_sample{self.step:02d}.bvh')
        sin_anim, joint_names, frametime = BVH.load(self.args.sin_path)
        skeleton = get_kinematic_chain(sin_anim.parents)
        eval_batch_size = 1
        # repr_6d = quat2repr6d(th.tensor(sin_anim.rotations.qs))
        # motion = np.concatenate([sin_anim.positions, repr_6d], axis=2)
        # motion = th.from_numpy(motion)
        # motion = motion.permute(1, 2, 0)  # n_frames x n_joints x n_feats  ==> n_joints x n_feats x n_frames
        # motion = motion.reshape(-1, motion.shape[-1]).unsqueeze(0) # n_joints x n_feats x n_frames (n_joints = 1)
        # motion = motion.to(th.float32)  # align with network dtype
        # sample = motion
        sample_fn = self.diffusion.p_sample_loop

        sample = sample_fn(
            self.ddp_model,
            (self.args.batch_size, self.ddp_model.njoints, self.ddp_model.nfeats, 334),
            clip_denoised=False,
            model_kwargs=self.eval_model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        sample = sample.permute(0,3,2,1)
        one_sample = sample[0].reshape(self.motion_shape[-1], -1, 9).cpu().numpy()
        # one_sample = sample[0].reshape(167, 63, 9).cpu().numpy()
        quats = repr6d2quat(th.tensor(one_sample[:, :, 3:])).cpu().numpy()
        anim = Animation(rotations=Quaternions(quats), positions=one_sample[:,:,:3],
                                orients=sin_anim.orients, offsets=sin_anim.offsets, parents=sin_anim.parents)
        BVH.save(os.path.expanduser(bvh_path), anim, joint_names, frametime, positions=True)  # "positions=True" is important for the dragon and does not harm the others
        xyz_samples = anim_pos(anim)  # n_frames x n_joints x 3  =>
        # print('xyz_samples', xyz_samples.shape)
        sample = xyz_samples  # n_frames x n_joints x 3  
        sample_tmp = sample.copy()
        # sample[:,:,2] = sample_tmp[:,:,1] *-1
        # sample[:,:,1] = sample_tmp[:,:,2]
        caption = 'evaluation sample'
        length = self.motion_shape[-1]
        animation_save_path = os.path.join(self.save_dir, f'eval_sample{self.step:02d}.gif')
        motion_to_plot = copy.deepcopy(sample)
        plot_3d_motion(animation_save_path, skeleton, motion_to_plot, dataset=self.args.dataset, title=caption, fps=20)
        self.ddp_model.train()
         
         
    def print_changed_lr(self, lr_saved):
        lr_cur = self.opt.param_groups[0]['lr']
        if lr_saved is not None:
            if abs(lr_saved - lr_cur) > 0.05 * lr_saved:
                print(f'step {self.total_step()}: lr_saved updated to ', lr_cur)
                lr_saved =lr_cur
        else:
            lr_saved = lr_cur
            print(f'step {self.total_step()}: lr_saved = ', lr_cur)
        return lr_saved

    def total_step(self):
        return self.step + self.resume_step\
            
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            self.eval_model_kwargs = micro_cond 
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # random resize
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset if self.data is not None else None
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def log_step(self):
        logger.logkv("step", self.total_step() + self.resume_step)
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{self.total_step():09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{self.total_step():09d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
