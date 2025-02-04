#!/usr/bin/env python

import os
import sys
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# -------------------------------------------------------------------------
# Add your project root to PYTHONPATH if needed
# -------------------------------------------------------------------------
project_root = "/iridisfs/scratch/tvtn1c23/MotionDiffusion-MoE"
text2motion_path = os.path.join(project_root, 'text2motion')
sys.path.append(project_root)
sys.path.append(text2motion_path)

import numpy as np
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions  # <--- your custom argument parser
from utils.plot_script import *
from models import MotionTransformer
from trainers import DDPMTrainer
from datasets1 import Text2MotionDataset, build_dataloader


def setup_for_distributed(is_master: bool):
    """
    Disable printing when not on master process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def build_models(opt, dim_pose: int):
    """
    Build the MotionTransformer model.
    """
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff
    )
    return encoder


def main():
    # ---------------------------------------------------------------------
    # 1. Parse command-line arguments from your custom parser.
    #    (Adjust if you want to add or remove arguments.)
    # ---------------------------------------------------------------------
    parser = TrainCompOptions()
    opt = parser.parse()  # e.g. '--name test --batch_size 128 --dataset_name t2m ...'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    # ---------------------------------------------------------------------
    # 2. Read environment variables set by torchrun
    #    e.g. torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 ...
    # ---------------------------------------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # ---------------------------------------------------------------------
    # 3. Initialize the distributed process group
    # ---------------------------------------------------------------------
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # ---------------------------------------------------------------------
    # 4. Setup device and printing
    # ---------------------------------------------------------------------
    torch.cuda.set_device(local_rank)
    opt.device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)
    setup_for_distributed(is_master)  # Only master prints

    # ---------------------------------------------------------------------
    # 5. Dataset-specific config
    # ---------------------------------------------------------------------
    if opt.dataset_name == 't2m':
        opt.data_root = '/iridisfs/scratch/tvtn1c23/HumanML3D/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir   = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir   = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.max_motion_length = 196
        dim_pose = 251
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError(f"Unknown dataset name: {opt.dataset_name}")

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std  = np.load(pjoin(opt.data_root, 'Std.npy'))
    train_split_file = pjoin(opt.data_root, 'train.txt')

    # ---------------------------------------------------------------------
    # 6. Build model and wrap with DistributedDataParallel (DDP)
    # ---------------------------------------------------------------------
    encoder = build_models(opt, dim_pose).to(opt.device)
    ddp_encoder = DDP(
        encoder,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    # ---------------------------------------------------------------------
    # 7. Create trainer
    # ---------------------------------------------------------------------
    trainer = DDPMTrainer(opt, ddp_encoder)

    # ---------------------------------------------------------------------
    # 8. Build dataset & distributed sampler & dataloader
    # ---------------------------------------------------------------------
    # 1) Create the Dataset
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)
    
    # 2) Create the distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,  # total number of processes
        rank=rank,               # index of the current process
        shuffle=True
    )
    
    # 3) Create a direct DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,  # how many samples per GPU
        sampler=train_sampler,      # pass the distributed sampler
        num_workers=4,
        drop_last=True,
        shuffle=False,             # Sampler handles shuffling
        pin_memory=True,           # optional
        persistent_workers=True    # optional
    )


    # ---------------------------------------------------------------------
    # 9. Training loop
    # ---------------------------------------------------------------------
    try:
        trainer.train(train_loader)
    except Exception as e:
        # Print any errors clearly (including non-master ranks)
        print(f"[Rank {rank}] Error: {e}", force=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
