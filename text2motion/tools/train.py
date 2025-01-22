import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to PYTHONPATH
project_root = '/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffusion-MoE'
text2motion_path = os.path.join(project_root, 'text2motion')
sys.path.append(project_root)
sys.path.append(text2motion_path)

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *
from models import MotionTransformer
from trainers import DDPMTrainer
from datasets1 import Text2MotionDataset, build_dataloader
import numpy as np
from os.path import join as pjoin

def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff
    )
    return encoder

def setup_for_distributed(is_master):
    """
    Disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main_worker(local_rank, nprocs, opt):
    """
    Spawns one process per GPU. local_rank is the GPU index within the node.
    """
    # 1. Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=opt.world_size, rank=opt.node_rank * nprocs + local_rank)
    torch.cuda.set_device(local_rank)

    is_master = (dist.get_rank() == 0)
    setup_for_distributed(is_master)  # Only master prints

    # 2. Build dataset config (same as single GPU)
    if opt.dataset_name == 't2m':
        opt.data_root = './HumanML3D/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))
    train_split_file = pjoin(opt.data_root, 'train.txt')

    # 3. Build model and wrap with DDP
    encoder = build_models(opt, dim_pose).cuda(local_rank)
    ddp_encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 4. Create trainer
    trainer = DDPMTrainer(opt, ddp_encoder)

    # 5. Build dataset and dataloader. We must use DistributedSampler.
    #    (If using your custom build_dataloader, ensure it can handle sampler.)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=opt.batch_size,
        drop_last=True,
        workers_per_gpu=4,
        shuffle=False,  # Turn off shuffle here, because the sampler does it
        sampler=train_sampler
    )

    # 6. Train
    trainer.train(train_loader)

    # 7. Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()

    # Example: if you run with torch.distributed.launch, it will pass local_rank as an arg.
    # Otherwise, define an argument or read from env variables.
    # For example:
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--node_rank', type=int, default=0, help="Rank of this node (0 if single-node)")
    args, unknown = parser.parse_known_args()
    opt.local_rank = args.local_rank
    opt.node_rank = args.node_rank

    # total number of processes = number_of_nodes * nprocs_per_node
    # If you used --nproc_per_node=4, world_size = 4 * number_of_nodes
    # This can also come from environment variables set by the launcher.
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))

    # For multi-GPU training, you typically do not want manual device assignment outside
    # the main_worker. So remove or ignore 'opt.device = torch.device("cuda")'
    # Instead, each process sets its own device.

    # Launch processes
    nprocs = torch.cuda.device_count()  # Usually #GPUs on the node
    mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, opt))
