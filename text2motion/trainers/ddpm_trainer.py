import torch
import torch.nn.functional as F
import random
import time
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
import math

# from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets1 import build_dataloader


class DDPMTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        

        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

        self.cfg_scale = args.cfg_scale if hasattr(args, 'cfg_scale') else 7.5

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()

        self.caption = caption
        self.motions = motions
        x_start = motions
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']
        self.mse_loss = output['mse_loss']
        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

    def generate_batch(self, caption, m_lens, dim_pose):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)
        T = min(m_lens.max(), self.encoder.num_frames)
        B = len(caption)
        
        # Use CFG sampling instead of regular sampling
        output = self.diffusion.p_sample_loop_with_cfg(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens,
                'text': caption
            },
            cfg_scale=self.cfg_scale
        )
        return output

    def generate(self, caption, m_lens, dim_pose, batch_size=8):
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
            print(output.shape)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output

    def backward_G(self):
        """Enhanced backward pass with diffusion-specific losses"""
        
        # 1. Basic noise prediction loss (enhanced)
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        
        # 2. Progressive denoising loss
        # Compute losses at multiple noise levels
        t_steps = torch.linspace(0, self.diffusion_steps-1, 10).long().to(self.device)
        prog_loss = 0.0
        for t in t_steps:
            t_batch = torch.full((self.motions.shape[0],), t, device=self.device)
            noise = torch.randn_like(self.motions)
            noisy_motion = self.diffusion.q_sample(self.motions, t_batch, noise=noise)
            pred_noise = self.encoder(noisy_motion, t_batch, text=self.caption)
            prog_loss += F.mse_loss(pred_noise, noise)
        prog_loss = prog_loss / len(t_steps)
        
        # 3. Time-aware motion consistency loss
        def compute_time_consistency(x, noise_levels):
            # Compute consistency across different noise levels
            consistency_loss = 0.0
            for t1, t2 in zip(noise_levels[:-1], noise_levels[1:]):
                x_t1 = self.diffusion.q_sample(x, t1.expand(x.shape[0]))
                x_t2 = self.diffusion.q_sample(x, t2.expand(x.shape[0]))
                pred_t1 = self.encoder(x_t1, t1.expand(x.shape[0]), text=self.caption)
                pred_t2 = self.encoder(x_t2, t2.expand(x.shape[0]), text=self.caption)
                consistency_loss += F.smooth_l1_loss(pred_t1, pred_t2)
            return consistency_loss / (len(noise_levels) - 1)
        
        time_steps = torch.linspace(100, 900, 5).long().to(self.device)
        loss_consistency = compute_time_consistency(self.motions, time_steps)
        
        # 4. Latent motion structure loss
        def compute_structure_loss(real_motion, fake_noise, t):
            # Get motion at different diffusion steps
            noisy_motion = self.diffusion.q_sample(real_motion, t)
            # Compute structural features (e.g., joint angles, velocities)
            def extract_features(x):
                joints = x.reshape(x.shape[0], x.shape[1], -1, 3)
                # Joint angles
                vectors = joints[:, :, 1:] - joints[:, :, :-1]
                angles = torch.acos(torch.sum(vectors[:, :, 1:] * vectors[:, :, :-1], dim=-1).clamp(-1, 1))
                return angles
            
            real_struct = extract_features(noisy_motion)
            fake_struct = extract_features(fake_noise)
            return F.mse_loss(fake_struct, real_struct)
        
        t_struct = torch.randint(0, self.diffusion_steps, (1,)).to(self.device)
        loss_structure = compute_structure_loss(self.motions, self.fake_noise, t_struct)
        
        # 5. Motion prior regularization
        def compute_prior_loss(motion):
            # Physics-based priors
            joints = motion.reshape(motion.shape[0], motion.shape[1], -1, 3)
            
            # Joint velocity smoothness
            velocity = joints[:, 1:] - joints[:, :-1]
            acc = velocity[:, 1:] - velocity[:, :-1]
            smooth_loss = torch.mean(torch.square(acc))
            
            # Joint angle limits
            vectors = joints[:, :, 1:] - joints[:, :, :-1]
            angles = torch.acos(torch.sum(vectors[:, :, 1:] * vectors[:, :, :-1], dim=-1).clamp(-1, 1))
            angle_loss = torch.mean(F.relu(angles - math.pi/2))
            
            return smooth_loss + 0.1 * angle_loss
        
        loss_prior = compute_prior_loss(self.fake_noise)
        
        # 6. Temporal coherence loss
        def compute_temporal_loss(motion, window_size=8):
            # Check temporal consistency in local windows
            temp_loss = 0.0
            for i in range(motion.shape[1] - window_size):
                window = motion[:, i:i+window_size]
                next_frame_pred = window[:, :-1].mean(dim=1)
                next_frame_true = motion[:, i+window_size]
                temp_loss += F.mse_loss(next_frame_pred, next_frame_true)
            return temp_loss / (motion.shape[1] - window_size)
        
        loss_temporal = compute_temporal_loss(self.fake_noise)
        
        # Combine all losses with weights
        total_loss = (
            1.0 * loss_mot_rec +      # Basic reconstruction
            0.5 * prog_loss +         # Progressive denoising
            0.3 * loss_consistency +  # Time consistency
            0.2 * loss_structure +    # Motion structure
            0.1 * loss_prior +        # Physics priors
            0.2 * loss_temporal +     # Temporal coherence
            self.mse_loss            # Original MSE loss
        )
        
        self.loss_mot_rec = total_loss
        
        # Store individual losses for logging
        loss_logs = OrderedDict({
            'loss_mot_rec': loss_mot_rec.item(),
            'loss_prog': prog_loss.item(),
            'loss_consistency': loss_consistency.item(),
            'loss_structure': loss_structure.item(),
            'loss_prior': loss_prior.item(),
            'loss_temporal': loss_temporal.item(),
            'loss_total': total_loss.item()
        })
        
        return loss_logs

    def update(self):
        """Enhanced update with gradient clipping and loss scaling"""
        self.zero_grad([self.opt_encoder])
        
        # Compute losses
        loss_logs = self.backward_G()
        
        # Scale gradients for better training stability
        self.loss_mot_rec.backward()
        
        # Clip gradients by value and norm
        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        
        # Update with gradient scaling
        for param in self.encoder.parameters():
            if param.grad is not None:
                param.grad *= 0.1  # Scale gradients for stability
        
        self.step([self.opt_encoder])
        
        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)
        # model_dir = '/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/t2m/t2m_new_ver2/model/latest.tar'
        # cur_epoch, it = self.load(model_dir)
        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True)
            # dist=self.opt.distributed,
            # num_gpus=len(self.opt.gpu_id))

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                caption, motions, m_lens = batch_data
                
                # Regular conditional training
                self.forward(batch_data)
                loss_logs = self.update()
                
                # Unconditional training (empty text)
                empty_caption = [""] * len(caption)
                uncond_batch = (empty_caption, motions, m_lens)
                self.forward(uncond_batch)
                uncond_loss_logs = self.update()
                
                # Combine logs
                for k, v in uncond_loss_logs.items():
                    loss_logs[f"uncond_{k}"] = v
                
                for k, v in loss_logs.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0 :
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                if it % self.opt.save_latest == 0 :
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)
