import torch
import torch.nn.functional as F
import time
from collections import OrderedDict
from os.path import join as pjoin
import math

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

# -------------------------------------------------------------------
# Adjust these imports as necessary for your codebase
# -------------------------------------------------------------------
from models.transformer import MotionTransformer
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
from datasets1 import build_dataloader
from utils.utils import print_current_loss


class DDPMTrainer(object):
    def __init__(self, args, encoder):
        """
        :param args: namespace of hyperparameters
        :param encoder: The model (possibly wrapped by DDP)
        """
        self.opt = args
        self.device = args.device

        # We'll store the wrapped model here, but access it via self._model() for custom calls
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

        # Move trainer submodules to device
        self.to(self.device)

        # Classifier-free guidance scale (optional)
        self.cfg_scale = getattr(args, 'cfg_scale', 7.5)

    def _model(self):
        """
        Helper to safely access the *underlying* model.
        If self.encoder is wrapped in DDP, returns self.encoder.module;
        otherwise returns self.encoder itself.
        """
        if hasattr(self.encoder, 'module'):
            return self.encoder.module
        return self.encoder

    def maybe_reset_all_moe_counters(self):
        """
        Safely call reset_all_moe_counters on the underlying model if it exists,
        avoiding the DDP wrapper issue.
        """
        m = self._model()
        if hasattr(m, "reset_all_moe_counters"):
            m.reset_all_moe_counters(m)

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
        """
        1. Prepares motions, captions.
        2. Samples random diffusion step t.
        3. Feeds to self.diffusion.training_losses(...).
        4. Extracts noise predictions & MoE loss (if any).
        """
        caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()

        self.caption = caption
        self.motions = motions

        B, T = motions.shape[:2]
        x_start = motions

        cur_len = torch.LongTensor(
            [min(T, m_len) for m_len in m_lens]
        ).to(self.device)

        # Sample random 't'
        t, _ = self.sampler.sample(B, x_start.device)

        # NOTE: We pass self._model() to diffusion so that the underlying model is used
        output = self.diffusion.training_losses(
            model=self._model(),  # safe to pass the wrapper; inside the forward it's normal
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )

        # Predicted noise vs real noise
        self.real_noise = output['target']
        self.fake_noise = output['pred']

        # If your code returns moe_loss
        self.moe_loss = output.get("moe_loss", 0.0)

        # Generate a mask (if your Transformer uses one)
        # Safely call the custom method on the underlying model
        model = self._model()
        if hasattr(model, "generate_src_mask"):
            self.src_mask = model.generate_src_mask(T, cur_len).to(x_start.device)
        else:
            # If the model does not have that method, fallback
            self.src_mask = torch.ones(B, T, device=x_start.device)

    def generate_batch(self, caption, m_lens, dim_pose):
        """
        Example: CFG sampling if your diffusion supports it.
        We call 'encode_text' on the underlying model.
        """
        m = self._model()  # underlying model
        if hasattr(m, "encode_text"):
            xf_proj, xf_out = m.encode_text(caption, self.device)
        else:
            # If model doesn't have encode_text, handle differently
            xf_proj, xf_out = None, None

        T = min(m_lens.max(), m.num_frames) if hasattr(m, "num_frames") else m_lens.max()
        B = len(caption)

        # If your diffusion code supports p_sample_loop_with_cfg
        output = self.diffusion.p_sample_loop_with_cfg(
            self._model(),  # can pass the wrapper
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
        """
        Generates motions in batches from text captions.
        """
        N = len(caption)
        cur_idx = 0
        self.eval_mode()
        all_output = []

        while cur_idx < N:
            end_idx = min(cur_idx + batch_size, N)
            batch_caption = caption[cur_idx:end_idx]
            batch_m_lens = m_lens[cur_idx:end_idx]

            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
            B = output.shape[0]
            print(output.shape)

            for i in range(B):
                all_output.append(output[i])

            cur_idx += batch_size

        return all_output

    def backward_G(self):
        """
        Compute a combined loss for:
          - Basic reconstruction (noise prediction)
          - Optionally MoE
        """
        # Basic noise-prediction loss
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)

        # Multiply by src_mask if you have one
        if hasattr(self, "src_mask"):
            loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        else:
            loss_mot_rec = loss_mot_rec.mean()

        # Combine with self.moe_loss if any
        total_loss = loss_mot_rec + (self.moe_loss if isinstance(self.moe_loss, torch.Tensor) else 0.0)
        self.loss_mot_rec = total_loss

        loss_logs = OrderedDict({
            'loss_mot_rec': loss_mot_rec.item(),
            'loss_moe': float(self.moe_loss) if isinstance(self.moe_loss, torch.Tensor) else self.moe_loss,
            'loss_total': total_loss.item()
        })

        return loss_logs

    def update(self):
        """
        1. zero_grad
        2. backward
        3. gradient clip
        4. optimizer step
        """
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()

        self.loss_mot_rec.backward()

        # Optionally clip gradients
        clip_grad_norm_(self._model().parameters(), max_norm=1.0)

        self.step([self.opt_encoder])
        return loss_logs

    def to(self, device):
        """ Move submodules (like self.mse_criterion) to device. """
        if self.opt.is_train:
            self.mse_criterion.to(device)
        model = self._model()
        model = model.to(device)

    def train_mode(self):
        # For DDP, calling self.encoder.train() is okay; it calls module.train() inside
       self._model().train()

    def eval_mode(self):
        self._model().eval()

    def save(self, file_name, ep, total_it):
        """
        Save optimizer + model state (DDP uses .module).
        """
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        # If wrapped in DDP, the real model is self.encoder.module
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except AttributeError:
            state['encoder'] = self.encoder.state_dict()

        torch.save(state, file_name)

    def load(self, model_dir):
        """
        Load optimizer + model state; returns last epoch and iteration.
        """
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        # If using DDP, load state into self.encoder.module
        try:
            self.encoder.module.load_state_dict(checkpoint['encoder'], strict=True)
        except AttributeError:
            self.encoder.load_state_dict(checkpoint['encoder'], strict=True)

        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        """
        Main training loop.
        `train_dataset` can be a DataLoader or a direct dataset, 
        depending on your usage.
        """
        self.to(self.device)
        self.opt_encoder = optim.Adam(self._model().parameters(), lr=self.opt.lr)

        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)

        start_time = time.time()

        # If you want to build a DataLoader here, do so; otherwise assume train_dataset is already a DataLoader
        train_loader = train_dataset

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()

            # Optionally reset MoE counters each epoch if your model uses them
            self.maybe_reset_all_moe_counters()

            for i, batch_data in enumerate(train_loader):
                # 1) Forward + update with text
                self.forward(batch_data)
                loss_logs = self.update()

                # 2) Optional: unconditional forward
                caption, motions, m_lens = batch_data
                empty_caption = [""] * len(caption)
                uncond_batch = (empty_caption, motions, m_lens)
                self.forward(uncond_batch)
                uncond_loss_logs = self.update()

                # Merge logs
                for k, v in uncond_loss_logs.items():
                    loss_logs[f"uncond_{k}"] = v

                # Accumulate
                for k, v in loss_logs.items():
                    logs[k] = logs.get(k, 0.0) + v

                it += 1

                # Print logs periodically
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                # Save "latest" periodically
                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # End of epoch: save "latest"
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # Save full checkpoint every N epochs
            if epoch % self.opt.save_every_e == 0:
                self.save(
                    pjoin(self.opt.model_dir, 'ckpt_e%03d.tar' % (epoch)),
                    epoch,
                    total_it=it
                )
