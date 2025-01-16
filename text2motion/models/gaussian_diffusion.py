"""
This code is borrowed from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
Enhanced with additional sampling schedules, samplers (DPMSolver), and adaptive schedule sampling.
"""

import enum
import math
import numpy as np
import torch as th
import torch
from abc import ABC, abstractmethod
import torch.distributed as dist


# ------------------------------------------------------------------------------------------
# Enhanced Beta Schedules
# ------------------------------------------------------------------------------------------

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    Beta schedules:
      - 'linear': from Ho et al (fixed schedule).
      - 'cosine': from Nichol & Dhariwal (cosine variant).
      - 'sqrt':   a square-root schedule (newly added).
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule_name == "sqrt":
        # A sqrt-based schedule: betas ~ (sqrt(t+1)-sqrt(t)) for t in [0,1]
        # This is just a demonstration; you can tweak the min/max range.
        max_beta = 0.02
        min_beta = 0.0001
        alphas = np.linspace(1.0, 0.0, num_diffusion_timesteps, dtype=np.float64)
        betas = (1 - alphas**2)
        # Normalize to [min_beta, max_beta]
        betas = (betas - betas.min()) / (betas.max() - betas.min())
        betas = betas * (max_beta - min_beta) + min_beta
        return betas

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)




# ------------------------------------------------------------------------------------------
# ScheduleSampler and New AdaptiveLossSampler
# ------------------------------------------------------------------------------------------

def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif name == "adaptive-loss":
        return AdaptiveLossSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Base class for samplers that adapt their weights based on observed losses.
    """
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Gather local losses and update the global reweighting.
        """
        # Gather from all processes if distributed
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad for all_gather
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Sub-classes should override to update their internal weighting from the full list
        of timesteps and losses across all ranks.
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    Original second-moment-based reweighting from the code:
    - If we haven't warmed up, just sample uniformly.
    - Otherwise, sample with probability ~ sqrt(E[loss^2]).

    A small 'uniform_prob' is used to ensure we still occasionally sample all timesteps.
    """
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        # Add uniform mixing
        weights *= (1 - self.uniform_prob)
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class AdaptiveLossSampler(LossAwareSampler):
    """
    Enhanced version of LossAwareSampler:
    - Maintains an exponential moving average (EMA) of losses for each timestep.
    - Probability ~ sqrt(EMA(loss^2)) with a configurable warmup fraction.
    """
    def __init__(self, diffusion, alpha=0.9, uniform_prob=0.001, warmup_ratio=0.2):
        """
        :param alpha: smoothing factor for EMA of squared losses.
        :param uniform_prob: fraction of uniform sampling to keep exploration.
        :param warmup_ratio: fraction of total steps to sample uniformly before applying EMA weighting.
        """
        self.diffusion = diffusion
        self.alpha = alpha
        self.uniform_prob = uniform_prob
        self.total_steps = diffusion.num_timesteps
        self.warmup_cutoff = int(self.total_steps * warmup_ratio)

        self.ema_losses = np.zeros([self.total_steps], dtype=np.float64)
        self.ema_counts = np.zeros([self.total_steps], dtype=np.float64)
        self._step_count = 0  # tracks how many times we've updated

    def weights(self):
        # If we haven't reached warmup yet, sample uniformly
        if self._step_count < self.warmup_cutoff:
            return np.ones([self.total_steps], dtype=np.float64)

        # Probability ~ sqrt(EMA of squared losses)
        w = np.sqrt(self.ema_losses / np.maximum(self.ema_counts, 1e-8))
        w_sum = w.sum() + 1e-8
        w = w / w_sum

        # Mix with uniform
        w = w * (1 - self.uniform_prob) + self.uniform_prob / self.total_steps
        return w

    def update_with_all_losses(self, ts, losses):
        self._step_count += 1
        for t, loss in zip(ts, losses):
            # update EMA of squared loss
            sq = loss ** 2
            self.ema_counts[t] = self.alpha * self.ema_counts[t] + (1 - self.alpha) * 1.0
            self.ema_losses[t]  = self.alpha * self.ema_losses[t]  + (1 - self.alpha) * sq


# ------------------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------------------

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to scalars.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x.pow(3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image. x is assumed in [-1,1].
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: shape [N, ...] with N = batch size.
    :return: shape [N, 1, ...] with the same number of dimensions as broadcast_shape.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ------------------------------------------------------------------------------------------
# Model Mean/Var Enums
# ------------------------------------------------------------------------------------------

class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()  # model predicts x_{t-1}
    START_X = enum.auto()     # model predicts x_0
    EPSILON = enum.auto()     # model predicts epsilon


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()          # raw MSE loss
    RESCALED_MSE = enum.auto() # MSE with rescaling
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


# ------------------------------------------------------------------------------------------
# The Core Diffusion Class (GaussianDiffusion)
# ------------------------------------------------------------------------------------------

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        cfg_scale=7.5,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.cfg_scale = cfg_scale

        # Use float64 for accuracy
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # --------------------------------------------------------------------------------------
    # Basic q(x_t|x_0) and posterior computations
    # --------------------------------------------------------------------------------------

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :return: (mean, variance, log_variance)
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0).
        If noise=None, random normal is used.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        posterior q(x_{t-1} | x_t, x_0).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # --------------------------------------------------------------------------------------
    # Model-based sampling
    # --------------------------------------------------------------------------------------

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        p(x_{t-1} | x_t).
        Model outputs either x_{t-1}, x_0, or epsilon depending on ModelMeanType.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, 2 * C, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x_0):
            if denoised_fn is not None:
                x_0 = denoised_fn(x_0)
            if clip_denoised:
                return x_0.clamp(-1, 1)
            return x_0

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape)
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    # --------------------------------------------------------------------------------------
    # p_sample and loops (DDPM / DDIM)
    # --------------------------------------------------------------------------------------

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        noise_fn=th.randn_like,
    ):
        """
        Sample x_{t-1} from p(x_{t-1}|x_t).
        cond_fn: gradient function that modifies the mean for classifier guidance, etc.
        noise_fn: function to generate random noise.
        """
        if model_kwargs is None:
            model_kwargs = {}
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = noise_fn(x.shape, device=x.device, dtype=x.dtype)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        if cond_fn is not None:
            grad = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
            out["mean"] = out["mean"].float() + out["variance"] * grad.float()

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        before_step_fn=None,
    ):
        """
        Generate samples from the model using the original DDPM approach.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            before_step_fn=before_step_fn,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        before_step_fn=None,
    ):
        """
        Yield intermediate samples from each diffusion timestep.
        before_step_fn: an optional function called before each sampling step,
                        for logging or other side-effects.
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if before_step_fn is not None:
                before_step_fn(i, img)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    # --------------------------------------------------------------------------------------
    # DDIM sampling
    # --------------------------------------------------------------------------------------

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} using DDIM. 
        See e.g. Song et al. for details on the DDIM method.
        """
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            # condition_score modifies 'pred_xstart' and 'mean'
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # model outputs eps -> re-derive
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)

        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Progressive generator for DDIM sampling.
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            out["pred_xstart"], x, t
        )
        return out

    # --------------------------------------------------------------------------------------
    # DPM-Solver Sampler (New!)
    # --------------------------------------------------------------------------------------

    def dpmsolver_sample_loop(
        self,
        model,
        shape,
        steps=20,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        A simple DPM-Solver sampler stub for demonstration.
        steps: the number of discretized solver steps (less than self.num_timesteps).
        """

        # This stub approach is a simple variant of DPM-Solver for illustration.
        # For a full implementation, see official references:
        #  https://arxiv.org/abs/2206.00927

        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            x = noise
        else:
            x = th.randn(*shape, device=device)

        all_ts = np.linspace(self.num_timesteps - 1, 0, steps, dtype=float)
        if progress:
            from tqdm.auto import tqdm
            all_ts = tqdm(all_ts)

        def get_model_output(xt, t_continuous):
            # t_continuous is a float in [0, num_timesteps-1]
            t_int = int(t_continuous + 0.5)
            t_tensor = th.tensor([t_int] * xt.shape[0], device=xt.device)
            with th.no_grad():
                out = self.p_mean_variance(model, xt, t_tensor, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
            return out

        for i, t_cur in enumerate(all_ts):
            # Suppose we do a single-step Euler approach
            out = get_model_output(x, t_cur)
            dt = -1.0 if i + 1 < len(all_ts) else 0.0  # naive step size
            # p_mean_variance returns x_{t-1}, but we do a direct ODE approach on x_t
            # We approximate the "velocity" as: v = x_{t-1} - x_t
            x_next = out["mean"]  # approximate next step
            x = x_next

        return x

    # --------------------------------------------------------------------------------------
    # Training Losses
    # --------------------------------------------------------------------------------------



    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        One term for the variational lower-bound, in bits-per-dim.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped,
            out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        
        model.reset_all_moe_counters(model)

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # freeze var for loss
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2).view(-1)
            terms["target"] = target
            terms["pred"]   = model_output
            # ------------------------------------------------------------------------------
            #  ADD THE MOE LOAD-BALANCING LOSS
            # ------------------------------------------------------------------------------
            moe_loss = 0.0
            # If your MoE class is SwitchMoELayer, gather the load-balancing loss:
            moe_loss = moe_loss + model.get_moe_loss(model)

            # Optionally store it in `terms` for logging:
            terms["moe_loss"] = moe_loss
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        The prior KL term in bits-per-dim for the variational lower-bound.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(qt_mean, qt_log_variance, 0.0, 0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound in bits-per-dim,
        plus other related quantities.
        """
        device = x_start.device
        batch_size = x_start.shape[0]
        vb = []
        xstart_mse = []
        mse = []

        for t_ in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t_] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model, x_start, x_t, t_batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def p_sample_with_cfg(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        cfg_scale=7.5,
    ):
        """
        Sample with classifier-free guidance
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Get unconditional model kwargs by setting text to empty
        uncond_kwargs = model_kwargs.copy()
        uncond_kwargs["text"] = [""] * len(model_kwargs["text"])  # Empty text
        uncond_kwargs["xf_proj"] = None  # Clear text embeddings
        uncond_kwargs["xf_out"] = None
        
        # Get both conditional and unconditional predictions
        out_cond = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, 
            denoised_fn=denoised_fn, model_kwargs=model_kwargs
        )
        out_uncond = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=uncond_kwargs
        )
        
        # Perform guidance
        e_t = out_uncond["pred_xstart"]
        e_t_cond = out_cond["pred_xstart"]
        
        # Combine predictions with guidance
        guided_pred = e_t + cfg_scale * (e_t_cond - e_t)
        
        # Update prediction
        out = out_cond.copy()
        out["pred_xstart"] = guided_pred
        
        # Compute new mean based on guided prediction
        new_mean, _, _ = self.q_posterior_mean_variance(
            x_start=guided_pred,
            x_t=x,
            t=t
        )
        out["mean"] = new_mean
        
        # Sample with guidance
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_with_cfg(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        cfg_scale=7.5,
    ):
        """
        Generate samples with classifier-free guidance
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is None:
            noise = torch.randn(*shape, device=device)
        x_t = noise

        if progress:
            from tqdm.auto import tqdm
            timesteps = tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling')
        else:
            timesteps = reversed(range(0, self.num_timesteps))

        for t in timesteps:
            t_batch = torch.tensor([t] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample_with_cfg(
                    model,
                    x_t,
                    t_batch,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    cfg_scale=cfg_scale,
                )
                x_t = out["sample"]
                
        return x_t
