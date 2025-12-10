import enum
import torch 
import numpy as np
import torch.nn as nn
import toml

from tqdm.auto import tqdm
from torch.nn import functional as F

# modified from DPS and DSG
# DPS: https://github.com/DPS2022/diffusion-posterior-sampling
# DSG: https://github.com/LingxiaoYang2023/DSG2024
   
   
def get_noise_schedule(schedule_name, num_diffusion_timesteps, beta_start = 0.0001, beta_end = 0.02):
    if schedule_name == "linear":
        schedule = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
        schedule = schedule.numpy()
        return schedule


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if not isinstance(arr, torch.Tensor):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:
        res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def load_spk_model(config_path: str, model_filename: str, device: torch.device = None):
    if device is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = toml.load(config_path)
    spk_model = mix_src_encoder(256, config).to(device)
    spk_model.load_state_dict(torch.load(model_filename, map_location=device))
    spk_model.eval()
    
    print(f"Speaker encoder loaded from {model_filename} on {device}")
    return spk_model


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def adjust_waveform_length_math(x: torch.Tensor, n_fft: int, hop_length: int, multiple: int = 8) -> torch.Tensor:
    if x.ndim == 3:
        x = x.squeeze(1)
    B, T = x.shape
    pad = n_fft // 2
    F_orig = (T + 2 * pad - n_fft) // hop_length + 1
    remainder = F_orig % multiple
    if remainder == 0:
        T_new = T
    else:
        F_target = F_orig - remainder
        T_new = F_target * hop_length - 2 * pad + n_fft - 1
    return x[:, :T_new]


def log_sum_exp(a, b, k):
    """
    Numerically stable Log-Sum-Exp function for smooth_max.
    Calculates log(exp(k*a) + exp(k*b)) / k
    """
    max_val = np.maximum(k * a, k * b)
    return (max_val + np.log(np.exp(k * a - max_val) + np.exp(k * b - max_val))) / k


def smooth_max(a, b, k):
    return log_sum_exp(a, b, k)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto() 
    L1 = enum.auto()
    
    RESCALED_MSE = (
        enum.auto()
    )  
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        steps = 1000, 
        noise_schedule = "linear",
        model_mean_type = None,
        model_var_type = None,
        loss_type = None,
        **kwargs):
        super().__init__()

        beta_start = kwargs.get('beta_start', 0.0001)
        beta_end = kwargs.get('beta_end', 0.02)
        self.betas = get_noise_schedule(schedule_name = noise_schedule, num_diffusion_timesteps = steps, beta_start = beta_start, beta_end = beta_end)
        self.num_timesteps = int(self.betas.shape[0])
        
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        # config
        self.config = kwargs.get('config_file')
        
        # paras
        self.clip_denoised = True
        self.input_sigma_t = False
        self.rescale_timesteps = False # True

        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0) 
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) 
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) 
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)) 

        # smooth_max schedule settings
        target_floor = 0.002 
        sharpness_k = 1000.0  
        self.posterior_std_dev = 0.09 * np.sqrt(self.posterior_variance) 
        self.smooth_max_schedule = smooth_max(self.posterior_std_dev, target_floor, k=sharpness_k)

        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )


    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

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
                
        elif self.loss_type == LossType.L1:
            # to model
            if self.config['t_mode'] == "sigma":
                model_output = model(x_t, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs)
            else:
                model_output = model(x_t, self._scale_timesteps(t, True), **model_kwargs)
            
            if isinstance(model_output, tuple):
                dec_wave, spk_losses = model_output
                model_output = dec_wave
                terms["spk_loss"] = spk_losses
            
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
        
            assert (
                model_output.shape == target.shape == x_start.shape
            ), f"model_output.shape: {model_output.shape}, target.shape: {target.shape}, x_start.shape: {x_start.shape}"
            terms["l1"] = mean_flat(torch.abs(target - model_output))
            terms["loss"] = terms["l1"]
            
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # to model
            if self.config['t_mode'] == "sigma":
                model_output = model(x_t, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs)
            else:
                model_output = model(x_t, self._scale_timesteps(t, True), **model_kwargs)
                
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
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
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self,
        model,
        x,
        t,
        clip_denoised,
        model_kwargs = None):
        """
        get p(x_{t-1} | x_t)
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, _ = x.shape[:2]
        assert t.shape == (B,)
        out = {}
        
        # model -> noise
        if self.config['t_mode'] == "sigma":
            model_output = model(x, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs).squeeze(1)
            model_output = model_output.unsqueeze(1)
            # if x.ndim == 3:
            #     x = x.squeeze(1)
        else:
            model_out = model(x.squeeze(1), self._scale_timesteps(t, True), **model_kwargs) 
            if isinstance(model_out, torch.Tensor):
                model_output = model_out.unsqueeze(1)
            elif isinstance(model_out, tuple):
                model_output = model_out[0].unsqueeze(1)
                spk_infos = model_out[1]
                out["spk_info"] = spk_infos

        # variance
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        # to tensor
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        # predict EPSILON
        def process_xstart(x):
            if self.clip_denoised:
                return x.clamp(-1, 1)
            return x

        # predict x(0)
        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )

        # p(x(t-1)|x(t),x(0))
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        
        out["mean"] = model_mean
        out["variance"] = model_variance
        out["log_variance"] = model_log_variance
        out["pred_xstart"] = pred_xstart

        return out


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


    def p_sample(
        self,
        model,
        x,
        t,
        model_kwargs = None,
        prior = None):

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised = self.clip_denoised,
            model_kwargs = model_kwargs
        )  

        noise = torch.randn_like(x)
        if prior is not None:
            noise = noise * prior[None, None, ...]
        
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"], "total_out": out}


    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        coef2 = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return coef1 * x_start + coef2 * noise


    def p_sample_loop(
        self, 
        model,
        shape,
        orig_x, 
        n_src, 
        clip_denoised,
        degradation,
        measurement = None,
        model_kwargs = None,
        device = None,
        save_path = None,
        **kwargs):
        
        if save_path is not None:
            save_root = save_path
        else:
            save_root = "/mnt/work/diff_infer"

        if device is None:
            device = next(model.parameters()).device

        # bar
        pbar = tqdm(list(range(self.num_timesteps))[::-1], ncols=80)

        # noise
        # x = torch.randn(*shape, device=device) # N(0,1)
        # x.requires_grad_(False)
        
        # measurement init
        # x = torch.concat([measurement for _ in range(n_src)], dim=-1) # measurement
        # x.requires_grad_(False)
        
        # x prior, given condition
        measurement = degradation(orig_x, n_src = n_src)
        t_last = torch.tensor(range(self.num_timesteps)[-1] * shape[0], device=device) 
        noisy_measurement = self.q_sample(measurement, t = t_last - 50) 
        x = torch.concat([noisy_measurement for _ in range(n_src)], dim=-1)

        snr = 0.00001
        xi = 1.0

        corrector = CorrectorVPConditional(
            degradation=degradation,
            n_src = n_src,
            snr=snr,
            xi=xi,
            sde=self,
            score_fn=model,
            device=device,
        )

        for i in pbar:
            t = torch.tensor([i] * shape[0], device=device) # size [1]
            x = x.requires_grad_()
            
            # single src
            x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) # [single src,...]
            x_stack = torch.cat(x_solo, dim=0)
            t = torch.tensor([i] * x_stack.size(0), device=device) 
            
            if model_kwargs is not None:
                label = torch.tensor(model_kwargs, dtype=torch.long).to(x_stack.device)
                model_kwargs_in = {"condition": label}
            else:
                model_kwargs_in = None
            
            # p sample
            out = self.p_sample(model,x_stack,t,model_kwargs_in)
            p_sample_update_vector = out['mean'] - x_stack
            
            # to one
            out_samples = out['sample'].reshape(1, 1, -1)
            out_means = out['mean'].reshape(1, 1, -1)   
            out_pred_xstart = out['pred_xstart'].reshape(1, 1, -1)           

            # Give condition
            measurement = degradation(orig_x, n_src = n_src)
            t = torch.tensor([i] * shape[0], device=device) 
            noisy_measurement = self.q_sample(measurement, t)

            # dps update_fn_recons update_fn_recons_combin
            if i < 200 and i >= 0: 
                x, distance = corrector.update_fn_recons(
                                    n_src,
                                    x_t = out_samples, 
                                    x_t_mean = out_means,
                                    measurement = measurement,
                                    noisy_measurement = noisy_measurement,
                                    x_prev  = x,
                                    x_0_hat = out_pred_xstart,
                                    time = t,
                                    total_out = out,
                                    save_path = save_root,
                                    schedule = 'hybrid')
                pbar.set_postfix({'distance': distance.item()}, refresh=False)
            x = x.detach() 
            out["sample"] = x


            yield out
            x = out["sample"]
            
        return out


    def p_sample_loop_group(
        self, 
        model, 
        shape,
        orig_x, 
        n_src, 
        clip_denoised,
        degradation,
        measurement = None,
        model_kwargs = None,
        device = None):
        """
        multiplt source models
        """
        if device is None:
            device = next(model[0].parameters()).device

        # bar
        pbar = tqdm(list(range(self.num_timesteps))[::-1], ncols=80)
        # pbar = tqdm(list(range(180))[::-1], ncols=80)

        # 1. noise
        # x = torch.randn(*shape, device=device)
        # x.requires_grad_(False)
        
        # 2. measurement init
        # x = torch.concat([measurement for _ in range(n_src)], dim=-1) 
        # x.requires_grad_(False)
        
        # 3. x prior, given condition
        measurement = degradation(orig_x, n_src = n_src)
        t_last = torch.tensor(range(self.num_timesteps)[-1] * shape[0], device=device) 
        noisy_measurement = self.q_sample(measurement, t = t_last - 50) 
        x = torch.concat([noisy_measurement for _ in range(n_src)], dim=-1)
        
        snr = 0.00001
        xi = 1.0

        corrector = CorrectorVPConditional(
            degradation=degradation,
            n_src = n_src,
            snr=snr,
            xi=xi,
            sde=self,
            score_fn=model,
            device=device,
        )

        for i in pbar:   
            t = torch.tensor([i] * shape[0], device=device) 
            x = x.requires_grad_()
            
            # single src
            x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) 
            
            # models
            out_samples = []
            out_means = []
            out_pred_xstart = []
            for model_one, model_kwargs_one, x_one in zip(model, model_kwargs, x_solo):
                if model_kwargs is not None:
                    label = torch.tensor(model_kwargs_one, dtype=torch.long).to(device)
                    model_kwargs_in = {"condition": label}
                else:
                    model_kwargs_in = None
                
                # p sample
                out = self.p_sample(model_one,x_one,t,model_kwargs_in)
    
                out_samples.append(out['sample'].reshape(1, 1, -1))
                out_means.append(out['mean'].reshape(1, 1, -1))
                out_pred_xstart.append(out['pred_xstart'].reshape(1, 1, -1))           

            out_samples = torch.cat(out_samples, dim=-1)
            out_means = torch.cat(out_means, dim=-1)
            out_pred_xstart = torch.cat(out_pred_xstart, dim=-1)
            
            # Give condition
            measurement = degradation(orig_x, n_src = n_src)
            t = torch.tensor([i] * shape[0], device=device) # size [1]
            noisy_measurement = self.q_sample(measurement, t)

            # update
            if i < 200 and i >= 0: 
                x, distance = corrector.update_fn_recons(
                                    n_src,
                                    x_t = out_samples, # update 
                                    x_t_mean = out_means,
                                    measurement = measurement,
                                    noisy_measurement = noisy_measurement,
                                    x_prev  = x,
                                    x_0_hat = out_pred_xstart,
                                    time = t,
                                    total_out = out,
                                    schedule = 'hybrid')
                pbar.set_postfix({'distance': distance.item()}, refresh=False)
            x = x.detach() 
            out["sample"] = x


            yield out
            x = out["sample"]
            
        return out

    
    def direct_sample(
        self, 
        model,
        shape,
        measurement, 
        n_src, 
        clip_denoised,
        degradation,
        model_kwargs = None,
        device = None,
        diffwave = False):

        if device is None:
            device = next(model.parameters()).device

        # bar
        pbar = tqdm(list(range(self.num_timesteps))[::-1], ncols=80)
        
        # 1. initial noise
        noise_list = [torch.randn(*shape, device=device) for _ in range(n_src)]
        x = torch.cat(noise_list, dim=-1) 
        x.requires_grad_(False)
        
        snr = 0.00001
        xi = None

        corrector = CorrectorVPConditional(
            degradation=degradation,
            n_src = n_src,
            snr=snr,
            xi=xi,
            sde=self,
            score_fn=model,
            device=device,
        )

        for i in pbar:
            t = torch.tensor([i] * shape[0], device=device) 
            
            with torch.no_grad(): # unconditional: 
                # single src
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) 
                x_stack = torch.cat(x_solo, dim=0)
                t = torch.tensor([i] * x_stack.size(0), device=device) 
                
                if model_kwargs is not None:
                    label = torch.tensor(model_kwargs, dtype=torch.long).to(x_stack.device)
                    model_kwargs_in = {"condition": label}
                else:
                    model_kwargs_in = {}
                
                out = self.p_sample(model,x_stack,t,model_kwargs_in)
                out_samples = out['sample'].reshape(1, 1, -1)
                out_means = out['mean'].reshape(1, 1, -1)   
                out_pred_xstart = out['pred_xstart'].reshape(1, 1, -1)  
                    
            y = measurement
            t = torch.tensor([i] * shape[0], device=device)

            # update 1
            coefficient = 0.5 
            total_log_sum = 0
            steps = 2
            for i in range(steps):
                new_samples = []
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)
                
                # log p(y | x)
                log_p_y_x = y - x_sum
                total_log_sum += log_p_y_x
                
                start = 0
                end = y.size(-1)
                while end <= x.size(-1):
                    new_sample = (
                        out_samples[:, :, start:end] + coefficient * total_log_sum
                    )
                    new_samples.append(new_sample)
                    start = end
                    end += y.size(-1)
                x = torch.cat(new_samples, dim=-1)

            # update 2
            threshold = 150
            if t[0] < threshold and t[0] > 0:
                # to batch
                x = x.squeeze(1)
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) # [single src,...]
                x_stack = torch.cat(x_solo, dim=0)
                t_b = torch.tensor([t] * x_stack.size(0), device=x_stack.device) 
                
                if model_kwargs is not None:    
                    label = torch.tensor(model_kwargs, dtype=torch.long).to(measurement.device)
                    model_kwargs_in = {"condition": label}
                else:
                    model_kwargs_in = {}
                    
                # get noise
                if diffwave:
                    x_in = x_stack.unsqueeze(1)
                    eps = model(x_in, _extract_into_tensor(self.betas, t, t.shape), **model_kwargs_in)
                    eps = eps.transpose(0, 1).squeeze(0) 
                else:
                    eps = model(x_stack, self._scale_timesteps(t_b, True), **model_kwargs_in)
                eps = eps.reshape(1, 1, -1)
                x = x.unsqueeze(1)
                
                # condition
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)
                condition = y - (x_sum) 

                # corrector
                x = corrector.langevin_corrector_sliced(x, t, eps, y, condition)
                
            out["sample"] = x.reshape(1, 1, -1)
            
            yield out
            x = out["sample"]

        return out


    def direct_sample_group(
        self, 
        model,
        shape,
        measurement, 
        n_src, 
        clip_denoised,
        degradation,
        model_kwargs = None,
        device = None):
        """
        for multiple source models
        """

        if device is None:
            device = next(model[0].parameters()).device

        # bar
        pbar = tqdm(list(range(self.num_timesteps))[::-1], ncols=80)
        
        # 1. initial noise
        noise_list = [torch.randn(*shape, device=device) for _ in range(n_src)]
        x = torch.cat(noise_list, dim=-1) 
        x.requires_grad_(False)

        snr = 0.00001
        xi = None

        corrector = CorrectorVPConditional(
            degradation=degradation,
            n_src = n_src,
            snr=snr,
            xi=xi,
            sde=self,
            score_fn=model,
            device=device,
        )

        for i in pbar:
            t = torch.tensor([i] * shape[0], device=device) 
            
            with torch.no_grad(): # unconditional: 
                # single src
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) # [single src,...]
                x_stack = torch.cat(x_solo, dim=0)
                t = torch.tensor([i] * shape[0], device=device) 
                
                # models
                out_samples = []
                out_means = []
                out_pred_xstart = []
                for model_one, model_kwargs_one, x_one in zip(model, model_kwargs, x_solo):
                    if model_kwargs is not None:
                        label = torch.tensor(model_kwargs_one, dtype=torch.long).to(device)
                        model_kwargs_in = {"condition": label}
                    else:
                        model_kwargs_in = None
                            
                    # p sample
                    out = self.p_sample(model_one,x_one,t,model_kwargs_in)

                    out_samples.append(out['sample'].reshape(1, 1, -1))
                    out_means.append(out['mean'].reshape(1, 1, -1))
                    out_pred_xstart.append(out['pred_xstart'].reshape(1, 1, -1))       
                
            out_samples = torch.cat(out_samples, dim=-1)
            out_means = torch.cat(out_means, dim=-1)
            out_pred_xstart = torch.cat(out_pred_xstart, dim=-1)
                    
            y = measurement
            t = torch.tensor([i] * shape[0], device=device)

            # update 1
            coefficient = 0.5 
            total_log_sum = 0
            steps = 2
            for i in range(steps):
                new_samples = []
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)
                
                # log p(y | x)
                log_p_y_x = y - x_sum
                total_log_sum += log_p_y_x
                
                start = 0
                end = y.size(-1)
                while end <= x.size(-1):
                    new_sample = (
                        out_samples[:, :, start:end] + coefficient * total_log_sum
                    )
                    new_samples.append(new_sample)
                    start = end
                    end += y.size(-1)
                x = torch.cat(new_samples, dim=-1)

            # update 2
            threshold = 150
            if t[0] < threshold and t[0] > 0:
                # to batch
                x = x.squeeze(1)
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) # [single src,...]
                x_stack = torch.cat(x_solo, dim=0)
                t_b = torch.tensor([t] * shape[0], device=x_stack.device) 
                
                # model score output
                eps_total = []
                for model_one, model_kwargs_one, x_one in zip(model, model_kwargs, x_solo):
                    if model_kwargs is not None:
                        label = torch.tensor(model_kwargs_one, dtype=torch.long).to(device)
                        model_kwargs_in = {"condition": label}
                    else:
                        model_kwargs_in = None
                            
                    # get noise
                    eps = model_one(x_one, self._scale_timesteps(t_b, True), **model_kwargs_in)
                    eps_total.append(eps)
                
                eps = torch.cat(eps_total, dim=0)
                eps = eps.reshape(1, 1, -1)
                x = x.unsqueeze(1)
                
                # condition
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)
                condition = y - (x_sum) 

                # corrector
                x = corrector.langevin_corrector_sliced(x, t, eps, y, condition)
                
            out["sample"] = x.reshape(1, 1, -1)
            
            yield out
            x = out["sample"]

        return out

    
    def p_sample_unconditional(
        self,         
        model,
        shape,
        device = None,
        model_kwargs = None,
        prior = None):
        if device is None:
            device = next(model.parameters()).device

        # noise
        x = torch.randn(*shape, device=device) # N(0,1)
        if prior is not None:
            x = x * prior[None,None,...]
        x.requires_grad_(False)

        # cut length
        if self.config['model_name'] == "Atten_unet" or self.config['model_name'] == "Atten_unet_2":
            x = adjust_waveform_length_math(
                x, 
                n_fft = self.config['model_cfg']['stft_params']['n_fft'],
                hop_length = self.config['model_cfg']['stft_params']['hop_length'],
                multiple = 8
            )
            x = x.unsqueeze(0)
         
        # bar
        pbar = tqdm(list(range(self.num_timesteps))[::-1], ncols=80)

        for i in pbar: 
            t = torch.tensor([i] * shape[0], device=device) # size [1]
            out = self.p_sample(model,x,t, model_kwargs, prior = prior)

            yield out
            x = out["sample"]

        return out

    def _scale_timesteps(self, t, rescale_timesteps = False):
        if rescale_timesteps:
            t_scaled = (t.float() * (1000.0 / self.num_timesteps)).long()
            return t_scaled
        return t


class CorrectorVPConditional():
    def __init__(self, degradation, n_src, xi, sde, snr, score_fn, device):
        self.degradation = degradation
        self.n_src = n_src
        self.xi = xi
        self.alphas = torch.from_numpy(sde.alphas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) 
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.recip_alphas = torch.from_numpy(1 / sde.sqrt_one_minus_alphas_cumprod).to(
            device
        )
        self.device = device
        self.save_file = []
        
        len_src = 64000
        self.source_m = [torch.zeros(1, 1, len_src, device=self.device) for _ in range(self.n_src)]
        self.source_v = [torch.zeros(1, 1, len_src, device=self.device) for _ in range(self.n_src)]
        self.diffusion_step_adam = 0
        

    def update_fn_recons(self, n_src, x_t, x_t_mean, x_prev, x_0_hat, measurement, **kwargs):
        save_root = kwargs.get('save_path')
        
        t = kwargs.get('time')
        lamda = torch.sqrt(1 - self.alphas[t]).float()
        beta = self.sde.betas[t]
        
        sigmas = self.sde.sqrt_one_minus_alphas_cumprod
        sigmas = torch.tensor(sigmas).float().to(self.device)
        s = torch.flip(sigmas, dims=(-1,))
        sigma = s[t]
        
        # total out
        total_out = kwargs.get('total_out')
        
        # 1. degradation
        num_sampling = 1
        norm = 0
        for _ in range(num_sampling):
            x_0_hat = x_0_hat # noisy input
            A_x = self.degradation(x_0_hat, self.n_src) # degradation
            
            # difference for gradient
            sig_diff = measurement - A_x
            norm_total = (torch.linalg.norm(sig_diff)) # l2

            # group norm
            seg_num =  512 
            norm_sig_group = self.segmented_l2_norm(sig_diff, num_segments=seg_num) * 0.05 
            
            # fft norm
            fft_diff = self.difference_fft(x_0_hat, measurement, self.n_src)
            norm_fft = (torch.linalg.norm(fft_diff)) * 0.1  
            
            # all
            norm = norm_sig_group + norm_total + norm_fft
            
            # only time
            # norm = norm_total 
            
            # time + group
            # norm = norm_total + norm_sig_group
            
            # time + stft
            # norm = norm_total + norm_fft      
            
        # obtain grad update
        grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        posterior_variance = self.sde.posterior_variance
        
        # gradient update
        schedule = kwargs.get('schedule', 'hybrid')
        # DPS
        if schedule == 'DPS':
            grad_chunk = torch.split(grad, grad.size(-1) // self.n_src, dim=-1)
            x_t_chunk = x_t.split(x_t.size(-1) // self.n_src, dim=-1) # x_t srsc
            x_0_hat_chunk = x_0_hat.split(x_0_hat.size(-1) // self.n_src, dim=-1) # x_0_hat src
            updated_x_sources = []
            all_s_i = []
            for i in range(self.n_src):
                grad_i = grad_chunk[i]
                x_t_i = x_t_chunk[i]
                
                # norm_grad_i = torch.linalg.norm(grad_i)  # norm sub grad
                # dir_grad_i = grad_i / (norm_grad_i + epsilon) # norm
                dir_grad_i = grad_i 
                
                # step size
                s_i = 1.5
                
                # update n src
                x_t_i = x_t_i - dir_grad_i * s_i
                
                updated_x_sources.append(x_t_i)
                all_s_i.append(s_i)
            x_t = torch.cat(updated_x_sources, dim=-1)

        # cqt from solve audio diffusion paper
        # from CQT-Diff paper: https://arxiv.org/abs/2210.15228
        elif schedule == 'cqt':
            epsilon = 1e-8
            grad_chunk = torch.split(grad, grad.size(-1) // self.n_src, dim=-1) # grad src
            x_t_chunk = x_t.split(x_t.size(-1) // self.n_src, dim=-1) # x_t srsc
            x_0_hat_chunk = x_0_hat.split(x_0_hat.size(-1) // self.n_src, dim=-1) # x_0_hat src
            updated_x_sources = []
            all_s_i = []
            for i in range(self.n_src):
                grad_i = grad_chunk[i]
                x_t_i = x_t_chunk[i]
                
                # norm sub grad
                norm_grad_i = torch.linalg.norm(grad_i)
                # norm_grad_i = torch.pow(norm_grad_i, 2)
                dir_grad_i = grad_i 

                b, c, h = x_t_i.shape
                xi = 0.001 * torch.sqrt(torch.tensor(c * h))
                sigma = self.sde.sqrt_one_minus_alphas_cumprod[t]
                s = xi / (norm_grad_i * sigma + 1e-6) # ?
                
                # update n src
                x_t_i = x_t_i - grad_i * s
                
                print('s', s)
                print('sigma', sigma)
                
                updated_x_sources.append(x_t_i)
            x_t = torch.cat(updated_x_sources, dim=-1)
        
        # DSG
        elif schedule == 'DSG':
            len_src = x_t.size(-1) // self.n_src
            grad_chunks = torch.split(grad, len_src, dim=-1)
            x_t_chunks = torch.split(x_t, len_src, dim=-1) # x_t srsc
            x_t_mean_chunks = torch.split(x_t_mean, len_src, dim=-1)
            x_0_hat_chunk = x_0_hat.split(x_0_hat.size(-1) // self.n_src, dim=-1) # x_0_hat src
            
            eps = 1e-8 
            updated_x_sources = []
            for i in range(self.n_src):
                grad_i = grad_chunks[i]          
                x_t_i = x_t_chunks[i]            
                x_t_mean_i = x_t_mean_chunks[i]   
                x_0_hat_i = x_0_hat_chunk[i]
                norm_dims = list(range(1, grad_i.ndim)) 

                # snr
                signal_power = float(torch.mean(x_0_hat_i.pow(2)).item())
                noise_tensor = x_t_i - x_0_hat_i
                noise_power = float(torch.mean(noise_tensor.pow(2)).item())
                snr_db = 10 * np.log10(signal_power / (noise_power + 1e-9))
                
                # 1. 
                grad_norm_i = torch.linalg.norm(grad_i, dim=norm_dims, keepdim=True)

                # 2. 
                b, c, h = x_t_i.shape
                r = torch.sqrt(torch.tensor(c * h)) * self.sde.posterior_std_dev[t] # 0.02, * posterior_variance[t]
                d_star_i = -r * grad_i / (grad_norm_i + eps)
                
                # 3. d_sample_i
                # d_sample_i = x_t_i - x_t_mean_i
                # guidance_rate = 0.5
                # mix_direction_i = d_sample_i + guidance_rate * (d_star_i - d_sample_i)
                # mix_direction_i = guidance_rate * d_star_i
                
                # no guidance here
                mix_direction_i = d_star_i 
                
                # 4. 
                mix_direction_norm_i = torch.linalg.norm(mix_direction_i, dim=norm_dims, keepdim=True)
                mix_step_i = (mix_direction_i / (mix_direction_norm_i + eps)) * r
                
                # 5. x_t_i
                x_t_i_updated = x_t_i + mix_step_i
                updated_x_sources.append(x_t_i_updated)  
            x_t = torch.cat(updated_x_sources, dim=-1)  
        
        elif schedule == 'hybrid':
            # hybrid using smooth max
            len_src = x_t.size(-1) // self.n_src
            grad_chunks = torch.split(grad, len_src, dim=-1)
            x_t_chunks = torch.split(x_t, len_src, dim=-1) # x_t srcs
            x_t_mean_chunks = torch.split(x_t_mean, len_src, dim=-1)
            x_0_hat_chunk = x_0_hat.split(x_0_hat.size(-1) // self.n_src, dim=-1) # x_0_hat src
            
            eps = 1e-8 
            updated_x_sources = []
            for i in range(self.n_src):
                grad_i = grad_chunks[i]          
                x_t_i = x_t_chunks[i]            
                x_t_mean_i = x_t_mean_chunks[i]   
                x_0_hat_i = x_0_hat_chunk[i]
                norm_dims = list(range(1, grad_i.ndim))

                # snr
                signal_power = float(torch.mean(x_0_hat_i.pow(2)).item())
                noise_tensor = x_t_i - x_0_hat_i
                noise_power = float(torch.mean(noise_tensor.pow(2)).item())
                snr_db = 10 * np.log10(signal_power / (noise_power + 1e-9))

                # update step
                b, c, h = x_t_i.shape
                r = torch.sqrt(torch.tensor(c * h)) * self.sde.smooth_max_schedule[t]
                
                norm_grad_i = torch.linalg.norm(grad_i)
                dir_grad_i = grad_i / (norm_grad_i + eps)
                x_t_i_updated = x_t_i - dir_grad_i * r

                updated_x_sources.append(x_t_i_updated)                                    
            x_t = torch.cat(updated_x_sources, dim=-1)  

        # saving states
        if_save_infos = False
        if if_save_infos:
            # ------------------- save grad -------------------
            save_dir = os.path.join(save_root, "cond_grads")
            os.makedirs(save_dir, exist_ok=True)

            step_tag = int(t.item())                        
            fname    = f"grad_step_{step_tag:04d}.pt"     
            torch.save(grad.cpu(), os.path.join(save_dir, fname))

            # ------------------- save x_0_hat -----------------
            save_dir_x0 = os.path.join(save_root, "total_x0hat")
            os.makedirs(save_dir_x0, exist_ok=True)
            
            x0_chunks = torch.split(x_0_hat.cpu(), x_0_hat.size(-1) // self.n_src, dim=-1)
            for i, chunk in enumerate(x0_chunks):
                fname = f"x0hat_step_{step_tag:04d}_src{i}.pt"
                torch.save(chunk, os.path.join(save_dir_x0, fname))     
                
            # ------------------- save x_t -----------------
            save_dir_xt = os.path.join(save_root, "total_xt")
            os.makedirs(save_dir_xt, exist_ok=True)
            
            xt_chunks = torch.split(x_t.cpu(), x_t.size(-1) // self.n_src, dim=-1)
            for i, chunk in enumerate(xt_chunks):
                fname = f"xt_step_{step_tag:04d}_src{i}.pt"
                torch.save(chunk, os.path.join(save_dir_xt, fname))         

        return x_t, norm


    def segmented_l2_norm(self, signal, num_segments=4):
        segments = torch.chunk(signal, num_segments, dim=-1)
        seg_norms = [torch.linalg.norm(seg) for seg in segments]
        return sum(seg_norms)


    def difference_fft(self, x, y, n_src, n_fft = 512, hop_length = 256, win_length = 512):
        min_sample_length = x.shape[-1] // n_src
        segments = torch.split(x, min_sample_length, dim=-1) 
        degraded = sum(segments)

        window = torch.hann_window(win_length, device=x.device)

        stft_degraded = torch.stft(degraded.squeeze(1), n_fft=n_fft, hop_length=hop_length, 
                                    win_length=win_length, window=window, return_complex=True)
        stft_y = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=hop_length, 
                            win_length=win_length, window=window, return_complex=True)
        
        spec_degraded = stft_degraded.abs()
        spec_y = stft_y.abs()
        
        diff = (spec_degraded - spec_y).abs()  
        
        return diff


    def update_fn_adaptive(
        self, n_src, x, x_prev, t, y, threshold=150, steps=1, model_kwargs_in=None, source_separation=False 
    ):
        x, condition = self.update_fn(n_src, x, x_prev, t, y, steps, source_separation)

        if t[0] < threshold and t[0] > 0:
            if self.sde.input_sigma_t:
                eps = self.score_fn(
                    x, _extract_into_tensor(self.sde.betas, t, t.shape)
                )
            else:
                model_kwargs = model_kwargs_in
                x = x.squeeze(1)
                
                # to batch
                x_solo = torch.split(x, x.size(-1) // n_src, dim=-1) 
                x_stack = torch.cat(x_solo, dim=0)
                t_b = torch.tensor([t] * x_stack.size(0), device=x_stack.device) 
                eps = self.score_fn(x_stack, self.sde._scale_timesteps(t_b, True), **model_kwargs)
                if isinstance(eps, tuple):
                    eps = eps[0]
                    eps = eps.reshape(1, 1, -1)
                else:
                    eps = eps.reshape(1, 1, -1)
                
                x = x.unsqueeze(1)
                
            if condition is None:
                segments = torch.split(x, y.size(-1), dim=-1) 
                x_sum = sum(segments)

                condition = y - (x_sum) 

            if source_separation:
                x = self.langevin_corrector_sliced(x, t, eps, y, condition)
            else:
                x = self.langevin_corrector(x, t, eps, y, condition)

        return x


    def update_fn(self, n_src, x, x_prev, t, y, steps, source_separation):
        if source_separation:
            coefficient = 0.1 
            total_log_sum = 0
            for i in range(steps):
                new_samples = []
                
                # noise
                # sigma = torch.sqrt(self.alphas[t]).float()
                # x_prev = x_prev + 0.05 * sigma * torch.randn_like(x_prev)

                segments = torch.split(x_prev, y.size(-1), dim=-1) 
                x_sum = sum(segments)

                # log p(y | x)
                log_p_y_x = y - x_sum
                total_log_sum += log_p_y_x

                start = 0
                end = y.size(-1)
                while end <= x_prev.size(-1):
                    new_sample = (
                        x["sample"][:, :, start:end] + coefficient * total_log_sum
                    )
                    new_samples.append(new_sample)
                    start = end
                    end += y.size(-1)
                x_prev = torch.cat(new_samples, dim=-1)
            condition = None

        return x_prev.float(), condition


    def langevin_corrector_sliced(self, x, t, eps, y, condition=None):
        alpha = self.alphas[t]
        corrected_samples = []

        start = 0
        end = y.size(-1)
        while end <= x.size(-1):
            score = self.recip_alphas[t] * eps[:, :, start:end]
            noise = torch.randn_like(x[:, :, start:end], device=x.device)
            grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            score_to_use = score + condition if condition is not None else score 
            x_new = (
                x[:, :, start:end]
                + step_size * score_to_use
                + torch.sqrt(2 * step_size) * noise
            )
            corrected_samples.append(x_new)
            start = end
            end += y.size(-1)

        return torch.cat(corrected_samples, dim=-1).float()


    def langevin_corrector(self, x, t, eps, y, condition=None):
        alpha = self.alphas[t]

        score = self.recip_alphas[t] * eps
        noise = torch.randn_like(x, device=x.device)
        grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha

        score_to_use = score + condition if condition is not None else score
        x_new = x + step_size * score_to_use + torch.sqrt(2 * step_size) * noise

        return x_new.float()
