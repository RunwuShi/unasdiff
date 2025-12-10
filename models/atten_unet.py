import os
import json
import toml
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange
from timm.models.layers import trunc_normal_
use_flash = False


try:
    from flash_attn import flash_attn_func
    use_flash = True
except ImportError as e:
    use_flash = False
    print("no attention:", e)

##############################preprocess##############################
class stftpreprocess(nn.Module):
    def __init__(self,                 
                n_fft: int,
                hop_length: int,
                win_length: int, 
                press,
                **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length        
        self.press = press
        

    def forward(self, x, spec_type = "complex"):
        """
        :param x: [B, wave length]
        :return: [B, F, T] complex
        """
        # std
        mix_std_ = 1
        if x.ndim == 3:
            x = x.squeeze(1)
            
        # stft
        spec_ori = torch.stft(x, 
                            n_fft = self.n_fft, 
                            hop_length = self.hop_length, 
                            win_length = self.win_length, 
                            window = torch.hann_window(self.win_length).pow(0.5).to(x.device), 
                            center=True,
                            return_complex=True)
        
        # compress complex
        if spec_type == "complex":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5)) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex
            elif self.press == "None":
                spec = spec_ori  # [B, F, T], complex
            else:
                spec = torch.pow(spec_ori.abs(), self.press) * torch.exp(1j * spec_ori.angle())  # [B, F, T], complex

        elif spec_type == "amplitude":
            if self.press == "log":
                spec = torch.log(torch.clamp(spec_ori.abs(), min=1e-5))   # [B, F, T], complex
            else:
                spec = torch.pow(spec_ori.abs(), self.press)  # [B, F, T], complex
        
        return spec, mix_std_


class stftpostprocess(nn.Module):
    def __init__(self,                 
                n_fft: int,
                hop_length: int,
                win_length: int, 
                press,
                **kwargs):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.press = press
        win = torch.hann_window(self.win_length).pow(0.5)
        self.register_buffer('window', win, persistent=False)

    def forward(self, x, length, mix_std = 1):
        if self.press == "None":
            spec = x  # [B, F, T], complex
        else:
            # reverse compression
            magnitude = x.abs()  # [B, F, T]
            phase = x.angle() 
            spec = torch.pow(magnitude + 1e-8, 1 / self.press) * torch.exp(1j * phase)  # [B, T, F]

        if spec.real.dtype != torch.float32:
            spec = spec.to(torch.complex64)

        wav = torch.istft(spec, 
                            n_fft = self.n_fft, 
                            hop_length = self.hop_length, 
                            win_length = self.win_length, 
                            window= self.window,
                            center=True,
                            length=length)
        wav = wav * mix_std

        return wav

##############################norm##############################
class LayerNorm1d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-1, -2)
        return x

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class RMSNorm1d(nn.RMSNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        # PyTorch 2.0+ 的 nn.RMSNorm 接口：normalized_shape, eps, elementwise_affine
        super().__init__(normalized_shape=num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)               # -> [B, T, C]
        x = F.rms_norm(x,                     # 函数式调用同样可行
                       normalized_shape=self.normalized_shape,
                       weight=self.weight if self.elementwise_affine else None,
                       eps=self.eps)
        x = x.transpose(-1, -2)               # -> [B, C, T]
        return x


class RMSNorm2d(nn.RMSNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(normalized_shape=num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)             # -> [B, H, W, C]
        x = F.rms_norm(
            x,
            normalized_shape=self.normalized_shape,
            weight=self.weight if self.elementwise_affine else None,
            eps=self.eps
        )
        x = x.permute(0, 3, 1, 2)             # -> [B, C, H, W]
        return x
    
##############################unet##############################
class DownSampler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.down_factor = kwargs.get('down_factor')
        self.layer = nn.Conv1d(
            in_channels=kwargs.get('dim'),
            out_channels=kwargs.get('dim'),
            kernel_size=int(kwargs.get('kernel_size')),
            stride=1,
            padding=(int(kwargs.get('kernel_size'))-1)//2,
            groups=kwargs.get('dim'),
        )

    def forward(self, x):
        x = self.layer(x) + x
        out = rearrange(x, 'b d (t dt) -> b dt t d', dt=self.down_factor)
        return out

class DownSamplerOnedim(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.down_factor = kwargs.get('down_factor')
        
    def forward(self, x):
        out = rearrange(x, 'b c (h dh) t -> b dh c h t', dh=self.down_factor)
        return out


def pad_audio_for_stft(
    x: torch.Tensor, 
    n_fft: int, 
    hop_length: int, 
    align_factor: int = 4,
    center: bool = True
):
    B, L = x.shape
    T = (L // hop_length) + 1
    remainder = T % align_factor 
    pad_samples = 0
    if remainder != 0:
        pad_frames = align_factor - remainder
        pad_samples = pad_frames * hop_length
        x = F.pad(x, (0, pad_samples), mode='constant', value=0) 
         
    return x, pad_samples
    
def precompute_freqs_cis_1d(dim: int, end: int, theta: float = 10000.0):
    positions = torch.arange(end) 
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(positions, freqs)
    cis = torch.polar(torch.ones_like(angles), angles) 
    return cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)] 
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)] 
    return freqs_cis.view(*shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) 
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def modulate1d(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


def modulate2d(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size,bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):    
        embeddings = self.embedding_table(labels)
        return embeddings


    
class FlashAttn(nn.Module):
    def __init__(self, dim, num_heads, bias=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        b, _, len_seq = x.size()
        x = x.transpose(-1,-2)
        qkv = self.to_qkv(x).chunk(3, dim=-1) # 3 * [b t c]

        # qkv
        q = rearrange(qkv[0], 'b t (h d) -> b t h d', h=self.heads)
        k = rearrange(qkv[1], 'b t (h d) -> b t h d', h=self.heads)
        v = rearrange(qkv[2], 'b t (h d) -> b t h d', h=self.heads)

        # rope
        N = len_seq  # len of sequence
        freqs_cis = precompute_freqs_cis_1d(self.dim // self.heads, N).to(x.device) # [b, t (len), dim // heads]
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # flashattention
        if use_flash:
            q = q.to(torch.float16).contiguous()
            k = k.to(torch.float16).contiguous()
            v = v.to(torch.float16).contiguous()
            x = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        else:
            qh = q.permute(0, 2, 1, 3).contiguous()  # [b, h, t, d]
            kh = k.permute(0, 2, 1, 3).contiguous()
            vh = v.permute(0, 2, 1, 3).contiguous()

            attn_logits = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [b,h,t,t]
            attn_probs  = F.softmax(attn_logits, dim=-1)                       # [b,h,t,t]
            out         = torch.matmul(attn_probs, vh)                         # [b,h,t,d]
            x = out.permute(0, 2, 1, 3).contiguous()                           # [b,t,h,d]

        # out
        x = rearrange(x, 'b t h d -> b t (h d)') # [b, channel dim, real len]
        x = self.to_out(x)
        x = x.transpose(-1,-2)
        
        return x
    
    
    
class Mlp1D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Conv1d(in_features, hidden_features, 1, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp2D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp2D_gate(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1, bias=bias[0])
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x_proj = self.fc1(x)                    
        x_val, x_gate = x_proj.chunk(2, dim=1) 
        x = x_val * F.silu(x_gate)    
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
    
# T & F path
class PixelUnshufflePath(nn.Module):
    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.downscale_factor
        B, C, F, T = x.shape
        assert T % r == 0
        x = x.view(B, C, F, T // r, r) # -> [B, C, F, T//r, r]
        x = x.permute(0, 1, 4, 2, 3) # -> [B, C, r, F, T//r]
        x = x.reshape(B, C * r, F, T // r)
        return x

class PixelShufflePath(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C*r, T']
        r = self.upscale_factor
        B, Cr, F, T_prime = x.shape
        assert Cr % r == 0
        C = Cr // r
        x = x.view(B, C, r, F, T_prime)    # [B, C, r, F, T']
        x = x.permute(0, 1, 3, 4, 2) # [B, C, F, T', r]
        x = x.reshape(B, C, F, T_prime * r)  # [B, C, F, T]
        return x


# full fre att
class FullTimeAtt(nn.Module):
    def __init__(self, hidden_size, num_heads, emb_dim, F_divide, sup_dim=8, F_shuffle_num=8, **kwargs):
        super().__init__()
        para = kwargs.get('para')
        orig_F = para['stft_params']['n_fft']//2 + 1
        orig_F = orig_F // F_divide

        # sup, recov conv
        F_sub_dim = orig_F // F_shuffle_num
        F_sub_channel = F_sub_dim * sup_dim
        downscale_factor = F_shuffle_num
        
        self.down_norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.down_unshuffle = PixelUnshufflePath(downscale_factor = downscale_factor)
        self.sup_conv = nn.Conv2d(hidden_size * downscale_factor, sup_dim * 2, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.up_norm = LayerNorm1d(F_sub_channel, affine=False, eps=1e-6)
        self.up_shuffle = PixelShufflePath(upscale_factor = downscale_factor)
        self.recov_conv = nn.Conv2d(sup_dim, hidden_size * downscale_factor, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)

        # att, flash attention
        self.attn = FlashAttn(dim = F_sub_channel, num_heads=num_heads, bias=True, **kwargs) 

        # ffn
        mlp_hidden_dim = 2 * hidden_size
        self.mlp = Mlp2D_gate(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)

        # norms
        self.norm1 = LayerNorm1d(F_sub_channel, affine=False, eps=1e-6)
        self.norm2 = LayerNorm2d(hidden_size, affine=False, eps=1e-6)

        # condition
        self.adaLN_modulation_att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 3 * F_sub_channel, bias=True)
        )
        linear = self.adaLN_modulation_att[-1]              
        nn.init.constant_(linear.weight, 0.0)            
        nn.init.constant_(linear.bias,   0.0)  

        self.adaLN_modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 3 * hidden_size, bias=True)
        )
        linear = self.adaLN_modulation_mlp[-1]              
        nn.init.constant_(linear.weight, 0.0)            
        nn.init.constant_(linear.bias,   0.0)  
        
        
    def forward(self, x, c):      
        B, C, F_dim, T = x.size()
        
        # down sample -> att
        residual = x
        x_att = x
        
        # downscale
        x_att = self.down_norm(x_att)
        x_att = x_att.transpose(-1, -2)
        x_att = self.down_unshuffle(x_att) 
        x_att = x_att.transpose(-1, -2) # [B, C*num_band, F/num_band, T]
        x_val, x_gate = self.sup_conv(x_att).chunk(2, dim=1)
        x_att = x_val * F.silu(x_gate) # [B, C_sup, F/num_band, T]
        # att reshape
        _, C_att, F_att,_ = x_att.size()
        
        # condition att
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation_att(c).chunk(3, dim=1)  
        x_att = rearrange(x_att, 'b c f t -> b (c f) t')
        x_att = modulate1d(self.norm1(x_att), shift_msa, scale_msa)
        
        # att
        x_att = self.attn(x_att)

        # condition att
        x_att = gate_msa.unsqueeze(-1) * x_att
        
        # upscale
        x_att = self.up_norm(x_att)
        x_att = rearrange(x_att, 'b (c f) t -> b c f t', c=C_att, f=F_att)
        # up sample
        x_att = self.recov_conv(x_att)
        x_att = x_att.transpose(-1, -2)
        x_att = self.up_shuffle(x_att)
        x_att = x_att.transpose(-1, -2)
        
        # residual
        x = residual + x_att
        
        # resi, mlp
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_mlp(c).chunk(3, dim=1)        
        
        residual = x
        mlp_in = modulate2d(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(mlp_in)
        # resi
        x_mlp = gate_mlp.unsqueeze(-1).unsqueeze(-1) * x_mlp
        x = residual + x_mlp
        
        return x


class PathAtt(nn.Module):
    def __init__(self, hidden_size, num_heads, downscale, downscale_post, downscale_factor, emb_dim, **kwargs):
        super().__init__()
        para = kwargs.get('para')
        
        # sup, recov conv
        self.downscale = downscale
        self.downscale_post = downscale_post
        if self.downscale:
            self.downscale_factor = downscale_factor
            self.down_up_norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
            self.down_unshuffle = PixelUnshufflePath(downscale_factor = downscale_factor)
            self.sup_conv = nn.Conv2d(hidden_size * downscale_factor, hidden_size * 2, kernel_size=1, stride=1, padding=0, bias=True)
        if self.downscale_post:
            self.up_shuffle = PixelShufflePath(upscale_factor = downscale_factor)
            self.recov_conv = nn.Conv2d(hidden_size, hidden_size * downscale_factor, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)

        # att, flash attention
        self.attn = FlashAttn(dim = hidden_size, num_heads=num_heads, bias=True, **kwargs) 

        # ffn
        mlp_hidden_dim = 2 * hidden_size
        self.mlp = Mlp2D_gate(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)

        # norms
        self.norm1 = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.norm2 = LayerNorm2d(hidden_size, affine=False, eps=1e-6)

        # condition
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 6 * hidden_size, bias=True)
        )
        linear = self.adaLN_modulation[-1]              
        nn.init.constant_(linear.weight, 0.0)            
        nn.init.constant_(linear.bias,   0.0)  
        
        
    def forward(self, x, c):
        # condition
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)        
        B, C, F_dim, T = x.size()
        
        # down sample -> att
        residual = x
        x_att = x
        
        # downscale
        if self.downscale:
            x_att = self.down_up_norm(x_att)
            x_att = self.down_unshuffle(x_att)
            x_val, x_gate = self.sup_conv(x_att).chunk(2, dim=1)
            x_att = x_val * F.silu(x_gate) 
            
        # condition att
        x_att = modulate2d(self.norm1(x_att), shift_msa, scale_msa)
        
        # att
        x_att = rearrange(x_att, 'b c f t -> (b t) c f')
        x_att = self.attn(x_att)
        
        # upscale
        if self.downscale_post:
            T_sup = T // self.downscale_factor 
            x_att = rearrange(x_att, '(b t) c f -> b c f t', b=B, t=T_sup)
            # condition att
            x_att = gate_msa.unsqueeze(-1).unsqueeze(-1) * x_att
            # up sample
            x_att = self.down_up_norm(x_att)
            x_att = self.recov_conv(x_att)
            x_att = self.up_shuffle(x_att)
        else:
            x_att = rearrange(x_att, '(b t) c f -> b c f t', b=B, t=T)
            # condition att
            x_att = gate_msa.unsqueeze(-1).unsqueeze(-1) * x_att
            
        # residual
        x = residual + x_att
        
        # resi, mlp
        residual = x
        mlp_in = modulate2d(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(mlp_in)
        # resi
        x_mlp = gate_mlp.unsqueeze(-1).unsqueeze(-1) * x_mlp
        x = residual + x_mlp
        
        return x
        

# u dit block
class U_DiTBlock(nn.Module):
    def __init__(self, hidden_size, emb_dim, num_heads, F_divide,
                 downscale = True, downscale_post = True,
                 fre_downscale_factor = 2, time_downscale_factor = 2, 
                 fulltimeatt = False, sup_dim = 8, F_shuffle_num=8, **kwargs):
        super().__init__()
        para = kwargs.get('para')
        # 1. full time att
        self.fulltimeatt = fulltimeatt
        if fulltimeatt:
            self.fullatt = FullTimeAtt(hidden_size = hidden_size, num_heads = num_heads, emb_dim = emb_dim, F_divide = F_divide, 
                                    sup_dim = sup_dim, F_shuffle_num = F_shuffle_num, **kwargs)

        # 2. intra fre att
        self.freatt = PathAtt(hidden_size, num_heads = num_heads, downscale = downscale, downscale_post = downscale_post,
                              downscale_factor = fre_downscale_factor, emb_dim = emb_dim, **kwargs)
        
        # 3. intra time att
        self.timeatt = PathAtt(hidden_size, num_heads = num_heads, downscale = downscale, downscale_post = downscale_post,
                               downscale_factor = time_downscale_factor, emb_dim = emb_dim, **kwargs)
        
    def forward(self, x, c):
        # 1. full time att
        if self.fulltimeatt:
            x = self.fullatt(x, c)  
        
        # 2. fre att
        x = self.freatt(x, c)
        
        # 3. time att
        x = x.transpose(-1, -2)   
        x = self.timeatt(x, c)
        x = x.transpose(-1, -2)   

        return x


class PixelUnshuffle1d_T(nn.Module):
    def __init__(self, downscale_factor: int):
        super(PixelUnshuffle1d_T, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.downscale_factor
        B, C, F, T = x.shape
        assert T % r == 0
        x = x.view(B, C, F, T // r, r) # -> [B, C, F, r, T//r]
        x = x.permute(0, 1, 4, 2, 3) # -> [B, C, r, F, T//r]
        x = x.reshape(B, C * r, F, T // r)
        return x


class PixelShuffle1d_T(nn.Module):
    def __init__(self, upscale_factor: int):
        super(PixelShuffle1d_T, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C*r, T'], 其中 T' = T // r
        r = self.upscale_factor
        B, Cr, F, T_prime = x.shape
        assert Cr % r == 0
        C = Cr // r
        x = x.view(B, C, r, F, T_prime)    # [B, C, r, F, T']
        x = x.permute(0, 1, 3, 4, 2) # [B, C, F, T', r]
        x = x.reshape(B, C, F, T_prime * r)  # [B, C, F, T]
        return x


class Downsample_T(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_T, self).__init__()
        self.body = nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = PixelUnshuffle1d_T(downscale_factor = 2)

    def forward(self, x):
        x = self.body(x)
        x = self.shuffle(x)
        return x


class Upsample_T(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_T, self).__init__()
        self.body = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = PixelShuffle1d_T(upscale_factor = 2)
        
    def forward(self, x):
        x = self.body(x)
        x = self.shuffle(x)
        return x
    
    
class Downsample_2D(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.body(x)
        x = self.shuffle(x)
        return x
    
    
class Upsample_2D(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        x = self.body(x)
        x = self.shuffle(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, emb_dim):
        super().__init__()
        self.norm_final = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # condition
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * hidden_size, bias=True)
        )
        linear = self.adaLN_modulation[-1]              
        nn.init.constant_(linear.weight, 0.0)            
        nn.init.constant_(linear.bias,   0.0)  
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate2d(self.norm_final(x), shift, scale)
        x = self.out_proj(x)
        return x


# model
class Atten_unet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # stft
        self.preprocess = stftpreprocess(n_fft = config['stft_params']['n_fft'], 
                                         hop_length = config['stft_params']['hop_length'], 
                                         win_length = config['stft_params']['win_length'], 
                                         press = 'None')

        # unet
        depth = [2,4,8,4,2] 
        hidden_size = config['model_params']['hidden_size'] 
        emb_dim = config['model_params']['emb_dim'] 
        num_heads = config['model_params']['num_heads'] 
        emb_num = config['model_params']['num_class'] 
        F = config['stft_params']['n_fft']//2 + 1
        
        # initial conv
        self.conv_1 = nn.Conv2d(2, hidden_size,kernel_size = 3,stride = 1,padding = 1, bias = False)
        
        # encoder 1        
        self.t_embedder_1 = TimestepEmbedder(emb_dim)
        self.y_embedder_1 = LabelEmbedder(emb_num, emb_dim, dropout_prob=0.1) 
        self.encoder_level_1 = nn.ModuleList([
            U_DiTBlock(hidden_size, emb_dim, num_heads, F_divide=1, 
                       downscale = True, downscale_post = False,
                       fre_downscale_factor = 1, time_downscale_factor = 1, 
                       sup_dim = 12, F_shuffle_num=4,
                       para = config, **kwargs) for _ in range(depth[0])])
        self.down1_2 = Downsample_2D(hidden_size) 

        # encoder 2
        self.t_embedder_2 = TimestepEmbedder(emb_dim)
        self.y_embedder_2 = LabelEmbedder(emb_num, emb_dim, dropout_prob=0.1)
        self.encoder_level_2 = nn.ModuleList([
            U_DiTBlock(hidden_size * 2, emb_dim, num_heads, F_divide=2, 
                       downscale = True, downscale_post = False,
                       fre_downscale_factor = 1, time_downscale_factor = 1,
                       sup_dim = 12, F_shuffle_num=4,
                       para = config, **kwargs) for _ in range(depth[1])])
        self.down2_3 = Downsample_2D(hidden_size * 2) 

        # latent
        self.t_embedder_3 = TimestepEmbedder(emb_dim)
        self.y_embedder_3 = LabelEmbedder(emb_num, emb_dim, dropout_prob=0.1)
        self.latent = nn.ModuleList([
            U_DiTBlock(hidden_size * 4, emb_dim, num_heads, F_divide=4,
                       downscale = True, downscale_post = False,
                       fre_downscale_factor = 1, time_downscale_factor = 1, 
                       fulltimeatt = True, sup_dim = 16, F_shuffle_num=4, 
                       para = config, **kwargs) for _ in range(depth[2])])

        # decoder 2
        self.t_embedder_4 = TimestepEmbedder(emb_dim)
        self.y_embedder_4 = LabelEmbedder(emb_num, emb_dim, dropout_prob=0.1)
        self.up3_2 = Upsample_2D(hidden_size * 4)
        self.reduce_chan_level2 = nn.Conv2d(int(hidden_size*4), int(hidden_size*2), kernel_size=1, bias=True)
        self.decoder_level_2 = nn.ModuleList([
            U_DiTBlock(hidden_size*2, emb_dim, num_heads, F_divide=2, 
                       downscale = True, downscale_post = False,
                       fre_downscale_factor = 1, time_downscale_factor = 1,
                       sup_dim = 12, F_shuffle_num=4,
                       para = config, **kwargs) for _ in range(depth[3])])

        # decoder 1
        self.t_embedder_5 = TimestepEmbedder(emb_dim)
        self.y_embedder_5 = LabelEmbedder(emb_num, emb_dim, dropout_prob=0.1)
        self.up2_1 = Upsample_2D(hidden_size * 2)
        self.reduce_chan_level1 = nn.Conv2d(int(hidden_size * 2), hidden_size, kernel_size=1, bias=True)
        self.decoder_level_1 = nn.ModuleList([
            U_DiTBlock(hidden_size, emb_dim, num_heads, F_divide=1, 
                       downscale = True, downscale_post = False,
                       fre_downscale_factor = 1, time_downscale_factor = 1, 
                       sup_dim = 12, F_shuffle_num=4,
                       para = config, **kwargs) for _ in range(depth[4])])
        
        # postprocess + final layer
        self.t_embedder_out = TimestepEmbedder(emb_dim)
        self.out_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.final_layer = FinalLayer(hidden_size, 2, emb_dim)
        
        # istft
        self.istft = stftpostprocess(n_fft = config['stft_params']['n_fft'], 
                                         hop_length = config['stft_params']['hop_length'], 
                                         win_length = config['stft_params']['win_length'], 
                                         press = 'None')
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
            #     nn.init.ones_(module.weight)
            #     nn.init.zeros_(module.bias)
                    
        self.apply(_basic_init)
            
            
    @autocast()  # amp
    def forward(self, x, t, **kwargs):
        if kwargs.get('cfg') is True:
            out = self.forward_with_cfg(x, t, **kwargs)
        else:   
            out = self.forward_function(x, t, **kwargs)
        
        # cfg
        # out = self.forward_with_cfg(x, t, **kwargs)
        
        # normal
        # out = self.forward_function(x, t, **kwargs)
        
        return out
    
    
    @autocast()  
    def forward_with_cfg(self, x, t, **kwargs):        
        # 1. unconditional
        batch_size = x.shape[0]
        unconditional_class_index = self.config['model_params']['num_class']
        y_unconditional_value_batched = torch.full(
            (batch_size,),
            unconditional_class_index,
            dtype=torch.long,  
            device=x.device)
        y_unconditional_for_forward = {'condition': y_unconditional_value_batched}
        uncond_model_out = self.forward_function(x, t, **y_unconditional_for_forward)
        
        # 2. with label
        cond_model_out = self.forward_function(x, t, **kwargs)
        
        # =1: condition
        # 0: nocondition
        cfg_scale = 5
        guided_waveform = uncond_model_out + cfg_scale * (cond_model_out - uncond_model_out)
        
        return guided_waveform

   
    @autocast()
    def forward_function(self, x, t, **kwargs):
        # condition
        if not kwargs or kwargs.get('condition') is None:
            y = torch.tensor([0]).to(x.device)
        else:
            y = kwargs.get('condition')
            
        # pad signal
        x, pad_len = pad_audio_for_stft(x, 
                                        self.config['stft_params']['n_fft'], 
                                        self.config['stft_params']['hop_length'])
        
        # spec
        x_in = x
        spec, mix_std = self.preprocess(x, "complex") # [B, F, T], complex
        spec_real_imag = torch.view_as_real(spec) 
        spec_mag = torch.abs(spec) # [256, 252]

        # conv
        spec_in = torch.stack([spec_real_imag[...,0], spec_real_imag[...,1]], dim = 1)
        x = self.conv_1(spec_in)
        
        # encoder 1
        t1 = self.t_embedder_1(t)    # [b, c dim]
        y1 = self.y_embedder_1(y, self.training)
        c1 = t1 + y1  
        out_enc_level1 = x
        for block in self.encoder_level_1:
            out_enc_level1 = block(out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # encoder 2
        t2 = self.t_embedder_2(t)    # [b, c dim]
        y2 = self.y_embedder_2(y, self.training)
        c2 = t2 + y2
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = block(out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # latent
        t3 = self.t_embedder_3(t)    # [b, c dim]
        y3 = self.y_embedder_3(y, self.training)
        c3 = t3 + y3
        latent = inp_enc_level3
        for block in self.latent:
            latent = block(latent, c3)

        # decoder 2
        t4 = self.t_embedder_4(t)    # [b, c dim]
        y4 = self.y_embedder_4(y, self.training)
        c4 = t4 + y4
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        for block in self.decoder_level_2:
            out_dec_level2 = block(out_dec_level2, c4)

        # decoder 1
        t5 = self.t_embedder_5(t)    # [b, c dim]
        y5 = self.y_embedder_5(y, self.training)
        c5 = t5 + y5
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = inp_dec_level1
        for block in self.decoder_level_1:
            out_dec_level1 = block(out_dec_level1, c5)

        # out conv
        c_out = self.t_embedder_out(t)
        x = self.out_conv(out_dec_level1)
        x = self.final_layer(x, c_out)
        
        # to wave
        x_real = x[:,0,:,:]
        x_imag = x[:,1,:,:]
        x_cplx = torch.complex(x_real.float(), x_imag.float()) # [b, F, T]
        
        x_cplx_s = x_cplx.squeeze(1)
        x_wave = self.istft(x_cplx_s, x_in.size(-1), 1)  # [B, T]
        x_wave = x_wave[:,:-pad_len]
        
        return x_wave
            
        
if __name__ == '__main__':
    config_path = os.path.join('./config/atten_unet_fsd/config.toml')  

    with open(config_path, 'r', encoding='utf-8') as f:
        ext = os.path.splitext(config_path)[1].lower()
        if ext in ('.json',):
            config = json.load(f)
        elif ext in ('.toml'):
            config = toml.load(f)
            
    # device, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Atten_unet(config["model_cfg"]).to(device)

    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}, numel={param.numel()}")

    # embed layer para num
    total = 0
    for name, param in model.named_parameters():
        if "timeatt.adaLN_modulation" in name:
            num = param.numel()
            total += num
    print(f"Total timeatt.adaLN_modulation params: {total}")

    paras =  sum(p.numel() for p in model.parameters())
    print('paras num', paras)
    
    # input
    batch = 1
    input_data = torch.randn(batch, 64000).to(device)

    # mac
    from thop import profile

    batch = batch
    device = next(model.parameters()).device
    dummy_input = torch.randn(batch, input_data.size(-1)).to(device)
    diffusion_step = torch.full((batch,), 0.02, dtype=torch.float32, device=device)

    # forward pass
    model_kwargs = {"condition":torch.tensor([0],device=device)}
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(input_data, diffusion_step, **model_kwargs)
    
    import torch.profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        output = model(input_data, diffusion_step, **model_kwargs)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
    
    print(output.shape)
