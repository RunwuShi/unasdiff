import os
import torch
from itertools import permutations

def _calculate_sisnr(estimate_source, source, EPS =1e-6):
    assert source.size() == estimate_source.size()
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    noise = estimate_source - proj
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)
    
    return sisnr



class SISNR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, estimate_source, source, EPS =1e-6):
        assert source.size() == estimate_source.size()
        source = source - torch.mean(source, axis = -1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
        ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
        proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
        noise = estimate_source - proj
        ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
        sisnr = 10 * torch.log10(ratio + EPS)
        
        return sisnr
        
        
class PIT_SISNR(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(PIT_SISNR, self).__init__()
        self.eps = eps
        self.sisnr_module = SISNR()

    def forward(self, estimate_sources, sources):
        B, n_src, T = estimate_sources.shape
        perms = list(itertools.permutations(range(n_src)))
        sisnr_all = []  
        
        for perm in perms:
            permuted_estimates = estimate_sources[:, perm, :]
            sisnr_perm = []
            for i in range(n_src):
                est = permuted_estimates[:, i, :]  # [B, T]
                tgt = sources[:, i, :]            # [B, T]
                sisnr_val = self.sisnr_module(est, tgt)  
                sisnr_perm.append(sisnr_val)
            sisnr_perm = torch.stack(sisnr_perm, dim=1)
            avg_sisnr = torch.mean(sisnr_perm, dim=1)
            sisnr_all.append(avg_sisnr)
        
        sisnr_all = torch.stack(sisnr_all, dim=0)
        best_sisnr, best_perm_idx = torch.max(sisnr_all, dim=0)
        best_perms = [perms[idx] for idx in best_perm_idx.cpu().tolist()]
        
        return best_sisnr
    
    
    
def pit_sisnr(est_sources, ref_sources, epsilon=1e-8):
    n_sources, _ = est_sources.shape
    perms = list(permutations(range(n_sources)))
    perms = torch.tensor(perms, device=est_sources.device, dtype=torch.long) # (n_perms, n_src)
    sisnr_perms = torch.zeros(len(perms), device=est_sources.device)
    
    for p_idx, p in enumerate(perms):
        permuted_est = est_sources[p, :]
        sisnr_vals = sisnr(permuted_est, ref_sources, epsilon)
        sisnr_perms[p_idx] = torch.mean(sisnr_vals)
        
    max_sisnr, best_perm_idx = torch.max(sisnr_perms, dim=0)
    best_perm = perms[best_perm_idx]
    
    return max_sisnr, best_perm


def sisnr(estimate, reference, epsilon=1e-8):
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)

    dot_product = torch.sum(estimate * reference, dim=-1, keepdim=True)
    norm_squared = torch.sum(reference * reference, dim=-1, keepdim=True)
    s_target = (dot_product / (norm_squared + epsilon)) * reference

    e_noise = estimate - s_target

    snr_val = torch.sum(s_target * s_target, dim=-1) / (torch.sum(e_noise * e_noise, dim=-1) + epsilon)
    
    return 10 * torch.log10(snr_val + epsilon)