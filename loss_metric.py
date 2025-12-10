import os
import torch
from itertools import permutations

def _calculate_sisnr(estimate_source, source, EPS =1e-6):
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    
    # e_noise = s' - s_target
    noise = estimate_source - proj
    
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)
    
    return sisnr



class SISNR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, estimate_source, source, EPS =1e-6):
        """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
        Args:
            source: torch tensor, [batch size, sequence length]
            estimate_source: torch tensor, [batch size, sequence length]
        Returns:
            SISNR, [batch size]
        """
        assert source.size() == estimate_source.size()

        # Step 1. Zero-mean norm
        source = source - torch.mean(source, axis = -1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

        # Step 2. SI-SNR
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
        
        proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
        
        # e_noise = s' - s_target
        noise = estimate_source - proj
        
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
        sisnr = 10 * torch.log10(ratio + EPS)
        
        return sisnr
        
        
class PIT_SISNR(torch.nn.Module):
    def __init__(self, eps=1e-6):
        """
        PIT_SISNR 用于计算多源分离问题中最佳排列下的 SI-SNR。
        输入:
            estimate_sources: Tensor, shape [B, n_src, T]，模型输出的分离结果
            sources: Tensor, shape [B, n_src, T]，真实的源信号
        输出:
            best_sisnr: Tensor, [B] 每个样本最佳排列下的平均 SI-SNR
            best_perms: List of tuples, 每个样本对应最佳排列的索引顺序
        """
        super(PIT_SISNR, self).__init__()
        self.eps = eps
        self.sisnr_module = SISNR()

    def forward(self, estimate_sources, sources):
        B, n_src, T = estimate_sources.shape
        # 枚举所有排列
        perms = list(itertools.permutations(range(n_src)))
        sisnr_all = []  # 用于存储每个排列下的平均 SISNR，形状为 [#perms, B]
        
        for perm in perms:
            # 根据排列 perm 重新排列估计的源，得到 [B, n_src, T]
            permuted_estimates = estimate_sources[:, perm, :]
            sisnr_perm = []
            for i in range(n_src):
                est = permuted_estimates[:, i, :]  # [B, T]
                tgt = sources[:, i, :]            # [B, T]
                sisnr_val = self.sisnr_module(est, tgt)  # 计算每个样本的 SISNR, [B]
                sisnr_perm.append(sisnr_val)
            # 将每个源对的 SISNR 叠加为 [B, n_src]，并取平均
            sisnr_perm = torch.stack(sisnr_perm, dim=1)
            avg_sisnr = torch.mean(sisnr_perm, dim=1)
            sisnr_all.append(avg_sisnr)
        
        # 将所有排列的 SISNR 结果堆叠成形状 [#perms, B]
        sisnr_all = torch.stack(sisnr_all, dim=0)
        # 对于每个样本，选择最佳排列（即平均 SISNR 最大的排列）
        best_sisnr, best_perm_idx = torch.max(sisnr_all, dim=0)
        best_perms = [perms[idx] for idx in best_perm_idx.cpu().tolist()]
        
        return best_sisnr
    
    
    
def pit_sisnr(est_sources, ref_sources, epsilon=1e-8):
    """
    Computes Permutation Invariant SI-SNR for a single example (no batch dimension).
    
    Args:
        est_sources (torch.Tensor): Estimated sources (n_src, T).
        ref_sources (torch.Tensor): Reference sources (n_src, T).
    
    Returns:
        tuple:
            - torch.Tensor: The best SI-SNR (scalar).
            - torch.Tensor: The indices of the best permutation (n_src,).
    """
    n_sources, _ = est_sources.shape
    
    # Generate all possible permutations of source indices
    perms = list(permutations(range(n_sources)))
    perms = torch.tensor(perms, device=est_sources.device, dtype=torch.long) # (n_perms, n_src)
    
    # Placeholder for SI-SNR for each permutation
    # Shape: (n_perms,)
    sisnr_perms = torch.zeros(len(perms), device=est_sources.device)
    
    for p_idx, p in enumerate(perms):
        # Permute the estimated sources according to the current permutation
        # p is something like (1, 0) for n_sources=2
        permuted_est = est_sources[p, :]
        
        # Calculate SI-SNR for this permutation
        # The sisnr helper function should handle (n_src, T) input
        # The result sisnr_vals will have shape (n_src,)
        sisnr_vals = sisnr(permuted_est, ref_sources, epsilon)
        
        # Average SI-SNR for this permutation
        sisnr_perms[p_idx] = torch.mean(sisnr_vals)
        
    # Find the permutation with the maximum average SI-SNR
    # max_sisnr is now a scalar tensor
    # best_perm_idx is now a scalar tensor holding the index of the best perm
    max_sisnr, best_perm_idx = torch.max(sisnr_perms, dim=0)
    
    # Get the actual permutation from the index
    # best_perm will have shape (n_src,)
    best_perm = perms[best_perm_idx]
    
    return max_sisnr, best_perm


def sisnr(estimate, reference, epsilon=1e-8):
    """
    Calculates the Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    
    Args:
        estimate (torch.Tensor): Estimated source signal (B, T).
        reference (torch.Tensor): Reference source signal (B, T).
        epsilon (float): A small value to prevent division by zero.
        
    Returns:
        torch.Tensor: The SI-SNR value (B,).
    """
    # Subtract mean to ensure zero-mean signals
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)

    # The scale-invariant part
    # s_target = <s', s>s / ||s||^2
    dot_product = torch.sum(estimate * reference, dim=-1, keepdim=True)
    norm_squared = torch.sum(reference * reference, dim=-1, keepdim=True)
    s_target = (dot_product / (norm_squared + epsilon)) * reference

    # e_noise = s' - s_target
    e_noise = estimate - s_target

    # SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    snr_val = torch.sum(s_target * s_target, dim=-1) / (torch.sum(e_noise * e_noise, dim=-1) + epsilon)
    
    return 10 * torch.log10(snr_val + epsilon)