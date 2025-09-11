"""Fitness metrics for evaluating fuzzy inference system denoising performance."""
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def fitness_ssim(denoised_img, ground_truth):
    """
    SSIM (Structural Similarity Index) fitness metric.
    
    Measures structural similarity between denoised and ground truth images.
    Range: [0, 1] where 1 = perfect similarity
    
    Good for: Preserving image structure and perceptual quality
    """
    return ssim(denoised_img, ground_truth)

def fitness_psnr(denoised_img, ground_truth):
    """
    PSNR (Peak Signal-to-Noise Ratio) fitness metric.
    
    Measures pixel-wise accuracy of denoising.
    Range: [0, ∞] in dB, higher = better (typically 20-50 dB)
    
    Good for: Measuring noise reduction effectiveness
    """
    return psnr(ground_truth, denoised_img)

def fitness_combined(denoised_img, ground_truth, w_ssim=0.7, w_psnr=0.3):
    """
    Combined SSIM + PSNR fitness metric.
    
    Balances structural preservation (SSIM) with noise reduction (PSNR).
    
    Args:
        denoised_img: Output from fuzzy inference system
        ground_truth: Original clean image
        w_ssim: Weight for SSIM component (default 0.7)
        w_psnr: Weight for PSNR component (default 0.3)
    
    Returns:
        Combined fitness score [0, 1] where higher = better
    """
    ssim_score = fitness_ssim(denoised_img, ground_truth)
    psnr_score = fitness_psnr(denoised_img, ground_truth) / 50.0  # Normalize PSNR to [0,1]
    return w_ssim * ssim_score + w_psnr * psnr_score

def get_fitness_function(metric_type):
    """
    Get the appropriate fitness function for GA optimization.
    
    Args:
        metric_type: 'ssim', 'psnr', or 'combined'
    
    Returns:
        Fitness function that the GA will use to evaluate fuzzy parameter quality
    """
    if metric_type == 'ssim':
        return fitness_ssim
    elif metric_type == 'psnr':
        return fitness_psnr
    elif metric_type == 'combined':
        return fitness_combined
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")