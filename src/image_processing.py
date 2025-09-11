"""Fuzzy-based image denoising functions optimized by genetic algorithms."""
import cv2
import numpy as np

def fuzzy_median_filter(img, kernel_size, alpha):
    """
    Improved fuzzy inference system for adaptive image denoising.
    
    This implements an optimized fuzzy-based denoising approach where:
    - kernel_size: Controls the neighborhood size for filtering
    - alpha: Controls the balance between noise reduction and detail preservation
    
    The fuzzy system adapts denoising strength based on local image properties.
    
    Args:
        img: Input noisy image
        kernel_size: Median filter kernel size (3, 5, or 7)
        alpha: Fuzzy inference parameter (0.2-0.7) controlling denoising strength
    
    Returns:
        Denoised image using optimized fuzzy inference system
    """
    # Ensure kernel size is valid for image size
    if img.shape[0] < kernel_size or img.shape[1] < kernel_size:
        kernel_size = 3
    
    try:
        # Stage 1: Improved fuzzy-based noise reduction
        # The fuzzy system determines denoising strategy based on alpha parameter
        
        if alpha <= 0.3:
            # Low alpha: Preserve details, light denoising
            # Fuzzy rule: IF noise_level is LOW THEN use edge_preserving_filter
            denoised = cv2.bilateralFilter(img, kernel_size, 40 + alpha * 30, 40 + alpha * 30)
            
        elif alpha <= 0.5:
            # Medium alpha: Balanced approach with improved blending
            # Fuzzy rule: IF noise_level is MEDIUM THEN combine bilateral AND median optimally
            bilateral = cv2.bilateralFilter(img, kernel_size, 50 + alpha * 20, 50 + alpha * 20)
            median = cv2.medianBlur(img, kernel_size)
            
            # Improved fuzzy membership function for blending
            blend_weight = (alpha - 0.3) / 0.2  # Maps 0.3-0.5 to 0-1
            # Non-linear blending for better results
            blend_weight = blend_weight ** 0.8  # Slightly favor bilateral filtering
            
            denoised = cv2.addWeighted(bilateral, 1 - blend_weight, median, blend_weight, 0)
            
        else:
            # High alpha: Strong denoising but preserve important edges
            # Fuzzy rule: IF noise_level is HIGH THEN use selective_aggressive_filtering
            # Use morphological opening to preserve important structures
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            
            # Combine median with morphological result
            median = cv2.medianBlur(img, kernel_size)
            denoised = cv2.addWeighted(median, 0.7, opened, 0.3, 0)
            
            # Additional smoothing for very high alpha values
            if alpha > 0.6:
                denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5 + (alpha - 0.6) * 0.5)
        
        # Stage 2: Improved fuzzy-based detail enhancement
        # Create multiple scales for fuzzy detail recovery
        gaussian_fine = cv2.GaussianBlur(denoised, (0, 0), 0.8 + alpha * 0.4)
        gaussian_coarse = cv2.GaussianBlur(denoised, (0, 0), 1.2 + alpha * 0.6)
        
        # Optimized fuzzy membership functions for enhancement strength
        # Less aggressive enhancement to avoid over-sharpening
        unsharp_strength_fine = 1.3 + alpha * 0.5    # Reduced from 1.8 + alpha * 1.2
        unsharp_strength_coarse = 1.1 + alpha * 0.3  # Reduced from 1.2 + alpha * 0.8
        
        # Apply fuzzy-controlled unsharp masking
        fine_details = cv2.addWeighted(denoised, unsharp_strength_fine, 
                                     gaussian_fine, -(unsharp_strength_fine - 1.0), 0)
        coarse_details = cv2.addWeighted(denoised, unsharp_strength_coarse, 
                                       gaussian_coarse, -(unsharp_strength_coarse - 1.0), 0)
        
        # Fuzzy combination of detail enhancements with better balance
        detail_enhanced = cv2.addWeighted(fine_details, 0.6, coarse_details, 0.4, 0)
        
        # Stage 3: Improved fuzzy adaptive blending based on local variance
        # Calculate local variance for fuzzy texture classification
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Local variance computation with better kernel
        local_mean = cv2.GaussianBlur(gray_img.astype(np.float32), (7, 7), 1.0)
        local_variance = cv2.GaussianBlur((gray_img.astype(np.float32) - local_mean) ** 2, (7, 7), 1.0)
        
        # Fuzzy membership function for texture classification
        variance_norm = cv2.normalize(local_variance, None, 0, 1, cv2.NORM_MINMAX)
        
        # Extend variance map to match image dimensions
        if len(img.shape) == 3:
            variance_weights = np.stack([variance_norm] * 3, axis=2)
        else:
            variance_weights = variance_norm
        
        # Improved fuzzy rules for adaptive blending:
        # More conservative blending weights
        original_weight = 0.15 + variance_weights * 0.25     # 0.15 to 0.4
        denoised_weight = 0.55 - variance_weights * 0.2      # 0.55 to 0.35
        enhanced_weight = 0.3 + alpha * 0.15                 # 0.3 to 0.45
        
        # Normalize fuzzy weights
        total_weight = original_weight + denoised_weight + enhanced_weight
        original_weight /= total_weight
        denoised_weight /= total_weight
        enhanced_weight /= total_weight
        
        # Final fuzzy-weighted combination
        result = (img.astype(np.float32) * original_weight + 
                 denoised.astype(np.float32) * denoised_weight + 
                 detail_enhanced.astype(np.float32) * enhanced_weight)
        
        # Ensure output is in valid range
        enhanced = np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f'Warning: Fuzzy denoising failed for kernel_size={kernel_size}, error: {e}')
        # Improved fallback strategy
        try:
            # Try bilateral filter first
            enhanced = cv2.bilateralFilter(img, kernel_size, 50, 50)
        except:
            # Ultimate fallback to simple median
            try:
                enhanced = cv2.medianBlur(img, kernel_size)
            except:
                enhanced = img.copy()
    
    return enhanced

def simple_denoising_filter(img, kernel_size, alpha):
    """
    Simple denoising filter for comparison with fuzzy approach.
    
    Args:
        img: Input noisy image
        kernel_size: Median filter kernel size
        alpha: Sharpening strength parameter
    
    Returns:
        Enhanced image using simple median + sharpening approach
    """
    # Ensure kernel size is valid for image size
    if img.shape[0] < kernel_size or img.shape[1] < kernel_size:
        kernel_size = 3
    
    try:
        # Simple median filtering
        median = cv2.medianBlur(img, kernel_size)
    except Exception as e:
        print(f'Warning: medianBlur failed for kernel_size={kernel_size}, error: {e}')
        median = img.copy()
    
    # Simple unsharp masking
    sharpened = cv2.addWeighted(img, 1.5, median, -0.5, 0)
    enhanced = cv2.addWeighted(median, 1-alpha, sharpened, alpha, 0)
    
    return enhanced