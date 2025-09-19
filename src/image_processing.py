"""Fuzzy-based image denoising functions optimized by genetic algorithms."""
import cv2
import numpy as np

def fuzzy_median_filter(img, kernel_size, alpha):
    """
    Noise-aware fuzzy inference system for adaptive image denoising.
    
    This implements an intelligent fuzzy approach that:
    - Automatically detects noise type (Gaussian vs Salt & Pepper)
    - Adapts denoising strategy based on noise characteristics
    - Uses appropriate filters for each noise type
    
    Args:
        img: Input noisy image
        kernel_size: Median filter kernel size (3, 5, or 7)
        alpha: Fuzzy inference parameter (0.2-0.8) controlling denoising strategy
    
    Returns:
        Denoised image using noise-aware fuzzy inference system
    """
    # Ensure kernel size is valid for image size
    if img.shape[0] < kernel_size or img.shape[1] < kernel_size:
        kernel_size = 3
    
    try:
        # Step 0: Conservative noise type detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Calculate noise characteristics
        local_mean = cv2.GaussianBlur(gray_img.astype(np.float32), (5, 5), 1.0)
        local_variance = cv2.GaussianBlur((gray_img.astype(np.float32) - local_mean) ** 2, (5, 5), 1.0)
        avg_variance = np.mean(local_variance)
        
        # More conservative salt & pepper detection
        total_pixels = gray_img.size
        very_dark = np.sum(gray_img < 20) / total_pixels
        very_bright = np.sum(gray_img > 235) / total_pixels
        impulse_ratio = very_dark + very_bright
        
        # Conservative threshold - only classify as SP if very obvious
        is_impulse_noise = impulse_ratio > 0.12
        
        if is_impulse_noise:
            # SIMPLIFIED RULE SET A: Salt & Pepper Noise Handling
            # Keep it simple - median filtering works best for impulse noise
            
            if alpha <= 0.6:
                # Conservative median filtering
                result = cv2.medianBlur(img, kernel_size)
            else:
                # High alpha: More aggressive median filtering
                result = cv2.medianBlur(img, kernel_size)
                
                # Only add morphological cleanup if kernel is large enough
                if kernel_size >= 5:
                    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_morph)
            
            # NO enhancement for impulse noise - it amplifies remaining artifacts
            
        else:
            # SIMPLIFIED RULE SET B: Gaussian Noise Handling  
            # Focus on proven approaches
            
            if alpha <= 0.4:
                # Low alpha: Edge-preserving bilateral filter
                result = cv2.bilateralFilter(img, kernel_size, 
                                           40 + alpha * 30,
                                           40 + alpha * 30)
            elif alpha <= 0.7:
                # Medium alpha: Balanced bilateral + median approach
                bilateral = cv2.bilateralFilter(img, kernel_size, 
                                              50 + (alpha - 0.4) * 25,
                                              50 + (alpha - 0.4) * 25)
                median = cv2.medianBlur(img, kernel_size)
                
                # Simple blending based on alpha
                median_weight = (alpha - 0.4) / 0.3 * 0.4  # Max 40% median
                result = cv2.addWeighted(bilateral, 1 - median_weight, 
                                       median, median_weight, 0)
            else:
                # High alpha: Stronger denoising
                result = cv2.medianBlur(img, kernel_size)
                
                # Light bilateral smoothing
                result = cv2.bilateralFilter(result, 3, 30, 30)
            
            # Conservative enhancement only for Gaussian noise
            if alpha > 0.5 and avg_variance > 100:
                gaussian = cv2.GaussianBlur(result, (0, 0), 1.0)
                enhancement_strength = 1.0 + (alpha - 0.5) * 0.15  # Very conservative: 1.0 to 1.075
                enhanced = cv2.addWeighted(result, enhancement_strength, 
                                         gaussian, -(enhancement_strength - 1.0), 0)
                # Very light blending
                result = cv2.addWeighted(result, 0.85, enhanced, 0.15, 0)
        
        # Ensure output is in valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f'Warning: Fuzzy denoising failed for kernel_size={kernel_size}, error: {e}')
        # Improved fallback strategy
        try:
            # Try bilateral filter first
            result = cv2.bilateralFilter(img, kernel_size, 50, 50)
        except:
            # Ultimate fallback to simple median
            try:
                result = cv2.medianBlur(img, kernel_size)
            except:
                result = img.copy()
    
    return result

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