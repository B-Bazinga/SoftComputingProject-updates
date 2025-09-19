"""Analysis tools for comparing fuzzy vs simple denoising performance."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from metrics import calculate_ssim, calculate_psnr, calculate_combined_fitness


def analyze_failure_regions(original, noisy, fuzzy_result, simple_result, image_name="test"):
    """
    Analyze where fuzzy system fails compared to simple system.
    
    Returns detailed analysis of performance differences at pixel level.
    """
    # Calculate error maps
    fuzzy_error = np.abs(original.astype(np.float32) - fuzzy_result.astype(np.float32))
    simple_error = np.abs(original.astype(np.float32) - simple_result.astype(np.float32))
    
    # Regions where fuzzy performs worse
    fuzzy_worse_mask = fuzzy_error > simple_error
    fuzzy_worse_regions = np.sum(fuzzy_worse_mask)
    total_pixels = original.size
    
    # Calculate metrics
    fuzzy_ssim = calculate_ssim(original, fuzzy_result)
    fuzzy_psnr = calculate_psnr(original, fuzzy_result)
    fuzzy_fitness = calculate_combined_fitness(original, fuzzy_result)
    
    simple_ssim = calculate_ssim(original, simple_result)
    simple_psnr = calculate_psnr(original, simple_result)
    simple_fitness = calculate_combined_fitness(original, simple_result)
    
    # Local variance analysis
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    local_mean = cv2.GaussianBlur(gray_original.astype(np.float32), (5, 5), 1.0)
    local_variance = cv2.GaussianBlur((gray_original.astype(np.float32) - local_mean) ** 2, (5, 5), 1.0)
    
    # Analyze failure patterns
    analysis = {
        'image_name': image_name,
        'metrics': {
            'fuzzy': {'ssim': fuzzy_ssim, 'psnr': fuzzy_psnr, 'fitness': fuzzy_fitness},
            'simple': {'ssim': simple_ssim, 'psnr': simple_psnr, 'fitness': simple_fitness}
        },
        'differences': {
            'ssim': fuzzy_ssim - simple_ssim,
            'psnr': fuzzy_psnr - simple_psnr,
            'fitness': fuzzy_fitness - simple_fitness
        },
        'failure_analysis': {
            'fuzzy_worse_pixels': fuzzy_worse_regions,
            'fuzzy_worse_percentage': (fuzzy_worse_regions / total_pixels) * 100,
            'avg_fuzzy_error': np.mean(fuzzy_error),
            'avg_simple_error': np.mean(simple_error),
            'avg_noise_level': np.mean(local_variance),
            'edge_regions_worse': analyze_edge_performance(original, fuzzy_worse_mask),
            'smooth_regions_worse': analyze_smooth_performance(local_variance, fuzzy_worse_mask)
        }
    }
    
    return analysis


def analyze_edge_performance(original, fuzzy_worse_mask):
    """Analyze fuzzy performance in edge regions."""
    # Detect edges using Canny
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    
    if edge_pixels == 0:
        return 0
    
    # Count fuzzy failures in edge regions
    edge_failures = np.sum(fuzzy_worse_mask & (edges > 0))
    return (edge_failures / edge_pixels) * 100 if edge_pixels > 0 else 0


def analyze_smooth_performance(variance_map, fuzzy_worse_mask):
    """Analyze fuzzy performance in smooth regions."""
    # Define smooth regions as low variance areas
    smooth_threshold = np.percentile(variance_map, 25)  # Bottom 25% variance
    smooth_regions = variance_map < smooth_threshold
    smooth_pixels = np.sum(smooth_regions)
    
    if smooth_pixels == 0:
        return 0
    
    # Count fuzzy failures in smooth regions
    smooth_failures = np.sum(fuzzy_worse_mask & smooth_regions)
    return (smooth_failures / smooth_pixels) * 100 if smooth_pixels > 0 else 0


def create_comparative_visualization(original, noisy, fuzzy_result, simple_result, 
                                   save_path=None, title="Denoising Comparison"):
    """Create visualization comparing fuzzy vs simple results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Original and noisy
    axes[0, 0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray' if len(noisy.shape) == 2 else None)
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')
    
    # Results
    axes[0, 2].imshow(fuzzy_result, cmap='gray' if len(fuzzy_result.shape) == 2 else None)
    axes[0, 2].set_title('Fuzzy Enhanced')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(simple_result, cmap='gray' if len(simple_result.shape) == 2 else None)
    axes[1, 0].set_title('Simple Enhanced')
    axes[1, 0].axis('off')
    
    # Error maps
    fuzzy_error = np.abs(original.astype(np.float32) - fuzzy_result.astype(np.float32))
    simple_error = np.abs(original.astype(np.float32) - simple_result.astype(np.float32))
    
    if len(fuzzy_error.shape) == 3:
        fuzzy_error = np.mean(fuzzy_error, axis=2)
        simple_error = np.mean(simple_error, axis=2)
    
    im1 = axes[1, 1].imshow(fuzzy_error, cmap='hot', vmin=0, vmax=np.max([fuzzy_error, simple_error]))
    axes[1, 1].set_title('Fuzzy Error Map')
    axes[1, 1].axis('off')
    
    im2 = axes[1, 2].imshow(simple_error, cmap='hot', vmin=0, vmax=np.max([fuzzy_error, simple_error]))
    axes[1, 2].set_title('Simple Error Map')
    axes[1, 2].axis('off')
    
    # Add colorbar
    plt.colorbar(im2, ax=axes[1, :], shrink=0.8, aspect=30)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_improvement_suggestions(analysis_results):
    """
    Generate specific improvement suggestions based on analysis results.
    """
    suggestions = []
    
    for result in analysis_results:
        if result['differences']['fitness'] < -0.01:  # Fuzzy significantly worse
            failure_pct = result['failure_analysis']['fuzzy_worse_percentage']
            edge_failure = result['failure_analysis']['edge_regions_worse']
            smooth_failure = result['failure_analysis']['smooth_regions_worse']
            
            if failure_pct > 60:
                suggestions.append(f"{result['image_name']}: Major fuzzy failure ({failure_pct:.1f}% worse pixels)")
                
                if edge_failure > 50:
                    suggestions.append(f"  - Edge preservation issue: {edge_failure:.1f}% of edge pixels worse")
                    
                if smooth_failure > 50:
                    suggestions.append(f"  - Over-processing smooth regions: {smooth_failure:.1f}% of smooth pixels worse")
    
    return suggestions


def print_analysis_summary(analysis_results):
    """Print comprehensive analysis summary."""
    print("\n" + "="*80)
    print("FUZZY VS SIMPLE DENOISING ANALYSIS SUMMARY")
    print("="*80)
    
    total_images = len(analysis_results)
    fuzzy_better_count = sum(1 for r in analysis_results if r['differences']['fitness'] > 0)
    
    avg_ssim_diff = np.mean([r['differences']['ssim'] for r in analysis_results])
    avg_psnr_diff = np.mean([r['differences']['psnr'] for r in analysis_results])
    avg_fitness_diff = np.mean([r['differences']['fitness'] for r in analysis_results])
    
    print(f"Total Images Analyzed: {total_images}")
    print(f"Fuzzy Better Than Simple: {fuzzy_better_count}/{total_images} ({fuzzy_better_count/total_images*100:.1f}%)")
    print(f"\nAverage Differences (Fuzzy - Simple):")
    print(f"  SSIM: {avg_ssim_diff:+.4f}")
    print(f"  PSNR: {avg_psnr_diff:+.2f} dB")
    print(f"  Fitness: {avg_fitness_diff:+.4f}")
    
    # Improvement suggestions
    suggestions = generate_improvement_suggestions(analysis_results)
    if suggestions:
        print(f"\n🔧 IMPROVEMENT SUGGESTIONS:")
        for suggestion in suggestions[:5]:  # Show top 5 suggestions
            print(f"   {suggestion}")
    
    print("="*80)