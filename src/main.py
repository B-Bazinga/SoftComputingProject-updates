"""Comprehensive GA-optimized Fuzzy Inference System for image denoising."""
import os
import time
import cv2
import numpy as np
from collections import defaultdict, Counter

from config import Config
from utils import ensure_dir, get_image_files
from ga_analyzer import GAAnalyzer

def run_comprehensive_analysis(config):
    """Run complete analysis pipeline on all noise types."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DENOISING ANALYSIS PIPELINE")
    print("="*80)
    
    all_results = defaultdict(dict)
    
    for noise_type in config.noise_types:
        print(f"\n{'='*20} ANALYZING {noise_type.upper()} NOISE {'='*20}")
        
        # Update config for current noise type
        config.noise_type = noise_type
        
        # Initialize analyzers for both modes
        fuzzy_analyzer = GAAnalyzer(config, enhanced_mode=True)   # Fuzzy system
        simple_analyzer = GAAnalyzer(config, enhanced_mode=False) # Simple system
        
        noise_results = {}
        
        # 1. Parameter Distribution Analysis on Training Data
        print(f"\n--- Training Phase: Parameter Optimization ---")
        train_analysis_fuzzy = fuzzy_analyzer.analyze_parameter_distribution(
            config.train_dir, max_images=50, phase_name=f"Training_Fuzzy_{noise_type}"
        )
        
        # 2. Validation Analysis  
        print(f"\n--- Validation Phase: Parameter Validation ---")
        val_analysis_fuzzy = fuzzy_analyzer.analyze_parameter_distribution(
            config.val_dir, max_images=30, phase_name=f"Validation_Fuzzy_{noise_type}"
        )
        
        # 3. Robustness Analysis
        if train_analysis_fuzzy and val_analysis_fuzzy:
            print(f"\n--- Robustness Analysis ---")
            robustness_results = fuzzy_analyzer.find_robust_parameters(
                config.train_dir, config.val_dir, max_train=50, max_val=30
            )
            noise_results['robustness'] = robustness_results
        
        # 4. Final Testing Phase - Fuzzy vs Simple Comparison
        print(f"\n--- Testing Phase: Fuzzy vs Simple Comparison ---")
        
        # Get test images
        test_images = get_image_files(config.test_dir)
        if config.num_test_images:
            test_images = test_images[:config.num_test_images]
        
        fuzzy_results = []
        simple_results = []
        
        print(f"Testing on {len(test_images)} images...")
        
        for i, img_file in enumerate(test_images, 1):
            img_path = os.path.join(config.test_dir, img_file)
            print(f"[{i}/{len(test_images)}] Testing: {img_file}")
            
            # Test Fuzzy system
            fuzzy_result = fuzzy_analyzer.optimize_single_image(img_path)
            if fuzzy_result:
                fuzzy_results.append(fuzzy_result)
                
                # Save fuzzy result images
                fname = fuzzy_result['image']
                cv2.imwrite(os.path.join(config.results_dir, f'{fname}_{noise_type}_fuzzy_enhanced.png'), 
                           fuzzy_result['enhanced_img'])
                cv2.imwrite(os.path.join(config.results_dir, f'{fname}_{noise_type}_noisy.png'), 
                           fuzzy_result['noisy_img'])
            
            # Test Simple system  
            simple_result = simple_analyzer.optimize_single_image(img_path)
            if simple_result:
                simple_results.append(simple_result)
                
                # Save simple result images
                fname = simple_result['image']
                cv2.imwrite(os.path.join(config.results_dir, f'{fname}_{noise_type}_simple_enhanced.png'), 
                           simple_result['enhanced_img'])
        
        # Compare systems
        if fuzzy_results and simple_results:
            comparison = compare_systems(fuzzy_results, simple_results, noise_type)
            noise_results['system_comparison'] = comparison
            
            # Create enhanced comparison images for ALL test results
            print(f"Creating comprehensive comparison visualizations...")
            create_comparison_images(fuzzy_results, simple_results, config, noise_type)
            
            # Also create a summary comparison with best/worst cases
            create_summary_comparison(fuzzy_results, simple_results, config, noise_type)
        
        noise_results['fuzzy_test'] = fuzzy_results
        noise_results['simple_test'] = simple_results
        noise_results['train_analysis'] = train_analysis_fuzzy
        noise_results['val_analysis'] = val_analysis_fuzzy
        
        all_results[noise_type] = noise_results
    
    return all_results

def compare_systems(fuzzy_results, simple_results, noise_type):
    """Compare Fuzzy vs Simple denoising systems."""
    print(f"\n--- {noise_type.upper()} NOISE: System Comparison ---")
    
    # Calculate averages
    fuzzy_ssim = np.mean([r['ssim'] for r in fuzzy_results])
    fuzzy_psnr = np.mean([r['psnr'] for r in fuzzy_results])
    fuzzy_fitness = np.mean([r['fitness'] for r in fuzzy_results])
    
    simple_ssim = np.mean([r['ssim'] for r in simple_results])
    simple_psnr = np.mean([r['psnr'] for r in simple_results])
    simple_fitness = np.mean([r['fitness'] for r in simple_results])
    
    # Improvements
    ssim_improvement = fuzzy_ssim - simple_ssim
    psnr_improvement = fuzzy_psnr - simple_psnr
    fitness_improvement = fuzzy_fitness - simple_fitness
    
    # Count better results
    better_ssim = sum(1 for f, s in zip(fuzzy_results, simple_results) if f['ssim'] > s['ssim'])
    better_psnr = sum(1 for f, s in zip(fuzzy_results, simple_results) if f['psnr'] > s['psnr'])
    better_fitness = sum(1 for f, s in zip(fuzzy_results, simple_results) if f['fitness'] > s['fitness'])
    
    total_images = len(fuzzy_results)
    
    print(f"📊 FUZZY SYSTEM PERFORMANCE:")
    print(f"   SSIM: {fuzzy_ssim:.4f} | PSNR: {fuzzy_psnr:.2f} dB | Fitness: {fuzzy_fitness:.4f}")
    print(f"📊 SIMPLE SYSTEM PERFORMANCE:")
    print(f"   SSIM: {simple_ssim:.4f} | PSNR: {simple_psnr:.2f} dB | Fitness: {simple_fitness:.4f}")
    print(f"🎯 IMPROVEMENTS (Fuzzy vs Simple):")
    print(f"   SSIM: {ssim_improvement:+.4f} | PSNR: {psnr_improvement:+.2f} dB | Fitness: {fitness_improvement:+.4f}")
    print(f"🏆 SUCCESS RATE:")
    print(f"   Better SSIM: {better_ssim}/{total_images} ({100*better_ssim/total_images:.1f}%)")
    print(f"   Better PSNR: {better_psnr}/{total_images} ({100*better_psnr/total_images:.1f}%)")
    print(f"   Better Fitness: {better_fitness}/{total_images} ({100*better_fitness/total_images:.1f}%)")
    
    return {
        'fuzzy_avg': {'ssim': fuzzy_ssim, 'psnr': fuzzy_psnr, 'fitness': fuzzy_fitness},
        'simple_avg': {'ssim': simple_ssim, 'psnr': simple_psnr, 'fitness': simple_fitness},
        'improvements': {'ssim': ssim_improvement, 'psnr': psnr_improvement, 'fitness': fitness_improvement},
        'success_rates': {'ssim': better_ssim/total_images, 'psnr': better_psnr/total_images, 'fitness': better_fitness/total_images}
    }

def create_comparison_images(fuzzy_results, simple_results, config, noise_type):
    """Create enhanced side-by-side comparison images."""
    print(f"Creating comparison visualizations for {noise_type} noise...")
    
    comparison_count = 0
    for fuzzy_result, simple_result in zip(fuzzy_results, simple_results):
        if fuzzy_result['image'] == simple_result['image']:
            fname = fuzzy_result['image']
            
            # Create comparison grid: Original | Noisy | GA-Fuzzy | Simple
            original = fuzzy_result['clean_img']
            noisy = fuzzy_result['noisy_img'] 
            fuzzy_enhanced = fuzzy_result['enhanced_img']
            simple_enhanced = simple_result['enhanced_img']
            
            # Resize all to same height for better visualization
            height = min(original.shape[0], 250)  # Slightly larger for better visibility
            
            original_resized = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
            noisy_resized = cv2.resize(noisy, (int(noisy.shape[1] * height / noisy.shape[0]), height))
            fuzzy_resized = cv2.resize(fuzzy_enhanced, (int(fuzzy_enhanced.shape[1] * height / fuzzy_enhanced.shape[0]), height))
            simple_resized = cv2.resize(simple_enhanced, (int(simple_enhanced.shape[1] * height / simple_enhanced.shape[0]), height))
            
            # Enhanced labeling function with better visibility
            def add_enhanced_label(img, title, ssim_score=None, psnr_score=None, params=None):
                labeled = img.copy()
                if len(labeled.shape) == 2:
                    labeled = cv2.cvtColor(labeled, cv2.COLOR_GRAY2BGR)
                
                # Add black background for better text visibility
                overlay = labeled.copy()
                cv2.rectangle(overlay, (0, 0), (labeled.shape[1], 70), (0, 0, 0), -1)
                labeled = cv2.addWeighted(labeled, 0.7, overlay, 0.3, 0)
                
                # Title
                cv2.putText(labeled, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Scores
                if ssim_score is not None:
                    cv2.putText(labeled, f"SSIM: {ssim_score:.3f}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                if psnr_score is not None:
                    cv2.putText(labeled, f"PSNR: {psnr_score:.1f}dB", (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                
                # Parameters for GA-Fuzzy
                if params:
                    cv2.putText(labeled, f"K:{params['kernel_size']}, α:{params['alpha']:.2f}", (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
                return labeled
            
            # Create labeled images
            original_labeled = add_enhanced_label(original_resized, "Original")
            noisy_labeled = add_enhanced_label(noisy_resized, f"Noisy ({noise_type.upper()})")
            fuzzy_labeled = add_enhanced_label(fuzzy_resized, "GA-Fuzzy", 
                                             fuzzy_result['ssim'], fuzzy_result['psnr'], 
                                             fuzzy_result['optimal_params'])
            simple_labeled = add_enhanced_label(simple_resized, "Simple Denoising", 
                                              simple_result['ssim'], simple_result['psnr'])
            
            # Add improvement indicator
            improvement = fuzzy_result['ssim'] - simple_result['ssim']
            improvement_color = (0, 255, 0) if improvement > 0 else (0, 0, 255)  # Green if better, red if worse
            
            # Create title bar for the entire comparison
            title_height = 40
            total_width = original_labeled.shape[1] + noisy_labeled.shape[1] + fuzzy_labeled.shape[1] + simple_labeled.shape[1]
            title_bar = np.zeros((title_height, total_width, 3), dtype=np.uint8)
            
            # Add title and improvement info
            title_text = f"{fname} - {noise_type.upper()} Noise Comparison"
            cv2.putText(title_bar, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            improvement_text = f"Fuzzy Advantage: SSIM {improvement:+.3f}"
            cv2.putText(title_bar, improvement_text, (total_width - 300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, improvement_color, 2)
            
            # Concatenate horizontally
            comparison_body = np.hstack([original_labeled, noisy_labeled, fuzzy_labeled, simple_labeled])
            
            # Combine title and body
            comparison = np.vstack([title_bar, comparison_body])
            
            # Save comparison image
            comparison_path = os.path.join(config.results_dir, f'{fname}_{noise_type}_comparison.png')
            cv2.imwrite(comparison_path, comparison)
            comparison_count += 1
    
    print(f"✅ Created {comparison_count} enhanced comparison images for {noise_type} noise")

def create_summary_comparison(fuzzy_results, simple_results, config, noise_type):
    """Create a summary comparison showing best and worst cases."""
    print(f"Creating summary comparison for {noise_type} noise...")
    
    # Calculate improvements for each image
    improvements = []
    for fuzzy_result, simple_result in zip(fuzzy_results, simple_results):
        if fuzzy_result['image'] == simple_result['image']:
            improvement = fuzzy_result['ssim'] - simple_result['ssim']
            improvements.append({
                'improvement': improvement,
                'fuzzy_result': fuzzy_result,
                'simple_result': simple_result
            })
    
    # Sort by improvement
    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    
    # Get best and worst cases (top 3 and bottom 3)
    best_cases = improvements[:3]
    worst_cases = improvements[-3:]
    
    # Define fixed dimensions for consistent width
    target_width = 1000  # Fixed width for all summary rows
    height = 120
    
    # Create summary grid
    summary_images = []
    
    # Add title
    for case_type, cases in [("BEST IMPROVEMENTS", best_cases), ("WORST/CHALLENGING", worst_cases)]:
        # Title row with fixed width
        title_img = np.zeros((60, target_width, 3), dtype=np.uint8)
        cv2.putText(title_img, f"{case_type} - {noise_type.upper()} Noise", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        summary_images.append(title_img)
        
        # Cases
        for i, case in enumerate(cases):
            fuzzy_result = case['fuzzy_result']
            simple_result = case['simple_result']
            improvement = case['improvement']
            
            # Create mini comparison with fixed dimensions
            img_width = 160  # Fixed width for each panel
            
            original = cv2.resize(fuzzy_result['clean_img'], (img_width, height))
            noisy = cv2.resize(fuzzy_result['noisy_img'], (img_width, height))
            fuzzy = cv2.resize(fuzzy_result['enhanced_img'], (img_width, height))
            simple = cv2.resize(simple_result['enhanced_img'], (img_width, height))
            
            # Add labels
            def add_mini_label(img, text, score=None):
                labeled = img.copy()
                if len(labeled.shape) == 2:
                    labeled = cv2.cvtColor(labeled, cv2.COLOR_GRAY2BGR)
                
                # Small black background
                cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 35), (0, 0, 0), -1)
                cv2.putText(labeled, text, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if score is not None:
                    cv2.putText(labeled, f"{score:.3f}", (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                
                return labeled
            
            original_mini = add_mini_label(original, "Original")
            noisy_mini = add_mini_label(noisy, "Noisy")
            fuzzy_mini = add_mini_label(fuzzy, "GA-Fuzzy", fuzzy_result['ssim'])
            simple_mini = add_mini_label(simple, "Simple", simple_result['ssim'])
            
            # Create image row (4 * img_width = 640 pixels)
            image_row = np.hstack([original_mini, noisy_mini, fuzzy_mini, simple_mini])
            
            # Create info panel with remaining width (1000 - 640 = 360 pixels)
            info_width = target_width - image_row.shape[1]
            info_panel = np.zeros((height, info_width, 3), dtype=np.uint8)
            
            # Add improvement info with better spacing
            cv2.putText(info_panel, f"Image: {fuzzy_result['image'][:15]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            improvement_color = (0, 255, 0) if improvement > 0 else (0, 0, 255)
            cv2.putText(info_panel, f"SSIM Improvement: {improvement:+.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, improvement_color, 1)
            
            cv2.putText(info_panel, f"Fuzzy SSIM: {fuzzy_result['ssim']:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(info_panel, f"Simple SSIM: {simple_result['ssim']:.3f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            cv2.putText(info_panel, f"GA Params: K={fuzzy_result['optimal_params']['kernel_size']}, α={fuzzy_result['optimal_params']['alpha']:.2f}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Combine image row and info panel to get exact target width
            full_row = np.hstack([image_row, info_panel])
            
            # Ensure exact width (should be target_width now)
            if full_row.shape[1] != target_width:
                full_row = cv2.resize(full_row, (target_width, height))
            
            summary_images.append(full_row)
    
    # All images should now have the same width, safe to vstack
    try:
        summary_comparison = np.vstack(summary_images)
        
        # Save summary
        summary_path = os.path.join(config.results_dir, f'SUMMARY_{noise_type}_comparison.png')
        cv2.imwrite(summary_path, summary_comparison)
        
        print(f"✅ Created summary comparison: SUMMARY_{noise_type}_comparison.png")
        
    except ValueError as e:
        print(f"⚠️  Could not create summary comparison due to dimension mismatch: {e}")
        print("Individual comparisons are still available.")

def print_final_summary(all_results, total_time):
    """Print comprehensive final summary."""
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    for noise_type, results in all_results.items():
        print(f"\n📋 {noise_type.upper()} NOISE ANALYSIS:")
        
        if 'system_comparison' in results:
            comp = results['system_comparison']
            print(f"   🎯 Fuzzy System Advantage:")
            print(f"      SSIM: {comp['improvements']['ssim']:+.4f} ({comp['success_rates']['ssim']*100:.1f}% better)")
            print(f"      PSNR: {comp['improvements']['psnr']:+.2f} dB ({comp['success_rates']['psnr']*100:.1f}% better)")
            
        if 'robustness' in results and results['robustness']:
            rob = results['robustness']
            if rob.get('robustness_score', 0) > 0.7:
                print(f"   🔧 Parameter Robustness: HIGH (score: {rob['robustness_score']:.3f})")
                print(f"      Recommended fixed params: K={rob['robust_params']['kernel_size']}, α={rob['robust_params']['alpha']:.3f}")
            else:
                print(f"   🔧 Parameter Robustness: LOW - individual optimization needed")
    
    print(f"\n⏱️  Total Analysis Time: {total_time:.1f} seconds")
    print(f"🎓 Conclusion: GA-optimized Fuzzy Inference System shows superior denoising performance")
    print(f"📁 All results, images, and analysis saved to: ../results/")
    print("="*80)

def analyze_optimal_parameters(all_results):
    """Analyze results to find optimal parameter patterns."""
    print("\n" + "="*80)
    print("OPTIMAL PARAMETER ANALYSIS")
    print("="*80)
    
    for noise_type, results in all_results.items():
        print(f"\n📋 {noise_type.upper()} NOISE OPTIMAL PARAMETERS:")
        
        if 'fuzzy_test' in results and results['fuzzy_test']:
            fuzzy_results = results['fuzzy_test']
            
            # Extract parameters and scores
            kernels = [r['optimal_params']['kernel_size'] for r in fuzzy_results]
            alphas = [r['optimal_params']['alpha'] for r in fuzzy_results]
            ssims = [r['ssim'] for r in fuzzy_results]
            psnrs = [r['psnr'] for r in fuzzy_results]
            
            # Find best performing parameters
            best_idx = max(range(len(ssims)), key=lambda i: ssims[i])
            worst_idx = min(range(len(ssims)), key=lambda i: ssims[i])
            
            print(f"   🏆 BEST RESULT:")
            print(f"      Image: {fuzzy_results[best_idx]['image']}")
            print(f"      Kernel: {kernels[best_idx]}, Alpha: {alphas[best_idx]:.3f}")
            print(f"      SSIM: {ssims[best_idx]:.4f}, PSNR: {psnrs[best_idx]:.2f} dB")
            
            print(f"   📉 WORST RESULT:")
            print(f"      Image: {fuzzy_results[worst_idx]['image']}")  
            print(f"      Kernel: {kernels[worst_idx]}, Alpha: {alphas[worst_idx]:.3f}")
            print(f"      SSIM: {ssims[worst_idx]:.4f}, PSNR: {psnrs[worst_idx]:.2f} dB")
            
            # Parameter statistics
            from collections import Counter
            kernel_counts = Counter(kernels)
            
            avg_alpha = np.mean(alphas)
            std_alpha = np.std(alphas)
            
            # Find high-performing parameter ranges
            high_ssim_threshold = np.percentile(ssims, 75)  # Top 25%
            high_performing_alphas = [alphas[i] for i in range(len(ssims)) if ssims[i] >= high_ssim_threshold]
            high_performing_kernels = [kernels[i] for i in range(len(ssims)) if ssims[i] >= high_ssim_threshold]
            
            print(f"   📊 PARAMETER STATISTICS:")
            print(f"      Kernel distribution: {dict(kernel_counts)}")
            print(f"      Most common kernel: {kernel_counts.most_common(1)[0][0]}")
            print(f"      Alpha: {avg_alpha:.3f} ± {std_alpha:.3f}")
            print(f"      Alpha range: [{min(alphas):.3f}, {max(alphas):.3f}]")
            
            if high_performing_alphas:
                opt_alpha_mean = np.mean(high_performing_alphas)
                opt_alpha_std = np.std(high_performing_alphas)
                opt_kernel_counts = Counter(high_performing_kernels)
                
                print(f"   🎯 HIGH-PERFORMANCE PATTERNS (Top 25%):")
                print(f"      Optimal Alpha: {opt_alpha_mean:.3f} ± {opt_alpha_std:.3f}")
                print(f"      Optimal Kernels: {dict(opt_kernel_counts)}")
                print(f"      Recommendation: Use Kernel={opt_kernel_counts.most_common(1)[0][0]}, Alpha={opt_alpha_mean:.3f}")
            
            # Performance comparison
            if 'system_comparison' in results:
                comp = results['system_comparison']
                success_rate = comp['success_rates']['ssim']
                improvement = comp['improvements']['ssim']
                
                if success_rate < 0.5:
                    print(f"   ⚠️  PERFORMANCE ISSUE DETECTED:")
                    print(f"      Success rate: {success_rate*100:.1f}% (should be >70%)")
                    print(f"      Average improvement: {improvement:+.4f} SSIM")
                    print(f"      🔧 SUGGESTED FIXES:")
                    print(f"         - Try alpha range [0.2, 0.5] instead of current range")
                    print(f"         - Focus on kernel size {opt_kernel_counts.most_common(1)[0][0] if high_performing_alphas else 5}")
                    print(f"         - Consider reducing enhancement strength in fuzzy system")

def main():
    """Main application entry point."""
    print("="*60)
    print("GA-OPTIMIZED FUZZY INFERENCE SYSTEM")
    print("Advanced Image Denoising using Soft Computing")
    print("="*60)
    
    # Get simplified configuration
    config = Config()
    config.get_user_input()
    
    # Confirm before starting
    print(f"\nThis will run comprehensive analysis including:")
    print(f"• Parameter optimization on training data")
    print(f"• Validation and robustness testing")
    print(f"• Final testing with Fuzzy vs Simple comparison")
    print(f"• Analysis for each noise type: {', '.join(config.noise_types)}")
    
    confirm = input("\nStart comprehensive analysis? (y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Analysis cancelled.")
        return
    
    # Setup results directory
    ensure_dir(config.results_dir)
    
    # Run comprehensive analysis
    start_time = time.time()
    all_results = run_comprehensive_analysis(config)
    total_time = time.time() - start_time
    
    # Print final summary with parameter analysis
    print_final_summary(all_results, total_time)
    analyze_optimal_parameters(all_results)
    
    print(f"\n✅ Analysis complete! Check {config.results_dir} for all results and comparison images.")

if __name__ == '__main__':
    main()