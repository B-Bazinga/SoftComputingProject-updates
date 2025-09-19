"""Genetic Algorithm parameter analysis and optimization pipeline."""
import os
import time
import csv
import statistics
import pickle
from collections import Counter, defaultdict

import cv2
import numpy as np

from config import Config
from utils import load_image, add_gaussian_noise, add_salt_pepper_noise, ensure_dir, get_image_files
from image_processing import fuzzy_median_filter, simple_denoising_filter
from genetic_algorithm import GeneticAlgorithm
from metrics import fitness_ssim, fitness_psnr, fitness_combined, get_fitness_function

class GAAnalyzer:
    """Genetic Algorithm parameter analyzer for soft computing optimization."""
    
    def __init__(self, config, enhanced_mode=True):
        self.config = config
        self.enhanced_mode = enhanced_mode
        self.filter_func = fuzzy_median_filter if enhanced_mode else simple_denoising_filter
        self.fitness_func = get_fitness_function(config.metric)
        
        # Analysis statistics
        self.parameter_distributions = defaultdict(list)
        self.fitness_distributions = defaultdict(list)
        self.convergence_analysis = []
        
    def optimize_single_image(self, img_path):
        """Optimize parameters for a single image using GA."""
        clean_img = load_image(img_path)
        if clean_img is None:
            return None
            
        fname = os.path.splitext(os.path.basename(img_path))[0]
        
        # Add noise based on configuration
        if self.config.noise_type == 'gaussian':
            noisy_img = add_gaussian_noise(clean_img, sigma=self.config.gaussian_sigma)
            noise_type = 'gaussian'
        elif self.config.noise_type == 'sp':
            noisy_img = add_salt_pepper_noise(clean_img, amount=self.config.sp_amount)
            noise_type = 'sp'
        else:
            return None
        
        # Run genetic algorithm optimization with noise-type awareness
        ga = GeneticAlgorithm(self.config.pop_size, self.config.generations, 
                             self.enhanced_mode, noise_type)
        
        if self.config.metric == 'combined':
            best, fitness_history = ga.evolve(
                noisy_img, clean_img, self.fitness_func, self.filter_func, fname, self.config.metric,
                w_ssim=self.config.w_ssim, w_psnr=self.config.w_psnr
            )
        else:
            best, fitness_history = ga.evolve(
                noisy_img, clean_img, self.fitness_func, self.filter_func, fname, self.config.metric
            )
        
        # Generate final enhanced image
        final_enhanced = self.filter_func(noisy_img, best.kernel_size, best.alpha)
        
        # Calculate comprehensive metrics
        ssim_score = fitness_ssim(final_enhanced, clean_img)
        psnr_score = fitness_psnr(final_enhanced, clean_img)
        
        # Analyze convergence
        convergence_gen = len(fitness_history)
        for i, fitness in enumerate(fitness_history):
            if i > 0 and abs(fitness - fitness_history[i-1]) < 1e-6:
                convergence_gen = i
                break
        
        return {
            'image': fname,
            'optimal_params': {'kernel_size': best.kernel_size, 'alpha': best.alpha},
            'fitness': best.fitness,
            'ssim': ssim_score,
            'psnr': psnr_score,
            'convergence_generation': convergence_gen,
            'fitness_history': fitness_history,
            'noisy_img': noisy_img,
            'enhanced_img': final_enhanced,
            'clean_img': clean_img
        }
    
    def test_with_fixed_params(self, img_path, params):
        """Test fixed parameters on a single image."""
        clean_img = load_image(img_path)
        if clean_img is None:
            return None
            
        fname = os.path.splitext(os.path.basename(img_path))[0]
        
        # Add noise
        if self.config.noise_type == 'gaussian':
            noisy_img = add_gaussian_noise(clean_img, sigma=self.config.gaussian_sigma)
        elif self.config.noise_type == 'sp':
            noisy_img = add_salt_pepper_noise(clean_img, amount=self.config.sp_amount)
        else:
            return None
        
        # Apply fixed parameters
        enhanced = self.filter_func(noisy_img, params['kernel_size'], params['alpha'])
        
        if self.config.metric == 'combined':
            fitness_score = self.fitness_func(enhanced, clean_img, self.config.w_ssim, self.config.w_psnr)
        else:
            fitness_score = self.fitness_func(enhanced, clean_img)
        
        ssim_score = fitness_ssim(enhanced, clean_img)
        psnr_score = fitness_psnr(enhanced, clean_img)
        
        return {
            'image': fname,
            'params': params,
            'fitness': fitness_score,
            'ssim': ssim_score,
            'psnr': psnr_score,
            'noisy_img': noisy_img,
            'enhanced_img': enhanced,
            'clean_img': clean_img
        }
    
    def analyze_parameter_distribution(self, dataset_dir, max_images=None, phase_name="Analysis"):
        """Analyze optimal parameter distribution across a dataset."""
        print(f"\n=== {phase_name} on {dataset_dir} ===")
        
        try:
            image_files = get_image_files(dataset_dir)
            if not image_files:
                print(f"No images found in {dataset_dir}")
                return None
                
            if max_images:
                image_files = image_files[:max_images]
                
            print(f"Analyzing {len(image_files)} images...")
            
            results = []
            
            for i, img_file in enumerate(image_files, 1):
                img_path = os.path.join(dataset_dir, img_file)
                print(f"[{i}/{len(image_files)}] Optimizing: {img_file}")
                
                result = self.optimize_single_image(img_path)
                if result is not None:
                    results.append(result)
                    
                    # Store for distribution analysis
                    self.parameter_distributions[phase_name].append(result['optimal_params'])
                    self.fitness_distributions[phase_name].append(result['fitness'])
            
            if not results:
                return None
            
            # Calculate statistics
            kernels = [r['optimal_params']['kernel_size'] for r in results]
            alphas = [r['optimal_params']['alpha'] for r in results]
            fitness_scores = [r['fitness'] for r in results]
            ssim_scores = [r['ssim'] for r in results]
            psnr_scores = [r['psnr'] for r in results]
            convergence_gens = [r['convergence_generation'] for r in results]
            
            kernel_dist = Counter(kernels)
            
            analysis_summary = {
                'phase': phase_name,
                'num_images': len(results),
                'parameter_stats': {
                    'kernel_distribution': dict(kernel_dist),
                    'most_common_kernel': kernel_dist.most_common(1)[0],
                    'alpha_mean': statistics.mean(alphas),
                    'alpha_std': statistics.stdev(alphas) if len(alphas) > 1 else 0,
                    'alpha_range': (min(alphas), max(alphas))
                },
                'performance_stats': {
                    'avg_fitness': statistics.mean(fitness_scores),
                    'std_fitness': statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0,
                    'avg_ssim': statistics.mean(ssim_scores),
                    'std_ssim': statistics.stdev(ssim_scores) if len(ssim_scores) > 1 else 0,
                    'avg_psnr': statistics.mean(psnr_scores),
                    'std_psnr': statistics.stdev(psnr_scores) if len(psnr_scores) > 1 else 0,
                    'best_fitness': max(fitness_scores),
                    'worst_fitness': min(fitness_scores)
                },
                'convergence_stats': {
                    'avg_convergence_gen': statistics.mean(convergence_gens),
                    'std_convergence_gen': statistics.stdev(convergence_gens) if len(convergence_gens) > 1 else 0,
                    'fastest_convergence': min(convergence_gens),
                    'slowest_convergence': max(convergence_gens)
                },
                'all_results': results
            }
            
            # Print analysis
            print(f"\n=== {phase_name} Summary ===")
            print(f"Images Processed: {len(results)}")
            print(f"Parameter Distribution:")
            print(f"  Kernel Sizes: {dict(kernel_dist)}")
            print(f"  Most Common Kernel: {kernel_dist.most_common(1)[0][0]} ({kernel_dist.most_common(1)[0][1]} images)")
            print(f"  Alpha: {statistics.mean(alphas):.3f} ± {statistics.stdev(alphas) if len(alphas) > 1 else 0:.3f}")
            print(f"  Alpha Range: [{min(alphas):.3f}, {max(alphas):.3f}]")
            print(f"Performance:")
            print(f"  Avg Fitness: {statistics.mean(fitness_scores):.4f} ± {statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0:.4f}")
            print(f"  Avg SSIM: {statistics.mean(ssim_scores):.4f} ± {statistics.stdev(ssim_scores) if len(ssim_scores) > 1 else 0:.4f}")
            print(f"  Avg PSNR: {statistics.mean(psnr_scores):.2f} ± {statistics.stdev(psnr_scores) if len(psnr_scores) > 1 else 0:.2f} dB")
            print(f"Convergence:")
            print(f"  Avg Generations: {statistics.mean(convergence_gens):.1f} ± {statistics.stdev(convergence_gens) if len(convergence_gens) > 1 else 0:.1f}")
            print(f"  Range: [{min(convergence_gens)}, {max(convergence_gens)}] generations")
            
            return analysis_summary
            
        except Exception as e:
            print(f"Error during {phase_name.lower()}: {e}")
            return None
    
    def compare_optimization_vs_fixed(self, test_dir, common_params, max_images=None):
        """Compare individual optimization vs using common parameters."""
        print(f"\n=== Optimization vs Fixed Parameters Comparison ===")
        
        try:
            image_files = get_image_files(test_dir)
            if not image_files:
                print(f"No images found in {test_dir}")
                return None
                
            if max_images:
                image_files = image_files[:max_images]
                
            print(f"Comparing on {len(image_files)} images...")
            print(f"Fixed parameters: Kernel={common_params['kernel_size']}, Alpha={common_params['alpha']:.3f}")
            
            optimized_results = []
            fixed_results = []
            
            for i, img_file in enumerate(image_files, 1):
                img_path = os.path.join(test_dir, img_file)
                print(f"[{i}/{len(image_files)}] Comparing: {img_file}")
                
                # Individual optimization
                opt_result = self.optimize_single_image(img_path)
                if opt_result:
                    optimized_results.append(opt_result)
                
                # Fixed parameters
                fixed_result = self.test_with_fixed_params(img_path, common_params)
                if fixed_result:
                    fixed_results.append(fixed_result)
            
            if not optimized_results or not fixed_results:
                return None
            
            # Calculate comparison statistics
            opt_fitness = [r['fitness'] for r in optimized_results]
            fixed_fitness = [r['fitness'] for r in fixed_results]
            
            opt_ssim = [r['ssim'] for r in optimized_results]
            fixed_ssim = [r['ssim'] for r in fixed_results]
            
            opt_psnr = [r['psnr'] for r in optimized_results]
            fixed_psnr = [r['psnr'] for r in fixed_results]
            
            # Performance differences
            fitness_improvements = [opt - fixed for opt, fixed in zip(opt_fitness, fixed_fitness)]
            ssim_improvements = [opt - fixed for opt, fixed in zip(opt_ssim, fixed_ssim)]
            psnr_improvements = [opt - fixed for opt, fixed in zip(opt_psnr, fixed_psnr)]
            
            comparison = {
                'num_images': len(optimized_results),
                'optimized_performance': {
                    'avg_fitness': statistics.mean(opt_fitness),
                    'avg_ssim': statistics.mean(opt_ssim),
                    'avg_psnr': statistics.mean(opt_psnr)
                },
                'fixed_performance': {
                    'avg_fitness': statistics.mean(fixed_fitness),
                    'avg_ssim': statistics.mean(fixed_ssim),
                    'avg_psnr': statistics.mean(fixed_psnr)
                },
                'improvements': {
                    'avg_fitness_gain': statistics.mean(fitness_improvements),
                    'avg_ssim_gain': statistics.mean(ssim_improvements),
                    'avg_psnr_gain': statistics.mean(psnr_improvements),
                    'better_fitness_count': sum(1 for imp in fitness_improvements if imp > 0),
                    'better_ssim_count': sum(1 for imp in ssim_improvements if imp > 0),
                    'better_psnr_count': sum(1 for imp in psnr_improvements if imp > 0)
                },
                'fixed_params': common_params,
                'optimized_results': optimized_results,
                'fixed_results': fixed_results
            }
            
            # Print comparison
            print(f"\n=== Comparison Results ===")
            print(f"Individual Optimization:")
            print(f"  Avg Fitness: {statistics.mean(opt_fitness):.4f}")
            print(f"  Avg SSIM: {statistics.mean(opt_ssim):.4f}")
            print(f"  Avg PSNR: {statistics.mean(opt_psnr):.2f} dB")
            print(f"Fixed Parameters:")
            print(f"  Avg Fitness: {statistics.mean(fixed_fitness):.4f}")
            print(f"  Avg SSIM: {statistics.mean(fixed_ssim):.4f}")
            print(f"  Avg PSNR: {statistics.mean(fixed_psnr):.2f} dB")
            print(f"Improvement Analysis:")
            print(f"  Avg Fitness Gain: {statistics.mean(fitness_improvements):+.4f}")
            print(f"  Avg SSIM Gain: {statistics.mean(ssim_improvements):+.4f}")
            print(f"  Avg PSNR Gain: {statistics.mean(psnr_improvements):+.2f} dB")
            print(f"  Images Improved: {sum(1 for imp in fitness_improvements if imp > 0)}/{len(fitness_improvements)} ({100*sum(1 for imp in fitness_improvements if imp > 0)/len(fitness_improvements):.1f}%)")
            
            return comparison
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            return None
    
    def find_robust_parameters(self, train_dir, val_dir, max_train=None, max_val=None):
        """Find parameters that work well across different image sets."""
        print(f"\n=== Finding Robust Parameters ===")
        
        # Analyze parameter distribution on training set
        train_analysis = self.analyze_parameter_distribution(train_dir, max_train, "Training")
        if not train_analysis:
            return None
        
        # Extract common parameters from training
        train_stats = train_analysis['parameter_stats']
        common_params = {
            'kernel_size': train_stats['most_common_kernel'][0],
            'alpha': train_stats['alpha_mean']
        }
        
        print(f"\nExtracted common parameters from training:")
        print(f"  Kernel: {common_params['kernel_size']} (most common)")
        print(f"  Alpha: {common_params['alpha']:.3f} (mean)")
        
        # Test these parameters on validation set
        if val_dir:
            val_comparison = self.compare_optimization_vs_fixed(val_dir, common_params, max_val)
            if val_comparison:
                # If fixed parameters work well (>80% of cases), they're robust
                success_rate = val_comparison['improvements']['better_fitness_count'] / val_comparison['num_images']
                
                if success_rate < 0.2:  # Fixed params work well in 80%+ cases
                    print(f"✅ Robust parameters found! Fixed params work well in {(1-success_rate)*100:.1f}% of cases")
                    return {
                        'robust_params': common_params,
                        'robustness_score': 1 - success_rate,
                        'train_analysis': train_analysis,
                        'val_comparison': val_comparison
                    }
                else:
                    print(f"⚠️  Parameters not robust. Individual optimization needed in {success_rate*100:.1f}% of cases")
        
        return {
            'robust_params': common_params,
            'robustness_score': 0,
            'train_analysis': train_analysis,
            'recommendation': 'Use individual optimization for each image'
        }
    
    def save_analysis_results(self, analysis_results, comparison_results=None):
        """Save comprehensive analysis results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save parameters
        if 'robust_params' in analysis_results:
            params_file = os.path.join(self.config.results_dir, f'robust_params_{timestamp}.pkl')
            with open(params_file, 'wb') as f:
                pickle.dump(analysis_results['robust_params'], f)
            print(f"Robust parameters saved to {params_file}")
        
        # Save detailed CSV log
        log_file = os.path.join(self.config.results_dir, f'ga_analysis_{timestamp}.csv')
        
        with open(log_file, 'w', newline='') as csvfile:
            fieldnames = [
                'phase', 'image_count', 'kernel_dist', 'common_kernel', 'alpha_mean', 'alpha_std',
                'avg_fitness', 'std_fitness', 'avg_ssim', 'std_ssim', 'avg_psnr', 'std_psnr',
                'avg_convergence', 'robustness_score'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Training analysis
            if 'train_analysis' in analysis_results:
                train = analysis_results['train_analysis']
                writer.writerow({
                    'phase': 'parameter_analysis',
                    'image_count': train['num_images'],
                    'kernel_dist': str(train['parameter_stats']['kernel_distribution']),
                    'common_kernel': train['parameter_stats']['most_common_kernel'][0],
                    'alpha_mean': train['parameter_stats']['alpha_mean'],
                    'alpha_std': train['parameter_stats']['alpha_std'],
                    'avg_fitness': train['performance_stats']['avg_fitness'],
                    'std_fitness': train['performance_stats']['std_fitness'],
                    'avg_ssim': train['performance_stats']['avg_ssim'],
                    'std_ssim': train['performance_stats']['std_ssim'],
                    'avg_psnr': train['performance_stats']['avg_psnr'],
                    'std_psnr': train['performance_stats']['std_psnr'],
                    'avg_convergence': train['convergence_stats']['avg_convergence_gen']
                })
            
            # Robustness analysis
            if 'robustness_score' in analysis_results:
                writer.writerow({
                    'phase': 'robustness_analysis',
                    'robustness_score': analysis_results['robustness_score']
                })
        
        print(f"Analysis results saved to {log_file}")
        return log_file