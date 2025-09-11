import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import random
import os
import argparse
import time
import statistics


# --- CONFIG (defaults, can be overridden by CLI) ---
DATASET_DIR = 'images/test/'
RESULTS_DIR = 'results/'
NOISE_TYPE = 'gaussian'
GAUSSIAN_SIGMA = 25
SP_AMOUNT = 0.1
POP_SIZE = 20
GENERATIONS = 20
METRIC = 'ssim'  # 'ssim', 'psnr', or 'combined'
W_SSIM = 0.7
W_PSNR = 0.3


# --- UTILS ---
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Warning: Image not found or unreadable: {path}')
        return None
    return img


def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(img, amount=0.1):
    noisy = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5)
    num_pepper = np.ceil(amount * img.size * 0.5)
    # Salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[coords] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[coords] = 0
    return noisy

def median_sharpen_filter(img, kernel_size, alpha):
    # Ensure kernel size is valid for image size
    if img.shape[0] < kernel_size or img.shape[1] < kernel_size:
        kernel_size = 3
    try:
        median = cv2.medianBlur(img, kernel_size)
    except Exception as e:
        print(f'Warning: medianBlur failed for kernel_size={kernel_size}, error: {e}')
        median = img.copy()
    sharpened = cv2.addWeighted(img, 1.5, median, -0.5, 0)
    enhanced = cv2.addWeighted(median, 1-alpha, sharpened, alpha, 0)
    return enhanced


def fitness_ssim(enhanced, ground_truth):
    return ssim(enhanced, ground_truth)

def fitness_psnr(enhanced, ground_truth):
    return psnr(ground_truth, enhanced)

def fitness_combined(enhanced, ground_truth, w_ssim=0.7, w_psnr=0.3):
    ssim_score = fitness_ssim(enhanced, ground_truth)
    psnr_score = fitness_psnr(enhanced, ground_truth) / 50.0  # Normalize PSNR to [0,1] (typical range)
    return w_ssim * ssim_score + w_psnr * psnr_score

# --- GA COMPONENTS ---
class Chromosome:
    def __init__(self, kernel_size, alpha):
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.fitness = None

    def mutate(self):
        if random.random() < 0.5:
            self.kernel_size = random.choice([3,5,7])
        else:
            self.alpha = min(max(self.alpha + random.uniform(-0.1, 0.1), 0), 1)

    def crossover(self, other):
        if random.random() < 0.5:
            return Chromosome(self.kernel_size, other.alpha)
        else:
            return Chromosome(other.kernel_size, self.alpha)

# --- MAIN PIPELINE ---

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_performance_summary(log, total_time):
    """Print comprehensive performance metrics after processing all images"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    # Basic stats
    total_images = len(log)
    avg_processing_time = statistics.mean([r['processing_time'] for r in log])
    total_processing_time = sum([r['processing_time'] for r in log])
    
    print(f"Total Images Processed: {total_images}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Total Processing Time: {total_processing_time:.2f} seconds")
    print(f"Average Time per Image: {avg_processing_time:.2f} seconds")
    print(f"Images per Second: {total_images/total_time:.2f}")
    
    # SSIM Statistics
    ssim_scores = [r['ssim'] for r in log]
    initial_ssim_scores = [r['initial_ssim'] for r in log]
    ssim_improvements = [r['improvement_ssim'] for r in log]
    
    print(f"\nSSIM METRICS:")
    print(f"  Initial SSIM - Mean: {statistics.mean(initial_ssim_scores):.4f}, Std: {statistics.stdev(initial_ssim_scores):.4f}")
    print(f"  Final SSIM   - Mean: {statistics.mean(ssim_scores):.4f}, Std: {statistics.stdev(ssim_scores):.4f}")
    print(f"  SSIM Improvement - Mean: {statistics.mean(ssim_improvements):.4f}, Max: {max(ssim_improvements):.4f}")
    print(f"  Best SSIM: {max(ssim_scores):.4f} (Image: {log[ssim_scores.index(max(ssim_scores))]['image']})")
    print(f"  Worst SSIM: {min(ssim_scores):.4f} (Image: {log[ssim_scores.index(min(ssim_scores))]['image']})")
    
    # PSNR Statistics
    psnr_scores = [r['psnr'] for r in log]
    initial_psnr_scores = [r['initial_psnr'] for r in log]
    psnr_improvements = [r['improvement_psnr'] for r in log]
    
    print(f"\nPSNR METRICS:")
    print(f"  Initial PSNR - Mean: {statistics.mean(initial_psnr_scores):.2f} dB, Std: {statistics.stdev(initial_psnr_scores):.2f} dB")
    print(f"  Final PSNR   - Mean: {statistics.mean(psnr_scores):.2f} dB, Std: {statistics.stdev(psnr_scores):.2f} dB")
    print(f"  PSNR Improvement - Mean: {statistics.mean(psnr_improvements):.2f} dB, Max: {max(psnr_improvements):.2f} dB")
    print(f"  Best PSNR: {max(psnr_scores):.2f} dB (Image: {log[psnr_scores.index(max(psnr_scores))]['image']})")
    print(f"  Worst PSNR: {min(psnr_scores):.2f} dB (Image: {log[psnr_scores.index(min(psnr_scores))]['image']})")
    
    # GA Optimization Statistics
    fitness_scores = [r['fitness'] for r in log]
    print(f"\nGENETIC ALGORITHM PERFORMANCE:")
    print(f"  Best Fitness: {max(fitness_scores):.4f} (Image: {log[fitness_scores.index(max(fitness_scores))]['image']})")
    print(f"  Worst Fitness: {min(fitness_scores):.4f} (Image: {log[fitness_scores.index(min(fitness_scores))]['image']})")
    print(f"  Average Fitness: {statistics.mean(fitness_scores):.4f}")
    print(f"  Fitness Std Dev: {statistics.stdev(fitness_scores):.4f}")
    
    # Parameter Analysis
    kernels = [r['kernel'] for r in log]
    alphas = [r['alpha'] for r in log]
    from collections import Counter
    kernel_counts = Counter(kernels)
    
    print(f"\nOPTIMAL PARAMETERS ANALYSIS:")
    print(f"  Kernel Size Distribution: {dict(kernel_counts)}")
    print(f"  Most Common Kernel: {kernel_counts.most_common(1)[0][0]} ({kernel_counts.most_common(1)[0][1]} images)")
    print(f"  Alpha Range: {min(alphas):.2f} - {max(alphas):.2f}")
    print(f"  Average Alpha: {statistics.mean(alphas):.2f} ± {statistics.stdev(alphas):.2f}")
    
    # Success Rate Analysis
    successful_improvements_ssim = sum(1 for imp in ssim_improvements if imp > 0)
    successful_improvements_psnr = sum(1 for imp in psnr_improvements if imp > 0)
    
    print(f"\nIMPROVEMENT SUCCESS RATE:")
    print(f"  SSIM Improvements: {successful_improvements_ssim}/{total_images} ({100*successful_improvements_ssim/total_images:.1f}%)")
    print(f"  PSNR Improvements: {successful_improvements_psnr}/{total_images} ({100*successful_improvements_psnr/total_images:.1f}%)")
    
    # Top Performers
    print(f"\nTOP 5 BEST ENHANCED IMAGES (by SSIM):")
    sorted_by_ssim = sorted(log, key=lambda x: x['ssim'], reverse=True)[:5]
    for i, result in enumerate(sorted_by_ssim, 1):
        print(f"  {i}. {result['image']}: SSIM={result['ssim']:.4f}, PSNR={result['psnr']:.2f}dB, Kernel={result['kernel']}, Alpha={result['alpha']:.2f}")
    
    print("="*80)


def process_image(img_path, noise_type='gaussian', sigma=25, sp_amount=0.1, metric='ssim', generations=20, w_ssim=0.7, w_psnr=0.3):
    start_time = time.time()
    clean_img = load_image(img_path)
    if clean_img is None:
        print(f'Skipping {img_path} (cannot load)')
        return None
    fname = os.path.splitext(os.path.basename(img_path))[0]
    # Add noise
    if noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(clean_img, sigma=sigma)
    elif noise_type == 'sp':
        noisy_img = add_salt_pepper_noise(clean_img, amount=sp_amount)
    else:
        print(f'Skipping {img_path} (unknown noise type)')
        return None
    # Save noisy image
    noisy_path = os.path.join(RESULTS_DIR, f'{fname}_noisy.png')
    cv2.imwrite(noisy_path, noisy_img)
    
    # Calculate initial metrics (before enhancement)
    initial_ssim = fitness_ssim(noisy_img, clean_img)
    initial_psnr = fitness_psnr(noisy_img, clean_img)
    
    # GA setup
    population = [Chromosome(random.choice([3,5,7]), random.uniform(0,1)) for _ in range(POP_SIZE)]
    
    best_fitness_history = []
    for gen in range(generations):
        # Evaluate fitness for all chromosomes
        for chrom in population:
            enhanced = median_sharpen_filter(noisy_img, chrom.kernel_size, chrom.alpha)
            if metric == 'ssim':
                chrom.fitness = fitness_ssim(enhanced, clean_img)
            elif metric == 'psnr':
                chrom.fitness = fitness_psnr(enhanced, clean_img)
            elif metric == 'combined':
                chrom.fitness = fitness_combined(enhanced, clean_img, w_ssim, w_psnr)
            else:
                print(f'Skipping {img_path} (unknown metric)')
                return None
        
        # Sort by fitness
        population.sort(key=lambda c: c.fitness, reverse=True)
        best_fitness_history.append(population[0].fitness)
        
        # Print best fitness for this generation
        print(f'{fname} Gen {gen+1}: Best {metric.upper()} = {population[0].fitness:.4f}, Kernel={population[0].kernel_size}, Alpha={population[0].alpha:.2f}')
        
        # Selection and reproduction
        parents = population[:10]
        offspring = []
        while len(offspring) < POP_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = p1.crossover(p2)
            child.mutate()
            offspring.append(child)
        population = offspring
    
    # Final evaluation to get the best solution
    for chrom in population:
        enhanced = median_sharpen_filter(noisy_img, chrom.kernel_size, chrom.alpha)
        if metric == 'ssim':
            chrom.fitness = fitness_ssim(enhanced, clean_img)
        elif metric == 'psnr':
            chrom.fitness = fitness_psnr(enhanced, clean_img)
        elif metric == 'combined':
            chrom.fitness = fitness_combined(enhanced, clean_img, w_ssim, w_psnr)
    
    population.sort(key=lambda c: c.fitness, reverse=True)
    best = population[0]
    final_enhanced = median_sharpen_filter(noisy_img, best.kernel_size, best.alpha)
    enhanced_path = os.path.join(RESULTS_DIR, f'{fname}_enhanced.png')
    cv2.imwrite(enhanced_path, final_enhanced)
    
    # Log results
    ssim_score = fitness_ssim(final_enhanced, clean_img)
    psnr_score = fitness_psnr(final_enhanced, clean_img)
    
    processing_time = time.time() - start_time
    improvement_ssim = ssim_score - initial_ssim
    improvement_psnr = psnr_score - initial_psnr
    
    return {
        'image': fname, 
        'kernel': best.kernel_size, 
        'alpha': best.alpha, 
        'ssim': ssim_score, 
        'psnr': psnr_score, 
        'metric': metric, 
        'fitness': best.fitness, 
        'noisy': noisy_path, 
        'enhanced': enhanced_path,
        'initial_ssim': initial_ssim,
        'initial_psnr': initial_psnr,
        'improvement_ssim': improvement_ssim,
        'improvement_psnr': improvement_psnr,
        'processing_time': processing_time,
        'generations_converged': len(best_fitness_history),
        'best_fitness_history': best_fitness_history
    }


def print_performance_summary(log, total_time):
    """Print comprehensive performance metrics after processing all images"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    # Basic stats
    total_images = len(log)
    avg_processing_time = statistics.mean([r['processing_time'] for r in log])
    total_processing_time = sum([r['processing_time'] for r in log])
    
    print(f"Total Images Processed: {total_images}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Total Processing Time: {total_processing_time:.2f} seconds")
    print(f"Average Time per Image: {avg_processing_time:.2f} seconds")
    print(f"Images per Second: {total_images/total_time:.2f}")
    
    # SSIM Statistics
    ssim_scores = [r['ssim'] for r in log]
    initial_ssim_scores = [r['initial_ssim'] for r in log]
    ssim_improvements = [r['improvement_ssim'] for r in log]
    
    print(f"\nSSIM METRICS:")
    print(f"  Initial SSIM - Mean: {statistics.mean(initial_ssim_scores):.4f}, Std: {statistics.stdev(initial_ssim_scores):.4f}")
    print(f"  Final SSIM   - Mean: {statistics.mean(ssim_scores):.4f}, Std: {statistics.stdev(ssim_scores):.4f}")
    print(f"  SSIM Improvement - Mean: {statistics.mean(ssim_improvements):.4f}, Max: {max(ssim_improvements):.4f}")
    print(f"  Best SSIM: {max(ssim_scores):.4f} (Image: {log[ssim_scores.index(max(ssim_scores))]['image']})")
    print(f"  Worst SSIM: {min(ssim_scores):.4f} (Image: {log[ssim_scores.index(min(ssim_scores))]['image']})")
    
    # PSNR Statistics
    psnr_scores = [r['psnr'] for r in log]
    initial_psnr_scores = [r['initial_psnr'] for r in log]
    psnr_improvements = [r['improvement_psnr'] for r in log]
    
    print(f"\nPSNR METRICS:")
    print(f"  Initial PSNR - Mean: {statistics.mean(initial_psnr_scores):.2f} dB, Std: {statistics.stdev(initial_psnr_scores):.2f} dB")
    print(f"  Final PSNR   - Mean: {statistics.mean(psnr_scores):.2f} dB, Std: {statistics.stdev(psnr_scores):.2f} dB")
    print(f"  PSNR Improvement - Mean: {statistics.mean(psnr_improvements):.2f} dB, Max: {max(psnr_improvements):.2f} dB")
    print(f"  Best PSNR: {max(psnr_scores):.2f} dB (Image: {log[psnr_scores.index(max(psnr_scores))]['image']})")
    print(f"  Worst PSNR: {min(psnr_scores):.2f} dB (Image: {log[psnr_scores.index(min(psnr_scores))]['image']})")
    
    # GA Optimization Statistics
    fitness_scores = [r['fitness'] for r in log]
    print(f"\nGENETIC ALGORITHM PERFORMANCE:")
    print(f"  Best Fitness: {max(fitness_scores):.4f} (Image: {log[fitness_scores.index(max(fitness_scores))]['image']})")
    print(f"  Worst Fitness: {min(fitness_scores):.4f} (Image: {log[fitness_scores.index(min(fitness_scores))]['image']})")
    print(f"  Average Fitness: {statistics.mean(fitness_scores):.4f}")
    print(f"  Fitness Std Dev: {statistics.stdev(fitness_scores):.4f}")
    
    # Parameter Analysis
    kernels = [r['kernel'] for r in log]
    alphas = [r['alpha'] for r in log]
    from collections import Counter
    kernel_counts = Counter(kernels)
    
    print(f"\nOPTIMAL PARAMETERS ANALYSIS:")
    print(f"  Kernel Size Distribution: {dict(kernel_counts)}")
    print(f"  Most Common Kernel: {kernel_counts.most_common(1)[0][0]} ({kernel_counts.most_common(1)[0][1]} images)")
    print(f"  Alpha Range: {min(alphas):.2f} - {max(alphas):.2f}")
    print(f"  Average Alpha: {statistics.mean(alphas):.2f} ± {statistics.stdev(alphas):.2f}")
    
    # Success Rate Analysis
    successful_improvements_ssim = sum(1 for imp in ssim_improvements if imp > 0)
    successful_improvements_psnr = sum(1 for imp in psnr_improvements if imp > 0)
    
    print(f"\nIMPROVEMENT SUCCESS RATE:")
    print(f"  SSIM Improvements: {successful_improvements_ssim}/{total_images} ({100*successful_improvements_ssim/total_images:.1f}%)")
    print(f"  PSNR Improvements: {successful_improvements_psnr}/{total_images} ({100*successful_improvements_psnr/total_images:.1f}%)")
    
    # Top Performers
    print(f"\nTOP 5 BEST ENHANCED IMAGES (by SSIM):")
    sorted_by_ssim = sorted(log, key=lambda x: x['ssim'], reverse=True)[:5]
    for i, result in enumerate(sorted_by_ssim, 1):
        print(f"  {i}. {result['image']}: SSIM={result['ssim']:.4f}, PSNR={result['psnr']:.2f}dB, Kernel={result['kernel']}, Alpha={result['alpha']:.2f}")
    
    print("="*80)
    
    parser = argparse.ArgumentParser(description='Soft Computing Image Enhancement Demo')
    parser.add_argument('--dataset', type=str, default=DATASET_DIR, help='Dataset directory')
    parser.add_argument('--results', type=str, default=RESULTS_DIR, help='Results directory')
    parser.add_argument('--noise', type=str, default=NOISE_TYPE, choices=['gaussian','sp'], help='Noise type')
    parser.add_argument('--sigma', type=int, default=GAUSSIAN_SIGMA, help='Gaussian noise sigma')
    parser.add_argument('--sp_amount', type=float, default=SP_AMOUNT, help='Salt & Pepper noise amount')
    parser.add_argument('--pop', type=int, default=POP_SIZE, help='GA population size')
    parser.add_argument('--gen', type=int, default=GENERATIONS, help='GA generations')
    parser.add_argument('--metric', type=str, default=METRIC, choices=['ssim','psnr','combined'], help='Fitness metric')
    parser.add_argument('--w_ssim', type=float, default=W_SSIM, help='Weight for SSIM in combined metric')
    parser.add_argument('--w_psnr', type=float, default=W_PSNR, help='Weight for PSNR in combined metric')
    args = parser.parse_args()
    DATASET_DIR = args.dataset
    RESULTS_DIR = args.results
    NOISE_TYPE = args.noise
    GAUSSIAN_SIGMA = args.sigma
    SP_AMOUNT = args.sp_amount
    POP_SIZE = args.pop
    GENERATIONS = args.gen
    METRIC = args.metric
    W_SSIM = args.w_ssim
    W_PSNR = args.w_psnr

    start_time = time.time()
    
    ensure_dir(RESULTS_DIR)
    image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    log = []
    for img_file in image_files:
        img_path = os.path.join(DATASET_DIR, img_file)
        print(f'Processing: {img_path}')
        result = process_image(img_path, noise_type=NOISE_TYPE, sigma=GAUSSIAN_SIGMA, sp_amount=SP_AMOUNT, metric=METRIC, generations=GENERATIONS, w_ssim=W_SSIM, w_psnr=W_PSNR)
        if result is not None:
            log.append(result)
    
    total_time = time.time() - start_time
    
    # Save log
    import csv
    log_path = os.path.join(RESULTS_DIR, 'results_log.csv')
    with open(log_path, 'w', newline='') as csvfile:
        fieldnames = ['image','kernel','alpha','ssim','psnr','metric','fitness','noisy','enhanced','initial_ssim','initial_psnr','improvement_ssim','improvement_psnr','processing_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in log:
            # Remove the fitness history for CSV (too complex for CSV format)
            csv_row = {k: v for k, v in row.items() if k != 'best_fitness_history'}
            writer.writerow(csv_row)
    
    print(f'Processed {len(log)} images. Log saved to {log_path}')
    
    # Print comprehensive performance summary
    if log:
        print_performance_summary(log, total_time)

if __name__ == '__main__':
    main()
