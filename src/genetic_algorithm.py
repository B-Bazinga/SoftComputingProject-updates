"""Genetic Algorithm implementation for parameter optimization."""
import random

class Chromosome:
    """Represents a solution in the genetic algorithm."""
    
    def __init__(self, kernel_size=None, alpha=None, enhanced_mode=True):
        self.enhanced_mode = enhanced_mode
        
        if enhanced_mode:
            # GA-Fuzzy mode: Optimized parameter ranges based on analysis
            if kernel_size is None:
                self.kernel_size = random.choice([3, 5, 7])
            else:
                self.kernel_size = kernel_size
            
            if alpha is None:
                # Better alpha range for fuzzy inference - more conservative values
                self.alpha = random.uniform(0.2, 0.7)  # Reduced from 0.1-0.85
            else:
                self.alpha = alpha
        else:
            # Simple denoising mode: Different parameter interpretation
            if kernel_size is None:
                self.kernel_size = random.choice([3, 5, 7])
            else:
                self.kernel_size = kernel_size
            
            if alpha is None:
                # Simple mode alpha is sharpening strength
                self.alpha = random.uniform(0.1, 0.4)  # Lower values for sharpening
            else:
                self.alpha = alpha
        
        self.fitness = None
    
    def mutate(self, mutation_rate=0.3):  # Increased mutation rate for better exploration
        """Mutate chromosome with improved parameter ranges."""
        if random.random() < mutation_rate:
            # Mutate kernel size
            if random.random() < 0.5:
                self.kernel_size = random.choice([3, 5, 7])
            
            # Mutate alpha with mode-specific ranges
            if not self.enhanced_mode:
                # Simple mode: conservative alpha for sharpening
                self.alpha = max(0.05, min(0.5, self.alpha + random.uniform(-0.1, 0.1)))
            else:
                # Fuzzy mode: better alpha range based on fuzzy logic
                if self.alpha < 0.3:  # Low alpha region - preserve details
                    self.alpha = max(0.1, min(0.4, self.alpha + random.uniform(-0.05, 0.1)))
                elif self.alpha < 0.6:  # Medium alpha region - balanced
                    self.alpha = max(0.2, min(0.7, self.alpha + random.uniform(-0.1, 0.1)))
                else:  # High alpha region - aggressive denoising
                    self.alpha = max(0.4, min(0.8, self.alpha + random.uniform(-0.1, 0.05)))
        
        return self
    
    def crossover(self, other):
        """Create offspring through crossover."""
        if random.random() < 0.5:
            return Chromosome(self.kernel_size, other.alpha, self.enhanced_mode)
        else:
            return Chromosome(other.kernel_size, self.alpha, self.enhanced_mode)

class GeneticAlgorithm:
    """Genetic Algorithm for optimizing image enhancement parameters."""
    
    def __init__(self, pop_size, generations, enhanced_mode=True):
        self.pop_size = pop_size
        self.generations = generations
        self.enhanced_mode = enhanced_mode
    
    def initialize_population(self):
        """Initialize population with random chromosomes."""
        population = []
        for _ in range(self.pop_size):
            kernel_size = random.choice([3, 5, 7])
            if self.enhanced_mode:
                alpha = random.uniform(0.2, 0.8)  # More focused range for enhanced mode
            else:
                alpha = random.uniform(0, 1)  # Full range for simple mode
            population.append(Chromosome(kernel_size, alpha, self.enhanced_mode))
        return population
    
    def evaluate_population(self, population, noisy_img, clean_img, fitness_func, filter_func, **kwargs):
        """Evaluate fitness for all chromosomes in population."""
        for chrom in population:
            enhanced = filter_func(noisy_img, chrom.kernel_size, chrom.alpha)
            if 'w_ssim' in kwargs and 'w_psnr' in kwargs:
                chrom.fitness = fitness_func(enhanced, clean_img, kwargs['w_ssim'], kwargs['w_psnr'])
            else:
                chrom.fitness = fitness_func(enhanced, clean_img)
    
    def select_parents(self, population, num_parents=10):
        """Select top performing chromosomes as parents."""
        population.sort(key=lambda c: c.fitness, reverse=True)
        return population[:num_parents]
    
    def reproduce(self, parents):
        """Create offspring through crossover and mutation."""
        offspring = []
        while len(offspring) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            child = p1.crossover(p2)
            child.mutate()
            offspring.append(child)
        return offspring
    
    def evolve(self, noisy_img, clean_img, fitness_func, filter_func, fname, metric_name, **kwargs):
        """
        Run the genetic algorithm evolution process.
        
        Args:
            noisy_img: Noisy input image
            clean_img: Ground truth clean image
            fitness_func: Fitness evaluation function
            filter_func: Image filtering function
            fname: Filename for progress reporting
            metric_name: Name of the metric being optimized
            **kwargs: Additional arguments for fitness function
        
        Returns:
            tuple: (best_chromosome, fitness_history)
        """
        population = self.initialize_population()
        best_fitness_history = []
        
        for gen in range(self.generations):
            # Evaluate fitness
            self.evaluate_population(population, noisy_img, clean_img, fitness_func, filter_func, **kwargs)
            
            # Sort and record best fitness
            population.sort(key=lambda c: c.fitness, reverse=True)
            best_fitness_history.append(population[0].fitness)
            
            # Print progress
            best = population[0]
            print(f'{fname} Gen {gen+1}: Best {metric_name.upper()} = {best.fitness:.4f}, '
                  f'Kernel={best.kernel_size}, Alpha={best.alpha:.2f}')
            
            # Selection and reproduction
            if gen < self.generations - 1:  # Don't reproduce on last generation
                parents = self.select_parents(population)
                population = self.reproduce(parents)
        
        # Final evaluation to ensure we have the best solution
        self.evaluate_population(population, noisy_img, clean_img, fitness_func, filter_func, **kwargs)
        population.sort(key=lambda c: c.fitness, reverse=True)
        
        return population[0], best_fitness_history