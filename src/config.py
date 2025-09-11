"""Simplified configuration for GA-optimized fuzzy denoising system."""
import os

class Config:
    def __init__(self):
        # Fixed paths - no user input needed
        self.train_dir = '../data/images/train/'
        self.val_dir = '../data/images/val/'
        self.test_dir = '../data/images/test/'
        self.results_dir = '../results/'
        
        # Always use combined metric
        self.metric = 'combined'
        
        # Default values
        self.noise_types = ['gaussian']  # Will be set by user
        self.gaussian_sigma = 25
        self.sp_amount = 0.1
        self.pop_size = 20
        self.generations = 20
        self.w_ssim = 0.7
        self.w_psnr = 0.3
        self.num_test_images = None
        
    def get_user_input(self):
        """Simplified configuration - only essential parameters"""
        print("=" * 60)
        print("GA-OPTIMIZED FUZZY INFERENCE SYSTEM")
        print("Image Denoising Configuration")
        print("=" * 60)
        
        # Noise configuration
        print("\n1. Noise Configuration:")
        print("Available options:")
        print("  1. Gaussian noise only")
        print("  2. Salt & Pepper noise only") 
        print("  3. Both noise types")
        
        noise_choice = input("Choose noise type [1]: ").strip() or "1"
        
        if noise_choice == "1":
            self.noise_types = ['gaussian']
            sigma_input = input(f"Gaussian sigma (1-50) [{self.gaussian_sigma}]: ").strip()
            if sigma_input:
                self.gaussian_sigma = int(sigma_input)
        elif noise_choice == "2":
            self.noise_types = ['sp']
            sp_input = input(f"Salt & Pepper amount (0.0-0.5) [{self.sp_amount}]: ").strip()
            if sp_input:
                self.sp_amount = float(sp_input)
        else:
            self.noise_types = ['gaussian', 'sp']
            sigma_input = input(f"Gaussian sigma (1-50) [{self.gaussian_sigma}]: ").strip()
            if sigma_input:
                self.gaussian_sigma = int(sigma_input)
            sp_input = input(f"Salt & Pepper amount (0.0-0.5) [{self.sp_amount}]: ").strip()
            if sp_input:
                self.sp_amount = float(sp_input)
        
        # GA configuration
        print("\n2. Genetic Algorithm Configuration:")
        pop_input = input(f"Population size (10-100) [{self.pop_size}]: ").strip()
        if pop_input:
            self.pop_size = int(pop_input)
            
        gen_input = input(f"Number of generations (5-50) [{self.generations}]: ").strip()
        if gen_input:
            self.generations = int(gen_input)
        
        # Fitness weights
        print("\n3. Combined Fitness Metric (SSIM + PSNR):")
        w_ssim_input = input(f"SSIM weight (0.0-1.0) [{self.w_ssim}]: ").strip()
        if w_ssim_input:
            self.w_ssim = float(w_ssim_input)
            self.w_psnr = 1.0 - self.w_ssim
        
        # Test images limit
        print("\n4. Testing Configuration:")
        num_images_input = input(f"Number of test images (leave empty for all): ").strip()
        if num_images_input:
            self.num_test_images = int(num_images_input)
        
        self.validate_config()
        self.print_config()
    
    def validate_config(self):
        """Validate configuration and check directories"""
        # Check if directories exist
        for dir_name, dir_path in [("Training", self.train_dir), 
                                  ("Validation", self.val_dir), 
                                  ("Test", self.test_dir)]:
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_name} directory not found: {dir_path}")
                # Create directory structure
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created {dir_path} - please add images and restart")
        
        # Validate parameters
        if not (1 <= self.gaussian_sigma <= 50):
            raise ValueError("Gaussian sigma must be between 1 and 50")
        
        if not (0.0 <= self.sp_amount <= 0.5):
            raise ValueError("Salt & Pepper amount must be between 0.0 and 0.5")
        
        if not (10 <= self.pop_size <= 100):
            raise ValueError("Population size must be between 10 and 100")
        
        if not (5 <= self.generations <= 50):
            raise ValueError("Generations must be between 5 and 50")
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "=" * 40)
        print("CONFIGURATION SUMMARY")
        print("=" * 40)
        print(f"Noise Types: {', '.join(self.noise_types).upper()}")
        if 'gaussian' in self.noise_types:
            print(f"Gaussian Sigma: {self.gaussian_sigma}")
        if 'sp' in self.noise_types:
            print(f"Salt & Pepper Amount: {self.sp_amount}")
        print(f"GA Population: {self.pop_size}")
        print(f"GA Generations: {self.generations}")
        print(f"Fitness: SSIM({self.w_ssim:.2f}) + PSNR({self.w_psnr:.2f})")
        if self.num_test_images:
            print(f"Test Images: {self.num_test_images}")
        else:
            print("Test Images: All available")
        print(f"Results: {self.results_dir}")
        print("=" * 40)