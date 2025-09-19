# GA-Optimized Fuzzy Inference System for Image Denoising

A comprehensive **Soft Computing** approach that combines **Genetic Algorithms** with **Fuzzy Inference Systems** for intelligent image denoising with **automatic noise type detection**.

## Overview

This project demonstrates advanced soft computing techniques for image enhancement:

- **🧬 Genetic Algorithm (GA)**: Evolutionary optimization of denoising parameters with noise-type awareness
- **🧠 Fuzzy Inference System (FIS)**: Intelligent adaptive filtering with automatic noise detection (Gaussian vs Salt & Pepper)
- **📊 Comprehensive Analysis**: Parameter distribution, robustness testing, and performance comparison
- **🔄 Multi-stage Processing**: Combines noise reduction with detail preservation
- **🎯 Noise-Type Aware**: Automatically detects and adapts to different noise characteristics

## Quick Start

### 1. Prerequisites

Ensure you have Python 3.7+ and `uv` installed.

#### Installing uv

**Linux/macOS:**
```bash
# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
# Install uv using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (All platforms):**
```bash
# Install via pip (if you prefer)
pip install uv
```

#### Project Setup

**Linux/macOS:**
```bash
# Clone the repository
git clone <repository-url>
cd SoftComputingProject

# Create and activate a virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies using uv
uv pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
# Clone the repository
git clone <repository-url>
cd SoftComputingProject

# Create and activate a virtual environment with uv
uv venv
.venv\Scripts\activate

# Install dependencies using uv
uv pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
# Clone the repository
git clone <repository-url>
cd SoftComputingProject

# Create and activate a virtual environment with uv
uv venv
.venv\Scripts\Activate.ps1

# Install dependencies using uv
uv pip install -r requirements.txt
```

**Required packages:**
- opencv-python >= 4.5.0
- numpy >= 1.20.0
- scikit-image >= 0.18.0
- scipy >= 1.7.0
- matplotlib

### 2. Data Setup

Organize your image dataset in the following structure:

```
data/
├── images/
│   ├── test/      # Test images for final evaluation
│   ├── train/     # Training images for parameter optimization
│   └── val/       # Validation images for robustness testing
```

**💡 Note:** The system works with clean images and automatically adds artificial noise (Gaussian or Salt & Pepper) for denoising evaluation.

### 3. Run the System

**Linux/macOS:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Run the main application
cd src
python main.py
```

**Windows (Command Prompt):**
```cmd
# Make sure virtual environment is activated
.venv\Scripts\activate

# Run the main application
cd src
python main.py
```

**Windows (PowerShell):**
```powershell
# Make sure virtual environment is activated
.venv\Scripts\Activate.ps1

# Run the main application
cd src
python main.py
```

The system will guide you through an interactive configuration process.

## 📁 Project Structure

```
SoftComputingProject/
├── src/                          # 🔧 Source code
│   ├── main.py                   # Main application entry point
│   ├── config.py                 # Interactive configuration system
│   ├── image_processing.py       # Fuzzy-based denoising algorithms
│   ├── genetic_algorithm.py      # GA optimization engine
│   ├── ga_analyzer.py           # Comprehensive analysis framework
│   ├── metrics.py               # Quality assessment metrics (SSIM, PSNR)
│   └── utils.py                 # Utility functions and image handling
├── data/                        # 📂 Dataset directory
│   └── images/                  # Clean images for denoising analysis
│       ├── test/                # Final testing images
│       ├── train/               # Parameter optimization images
│       └── val/                 # Validation images
├── results/                     # 📈 Output directory (auto-created)
│   ├── *_fuzzy_enhanced.png     # GA-optimized fuzzy denoising results
│   ├── *_simple_enhanced.png    # Simple denoising results  
│   ├── *_noisy.png              # Artificially corrupted input images
│   ├── *_comparison.png         # Side-by-side comparisons
│   └── SUMMARY_*_comparison.png # Best/worst case summaries
├── docs/                        # 📚 Documentation
│   ├── Concept.md               # Comprehensive theoretical guide
│   └── Report.md                # Technical project report
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Configuration Options

The system provides an interactive configuration interface asking for:

### 1. Noise Configuration
- **Gaussian noise only**: Standard sensor noise simulation
- **Salt & Pepper noise only**: Dead pixel simulation  
- **Both noise types**: Comprehensive evaluation

### 2. Genetic Algorithm Parameters
- **Population size**: 10-100 (recommended: 20-30)
- **Generations**: 5-50 (recommended: 15-25)

### 3. Fitness Metrics
- **SSIM weight**: Structural similarity importance (0.0-1.0)
- **PSNR weight**: Automatically calculated as (1 - SSIM weight)

### 4. Processing Limits
- **Test image count**: Limit for faster evaluation (optional)

## How It Works

### Automatic Noise Detection
```
Input Image → Noise Characteristics Analysis → Fuzzy Rule Set Selection
```

### Phase 1: Training (Parameter Learning)
```
Training Images → Add Noise → GA Optimization (Noise-Type Aware) → Extract Optimal Parameters
```

### Phase 2: Validation (Robustness Testing)
```
Validation Images → Test Fixed vs Individual Parameters → Robustness Score
```

### Phase 3: Testing (Final Comparison)
```
Test Images → GA-Fuzzy vs Simple Denoising → Performance Analysis
```

## 📊 Key Features

### Genetic Algorithm Optimization
- **Chromosome encoding**: `{kernel_size: [3,5,7], alpha: [0.2,0.9]}`
- **Noise-type aware**: Different parameter ranges for Gaussian vs Salt & Pepper noise
- **Fitness evaluation**: Combined SSIM + PSNR metrics
- **Advanced operators**: Crossover, mutation with adaptive rates

### Fuzzy Inference System
- **Automatic Noise Detection**: Distinguishes between Gaussian and Salt & Pepper noise
- **Rule Set A (Salt & Pepper)**: Median filtering with morphological cleanup
- **Rule Set B (Gaussian)**: Bilateral filtering with adaptive enhancement
- **Low Alpha (0.2-0.4)**: Detail preservation mode
- **Medium Alpha (0.4-0.7)**: Balanced denoising
- **High Alpha (0.7-0.9)**: Aggressive noise reduction

### Comprehensive Analysis
- **Parameter Distribution**: Analyze optimal parameters across datasets
- **Robustness Testing**: Find parameters that work across different images
- **Performance Comparison**: GA-Fuzzy vs Simple denoising evaluation
- **Visual Results**: Side-by-side comparison images

## 🎮 Usage Example

**Linux/macOS:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the application
cd src
python main.py
```

**Windows:**
```cmd
# Activate virtual environment
.venv\Scripts\activate

# Run the application
cd src
python main.py
```

**Sample interaction:**
```
GA-OPTIMIZED FUZZY INFERENCE SYSTEM
Image Denoising Configuration
============================================================

1. Noise Configuration:
Available options:
  1. Gaussian noise only
  2. Salt & Pepper noise only
  3. Both noise types
Choose noise type [1]: 3

2. Genetic Algorithm Configuration:
Population size (10-100) [20]: 25
Number of generations (5-50) [20]: 20

3. Combined Fitness Metric (SSIM + PSNR):
SSIM weight (0.0-1.0) [0.7]: 0.65

Start comprehensive analysis? (y/n) [y]: y
```

## 📈 Expected Results

### Performance Improvements
- **Gaussian Noise**: SSIM +0.02 to +0.05 (2-5% better), PSNR +1 to +3 dB
- **Salt & Pepper Noise**: Dramatic improvement with specialized processing
- **Overall Success Rate**: 70-90% of images show improvement over simple methods
- **Adaptive Processing**: System automatically selects optimal strategy based on noise type

### Output Files
- **Enhanced Images**: `{filename}_{noise}_fuzzy_enhanced.png`
- **Comparison Images**: Side-by-side Original|Noisy|GA-Fuzzy|Simple
- **Summary Reports**: Best and worst case performance analysis
- **Analysis Logs**: Detailed CSV reports with statistics

## 📚 Theoretical Background

For detailed explanations of the algorithms and concepts, see:
- **[docs/Concept.md](docs/Concept.md)**: Comprehensive guide to GA, Fuzzy Logic, and Image Denoising theory
- **[docs/Report.md](docs/Report.md)**: Technical project report with results and analysis

## 🔬 Scientific Contribution

This project demonstrates:

1. **Novel Integration**: GA + Fuzzy Logic with automatic noise type detection
2. **Adaptive Processing**: Image and noise-specific parameter tuning
3. **Multi-objective Optimization**: Balancing SSIM and PSNR metrics
4. **Comprehensive Evaluation**: Statistical and visual performance analysis
5. **No Training Required**: Direct optimization on target images
6. **Intelligent Noise Handling**: Specialized algorithms for different noise types

## Applications

- 🏥 **Medical Image Enhancement**: X-rays, MRI, CT scans
- 🛰️ **Satellite Image Processing**: Remote sensing data improvement
- 📷 **Digital Photography**: Noise reduction in low-light conditions
- 🎥 **Video Processing**: Real-time denoising applications
- 🏭 **Industrial Quality Control**: Inspection image enhancement

## Key Advantages

- ✅ **No Training Phase**: Works immediately on any image
- ✅ **Adaptive**: Optimizes parameters per image characteristics
- ✅ **Intelligent**: Fuzzy rules provide explainable decisions
- ✅ **Noise-Type Aware**: Automatically detects and handles different noise types
- ✅ **Robust**: Excellent performance on both Gaussian and Salt & Pepper noise
- ✅ **Comprehensive**: Complete analysis and comparison framework
- ✅ **Extensible**: Easy to add new fuzzy rules or GA operators

## 🚀 Recent Improvements

### Major Enhancement: Noise-Aware Processing
- **Automatic Detection**: System distinguishes between Gaussian and Salt & Pepper noise using statistical analysis
- **Specialized Processing**: Different fuzzy rule sets for optimal handling of each noise type
- **Improved Performance**: Significant improvements in both noise types through targeted approaches
- **Smart GA**: Genetic algorithm now uses noise-specific parameter ranges for better optimization

### Latest Improvements: Conservative & Analysis-Based Approach
- **Empirical Optimization**: Parameter ranges updated based on real performance analysis
- **Conservative Enhancement**: Reduced over-processing to minimize artifacts
- **Kernel Preferences**: Gaussian favors kernel=7, Salt & Pepper favors kernel=3
- **Alpha Tuning**: More conservative ranges (Gaussian: 0.2-0.6, SP: 0.4-0.8)
- **Simplified Logic**: Reduced complexity for better reliability

## 🔧 Technical Implementation Details

### Noise Detection Algorithm
The system automatically identifies noise type using:
- **Impulse Ratio Analysis**: Counts pixels with extreme values (< 20 or > 235)  
- **Variance Analysis**: Calculates local image variance patterns  
- **Conservative Threshold**: If impulse ratio > 12%, classifies as Salt & Pepper noise

### Fuzzy Rule Sets

#### Rule Set A: Salt & Pepper Noise (Specialized)
```
IF impulse_noise_detected THEN
    IF alpha <= 0.6 THEN use standard_median_filter
    IF alpha > 0.6 THEN use median + morphological_cleanup (kernel >= 5 only)
    enhancement = NONE (avoid artifact amplification)
```

#### Rule Set B: Gaussian Noise (Conservative)
```
IF gaussian_noise_detected THEN
    IF alpha <= 0.4 THEN use bilateral_filter
    IF alpha <= 0.7 THEN use bilateral + median_blend (max 40% median)
    IF alpha > 0.7 THEN use median + light_bilateral_smoothing
    enhancement = conservative_unsharp_masking (only if variance > 100)
```

### Genetic Algorithm Adaptations
- **Gaussian GA**: Kernel weights [0.1, 0.3, 0.6] (favoring kernel=7), Alpha range [0.2, 0.6]  
- **Salt & Pepper GA**: Kernel weights [0.8, 0.15, 0.05] (favoring kernel=3), Alpha range [0.4, 0.8]
- **Conservative Approach**: Reduced parameter ranges based on empirical analysis
- **Mutation**: Noise-type specific parameter bounds with conservative enhancement

## 🔧 Troubleshooting

### Common Issues

1. **No images found**: Ensure images are in `data/images/test/` directory
2. **Memory errors**: Reduce population size or number of test images
3. **Slow performance**: Decrease generations or use smaller image sizes
4. **Poor results**: Try different SSIM/PSNR weight combinations

### Performance Tips

- Start with 10-20 test images for initial evaluation
- Use population size 20-30 and 15-20 generations for good results
- SSIM weight 0.6-0.8 generally provides best perceptual quality
- **Salt & Pepper Noise**: System uses conservative alpha (0.4-0.8) and strongly favors kernel=3
- **Gaussian Noise**: System favors larger kernels (kernel=7) and conservative alpha (0.2-0.6) 
- **Conservative Enhancement**: Minimal enhancement reduces artifact amplification

## 🙋‍♂️ Support

For issues or questions:
1. Check the comprehensive documentation in `docs/Concept.md`
2. Read the technical report in `docs/Report.md`
3. Verify your dataset structure matches the expected format
4. Ensure all dependencies are correctly installed

---

**🎓 Educational Note**: This project showcases soft computing techniques as an alternative to deep learning approaches, demonstrating how evolutionary algorithms and fuzzy logic can solve complex image processing problems without requiring large datasets or training phases.