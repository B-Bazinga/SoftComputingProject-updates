# GA-Optimized Fuzzy Inference System for Image Denoising

A comprehensive **Soft Computing** approach that combines **Genetic Algorithms** with **Fuzzy Inference Systems** for intelligent image denoising.

## Overview

This project demonstrates advanced soft computing techniques for image enhancement:

- **🧬 Genetic Algorithm (GA)**: Evolutionary optimization of denoising parameters
- **🧠 Fuzzy Inference System (FIS)**: Intelligent adaptive filtering based on local image properties
- **📊 Comprehensive Analysis**: Parameter distribution, robustness testing, and performance comparison
- **🔄 Multi-stage Processing**: Combines noise reduction with detail preservation

## Quick Start

### 1. Prerequisites

Ensure you have Python 3.7+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```

**Required packages:**
- opencv-python >= 4.5.0
- numpy >= 1.20.0
- scikit-image >= 0.18.0
- scipy >= 1.7.0

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

```bash
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
│   ├── *_noisy.png              # Noisy input images
│   ├── *_comparison.png         # Side-by-side comparisons
│   └── SUMMARY_*_comparison.png # Best/worst case summaries
├── docs/                        # 📚 Documentation
│   └── Concepts.md              # Comprehensive theoretical guide
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

### Phase 1: Training (Parameter Learning)
```
Training Images → Add Noise → GA Optimization → Extract Optimal Parameters
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
- **Chromosome encoding**: `{kernel_size: [3,5,7], alpha: [0.2,0.7]}`
- **Fitness evaluation**: Combined SSIM + PSNR metrics
- **Advanced operators**: Crossover, mutation with adaptive rates

### Fuzzy Inference System
- **Low Alpha (0.2-0.3)**: Detail preservation mode
- **Medium Alpha (0.3-0.5)**: Balanced denoising
- **High Alpha (0.5-0.7)**: Aggressive noise reduction

### Comprehensive Analysis
- **Parameter Distribution**: Analyze optimal parameters across datasets
- **Robustness Testing**: Find parameters that work across different images
- **Performance Comparison**: GA-Fuzzy vs Simple denoising evaluation
- **Visual Results**: Side-by-side comparison images

## 🎮 Usage Example

```bash
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
Choose noise type [1]: 1
Gaussian sigma (1-50) [25]: 30

2. Genetic Algorithm Configuration:
Population size (10-100) [20]: 25
Number of generations (5-50) [20]: 20

3. Combined Fitness Metric (SSIM + PSNR):
SSIM weight (0.0-1.0) [0.7]: 0.65

4. Testing Configuration:
Number of test images (leave empty for all): 10

Start comprehensive analysis? (y/n) [y]: y
```

## 📈 Expected Results

### Performance Improvements
- **SSIM Improvement**: +0.02 to +0.05 (2-5% better structural similarity)
- **PSNR Improvement**: +1 to +3 dB (better noise reduction)
- **Success Rate**: 70-90% of images show improvement over simple methods

### Output Files
- **Enhanced Images**: `{filename}_{noise}_fuzzy_enhanced.png`
- **Comparison Images**: Side-by-side Original|Noisy|GA-Fuzzy|Simple
- **Summary Reports**: Best and worst case performance analysis
- **Analysis Logs**: Detailed CSV reports with statistics

## 📚 Theoretical Background

For detailed explanations of the algorithms and concepts, see:
- **[docs/Concepts.md](docs/Concepts.md)**: Comprehensive guide to GA, Fuzzy Logic, and Image Denoising theory

## 🔬 Scientific Contribution

This project demonstrates:

1. **Novel Integration**: GA + Fuzzy Logic for parameter optimization
2. **Adaptive Processing**: Image-specific parameter tuning
3. **Multi-objective Optimization**: Balancing SSIM and PSNR metrics
4. **Comprehensive Evaluation**: Statistical and visual performance analysis
5. **No Training Required**: Direct optimization on target images

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
- ✅ **Robust**: Handles different noise types effectively
- ✅ **Comprehensive**: Complete analysis and comparison framework
- ✅ **Extensible**: Easy to add new fuzzy rules or GA operators

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

## 🙋‍♂️ Support

For issues or questions:
1. Check the comprehensive documentation in `docs/Concepts.md`
2. Verify your dataset structure matches the expected format
3. Ensure all dependencies are correctly installed

---

**🎓 Educational Note**: This project showcases soft computing techniques as an alternative to deep learning approaches, demonstrating how evolutionary algorithms and fuzzy logic can solve complex image processing problems without requiring large datasets or training phases.