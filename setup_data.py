"""Setup script to create proper data directory structure."""
import os

def setup_data_directories():
    """Create data directory structure if it doesn't exist."""
    directories = [
        'data/images/train',
        'data/images/val', 
        'data/images/test',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created/verified: {directory}/")
    
    # Create README files explaining what goes in each directory
    readme_content = {
        'data/images/train/README.md': """# Training Images
Place clean images here for parameter optimization training.
The system will add artificial noise and optimize GA parameters on these images.
""",
        'data/images/val/README.md': """# Validation Images  
Place clean images here for parameter validation and robustness testing.
Used to test if parameters learned from training work on different images.
""",
        'data/images/test/README.md': """# Test Images
Place clean images here for final performance evaluation.
The system will compare GA-Fuzzy vs Simple denoising on these images.
""",
        'results/README.md': """# Results Directory
This directory contains all output files:
- *_fuzzy_enhanced.png: GA-optimized fuzzy denoising results
- *_simple_enhanced.png: Simple denoising results  
- *_noisy.png: Artificially corrupted input images
- *_comparison.png: Side-by-side comparisons
- SUMMARY_*_comparison.png: Best/worst case analysis
"""
    }
    
    for filepath, content in readme_content.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Created: {filepath}")

if __name__ == '__main__':
    print("Setting up data directories...")
    setup_data_directories()
    print("\n🎉 Setup complete!")
    print("\n📁 Next steps:")
    print("1. Add clean images to data/images/train/, data/images/val/, and data/images/test/")
    print("2. Run: cd src && python main.py")
