"""Utility functions for image loading and noise generation."""
import cv2
import numpy as np
import os

def load_image(path):
    """Load image in grayscale."""
    # Handle .mat files (MATLAB format)
    if path.lower().endswith('.mat'):
        try:
            from scipy.io import loadmat
            mat_data = loadmat(path)
            # Try common MATLAB variable names for images
            for key in ['image', 'img', 'data', 'I']:
                if key in mat_data:
                    img = mat_data[key]
                    break
            else:
                # If no common keys, use the first non-metadata key
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if keys:
                    img = mat_data[keys[0]]
                else:
                    print(f'Warning: No image data found in {path}')
                    return None
            
            # Convert to proper image format
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            # Ensure grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            return img
        except ImportError:
            print("Warning: scipy not installed. Cannot load .mat files. Install with: pip install scipy")
            return None
        except Exception as e:
            print(f'Warning: Error loading .mat file {path}: {e}')
            return None
    else:
        # Handle regular image files
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Warning: Image not found or unreadable: {path}')
            return None
        return img

def add_gaussian_noise(img, sigma=25):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(img, amount=0.1):
    """Add salt and pepper noise to image."""
    if img is None:
        raise ValueError("Input image is None")
    
    noisy = np.copy(img)
    num_salt = int(np.ceil(amount * img.size * 0.5))
    num_pepper = int(np.ceil(amount * img.size * 0.5))
    
    # Salt noise - ensure coordinates are within bounds
    if num_salt > 0:
        coords = tuple([np.random.randint(0, i, num_salt) for i in img.shape])
        noisy[coords] = 255
    
    # Pepper noise - ensure coordinates are within bounds
    if num_pepper > 0:
        coords = tuple([np.random.randint(0, i, num_pepper) for i in img.shape])
        noisy[coords] = 0
    
    return noisy

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_image_files(directory):
    """Get list of image files from directory."""
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.mat')
    return [f for f in os.listdir(directory) if f.lower().endswith(extensions)]