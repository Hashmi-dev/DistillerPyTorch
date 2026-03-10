#!/usr/bin/env python3
"""
Download CIFAR-10 dataset for Distiller
This script downloads CIFAR-10 to the specified directory and sets up the environment.
"""

import os
import sys
import urllib.request
import tarfile
import pickle
import shutil
from pathlib import Path

def download_cifar10(data_dir="/home/numair/Distiller/Cifar10"):
    """Download CIFAR-10 dataset if it doesn't exist."""
    
    # Create directory if it doesn't exist
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cifar10_dir = data_dir / "cifar-10-batches-py"
    
    # Check if CIFAR-10 already exists
    if cifar10_dir.exists() and len(list(cifar10_dir.glob("*"))) > 5:
        print(f"CIFAR-10 dataset already exists at {cifar10_dir}")
        return str(cifar10_dir)
    
    print("Downloading CIFAR-10 dataset...")
    
    # CIFAR-10 URL
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = data_dir / "cifar-10-python.tar.gz"
    
    try:
        # Download the dataset
        print("Downloading from:", url)
        urllib.request.urlretrieve(url, tar_path)
        print(f"Downloaded to {tar_path}")
        
        # Extract the dataset
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        # Remove the tar file to save space
        tar_path.unlink()
        print("Extraction complete!")
        
        # Verify the extraction
        if cifar10_dir.exists():
            files = list(cifar10_dir.glob("*"))
            print(f"CIFAR-10 dataset extracted successfully!")
            print(f"Files in dataset: {[f.name for f in files]}")
        else:
            print("Error: Extraction failed!")
            return None
            
        return str(cifar10_dir)
        
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")
        if tar_path.exists():
            tar_path.unlink()
        return None

def verify_cifar10(data_dir="/home/numair/Distiller/Cifar10"):
    """Verify CIFAR-10 dataset integrity."""
    cifar10_dir = Path(data_dir) / "cifar-10-batches-py"
    
    expected_files = [
        "batches.meta",
        "data_batch_1",
        "data_batch_2", 
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch"
    ]
    
    if not cifar10_dir.exists():
        return False
        
    for file in expected_files:
        if not (cifar10_dir / file).exists():
            print(f"Missing file: {file}")
            return False
    
    print("CIFAR-10 dataset verification successful!")
    return True

def test_torch_import():
    """Test if PyTorch can be imported and used."""
    try:
        import torch
        import torchvision
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"✓ TorchVision {torchvision.__version__} imported successfully")
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        print(f"✓ Basic tensor operations work: {x.shape}")
        return True
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

def setup_environment():
    """Set up the environment for running Distiller."""
    print("Setting up Distiller environment...")
    
    # Test PyTorch
    torch_works = test_torch_import()
    
    # Download CIFAR-10
    cifar_path = download_cifar10()
    
    if cifar_path and verify_cifar10():
        print(f"\n✓ CIFAR-10 dataset ready at: {cifar_path}")
        print(f"✓ You can now run Distiller experiments with CIFAR-10")
        print(f"\nExample command:")
        print(f"python compress_classifier.py --arch resnet20_cifar /home/numair/Distiller/Cifar10 --epochs 1 --lr 0.1")
        return True
    else:
        print("\n✗ Failed to set up CIFAR-10 dataset")
        return False

if __name__ == "__main__":
    setup_environment()
