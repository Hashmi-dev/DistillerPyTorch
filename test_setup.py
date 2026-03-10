#!/usr/bin/env python3
"""
Test script to verify Distiller setup without PyTorch dependencies
"""

import os
import sys
import pickle
from pathlib import Path

def test_cifar10_loading():
    """Test loading CIFAR-10 data without PyTorch."""
    cifar_path = Path("/home/numair/Distiller/Cifar10/cifar-10-batches-py")
    
    if not cifar_path.exists():
        print("❌ CIFAR-10 dataset not found!")
        return False
    
    try:
        # Load a batch to test
        batch_file = cifar_path / "data_batch_1" 
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
        
        print("✅ CIFAR-10 dataset loaded successfully!")
        print(f"   Batch shape: {batch_data[b'data'].shape}")
        print(f"   Number of samples: {len(batch_data[b'labels'])}")
        
        # Load metadata
        meta_file = cifar_path / "batches.meta"
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
        
        label_names = [name.decode('utf-8') for name in meta[b'label_names']]
        print(f"   Classes: {label_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CIFAR-10: {e}")
        return False

def test_distiller_import():
    """Test importing Distiller components without PyTorch."""
    try:
        # Add the distiller path
        sys.path.insert(0, '/home/numair/Distiller/DistillerPyTorch')
        
        # Test basic imports that don't require torch
        print("Testing Distiller imports...")
        
        # Test configuration
        import distiller.config
        print("✅ distiller.config imported")
        
        # Test utilities that might work
        import distiller.utils
        print("✅ distiller.utils imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Distiller import failed: {e}")
        return False

def create_simple_training_script():
    """Create a simple script to test the setup."""
    script_content = '''#!/usr/bin/env python3
"""
Simple CIFAR-10 classifier using Distiller - Basic version
This is a simplified version that should work once PyTorch is fixed.
"""

import os
import sys
import argparse

# Add the Distiller path
sys.path.insert(0, '/home/numair/Distiller/DistillerPyTorch')

def main():
    parser = argparse.ArgumentParser(description='Simple CIFAR-10 Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', default='resnet18', help='model architecture')
    parser.add_argument('--epochs', default=1, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    
    args = parser.parse_args()
    
    print(f"Starting training with:")
    print(f"  Architecture: {args.arch}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    
    try:
        import torch
        import torchvision
        print("✅ PyTorch successfully imported!")
        
        # Import Distiller
        import distiller
        import distiller.apputils as apputils
        print("✅ Distiller successfully imported!")
        
        # You can add actual training code here once PyTorch works
        print("🚀 Ready to start training!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please fix PyTorch installation first.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
'''
    
    with open('/home/numair/Distiller/DistillerPyTorch/simple_train.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('/home/numair/Distiller/DistillerPyTorch/simple_train.py', 0o755)
    print("✅ Created simple_train.py script")

def main():
    print("=== Distiller Setup Verification ===")
    print()
    
    # Test CIFAR-10
    print("1. Testing CIFAR-10 dataset...")
    cifar_ok = test_cifar10_loading()
    print()
    
    # Test Distiller
    print("2. Testing Distiller imports...")
    distiller_ok = test_distiller_import()
    print()
    
    # Create training script
    print("3. Creating training script...")
    create_simple_training_script()
    print()
    
    print("=== Summary ===")
    print(f"📊 CIFAR-10 dataset: {'✅ Ready' if cifar_ok else '❌ Not ready'}")
    print(f"🧠 Distiller library: {'✅ Working' if distiller_ok else '❌ Issues'}")
    print(f"🐍 PyTorch: ❌ Needs fixing (library compatibility issue)")
    print()
    
    if cifar_ok:
        print("🎉 CIFAR-10 is ready for use!")
        print("📁 Dataset location: /home/numair/Distiller/Cifar10/cifar-10-batches-py")
        print()
        print("Next steps to fix PyTorch:")
        print("1. Try: pip uninstall torch torchvision -y")
        print("2. Then: pip install torch==2.1.0 torchvision==0.16.0")
        print("3. Or use system PyTorch if available")
        print()
        print("Once PyTorch works, you can run:")
        print("python simple_train.py /home/numair/Distiller/Cifar10 --arch resnet18 --epochs 1")

if __name__ == "__main__":
    main()
