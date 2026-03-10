#!/usr/bin/env python3
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
