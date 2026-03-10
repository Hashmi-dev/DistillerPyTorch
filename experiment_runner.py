#!/usr/bin/env python3
"""
Distiller Experiment Runner
Run this when PyTorch compatibility is resolved.
"""

import subprocess
import sys
from pathlib import Path

def run_basic_training():
    """Run basic CIFAR-10 training example."""
    
    print("🚀 Running Basic CIFAR-10 Training")
    print("=" * 40)
    
    # Change to examples directory
    examples_dir = Path("/home/numair/Distiller/DistillerPyTorch/examples/classifier_compression")
    
    if not examples_dir.exists():
        print("❌ Classifier compression example not found")
        return False
    
    # Command from the README
    cmd = [
        "python", "compress_classifier.py",
        "--arch", "simplenet_cifar",
        "/home/numair/Distiller/Cifar10",
        "-p", "30",
        "-j", "1", 
        "--lr", "0.01",
        "--epochs", "2"  # Just 2 epochs for demo
    ]
    
    print(f"📋 Command: {' '.join(cmd)}")
    print(f"📂 Working dir: {examples_dir}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=examples_dir,
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("📊 Output:")
            print(result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ Training failed:")
            print(result.stderr[-500:])
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Error running training: {e}")
    
    return result.returncode == 0

def run_sensitivity_analysis():
    """Run sensitivity analysis example."""
    print("\n🔍 Running Sensitivity Analysis")  
    print("=" * 35)
    
    # This would run sensitivity analysis on a pre-trained model
    examples_dir = Path("/home/numair/Distiller/DistillerPyTorch/examples/sensitivity-analysis")
    
    if not examples_dir.exists():
        print("❌ Sensitivity analysis example not found")
        return False
    
    print("📋 This example analyzes layer sensitivity to pruning")
    print("📊 It helps determine which layers can be pruned more aggressively")
    
    # Command would be something like:
    cmd_info = """
    python sensitivity_analysis.py --arch resnet20_cifar 
           --resume path/to/checkpoint.pth.tar 
           /home/numair/Distiller/Cifar10
    """
    
    print(f"📋 Typical command: {cmd_info.strip()}")
    return True

if __name__ == "__main__":
    print("🔧 Distiller Experiment Runner")
    print("This script will work once PyTorch compatibility is resolved.\n")
    
    # Check if we can import torch
    try:
        import torch
        print("✅ PyTorch is available!")
        
        # Run experiments
        run_basic_training()
        run_sensitivity_analysis()
        
    except Exception as e:
        print(f"❌ PyTorch not available: {e}")
        print("\n🔧 To fix PyTorch compatibility:")
        print("1. Use a glibc-based system (Ubuntu/Debian)")
        print("2. Use conda instead of pip")
        print("3. Use Docker with official PyTorch image")
        
        print("\n📖 Example Docker command:")
        print("docker run -it --rm -v $(pwd):/workspace pytorch/pytorch:latest bash")
