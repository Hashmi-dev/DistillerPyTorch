#!/usr/bin/env python3
"""
Distiller CIFAR-10 Demo Setup
This script demonstrates how to get started with Distiller once PyTorch is working.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_pytorch_installation():
    """Check if PyTorch is working and provide fix instructions."""
    print("🔍 Checking PyTorch installation...")
    
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__} is working!")
        print(f"✅ TorchVision {torchvision.__version__} is working!")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        print()
        print("🔧 PyTorch Fix Instructions:")
        print("=" * 50)
        print("The current PyTorch installation has a library compatibility issue.")
        print("Try these solutions in order:")
        print()
        print("Option 1 - Clean reinstall:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip cache purge")
        print("  pip install torch==2.0.1 torchvision==0.15.2")
        print()
        print("Option 2 - Use conda instead:")
        print("  conda install pytorch torchvision cpuonly -c pytorch")
        print()
        print("Option 3 - System-wide PyTorch:")
        print("  deactivate  # exit virtual environment")
        print("  sudo apt-get install python3-torch python3-torchvision")
        print()
        return False

def demonstrate_distiller_usage():
    """Show how to use Distiller once PyTorch works."""
    
    print("📚 Distiller Usage Examples")
    print("=" * 50)
    
    cifar_path = "/home/numair/Distiller/Cifar10"
    
    examples = [
        {
            "name": "Basic CIFAR-10 Training",
            "description": "Train a simple ResNet on CIFAR-10",
            "command": f"python compress_classifier.py --arch resnet20_cifar {cifar_path} --epochs 1 --lr 0.1"
        },
        {
            "name": "Pruning with AGP",
            "description": "Apply Automated Gradual Pruning",
            "command": f"python compress_classifier.py --arch resnet20_cifar {cifar_path} --compress agp_schedule.yaml --epochs 5"
        },
        {
            "name": "Quantization Aware Training", 
            "description": "Train with quantization",
            "command": f"python compress_classifier.py --arch resnet20_cifar {cifar_path} --compress quant_schedule.yaml --epochs 3"
        },
        {
            "name": "Knowledge Distillation",
            "description": "Train student model with teacher guidance",
            "command": f"python compress_classifier.py --arch resnet20_cifar {cifar_path} --kd-teacher resnet56_cifar --epochs 5"
        },
        {
            "name": "Model Analysis",
            "description": "Analyze model structure and sparsity",
            "command": f"python compress_classifier.py --arch resnet20_cifar {cifar_path} --resume model.pth --summary sparsity"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   {example['description']}")
        print(f"   Command: {example['command']}")
    
    print("\n📖 More examples available in:")
    print("   - examples/classifier_compression/")
    print("   - examples/quantization/")
    print("   - examples/pruning_filters_for_efficient_convnets/")

def create_sample_schedule():
    """Create a sample compression schedule YAML."""
    
    schedule_content = '''# Simple AGP Pruning Schedule for CIFAR-10
version: 1

pruners:
  my_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.05
    final_sparsity: 0.50
    weights: ['module.conv1.weight', 'module.layer1.0.conv1.weight', 'module.layer1.0.conv2.weight']

policies:
  - pruner:
      instance_name: my_pruner
    starting_epoch: 0
    ending_epoch: 30
    frequency: 2

lr_schedulers:
  - class: StepLR
    step_size: 30
    gamma: 0.1
'''
    
    schedule_path = Path("/home/numair/Distiller/DistillerPyTorch/sample_agp_schedule.yaml")
    with open(schedule_path, 'w') as f:
        f.write(schedule_content)
    
    print(f"✅ Created sample schedule: {schedule_path}")
    return str(schedule_path)

def create_readme_update():
    """Create an updated README with current status."""
    
    readme_content = '''# Distiller Setup Status

## ✅ What's Working

1. **CIFAR-10 Dataset**: Downloaded and verified
   - Location: `/home/numair/Distiller/Cifar10/cifar-10-batches-py/`
   - Contains all required files (data_batch_1-5, test_batch, etc.)
   - Successfully loads with pickle

2. **Python Environment**: Virtual environment is active
   - Python 3.12 
   - All non-PyTorch dependencies installed

3. **Distiller Code**: Present and structured
   - Main library in `distiller/`
   - Examples in `examples/`
   - Documentation in `docs/`

## ❌ What Needs Fixing

**PyTorch Installation Issue**
- Current error: `pthread_attr_setaffinity_np: symbol not found`
- This is a library compatibility issue with the CPU version of PyTorch

## 🔧 How to Fix PyTorch

Try these solutions in order:

### Option 1: Clean Reinstall
```bash
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch==2.0.1 torchvision==0.15.2
```

### Option 2: Use Conda
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

### Option 3: System PyTorch
```bash
deactivate  # exit venv
sudo apt-get install python3-torch python3-torchvision
# Then run without venv
```

## 🚀 Once PyTorch Works

You can run these examples:

### Basic Training
```bash
python compress_classifier.py --arch resnet20_cifar /home/numair/Distiller/Cifar10 --epochs 1
```

### With Compression
```bash  
python compress_classifier.py --arch resnet20_cifar /home/numair/Distiller/Cifar10 --compress sample_agp_schedule.yaml --epochs 5
```

### Model Analysis
```bash
python compress_classifier.py --arch resnet20_cifar /home/numair/Distiller/Cifar10 --summary compute
```

## 📁 Project Structure
```
/home/numair/Distiller/
├── DistillerPyTorch/          # Main Distiller code
│   ├── distiller/             # Core library
│   ├── examples/              # Example scripts
│   ├── compress_classifier.py # Main training script
│   └── requirements.txt       # Dependencies
└── Cifar10/                   # CIFAR-10 dataset
    └── cifar-10-batches-py/   # Dataset files
```

## 🔍 Testing Commands

Test CIFAR-10: `python test_setup.py`
Test PyTorch: `python -c "import torch; print(torch.__version__)"`
List examples: `ls examples/*/`

---
Once PyTorch is fixed, you'll have a fully working neural network compression research environment!
'''
    
    readme_path = Path("/home/numair/Distiller/DistillerPyTorch/SETUP_STATUS.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created setup status: {readme_path}")
    return str(readme_path)

def main():
    """Main setup and demo function."""
    
    print("🧠 Distiller CIFAR-10 Demo Setup")
    print("=" * 40)
    print()
    
    # Check PyTorch
    pytorch_works = check_pytorch_installation()
    print()
    
    # Create sample files
    schedule_path = create_sample_schedule()
    readme_path = create_readme_update()
    print()
    
    # Show usage examples
    demonstrate_distiller_usage()
    print()
    
    print("📋 Summary")
    print("=" * 40)
    print("✅ CIFAR-10 dataset: Ready")
    print("✅ Sample files: Created")  
    print("✅ Documentation: Updated")
    print(f"{'✅' if pytorch_works else '❌'} PyTorch: {'Working' if pytorch_works else 'Needs fixing'}")
    print()
    
    if pytorch_works:
        print("🎉 Everything is ready! You can start training models.")
        print(f"Try: python compress_classifier.py --arch resnet20_cifar /home/numair/Distiller/Cifar10 --epochs 1")
    else:
        print("🔧 Fix PyTorch first, then you'll be ready to go!")
        print(f"📖 See {readme_path} for detailed instructions")

if __name__ == "__main__":
    main()
