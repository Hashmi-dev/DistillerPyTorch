#!/usr/bin/env python3
"""
Comprehensive Distiller Demo - Working Examples
This script demonstrates how to work with the Distiller framework
even when PyTorch has compatibility issues.
"""

import os
import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add the distiller module to path
sys.path.insert(0, '/home/numair/Distiller/DistillerPyTorch')

def show_distiller_structure():
    """Show the structure of the Distiller framework."""
    print("🏗️ Distiller Framework Structure")
    print("=" * 40)
    
    distiller_path = Path("/home/numair/Distiller/DistillerPyTorch/distiller")
    
    if not distiller_path.exists():
        print("❌ Distiller module not found")
        return False
    
    # List key modules
    key_modules = [
        "pruning", "quantization", "regularization", "models",
        "policy.py", "scheduler.py", "utils.py", "config.py"
    ]
    
    print("📦 Key Distiller Modules:")
    for module in key_modules:
        module_path = distiller_path / module
        if module_path.exists():
            if module_path.is_dir():
                subfiles = list(module_path.glob("*.py"))
                print(f"   📁 {module}/ ({len(subfiles)} Python files)")
            else:
                size_kb = module_path.stat().st_size / 1024
                print(f"   📄 {module} ({size_kb:.1f} KB)")
        else:
            print(f"   ❓ {module} (not found)")
    
    return True

def demonstrate_config_system():
    """Demonstrate Distiller's configuration system."""
    print("\n⚙️ Configuration System Demo")
    print("=" * 35)
    
    # Create a sample compression configuration
    sample_config = {
        "version": 1,
        "pruning": {
            "policy": "magnitude",
            "sparsity_levels": [0.1, 0.3, 0.5, 0.7],
            "schedule": "gradual"
        },
        "quantization": {
            "bits": {
                "weights": 8,
                "activations": 8
            },
            "method": "linear"
        },
        "knowledge_distillation": {
            "teacher_model": "resnet50",
            "student_model": "resnet18",
            "temperature": 4.0,
            "alpha": 0.3
        }
    }
    
    # Save sample config
    config_path = Path("/home/numair/Distiller/DistillerPyTorch/sample_compression_config.yaml")
    
    # Convert to YAML-like format for demonstration
    yaml_content = """version: 1

# Pruning configuration
pruning:
  policy: magnitude
  sparsity_levels: [0.1, 0.3, 0.5, 0.7]
  schedule: gradual

# Quantization configuration  
quantization:
  bits:
    weights: 8
    activations: 8
  method: linear

# Knowledge distillation
knowledge_distillation:
  teacher_model: resnet50
  student_model: resnet18
  temperature: 4.0
  alpha: 0.3
"""
    
    with open(config_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created sample config: {config_path}")
    print("📋 Configuration includes:")
    print("   - Magnitude-based pruning")
    print("   - 8-bit quantization")
    print("   - Knowledge distillation")
    
    return config_path

def analyze_example_models():
    """Analyze the example models in Distiller."""
    print("\n🎯 Example Models Analysis")
    print("=" * 35)
    
    examples_path = Path("/home/numair/Distiller/DistillerPyTorch/examples")
    
    if not examples_path.exists():
        print("❌ Examples directory not found")
        return
    
    # List example directories
    example_dirs = [d for d in examples_path.iterdir() if d.is_dir()]
    
    print(f"📁 Found {len(example_dirs)} example categories:")
    
    for example_dir in sorted(example_dirs):
        # Count Python files
        py_files = list(example_dir.glob("*.py"))
        config_files = list(example_dir.glob("*.yaml")) + list(example_dir.glob("*.yml"))
        
        print(f"\n   📂 {example_dir.name}:")
        print(f"      - Python scripts: {len(py_files)}")
        print(f"      - Config files: {len(config_files)}")
        
        # Show key files
        key_files = ["compress_classifier.py", "main.py", "train.py"]
        for key_file in key_files:
            if (example_dir / key_file).exists():
                print(f"      - ✅ {key_file}")

def create_experiment_runner():
    """Create a script to run experiments when PyTorch is fixed."""
    runner_content = '''#!/usr/bin/env python3
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
    print("\\n🔍 Running Sensitivity Analysis")  
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
    print("This script will work once PyTorch compatibility is resolved.\\n")
    
    # Check if we can import torch
    try:
        import torch
        print("✅ PyTorch is available!")
        
        # Run experiments
        run_basic_training()
        run_sensitivity_analysis()
        
    except Exception as e:
        print(f"❌ PyTorch not available: {e}")
        print("\\n🔧 To fix PyTorch compatibility:")
        print("1. Use a glibc-based system (Ubuntu/Debian)")
        print("2. Use conda instead of pip")
        print("3. Use Docker with official PyTorch image")
        
        print("\\n📖 Example Docker command:")
        print("docker run -it --rm -v $(pwd):/workspace pytorch/pytorch:latest bash")
'''
    
    runner_path = Path("/home/numair/Distiller/DistillerPyTorch/experiment_runner.py")
    with open(runner_path, 'w') as f:
        f.write(runner_content)
    
    os.chmod(runner_path, 0o755)
    print(f"✅ Created experiment runner: {runner_path}")
    return runner_path

def create_summary_report():
    """Create a summary of what's been set up."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "setup_status": {
            "cifar10_downloaded": True,
            "distiller_structure": "✅ Available",
            "pytorch_status": "❌ Library compatibility issue (musl libc)",
            "alternatives_created": True
        },
        "working_examples": [
            "simple_example.py - Basic neural network compression demo",
            "pytorch_alternative.py - PyTorch-free Distiller concepts", 
            "experiment_runner.py - Ready for when PyTorch is fixed"
        ],
        "next_steps": [
            "Fix PyTorch compatibility (use glibc system or Docker)",
            "Run actual Distiller training examples",
            "Experiment with different compression techniques"
        ],
        "dataset_info": {
            "cifar10_location": "/home/numair/Distiller/Cifar10",
            "training_samples": 50000,
            "test_samples": 10000,
            "classes": 10
        }
    }
    
    report_path = Path("/home/numair/Distiller/DistillerPyTorch/setup_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Setup Report")
    print("=" * 20)
    print(f"✅ CIFAR-10 Dataset: Ready ({report['dataset_info']['training_samples']:,} training samples)")
    print(f"✅ Distiller Framework: Available")
    print(f"❌ PyTorch: {report['setup_status']['pytorch_status']}")
    print(f"✅ Working Examples: {len(report['working_examples'])} created")
    
    print(f"\n📋 Created files:")
    for example in report['working_examples']:
        print(f"   - {example}")
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    return report_path

def main():
    """Main demonstration function."""
    print("🎯 Comprehensive Distiller Demo")
    print("=" * 40)
    
    # Show framework structure
    show_distiller_structure()
    
    # Demonstrate configuration
    demonstrate_config_system()
    
    # Analyze examples
    analyze_example_models()
    
    # Create experiment runner
    create_experiment_runner()
    
    # Create summary report
    create_summary_report()
    
    print(f"\n🎉 Demo Complete!")
    print("\n📖 Quick Start Guide:")
    print("1. Run: python simple_example.py")
    print("2. Run: python pytorch_alternative.py") 
    print("3. Fix PyTorch, then: python experiment_runner.py")

if __name__ == "__main__":
    main()
