#!/usr/bin/env python3
"""
PyTorch compatibility fix for musl libc systems
This script attempts to install a working version of PyTorch
"""

import os
import subprocess
import sys
from pathlib import Path

def install_pytorch_musl():
    """Install PyTorch compatible with musl libc."""
    
    print("🔧 Attempting to fix PyTorch compatibility for musl libc...")
    
    # Method 1: Try installing from different sources
    commands_to_try = [
        # Try installing a version built for broader compatibility
        ["pip", "install", "torch==1.13.1+cpu", "torchvision==0.14.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu", "--force-reinstall"],
        
        # Try CPU-only version without specific CPU extensions
        ["pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu", "--no-deps", "--force-reinstall"],
    ]
    
    for i, cmd in enumerate(commands_to_try, 1):
        print(f"\n🔄 Trying method {i}: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Installation successful!")
            return test_pytorch()
        except subprocess.CalledProcessError as e:
            print(f"❌ Method {i} failed: {e}")
            print(f"Error output: {e.stderr}")
    
    return False

def test_pytorch():
    """Test if PyTorch works."""
    try:
        print("\n🧪 Testing PyTorch import...")
        import torch
        import torchvision
        
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"✅ TorchVision {torchvision.__version__} imported successfully")
        
        # Test basic operations
        x = torch.randn(2, 3)
        print(f"✅ Basic tensor operations work: {x.shape}")
        
        # Test CIFAR-10 loading with torchvision
        from torchvision import datasets, transforms
        print("✅ TorchVision datasets available")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def create_pytorch_alternative():
    """Create a minimal PyTorch alternative for demonstration."""
    
    alternative_content = '''#!/usr/bin/env python3
"""
Minimal PyTorch-like interface for basic neural network operations
This provides a basic framework when full PyTorch isn't available
"""

import numpy as np
from pathlib import Path
import pickle

class Tensor:
    """Simple tensor class mimicking PyTorch tensors."""
    
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape
    
    def __str__(self):
        return f"Tensor({self.data})"
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)

def randn(*shape):
    """Create random tensor."""
    return Tensor(np.random.randn(*shape))

def zeros(*shape):
    """Create zero tensor."""
    return Tensor(np.zeros(shape))

class CIFAR10Dataset:
    """Simple CIFAR-10 dataset loader."""
    
    def __init__(self, root="/home/numair/Distiller/Cifar10", train=True):
        self.root = Path(root)
        self.train = train
        self.data, self.targets = self._load_data()
    
    def _load_data(self):
        """Load CIFAR-10 data."""
        cifar_dir = self.root / "cifar-10-batches-py"
        
        data_list = []
        targets_list = []
        
        if self.train:
            # Load training batches
            for i in range(1, 6):
                batch_file = cifar_dir / f"data_batch_{i}"
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                data_list.append(batch[b'data'])
                targets_list.extend(batch[b'labels'])
        else:
            # Load test batch
            test_file = cifar_dir / "test_batch"
            with open(test_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            data_list.append(batch[b'data'])
            targets_list.extend(batch[b'labels'])
        
        # Combine all data
        data = np.vstack(data_list)
        targets = np.array(targets_list)
        
        # Reshape to image format
        data = data.reshape(-1, 3, 32, 32)
        
        return data, targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SimpleNet:
    """Simple neural network for demonstration."""
    
    def __init__(self, num_classes=10):
        self.conv1_weight = randn(32, 3, 3, 3)
        self.conv1_bias = randn(32)
        self.conv2_weight = randn(64, 32, 3, 3)  
        self.conv2_bias = randn(64)
        self.fc_weight = randn(num_classes, 64 * 8 * 8)
        self.fc_bias = randn(num_classes)
    
    def forward(self, x):
        # This is a simplified forward pass
        print(f"Input shape: {x.shape}")
        return f"Output logits (shape would be: [{x.shape[0]}, 10])"
    
    def parameters(self):
        """Return model parameters."""
        return {
            'conv1.weight': self.conv1_weight,
            'conv1.bias': self.conv1_bias,
            'conv2.weight': self.conv2_weight,
            'conv2.bias': self.conv2_bias,
            'fc.weight': self.fc_weight,
            'fc.bias': self.fc_bias
        }

def demonstrate_distiller_concepts():
    """Demonstrate core Distiller concepts."""
    print("\\n🚀 Distiller Concepts Demonstration")
    print("=" * 50)
    
    # Create dataset
    print("📊 Loading CIFAR-10 dataset...")
    try:
        dataset = CIFAR10Dataset(train=True)
        print(f"✅ Training set: {len(dataset)} samples")
        
        # Show sample
        sample_data, sample_label = dataset[0]
        print(f"   Sample shape: {sample_data.shape}")
        print(f"   Sample label: {sample_label}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # Create model
    print("\\n🧠 Creating neural network...")
    model = SimpleNet()
    params = model.parameters()
    
    total_params = sum(param.data.size for param in params.values())
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Demonstrate pruning
    print("\\n✂️ Structured Pruning Simulation:")
    conv1_weight = params['conv1.weight'].data
    print(f"   Original conv1 shape: {conv1_weight.shape}")
    
    # Filter-wise pruning (remove least important filters)
    filter_norms = np.linalg.norm(conv1_weight.reshape(32, -1), axis=1)
    keep_filters = 24  # Keep 24 out of 32 filters
    keep_indices = np.argsort(filter_norms)[-keep_filters:]
    
    pruned_weight = conv1_weight[keep_indices]
    print(f"   Pruned conv1 shape: {pruned_weight.shape}")
    print(f"   Filters removed: {32 - keep_filters}")
    print(f"   Parameter reduction: {(1 - pruned_weight.size / conv1_weight.size):.1%}")
    
    # Demonstrate quantization
    print("\\n⚖️ Quantization Simulation:")
    fc_weight = params['fc.weight'].data
    print(f"   Original FC weights: {fc_weight.dtype} ({fc_weight.nbytes:,} bytes)")
    
    # 8-bit quantization
    weight_min, weight_max = fc_weight.min(), fc_weight.max()
    scale = (weight_max - weight_min) / 255
    zero_point = -weight_min / scale
    quantized_weight = np.clip(np.round(fc_weight / scale + zero_point), 0, 255).astype(np.uint8)
    
    print(f"   Quantized FC weights: {quantized_weight.dtype} ({quantized_weight.nbytes:,} bytes)")
    print(f"   Memory reduction: {fc_weight.nbytes / quantized_weight.nbytes:.1f}x")
    print(f"   Quantization scale: {scale:.6f}")
    
    print("\\n🎯 Distiller Features Demonstrated:")
    print("   ✅ Dataset loading (CIFAR-10)")
    print("   ✅ Model parameter inspection") 
    print("   ✅ Structured pruning (filter removal)")
    print("   ✅ Quantization (8-bit)")
    print("   ✅ Memory and parameter analysis")

if __name__ == "__main__":
    demonstrate_distiller_concepts()
'''
    
    alt_file = Path("/home/numair/Distiller/DistillerPyTorch/pytorch_alternative.py")
    with open(alt_file, 'w') as f:
        f.write(alternative_content)
    
    os.chmod(alt_file, 0o755)
    print(f"✅ Created PyTorch alternative at: {alt_file}")
    return alt_file

if __name__ == "__main__":
    # Try to install working PyTorch
    if not install_pytorch_musl():
        print("\n⚠️ PyTorch installation failed. Creating alternative...")
        alt_file = create_pytorch_alternative()
        
        print(f"\n📝 You can run the alternative demo with:")
        print(f"   python {alt_file}")
        
        print(f"\n🔧 To fix PyTorch compatibility:")
        print("1. Consider using a glibc-based system/container")
        print("2. Use conda instead of pip")
        print("3. Compile PyTorch from source")
    else:
        print("\n🎉 PyTorch is now working!")
        print("You can run the original Distiller examples.")
