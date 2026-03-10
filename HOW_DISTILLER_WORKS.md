# How Distiller Works: Complete Guide to Schedule YAML Files

## Overview: The Big Picture

Distiller integrates with PyTorch training to perform **neural network compression during training**. Here's the workflow:

1. **Start with a PyTorch model** (pre-trained or from scratch)
2. **Define a compression schedule** in a YAML file
3. **Train the model** while Distiller applies compression techniques
4. **Get a compressed model** that maintains accuracy but uses less resources

The **scheduler YAML** is the control center - it defines:
- **What** to compress (which layers, which weights)
- **How** to compress (pruning method, quantization, etc.)
- **When** to compress (which epochs, frequency)
- **How much** to compress (sparsity levels, compression ratios)

## The Core Workflow

```
Original Model → Distiller + Schedule YAML → Compressed Model
     ↓                      ↓                        ↓
  Full weights      Compression policies       Sparse/quantized weights
  100% compute      Applied during training    Reduced compute/memory
```

## Schedule YAML Structure: Anatomy

Every scheduler YAML has this basic structure:

```yaml
version: 1                    # Always 1

pruners:                      # Define compression algorithms
  my_pruner:
    class: 'AutomatedGradualPruner'
    # ... parameters ...

quantizers:                   # Define quantization (optional)
  my_quantizer:
    # ... parameters ...

policies:                     # Define WHEN to apply compression
  - pruner:
      instance_name: my_pruner
    starting_epoch: 0
    ending_epoch: 50
    frequency: 2

lr_schedulers:               # Optional: learning rate schedules
  # ... learning rate policies ...

extensions:                  # Optional: model transformations
  # ... thinning, knowledge distillation ...
```

## Key Components Explained

### 1. Pruners: The "What" and "How"

Pruners define compression algorithms. Main types:

#### AutomatedGradualPruner (AGP) - Most Common
```yaml
pruners:
  my_agp_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.05      # Start with 5% weights pruned
    final_sparsity: 0.80        # End with 80% weights pruned
    weights: [                  # Which layer weights to prune
      'module.conv1.weight',
      'module.layer1.0.conv1.weight',
      'module.fc.weight'
    ]
```

**What AGP does:**
- Gradually increases sparsity from `initial_sparsity` to `final_sparsity`
- Uses magnitude-based pruning (removes smallest weights)
- Follows a polynomial decay schedule

#### Structure Pruning (Filters, Channels)
```yaml
pruners:
  filter_pruner:
    class: 'L1RankedStructureParameterPruner'
    group_type: Filters         # Remove entire filters
    desired_sparsity: 0.30      # Remove 30% of filters
    weights: ['module.conv1.weight']
```

#### Sensitivity Pruning
```yaml
pruners:
  sensitivity_pruner:
    class: 'SensitivityPruner'
    sensitivities:              # Layer-specific pruning ratios
      'module.conv1.weight': 0.40
      'module.conv2.weight': 0.70
      'module.fc.weight': 0.90
```

### 2. Policies: The "When"

Policies control **when** pruners are applied:

```yaml
policies:
  - pruner:
      instance_name: my_agp_pruner
    starting_epoch: 10          # Start pruning at epoch 10
    ending_epoch: 100           # Stop pruning at epoch 100
    frequency: 2                # Apply every 2 epochs
```

**Timeline example:**
- Epochs 0-9: Normal training
- Epoch 10: Apply pruning (5% sparsity)
- Epoch 12: Apply pruning (~10% sparsity)
- ...
- Epoch 100: Apply final pruning (80% sparsity)
- Epochs 101+: Only training, no more pruning

### 3. Multiple Pruners: Hybrid Compression

You can combine different compression techniques:

```yaml
pruners:
  # Element-wise pruning for conv layers
  conv_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.05
    final_sparsity: 0.70
    weights: ['module.conv1.weight', 'module.conv2.weight']
  
  # Row pruning for fully connected layer
  fc_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.10
    final_sparsity: 0.90
    weights: ['module.fc.weight']

policies:
  - pruner:
      instance_name: conv_pruner
    starting_epoch: 0
    frequency: 2
  - pruner:
      instance_name: fc_pruner
    starting_epoch: 10
    frequency: 3
```

## Real Example: ResNet20 on CIFAR-10

Let's analyze a real schedule from the attachments:

```yaml
version: 1
pruners:
  # Filter pruning to reduce compute
  filter_pruner:
    class: 'L1RankedStructureParameterPruner'
    group_type: Filters
    desired_sparsity: 0.50
    weights: [
      'module.layer1.0.conv1.weight',
      'module.layer2.0.conv1.weight'
    ]
  
  # Fine-grained pruning for memory reduction
  fine_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.05
    final_sparsity: 0.85
    weights: [
      'module.layer1.0.conv2.weight',
      'module.layer2.0.conv2.weight',
      'module.layer3.0.conv1.weight'
    ]

policies:
  # Start filter pruning early
  - pruner:
      instance_name: filter_pruner
    starting_epoch: 0
    ending_epoch: 40
    frequency: 10
  
  # Fine pruning throughout training
  - pruner:
      instance_name: fine_pruner
    starting_epoch: 0
    ending_epoch: 120
    frequency: 2
```

**What happens:**
1. **Filter pruning**: Removes 50% of filters from specific layers (reduces computation)
2. **Fine-grained pruning**: Gradually prunes 85% of individual weights (reduces memory)
3. **Training continues**: Model learns to work with compressed structure

## Inputs, Outputs, and What They Mean

### Inputs to Distiller:
1. **PyTorch model** (pre-trained or untrained)
2. **Training dataset** (CIFAR-10, ImageNet, etc.)
3. **Schedule YAML** (defines compression strategy)
4. **Training hyperparameters** (learning rate, epochs, etc.)

### Outputs from Distiller:
1. **Compressed model checkpoint** (`.pth.tar` file)
2. **Training logs** (loss, accuracy curves)
3. **Compression statistics** (sparsity tables, MACs reduction)
4. **Performance metrics** (accuracy, model size, inference speed)

### Understanding the Statistics

When Distiller runs, it outputs tables like this:
```
+----+-------------------------------------+----------------+---------------+----------------+------------+------------+----------+----------+----------+------------+
|    | Name                                | Shape          |   NNZ (dense) |   NNZ (sparse) |   Cols (%) |   Rows (%) |   Ch (%) |   2D (%) |   3D (%) |   Fine (%) |
|----+-------------------------------------+----------------+---------------+----------------+------------+------------+----------+----------+----------+------------|
|  0 | module.conv1.weight                 | (16, 3, 3, 3)  |           432 |            432 |    0.00000 |    0.00000 |  0.00000 |  0.00000 |  0.00000 |    0.00000 |
|  1 | module.layer1.0.conv1.weight        | (16, 16, 3, 3) |          2304 |            461 |    0.00000 |    0.00000 |  6.25000 | 80.01302 |  7.81250 |   79.99566 |
```

**Column meanings:**
- **NNZ (dense/sparse)**: Non-zero values before/after compression
- **Fine (%)**: Element-wise sparsity (% of individual weights pruned)
- **2D (%)**: 2D structure sparsity (% of filters/kernels pruned)
- **3D (%)**: 3D structure sparsity (% of channels pruned)
- **Cols/Rows (%)**: Column/row-wise sparsity

## How to Create Your Own Schedule

### Step 1: Choose Compression Strategy

**For compute reduction (faster inference):**
- Use **filter pruning** or **channel pruning**
- Target conv layers with many filters

**For memory reduction (smaller models):**
- Use **fine-grained (element-wise) pruning**
- Target large weight matrices (FC layers)

**For both:**
- Combine multiple techniques (hybrid approach)

### Step 2: Determine Layer Names

Run this to see your model's layer names:
```python
import torch
model = torch.load('your_model.pth')
for name, param in model.named_parameters():
    print(f"'{name}': {param.shape}")
```

### Step 3: Set Sparsity Levels

**Conservative (maintain accuracy):**
- Conv layers: 30-60% sparsity
- FC layers: 80-95% sparsity

**Aggressive (high compression):**
- Conv layers: 70-90% sparsity
- FC layers: 95-99% sparsity

### Step 4: Design the Timeline

**Gradual approach:**
- Start pruning early (epoch 0-10)
- End pruning at 70-80% of total epochs
- Use frequency 1-3 for fine control

**Rapid approach:**
- Start pruning at 20-30% of training
- End pruning at 80% of training
- Use higher frequency (5-10)

## Sample Schedule Template

```yaml
version: 1

pruners:
  # Replace with your layer names and desired sparsity
  main_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.05
    final_sparsity: 0.70        # Adjust based on your needs
    weights: [
      'module.layer1.weight',   # Replace with actual layer names
      'module.layer2.weight',
      # Add more layers...
    ]

policies:
  - pruner:
      instance_name: main_pruner
    starting_epoch: 0
    ending_epoch: 80            # Adjust based on total epochs
    frequency: 2                # Prune every 2 epochs

lr_schedulers:
  - class: StepLR
    step_size: 30
    gamma: 0.1
```

## Running with Your Schedule

```bash
# For CIFAR-10 with ResNet20
python compress_classifier.py \
    --arch resnet20_cifar \
    /path/to/cifar10 \
    --epochs 120 \
    --compress your_schedule.yaml \
    --lr 0.1 \
    -p 50

# For ImageNet with ResNet50
python compress_classifier.py \
    --arch resnet50 \
    --pretrained \
    /path/to/imagenet \
    --epochs 90 \
    --compress your_schedule.yaml \
    --lr 0.01 \
    -j 8
```

## Key Takeaways

1. **Schedule YAML is the brain** - it controls all compression
2. **Pruners define algorithms** - AGP is most versatile
3. **Policies define timing** - when and how often to compress
4. **Multiple techniques can be combined** - hybrid approaches work best
5. **Results show sparsity statistics** - understand what each metric means
6. **Compression happens during training** - not after training

The beauty of Distiller is that it seamlessly integrates compression into the training loop, allowing the model to adapt to the compression as it learns!
