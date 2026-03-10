# How Distiller Works: Complete Workflow & Schedule YAML Guide

## The Big Picture: Distiller's Role in Neural Network Compression

Distiller is a **compression scheduler** that sits on top of PyTorch training. Here's how it works:

```
1. PyTorch Model Training (Normal)
   ↓
2. Distiller Schedule YAML (The Magic Happens Here)
   ↓
3. Compressed Model Output
```

## The Core Workflow

### Step 1: Start with a PyTorch Model
- You have a normal PyTorch model (ResNet, VGG, etc.)
- The model can be pre-trained or training from scratch

### Step 2: Define Compression Schedule (YAML File)
- **This is the heart of Distiller** - the schedule YAML file
- It defines WHAT to compress, WHEN to compress it, and HOW MUCH
- Different compression techniques: pruning, quantization, regularization

### Step 3: Training with Compression
- Distiller intercepts the normal PyTorch training loop
- At specific epochs/iterations, it applies compression according to your schedule
- The model continues training with compressed weights

### Step 4: Final Compressed Model
- You get a model that maintains accuracy but is smaller/faster
- Can be deployed for inference with significant resource savings

## The Schedule YAML: The Control Center

The schedule YAML file is where you specify:
- **What layers to compress** (which weights/filters)
- **When to apply compression** (epochs, frequency)  
- **How much to compress** (sparsity levels, quantization bits)
- **Which compression technique** (pruning method, quantization scheme)

## Main Components of a Schedule YAML

### 1. **Pruners** - Define HOW to compress
```yaml
pruners:
  my_pruner:
    class: 'AutomatedGradualPruner'  # The compression algorithm
    initial_sparsity: 0.05          # Start with 5% sparsity
    final_sparsity: 0.80            # End with 80% sparsity
    weights: ['conv1.weight', 'fc.weight']  # Which layers to compress
```

### 2. **Policies** - Define WHEN to apply compression
```yaml
policies:
  - pruner:
      instance_name: my_pruner
    starting_epoch: 5    # Start compression at epoch 5
    ending_epoch: 50     # Stop compression at epoch 50
    frequency: 2         # Apply every 2 epochs
```

### 3. **Extensions** (Optional) - Post-processing
```yaml
extensions:
  net_thinner:
    class: 'FilterRemover'  # Actually remove pruned filters
    thinning_func_str: remove_filters
```

## Types of Compression Techniques

### 1. **Pruning** (Remove weights/filters)
- **Element-wise**: Remove individual weights (fine-grained)
- **Structured**: Remove entire filters/channels (coarse-grained)
- **Gradual**: Slowly increase sparsity during training

### 2. **Quantization** (Reduce precision)
- **Post-training**: Quantize after training
- **Quantization-aware**: Train with quantization in mind

### 3. **Regularization** (Encourage sparsity)
- **L1/L2**: Add penalty terms to loss function
- **Group**: Encourage structured sparsity

## Real Example Breakdown

Let's look at a real schedule from the ResNet20 example:

```yaml
version: 1
pruners:
  layer_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.05
    final_sparsity: 0.80
    weights: [
      'module.layer1.0.conv1.weight',
      'module.layer1.0.conv2.weight',
      'module.layer2.0.conv1.weight'
    ]

policies:
  - pruner:
      instance_name: layer_pruner
    starting_epoch: 0
    ending_epoch: 120
    frequency: 1
```

**What this does:**
1. **Targets specific layers**: Only compresses the specified conv layers
2. **Gradual compression**: Starts at 5% sparsity, gradually increases to 80%
3. **Timeline**: Applies compression every epoch from 0 to 120
4. **Result**: 80% of weights in those layers become zero (removed)

## Input/Output Analysis

### **Inputs to Distiller:**
1. **PyTorch Model**: Your neural network architecture
2. **Training Data**: Dataset for training/fine-tuning
3. **Schedule YAML**: The compression recipe (most important!)
4. **Training Parameters**: Learning rate, epochs, etc.

### **Process:**
- Distiller hooks into PyTorch's training loop
- At scheduled intervals, applies compression transformations
- Model continues training to recover from compression

### **Outputs from Distiller:**
1. **Compressed Model**: Smaller model with maintained accuracy
2. **Compression Statistics**: Sparsity reports, parameter counts
3. **Performance Metrics**: Accuracy, inference speed, model size

## Key Distiller Classes Explained

### AutomatedGradualPruner (AGP)
- **Most popular pruning method**
- Gradually increases sparsity from initial to final value
- Based on magnitude-based pruning (removes smallest weights)

### SensitivityPruner
- Uses pre-computed sensitivity analysis
- Different sparsity levels for different layers
- More fine-tuned control

### L1RankedStructureParameterPruner
- Removes entire structures (filters, channels)
- Based on L1-norm ranking
- Creates actually smaller models

## The Magic of Gradual Pruning

Instead of removing 80% of weights at once (which would break the model), AGP:
1. **Week 1**: Remove 5% of smallest weights
2. **Week 2**: Remove 10% of smallest weights
3. **...**
4. **Week N**: Remove 80% of smallest weights

The model has time to adapt and maintain accuracy!

## Creating Your Own Schedule

### Step 1: Choose Your Target Layers
```python
# Get layer names from your model
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"Layer: {name}, Shape: {param.shape}")
```

### Step 2: Define Compression Strategy
- **High sparsity**: 70-90% for less important layers
- **Low sparsity**: 20-50% for critical layers (first/last layers)
- **Timeline**: Longer gradual pruning = better accuracy retention

### Step 3: Test and Iterate
- Start conservative (lower sparsity)
- Monitor accuracy during training
- Adjust sparsity levels based on results

## Common Patterns

### Pattern 1: Conservative Pruning
```yaml
initial_sparsity: 0.0
final_sparsity: 0.50
```

### Pattern 2: Aggressive Pruning
```yaml
initial_sparsity: 0.1
final_sparsity: 0.90
```

### Pattern 3: Layer-Specific Pruning
```yaml
# Different pruners for different layer types
conv_pruner:
  final_sparsity: 0.80
fc_pruner:
  final_sparsity: 0.95
```

## The Schedule YAML is Everything

The beauty of Distiller is that **everything is controlled by the YAML schedule**:
- Want different compression? Change the YAML
- Want different timeline? Change the YAML  
- Want to target different layers? Change the YAML

This makes it incredibly flexible and reproducible!
