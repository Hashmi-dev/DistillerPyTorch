# 🚀 THE COMPLETE DISTILLER GUIDE: How Schedule YAML Controls Everything

## 🎯 The Big Picture: What is Distiller?

**Distiller is a neural network compression scheduler that sits on top of PyTorch training.** The key insight is that **everything is controlled by a single YAML schedule file**.

```
Normal PyTorch Training:  Model → Training Loop → Trained Model
With Distiller:          Model → Training Loop + Schedule YAML → Compressed Model
```

## 🔑 The Core Insight: It's All About the Schedule YAML

The schedule YAML is the **control center** that defines:
- **WHAT** to compress (which layers/weights)
- **WHEN** to compress (epochs, frequency)  
- **HOW** to compress (algorithm, sparsity levels)
- **POST-PROCESSING** (remove filters, optimize structure)

**Change the YAML → Change everything about compression**

## 📊 Real Results: ResNet20 Example

From our analysis of a real Distiller schedule:

| Metric | Baseline | Compressed | Improvement |
|--------|----------|------------|-------------|
| **Accuracy** | 91.78% | 91.34% | **-0.44%** (tiny loss!) |
| **Model Size** | 270,896 params | 120,000 params | **2.3x smaller** |
| **Speed** | 40.8M MACs | 30.7M MACs | **1.3x faster** |
| **Sparsity** | 0% | 46.3% | **46% of weights removed** |

**🎯 Result: Significant compression with minimal accuracy loss!**

## 🧩 The Three Components of Schedule YAML

### 1. 🔧 **PRUNERS** - Define HOW to compress

```yaml
pruners:
  my_pruner:
    class: 'AutomatedGradualPruner'    # The algorithm
    initial_sparsity: 0.05             # Start: 5% weights removed
    final_sparsity: 0.80               # End: 80% weights removed
    weights: ['conv1.weight', 'fc.weight']  # Target layers
```

**What this means:**
- Gradually increase sparsity from 5% to 80% over training
- Only compress the specified layers
- Use magnitude-based pruning (remove smallest weights)

### 2. ⏰ **POLICIES** - Define WHEN to apply compression  

```yaml
policies:
  - pruner:
      instance_name: my_pruner
    starting_epoch: 5      # Start after 5 epochs of normal training
    ending_epoch: 50       # Complete compression by epoch 50
    frequency: 2           # Apply every 2 epochs
```

**What this means:**
- Let model train normally for 5 epochs first
- Apply compression every 2 epochs from epoch 5 to 50
- Total: 23 compression applications over 45 epochs

### 3. 🔧 **EXTENSIONS** - Define post-processing

```yaml
extensions:
  net_thinner:
    class: 'FilterRemover'             # Actually remove pruned filters
    thinning_func_str: remove_filters  # The removal function
```

**What this means:**
- Don't just zero out weights, physically remove them
- Creates actually smaller model files
- Enables real speedup during inference

## 🧠 Key Concepts That Make It Work

### 📈 Gradual Pruning
Instead of removing 80% of weights at once (would break the model):
```
Week 1: Remove 5% of smallest weights  → Model adapts
Week 2: Remove 10% of smallest weights → Model adapts  
...
Week N: Remove 80% of smallest weights → Final compressed model
```

### 🎯 Layer-Specific Strategy
Different layers get different treatment:
- **First layers**: No compression (critical for accuracy)
- **Middle layers**: High compression (can handle it)  
- **Last layers**: Careful compression (affects final output)

### 🏗️ Structured vs Unstructured
- **Unstructured**: Remove individual weights (high compression)
- **Structured**: Remove entire filters/channels (actually faster)

## 🔄 The Complete Workflow

1. **📁 Start**: You have a PyTorch model
2. **📝 Design**: Write a schedule YAML file
3. **🚀 Run**: `python compress_classifier.py --schedule your_schedule.yaml`
4. **🔄 Process**: Distiller hooks into PyTorch training:
   - Normal training for initial epochs
   - Gradual compression at scheduled intervals
   - Model continues training to adapt
5. **🎯 Result**: Compressed model with maintained accuracy

## 📚 Sample Schedules for Different Use Cases

### 🔰 Beginner: Simple Pruning (60% compression)
```yaml
version: 1
pruners:
  simple_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.0
    final_sparsity: 0.6
    weights: ['conv1.weight', 'conv2.weight', 'fc.weight']

policies:
  - pruner:
      instance_name: simple_pruner
    starting_epoch: 5
    ending_epoch: 30
    frequency: 1
```

### 🎯 Intermediate: Multi-Layer Strategy
```yaml
version: 1
pruners:
  conv_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.05
    final_sparsity: 0.75              # Conv layers: 75% compression
    weights: ['features.0.weight', 'features.3.weight']
  
  fc_pruner:
    class: 'AutomatedGradualPruner'
    initial_sparsity: 0.1
    final_sparsity: 0.9               # FC layers: 90% compression
    weights: ['classifier.0.weight', 'classifier.3.weight']

policies:
  - pruner:
      instance_name: conv_pruner
    starting_epoch: 10
    ending_epoch: 80
    frequency: 2
  - pruner:
      instance_name: fc_pruner  
    starting_epoch: 15
    ending_epoch: 70
    frequency: 3
```

### 🚀 Advanced: Structured Pruning (Actually Faster)
```yaml
version: 1
pruners:
  filter_pruner:
    class: 'L1RankedStructureParameterPruner'
    group_type: 'Filters'
    desired_sparsity: 0.5             # Remove 50% of filters
    weights: ['layer1.0.conv1.weight', 'layer2.0.conv1.weight']

policies:
  - pruner:
      instance_name: filter_pruner
    epochs: [20, 40, 60]              # Apply at specific epochs

extensions:
  net_thinner:
    class: 'FilterRemover'
    thinning_func_str: remove_filters
```

## 🎯 How to Create Your Own Schedule

### Step 1: Understand Your Model
```python
# Get layer names from your model
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"Layer: {name}, Shape: {param.shape}")
```

### Step 2: Design Compression Strategy
- **Conservative start**: 50-60% sparsity
- **Protect critical layers**: First/last layers  
- **Timeline**: Longer gradual compression = better accuracy
- **Test incrementally**: Start small, increase compression

### Step 3: Choose Techniques
- **Element-wise pruning**: High compression ratios
- **Structured pruning**: Real speedup benefits
- **Hybrid approach**: Different techniques for different layers

## 📊 Expected Results by Compression Level

| Sparsity | Model Size | Accuracy Impact | Use Case |
|----------|------------|-----------------|----------|
| **50%** | 2x smaller | Minimal (<1%) | Conservative |
| **70%** | 3.3x smaller | Small (1-2%) | Balanced |
| **90%** | 10x smaller | Moderate (2-5%) | Aggressive |

## 🚀 Next Steps

1. **📖 Study Examples**: Look at `examples/*/schedule.yaml` files
2. **🧪 Start Simple**: Use basic AGP pruning first
3. **📊 Monitor Results**: Track accuracy vs compression tradeoffs
4. **🔧 Add Thinning**: Use FilterRemover for real speedup
5. **🎯 Iterate**: Gradually increase compression based on results

## 🔑 Key Takeaway

**The schedule YAML is everything in Distiller.** It's the "recipe" that controls all aspects of compression. Master the YAML format, understand the key concepts (gradual pruning, layer selection, timing), and you can compress any PyTorch model effectively.

The beauty is in the flexibility: same Distiller code, different YAML file = completely different compression strategy!
