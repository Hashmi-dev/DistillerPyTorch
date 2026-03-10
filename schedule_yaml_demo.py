#!/usr/bin/env python3
"""
DISTILLER SCHEDULE YAML DEMO
==========================

This script demonstrates how the Distiller scheduler YAML file controls
neural network compression. It shows the complete workflow and breaks down
exactly what happens at each step.
"""

import yaml
import numpy as np
from pathlib import Path

def analyze_schedule_yaml(yaml_file):
    """Analyze a Distiller schedule YAML and explain what it does."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING SCHEDULE: {yaml_file}")
    print(f"{'='*60}")
    
    try:
        with open(yaml_file, 'r') as f:
            schedule = yaml.safe_load(f)
        
        print(f"\n🔍 SCHEDULE VERSION: {schedule.get('version', 'Not specified')}")
        
        # Analyze Pruners
        if 'pruners' in schedule:
            print(f"\n📊 COMPRESSION ALGORITHMS (PRUNERS):")
            print("-" * 40)
            
            for pruner_name, config in schedule['pruners'].items():
                print(f"\nPruner: '{pruner_name}'")
                print(f"  📈 Algorithm: {config.get('class', 'Unknown')}")
                print(f"  🎯 Initial Sparsity: {config.get('initial_sparsity', 0):.1%}")
                print(f"  🎯 Final Sparsity: {config.get('final_sparsity', 0):.1%}")
                
                if 'weights' in config:
                    print(f"  🎯 Target Layers ({len(config['weights'])} layers):")
                    for i, weight in enumerate(config['weights'][:5]):  # Show first 5
                        print(f"      {i+1}. {weight}")
                    if len(config['weights']) > 5:
                        print(f"      ... and {len(config['weights'])-5} more layers")
                
                # Calculate compression impact
                initial = config.get('initial_sparsity', 0)
                final = config.get('final_sparsity', 0)
                compression_ratio = 1 / (1 - final) if final < 1 else float('inf')
                print(f"  📦 Compression Impact:")
                print(f"      - Removes {final:.1%} of weights")
                print(f"      - Model size reduction: {compression_ratio:.1f}x")
                print(f"      - Memory savings: {final:.1%}")
        
        # Analyze Policies (When compression happens)
        if 'policies' in schedule:
            print(f"\n⏰ COMPRESSION TIMELINE (POLICIES):")
            print("-" * 40)
            
            for i, policy in enumerate(schedule['policies']):
                print(f"\nPolicy {i+1}:")
                
                if 'pruner' in policy:
                    pruner_info = policy['pruner']
                    print(f"  🔧 Uses Pruner: {pruner_info.get('instance_name', 'Unknown')}")
                
                start_epoch = policy.get('starting_epoch', 0)
                end_epoch = policy.get('ending_epoch', 'Unknown')
                frequency = policy.get('frequency', 1)
                
                print(f"  📅 Timeline: Epoch {start_epoch} → {end_epoch}")
                print(f"  🔄 Frequency: Every {frequency} epoch(s)")
                
                # Calculate total applications
                if isinstance(end_epoch, (int, float)) and isinstance(start_epoch, (int, float)):
                    total_epochs = end_epoch - start_epoch + 1
                    applications = max(1, total_epochs // frequency)
                    print(f"  📊 Total Applications: ~{applications} times")
        
        # Analyze Extensions (Post-processing)
        if 'extensions' in schedule:
            print(f"\n🔧 POST-PROCESSING (EXTENSIONS):")
            print("-" * 40)
            
            for ext_name, config in schedule['extensions'].items():
                print(f"\nExtension: '{ext_name}'")
                print(f"  🛠️  Type: {config.get('class', 'Unknown')}")
                print(f"  ⚙️  Function: {config.get('thinning_func_str', 'Default')}")
                
                if 'FilterRemover' in str(config.get('class', '')):
                    print(f"  📦 Effect: Actually removes pruned filters (smaller model)")
                elif 'Thinning' in str(config.get('class', '')):
                    print(f"  📦 Effect: Optimizes model structure")
        
        # Summary
        print(f"\n🎯 SCHEDULE SUMMARY:")
        print("-" * 20)
        
        total_pruners = len(schedule.get('pruners', {}))
        total_policies = len(schedule.get('policies', []))
        total_extensions = len(schedule.get('extensions', {}))
        
        print(f"📊 Total Compression Algorithms: {total_pruners}")
        print(f"⏰ Total Policies/Timelines: {total_policies}")
        print(f"🔧 Total Post-Processing Steps: {total_extensions}")
        
        return schedule
        
    except Exception as e:
        print(f"❌ Error analyzing schedule: {e}")
        return None

def demonstrate_compression_timeline():
    """Show how compression progresses over training epochs."""
    
    print(f"\n{'='*60}")
    print("COMPRESSION TIMELINE DEMONSTRATION")
    print(f"{'='*60}")
    
    # Simulate AutomatedGradualPruner progression
    initial_sparsity = 0.05  # 5%
    final_sparsity = 0.80    # 80%
    start_epoch = 0
    end_epoch = 50
    frequency = 2
    
    print(f"\n📊 Simulating AutomatedGradualPruner:")
    print(f"   Initial sparsity: {initial_sparsity:.1%}")
    print(f"   Final sparsity: {final_sparsity:.1%}")
    print(f"   Timeline: Epoch {start_epoch} to {end_epoch}")
    print(f"   Frequency: Every {frequency} epochs")
    
    print(f"\n⏰ COMPRESSION PROGRESSION:")
    print("-" * 40)
    print(f"{'Epoch':<8} {'Sparsity':<10} {'Weights Kept':<15} {'Model Size':<12}")
    print("-" * 40)
    
    for epoch in range(start_epoch, end_epoch + 1, frequency):
        # Calculate current sparsity (linear interpolation)
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * progress
        
        weights_kept = 1 - current_sparsity
        model_size_percent = weights_kept * 100
        
        print(f"{epoch:<8} {current_sparsity:<9.1%} {weights_kept:<14.1%} {model_size_percent:<11.1f}%")
    
    print("-" * 40)
    print(f"Final Result: Model is {1/(1-final_sparsity):.1f}x smaller!")

def create_sample_schedules():
    """Create sample schedule YAMLs for different use cases."""
    
    print(f"\n{'='*60}")
    print("CREATING SAMPLE SCHEDULE YAMLS")
    print(f"{'='*60}")
    
    # 1. Simple Pruning Schedule
    simple_schedule = {
        'version': 1,
        'pruners': {
            'simple_pruner': {
                'class': 'AutomatedGradualPruner',
                'initial_sparsity': 0.0,
                'final_sparsity': 0.6,
                'weights': [
                    'conv1.weight',
                    'conv2.weight', 
                    'fc.weight'
                ]
            }
        },
        'policies': [
            {
                'pruner': {
                    'instance_name': 'simple_pruner'
                },
                'starting_epoch': 5,
                'ending_epoch': 30,
                'frequency': 1
            }
        ]
    }
    
    # 2. Advanced Multi-Layer Schedule
    advanced_schedule = {
        'version': 1,
        'pruners': {
            'conv_pruner': {
                'class': 'AutomatedGradualPruner',
                'initial_sparsity': 0.05,
                'final_sparsity': 0.75,
                'weights': [
                    'features.0.weight',
                    'features.3.weight',
                    'features.6.weight'
                ]
            },
            'fc_pruner': {
                'class': 'AutomatedGradualPruner', 
                'initial_sparsity': 0.1,
                'final_sparsity': 0.9,
                'weights': [
                    'classifier.0.weight',
                    'classifier.3.weight'
                ]
            }
        },
        'policies': [
            {
                'pruner': {
                    'instance_name': 'conv_pruner'
                },
                'starting_epoch': 10,
                'ending_epoch': 80,
                'frequency': 2
            },
            {
                'pruner': {
                    'instance_name': 'fc_pruner'
                },
                'starting_epoch': 15,
                'ending_epoch': 70, 
                'frequency': 3
            }
        ],
        'extensions': {
            'net_thinner': {
                'class': 'FilterRemover',
                'thinning_func_str': 'remove_filters'
            }
        }
    }
    
    # 3. Structured Pruning Schedule
    structured_schedule = {
        'version': 1,
        'pruners': {
            'filter_pruner': {
                'class': 'L1RankedStructureParameterPruner',
                'group_type': 'Filters',
                'desired_sparsity': 0.5,
                'weights': [
                    'layer1.0.conv1.weight',
                    'layer1.0.conv2.weight',
                    'layer2.0.conv1.weight'
                ]
            }
        },
        'policies': [
            {
                'pruner': {
                    'instance_name': 'filter_pruner'
                },
                'epochs': [20, 40, 60]
            }
        ],
        'extensions': {
            'net_thinner': {
                'class': 'FilterRemover',
                'thinning_func_str': 'remove_filters'
            }
        }
    }
    
    schedules = {
        'simple_pruning_schedule.yaml': simple_schedule,
        'advanced_multi_layer_schedule.yaml': advanced_schedule, 
        'structured_pruning_schedule.yaml': structured_schedule
    }
    
    for filename, schedule in schedules.items():
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            yaml.dump(schedule, f, default_flow_style=False, indent=2)
        
        print(f"\n✅ Created: {filename}")
        print(f"   Purpose: {get_schedule_purpose(filename)}")
    
    return list(schedules.keys())

def get_schedule_purpose(filename):
    """Get the purpose description for each schedule type."""
    purposes = {
        'simple_pruning_schedule.yaml': 'Basic element-wise pruning (good for beginners)',
        'advanced_multi_layer_schedule.yaml': 'Different compression for conv vs FC layers',
        'structured_pruning_schedule.yaml': 'Remove entire filters (creates smaller models)'
    }
    return purposes.get(filename, 'Unknown purpose')

def main():
    """Main demonstration function."""
    
    print("🚀 DISTILLER SCHEDULE YAML COMPLETE DEMO")
    print("=" * 60)
    print("This demo shows how Distiller's schedule YAML controls everything!")
    
    # 1. Create sample schedules
    sample_files = create_sample_schedules()
    
    # 2. Analyze each schedule
    for yaml_file in sample_files:
        if Path(yaml_file).exists():
            analyze_schedule_yaml(yaml_file)
    
    # 3. Demonstrate compression timeline
    demonstrate_compression_timeline()
    
    # 4. Analyze real schedules from examples if available
    examples_dir = Path('examples')
    if examples_dir.exists():
        print(f"\n{'='*60}")
        print("ANALYZING REAL EXAMPLE SCHEDULES")
        print(f"{'='*60}")
        
        # Look for YAML files in examples
        yaml_files = list(examples_dir.rglob('*.yaml'))
        for yaml_file in yaml_files[:3]:  # Analyze first 3
            analyze_schedule_yaml(yaml_file)
    
    print(f"\n🎯 KEY TAKEAWAYS:")
    print("=" * 20)
    print("1. The YAML schedule controls EVERYTHING in Distiller")
    print("2. Pruners define HOW to compress (algorithm + targets)")
    print("3. Policies define WHEN to compress (timeline)")
    print("4. Extensions define post-processing (optimization)")
    print("5. Different schedules = different compression strategies")
    
    print(f"\n📚 NEXT STEPS:")
    print("1. Modify the sample YAML files to match your model")
    print("2. Experiment with different sparsity levels")
    print("3. Try different pruning algorithms")
    print("4. Run actual compression with: python compress_classifier.py --schedule your_schedule.yaml")

if __name__ == '__main__':
    main()
