import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

def load_classification_data(results_file="benchmark_classes_results.json"):
    """Load and validate classification benchmark data."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded data from {results_file}")
        print(f"📊 Data structure: {list(data.keys())}")
        print(f"📈 Total samples: {data['metadata']['total_samples']}")
        print(f"📋 Results entries: {len(data.get('results', []))}")
        
        return data
    except FileNotFoundError:
        print(f"❌ File {results_file} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None

def analyze_classification_results(results):
    """Analyze classification results to compute accuracy metrics."""
    base_results = [r for r in results if r['base']]
    non_base_results = [r for r in results if not r['base']]
    
    base_correct = sum(1 for r in base_results if r['correct'])
    non_base_correct = sum(1 for r in non_base_results if r['correct'])
    
    base_accuracy = base_correct / len(base_results) if base_results else 0
    non_base_accuracy = non_base_correct / len(non_base_results) if non_base_results else 0
    
    # Running accuracy calculation
    step_metrics = []
    base_running_correct = 0
    non_base_running_correct = 0
    base_count = 0
    non_base_count = 0
    
    for i, result in enumerate(results, 1):
        if result['base']:
            base_count += 1
            if result['correct']:
                base_running_correct += 1
            base_running_acc = base_running_correct / base_count
        else:
            non_base_count += 1
            if result['correct']:
                non_base_running_correct += 1
            non_base_running_acc = non_base_running_correct / non_base_count if non_base_count > 0 else 0
        
        step_metrics.append({
            'step': i,
            'base_accuracy': base_running_acc if base_count > 0 else 0,
            'non_base_accuracy': non_base_running_acc,
            'base_samples': base_count,
            'non_base_samples': non_base_count
        })
    
    return step_metrics, base_accuracy, non_base_accuracy

def plot_accuracy_comparison(data, save_dir="./"):
    """Plot base vs grammar model accuracy comparison."""
    results = data.get('results', [])
    if not results:
        print("❌ No results found for accuracy plot.")
        return
    
    step_metrics, base_final_acc, non_base_final_acc = analyze_classification_results(results)
    
    # Extract data for plotting
    steps = [m["step"] for m in step_metrics]
    base_accuracy = [m["base_accuracy"] for m in step_metrics]
    non_base_accuracy = [m["non_base_accuracy"] for m in step_metrics]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot running accuracy
    ax.plot(steps, base_accuracy, 'b-o', label='Base Model', linewidth=2, markersize=3, alpha=0.8)
    ax.plot(steps, non_base_accuracy, 'r-s', label='Grammar Model', linewidth=2, markersize=3, alpha=0.8)
    
    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Classification Accuracy: Base vs Grammar Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add final accuracy text
    improvement = non_base_final_acc - base_final_acc
    stats_text = f'Final Classification Accuracy:\nBase: {base_final_acc:.1%}\nGrammar: {non_base_final_acc:.1%}\nImprovement: {improvement:+.1%}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / 'classification_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Classification accuracy plot saved as '{save_path}'")
    plt.show()

def plot_confusion_matrices(data, save_dir="./"):
    """Plot confusion matrices for base and grammar models."""
    results = data.get('results', [])
    if not results:
        print("❌ No results found for confusion matrix.")
        return
    
    base_results = [r for r in results if r['base']]
    non_base_results = [r for r in results if not r['base']]
    
    # Get all unique labels
    all_labels = set()
    for r in results:
        all_labels.add(r['expected'])
        all_labels.add(r['predicted'])
    all_labels = sorted(list(all_labels))
    
    def create_confusion_matrix(results_subset, labels):
        matrix = np.zeros((len(labels), len(labels)))
        for r in results_subset:
            expected_idx = labels.index(r['expected'])
            predicted_idx = labels.index(r['predicted'])
            matrix[expected_idx, predicted_idx] += 1
        return matrix
    
    # Create confusion matrices
    base_matrix = create_confusion_matrix(base_results, all_labels)
    non_base_matrix = create_confusion_matrix(non_base_results, all_labels)
    
    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Base model confusion matrix
    sns.heatmap(base_matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=all_labels, yticklabels=all_labels, ax=ax1)
    ax1.set_title('Base Model Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Expected')
    
    # Grammar model confusion matrix
    sns.heatmap(non_base_matrix, annot=True, fmt='g', cmap='Reds', 
                xticklabels=all_labels, yticklabels=all_labels, ax=ax2)
    ax2.set_title('Grammar Model Confusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Expected')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / 'confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrices saved as '{save_path}'")
    plt.show()

def plot_class_performance(data, save_dir="./"):
    """Plot per-class performance comparison."""
    results = data.get('results', [])
    if not results:
        print("❌ No results found for class performance.")
        return
    
    base_results = [r for r in results if r['base']]
    non_base_results = [r for r in results if not r['base']]
    
    # Calculate per-class accuracy
    def calculate_class_accuracy(results_subset):
        class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results_subset:
            expected = r['expected']
            class_stats[expected]['total'] += 1
            if r['correct']:
                class_stats[expected]['correct'] += 1
        
        class_accuracy = {}
        for class_name, stats in class_stats.items():
            class_accuracy[class_name] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return class_accuracy
    
    base_class_acc = calculate_class_accuracy(base_results)
    non_base_class_acc = calculate_class_accuracy(non_base_results)
    
    # Get all classes
    all_classes = sorted(set(list(base_class_acc.keys()) + list(non_base_class_acc.keys())))
    
    # Prepare data for plotting
    base_values = [base_class_acc.get(cls, 0) for cls in all_classes]
    non_base_values = [non_base_class_acc.get(cls, 0) for cls in all_classes]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(all_classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_values, width, label='Base Model', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, non_base_values, width, label='Grammar Model', color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Emotion Classes', fontsize=12)
    ax.set_title('Per-Class Classification Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / 'class_performance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Per-class performance plot saved as '{save_path}'")
    plt.show()

def plot_timing_analysis(data, save_dir="./"):
    """Plot timing analysis for classification models."""
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    
    if not results:
        print("❌ No results found for timing analysis.")
        return
    
    # Extract timing data
    base_durations = [r['duration'] for r in results if r['base']]
    non_base_durations = [r['duration'] for r in results if not r['base']]
    all_durations = [r['duration'] for r in results]
    sample_numbers = list(range(1, len(results) + 1))
    
    # Create timing plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Duration over time by model type
    base_samples = [i for i, r in enumerate(results, 1) if r['base']]
    non_base_samples = [i for i, r in enumerate(results, 1) if not r['base']]
    
    if base_samples:
        ax1.scatter(base_samples, base_durations, color='blue', alpha=0.6, label='Base Model', s=30)
    if non_base_samples:
        ax1.scatter(non_base_samples, non_base_durations, color='red', alpha=0.6, label='Grammar Model', s=30)
    
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Duration (seconds)')
    ax1.set_title('Processing Time by Model Type', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Duration comparison histogram
    if base_durations and non_base_durations:
        ax2.hist([base_durations, non_base_durations], 
                bins=20, alpha=0.7, label=['Base Model', 'Grammar Model'], 
                color=['blue', 'red'], edgecolor='black')
        
        base_avg = np.mean(base_durations)
        non_base_avg = np.mean(non_base_durations)
        ax2.axvline(base_avg, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax2.axvline(non_base_avg, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Duration (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Duration Distribution Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative timing
    cumulative_times = np.cumsum(all_durations)
    ax3.plot(sample_numbers, cumulative_times, 'g-', linewidth=2)
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Cumulative Time (seconds)')
    ax3.set_title('Cumulative Processing Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Timing statistics summary
    ax4.axis('off')
    
    total_time = metadata.get('total_time', sum(all_durations))
    base_time = sum(base_durations)
    non_base_time = sum(non_base_durations)
    
    timing_summary = f"""
⏱️ TIMING ANALYSIS SUMMARY

📊 Overall Statistics:
   Total Time: {total_time:.2f}s
   Total Samples: {len(results)}
   Avg per Sample: {total_time/len(results):.3f}s

🔵 Base Model:
   Total Time: {base_time:.2f}s
   Samples: {len(base_durations)}
   Avg Time: {np.mean(base_durations):.3f}s
   Std Dev: {np.std(base_durations):.3f}s

🔴 Grammar Model:
   Total Time: {non_base_time:.2f}s
   Samples: {len(non_base_durations)}
   Avg Time: {np.mean(non_base_durations):.3f}s
   Std Dev: {np.std(non_base_durations):.3f}s

📈 Performance:
   Grammar vs Base Ratio: {np.mean(non_base_durations)/np.mean(base_durations):.2f}x
   Time Difference: {np.mean(non_base_durations)-np.mean(base_durations):+.3f}s
    """
    
    ax4.text(0.05, 0.95, timing_summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / 'timing_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"⏱️ Timing analysis plot saved as '{save_path}'")
    plt.show()

def plot_comprehensive_dashboard(data, save_dir="./"):
    """Create a comprehensive classification dashboard."""
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    
    if not results:
        print("❌ Insufficient data for dashboard.")
        return
    
    _, base_accuracy, non_base_accuracy = analyze_classification_results(results)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall accuracy comparison
    models = ['Base Model', 'Grammar Model']
    accuracies = [base_accuracy, non_base_accuracy]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Classification Accuracy', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add value labels
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Error distribution pie chart
    base_results = [r for r in results if r['base']]
    non_base_results = [r for r in results if not r['base']]
    
    non_base_correct = sum(1 for r in non_base_results if r['correct'])
    non_base_incorrect = len(non_base_results) - non_base_correct
    
    if len(non_base_results) > 0:
        ax2.pie([non_base_correct, non_base_incorrect], 
                labels=[f'Correct\n({non_base_correct})', f'Incorrect\n({non_base_incorrect})'],
                colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Grammar Model Results\n(Total: {len(non_base_results)} samples)', fontweight='bold')
    
    # 3. Improvement visualization
    improvement = non_base_accuracy - base_accuracy
    improvement_color = 'lightgreen' if improvement > 0 else 'lightcoral'
    
    ax3.bar(['Accuracy\nImprovement'], [improvement], color=improvement_color, alpha=0.8, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Accuracy Improvement')
    ax3.set_title('Grammar Model Performance Gain', fontweight='bold')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:+.1%}'.format(y)))
    
    # Add value label
    ax3.text(0, improvement + (0.01 if improvement > 0 else -0.02),
            f'{improvement:+.1%}', ha='center', va='bottom' if improvement > 0 else 'top', 
            fontweight='bold', fontsize=12)
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance summary
    ax4.axis('off')
    
    total_time = metadata.get('total_time', 0)
    base_durations = [r['duration'] for r in results if r['base']]
    non_base_durations = [r['duration'] for r in results if not r['base']]
    base_time = sum(base_durations)
    non_base_time = sum(non_base_durations)
    
    performance_text = f"""
🎯 CLASSIFICATION PERFORMANCE SUMMARY

📊 Model Comparison:
   • Base Accuracy: {base_accuracy:.1%}
   • Grammar Accuracy: {non_base_accuracy:.1%}
   • Improvement: {improvement:+.1%}

⏱️ Timing Analysis:
   • Total Time: {total_time:.1f}s
   • Base Model Time: {base_time:.1f}s
   • Grammar Model Time: {non_base_time:.1f}s
   • Speed Ratio: {non_base_time/base_time:.2f}x

📈 Key Insights:
   • Total Samples: {len(results)}
   • Base Samples: {len(base_results)}
   • Grammar Samples: {len(non_base_results)}
   • Processing overhead: {non_base_time/base_time:.2f}x slower
    """
    
    ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / 'classification_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Classification dashboard saved as '{save_path}'")
    plt.show()

def generate_classification_report(data, save_dir="./"):
    """Generate a comprehensive classification analysis report."""
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    
    # Create output directory if it doesn't exist
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 COMPREHENSIVE CLASSIFICATION BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    _, base_accuracy, non_base_accuracy = analyze_classification_results(results)
    total_samples = metadata.get('total_samples', len(results))
    base_samples = len([r for r in results if r['base']])
    non_base_samples = len([r for r in results if not r['base']])
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total samples: {total_samples}")
    print(f"   Base model samples: {base_samples}")
    print(f"   Grammar model samples: {non_base_samples}")
    print(f"   Batch size: {metadata.get('batch_size', 'Unknown')}")
    print(f"   Total batches: {metadata.get('total_batches', 'Unknown')}")
    
    print(f"\n🎯 CLASSIFICATION RESULTS:")
    print(f"   Base Model Accuracy: {base_accuracy:.1%}")
    print(f"   Grammar Model Accuracy: {non_base_accuracy:.1%}")
    improvement = non_base_accuracy - base_accuracy
    print(f"   Improvement: {improvement:+.1%}")
    
    # Timing analysis
    total_time = metadata.get('total_time', 0)
    base_durations = [r['duration'] for r in results if r['base']]
    non_base_durations = [r['duration'] for r in results if not r['base']]
    base_time = sum(base_durations)
    non_base_time = sum(non_base_durations)
    
    print(f"\n⏱️ TIMING ANALYSIS:")
    print(f"   Total processing time: {total_time:.1f}s")
    print(f"   Base model time: {base_time:.1f}s")
    print(f"   Grammar model time: {non_base_time:.1f}s")
    if base_time > 0:
        print(f"   Grammar overhead: {non_base_time/base_time:.2f}x")
    print(f"   Time per sample: {total_time/total_samples:.3f}s")
    
    # Class distribution
    all_classes = set(r['expected'] for r in results)
    print(f"\n📋 EMOTION CLASSES DETECTED:")
    print(f"   Total classes: {len(all_classes)}")
    print(f"   Classes: {', '.join(sorted(all_classes))}")
    
    print(f"\n📁 All plots saved to: {output_dir.absolute()}")
    print("=" * 60)

def main():
    """Main function to run all classification analysis and plotting."""
    print("🚀 Starting Comprehensive Classification Benchmark Analysis")
    print("=" * 65)
    
    # Load data
    data = load_classification_data()
    if not data:
        return
    
    # Create output directory
    output_dir = Path("./classification_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    print("\n📈 Generating accuracy comparison plots...")
    plot_accuracy_comparison(data, output_dir)
    
    print("\n📊 Generating confusion matrices...")
    plot_confusion_matrices(data, output_dir)
    
    print("\n📊 Generating per-class performance analysis...")
    plot_class_performance(data, output_dir)
    
    print("\n⏱️ Generating timing analysis...")
    plot_timing_analysis(data, output_dir)
    
    print("\n📊 Generating comprehensive dashboard...")
    plot_comprehensive_dashboard(data, output_dir)
    
    # Generate comprehensive report
    print("\n🔍 Generating comprehensive report...")
    generate_classification_report(data, output_dir)
    
    print(f"\n✅ Analysis complete! All outputs saved to '{output_dir.absolute()}'")

if __name__ == "__main__":
    main() 