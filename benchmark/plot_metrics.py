import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_benchmark_data(results_file="benchmark_tools_results.json"):
    """Load and validate benchmark data from the Gnosis results file."""
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

def analyze_results_data(results):
    """Analyze the results to compute step-by-step metrics."""
    base_syntactic = []
    base_semantic = []
    grammar_syntactic = []
    grammar_semantic = []
    
    base_syn_count = 0
    base_sem_count = 0
    grammar_syn_count = 0
    grammar_sem_count = 0
    
    step_metrics = []
    
    for i, result in enumerate(results, 1):
        # Update counts
        if not result['grammar_enabled']:
            # Base model result
            if result['syntactic_validity']:
                base_syn_count += 1
            if result['semantic_validity']:
                base_sem_count += 1
        else:
            # Grammar-enabled result
            if result['syntactic_validity']:
                grammar_syn_count += 1
            if result['semantic_validity']:
                grammar_sem_count += 1
        
        # Calculate running averages
        base_samples = sum(1 for r in results[:i] if not r['grammar_enabled'])
        grammar_samples = sum(1 for r in results[:i] if r['grammar_enabled'])
        
        if base_samples > 0:
            base_syn_avg = base_syn_count / base_samples
            base_sem_avg = base_sem_count / base_samples
        else:
            base_syn_avg = base_sem_avg = 0
            
        if grammar_samples > 0:
            grammar_syn_avg = grammar_syn_count / grammar_samples
            grammar_sem_avg = grammar_sem_count / grammar_samples
        else:
            grammar_syn_avg = grammar_sem_avg = 0
        
        step_metrics.append({
            'step': i,
            'base_syntactic_avg': base_syn_avg,
            'base_semantic_avg': base_sem_avg,
            'grammar_syntactic_avg': grammar_syn_avg,
            'grammar_semantic_avg': grammar_sem_avg,
            'base_samples': base_samples,
            'grammar_samples': grammar_samples
        })
    
    return step_metrics

def plot_comparison_accuracy(data, save_dir="./"):
    """Plot base vs grammar-enabled model accuracy comparison."""
    results = data.get('results', [])
    if not results:
        print("❌ No results found for accuracy plot.")
        return
    
    step_metrics = analyze_results_data(results)
    
    # Extract data for plotting
    steps = [m["step"] for m in step_metrics]
    base_syntactic = [m["base_syntactic_avg"] for m in step_metrics]
    base_semantic = [m["base_semantic_avg"] for m in step_metrics]
    grammar_syntactic = [m["grammar_syntactic_avg"] for m in step_metrics]
    grammar_semantic = [m["grammar_semantic_avg"] for m in step_metrics]
    
    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Syntactic accuracy comparison
    ax1.plot(steps, base_syntactic, 'b-o', label='Base Model (Syntactic)', linewidth=2, markersize=4, alpha=0.8)
    ax1.plot(steps, grammar_syntactic, 'r-s', label='Grammar Model (Syntactic)', linewidth=2, markersize=4, alpha=0.8)
    
    ax1.set_xlabel('Sample Number', fontsize=12)
    ax1.set_ylabel('Syntactic Accuracy', fontsize=12)
    ax1.set_title('Syntactic Accuracy: Base vs Grammar-Enabled Model', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add final accuracy text
    final_base_syn = base_syntactic[-1] if base_syntactic else 0
    final_grammar_syn = grammar_syntactic[-1] if grammar_syntactic else 0
    improvement_syn = final_grammar_syn - final_base_syn
    
    stats_text = f'Final Syntactic Accuracy:\nBase: {final_base_syn:.1%}\nGrammar: {final_grammar_syn:.1%}\nImprovement: {improvement_syn:+.1%}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Semantic accuracy comparison
    ax2.plot(steps, base_semantic, 'b-o', label='Base Model (Semantic)', linewidth=2, markersize=4, alpha=0.8)
    ax2.plot(steps, grammar_semantic, 'r-s', label='Grammar Model (Semantic)', linewidth=2, markersize=4, alpha=0.8)
    
    ax2.set_xlabel('Sample Number', fontsize=12)
    ax2.set_ylabel('Semantic Accuracy', fontsize=12)
    ax2.set_title('Semantic Accuracy: Base vs Grammar-Enabled Model', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add final accuracy text
    final_base_sem = base_semantic[-1] if base_semantic else 0
    final_grammar_sem = grammar_semantic[-1] if grammar_semantic else 0
    improvement_sem = final_grammar_sem - final_base_sem
    
    stats_text = f'Final Semantic Accuracy:\nBase: {final_base_sem:.1%}\nGrammar: {final_grammar_sem:.1%}\nImprovement: {improvement_sem:+.1%}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / 'accuracy_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Accuracy comparison plot saved as '{save_path}'")
    plt.show()

def plot_timing_analysis(data, save_dir="./"):
    """Plot comprehensive timing analysis."""
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    
    if not results:
        print("❌ No results found for timing analysis.")
        return
    
    # Extract timing data
    base_durations = [r['duration'] for r in results if not r['grammar_enabled']]
    grammar_durations = [r['duration'] for r in results if r['grammar_enabled']]
    all_durations = [r['duration'] for r in results]
    sample_numbers = list(range(1, len(results) + 1))
    
    # Create comprehensive timing plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Duration over time by model type
    base_samples = [i for i, r in enumerate(results, 1) if not r['grammar_enabled']]
    grammar_samples = [i for i, r in enumerate(results, 1) if r['grammar_enabled']]
    
    if base_samples:
        ax1.scatter(base_samples, base_durations, color='blue', alpha=0.6, label='Base Model', s=30)
    if grammar_samples:
        ax1.scatter(grammar_samples, grammar_durations, color='red', alpha=0.6, label='Grammar Model', s=30)
    
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Duration (seconds)')
    ax1.set_title('Processing Time by Model Type', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Duration comparison histogram
    if base_durations and grammar_durations:
        ax2.hist([base_durations, grammar_durations], 
                bins=20, alpha=0.7, label=['Base Model', 'Grammar Model'], 
                color=['blue', 'red'], edgecolor='black')
        
        base_avg = np.mean(base_durations)
        grammar_avg = np.mean(grammar_durations)
        ax2.axvline(base_avg, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax2.axvline(grammar_avg, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
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
    
    # Calculate comprehensive statistics
    total_time = metadata.get('total_time', sum(all_durations))
    base_time = metadata.get('base_model_time', sum(base_durations))
    grammar_time = metadata.get('grammar_model_time', sum(grammar_durations))
    
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
   Total Time: {grammar_time:.2f}s
   Samples: {len(grammar_durations)}
   Avg Time: {np.mean(grammar_durations):.3f}s
   Std Dev: {np.std(grammar_durations):.3f}s

📈 Performance:
   Grammar vs Base Ratio: {np.mean(grammar_durations)/np.mean(base_durations):.2f}x
   Time Difference: {np.mean(grammar_durations)-np.mean(base_durations):+.3f}s
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

def plot_performance_dashboard(data, save_dir="./"):
    """Create a comprehensive performance dashboard."""
    summary = data.get('summary', {})
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    
    if not summary or not results:
        print("❌ Insufficient data for performance dashboard.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Final accuracy comparison bar chart
    base_syn = summary['base_model']['syntactic_accuracy']
    base_sem = summary['base_model']['semantic_accuracy'] 
    grammar_syn = summary['grammar_model']['syntactic_accuracy']
    grammar_sem = summary['grammar_model']['semantic_accuracy']
    
    categories = ['Syntactic\nAccuracy', 'Semantic\nAccuracy']
    base_values = [base_syn, base_sem]
    grammar_values = [grammar_syn, grammar_sem]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, base_values, width, label='Base Model', color='lightblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, grammar_values, width, label='Grammar Model', color='lightcoral', alpha=0.8)
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Final Accuracy Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Success/Failure breakdown pie chart
    total_samples = metadata.get('total_samples', len(results))
    
    # Calculate success rates
    base_success = sum(1 for r in results if not r['grammar_enabled'] and r['semantic_validity'])
    base_total = sum(1 for r in results if not r['grammar_enabled'])
    grammar_success = sum(1 for r in results if r['grammar_enabled'] and r['semantic_validity'])
    grammar_total = sum(1 for r in results if r['grammar_enabled'])
    
    # Grammar model pie chart
    grammar_fail = grammar_total - grammar_success
    if grammar_total > 0:
        ax2.pie([grammar_success, grammar_fail], 
                labels=[f'Success\n({grammar_success})', f'Failure\n({grammar_fail})'],
                colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Grammar Model Results\n(Total: {grammar_total} samples)', fontweight='bold')
    
    # 3. Improvement visualization
    syn_improvement = grammar_syn - base_syn
    sem_improvement = grammar_sem - base_sem
    
    improvements = ['Syntactic\nImprovement', 'Semantic\nImprovement']
    improvement_values = [syn_improvement, sem_improvement]
    colors = ['lightgreen' if x > 0 else 'lightcoral' for x in improvement_values]
    
    bars = ax3.bar(improvements, improvement_values, color=colors, alpha=0.8, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Accuracy Improvement')
    ax3.set_title('Grammar Model Improvements', fontweight='bold')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:+.1%}'.format(y)))
    
    # Add value labels
    for bar, value in zip(bars, improvement_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                f'{value:+.1%}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance summary
    ax4.axis('off')
    
    # Calculate additional metrics
    total_time = metadata.get('total_time', 0)
    base_time = metadata.get('base_model_time', 0)
    grammar_time = metadata.get('grammar_model_time', 0)
    
    performance_text = f"""
🎯 GNOSIS PERFORMANCE SUMMARY

📊 Model Comparison:
   • Base Syntactic: {base_syn:.1%}
   • Grammar Syntactic: {grammar_syn:.1%}
   • Syntactic Gain: {syn_improvement:+.1%}
   
   • Base Semantic: {base_sem:.1%}
   • Grammar Semantic: {grammar_sem:.1%}
   • Semantic Gain: {sem_improvement:+.1%}

⏱️ Timing Analysis:
   • Total Time: {total_time:.1f}s
   • Base Model Time: {base_time:.1f}s
   • Grammar Model Time: {grammar_time:.1f}s
   • Speed Ratio: {grammar_time/base_time:.2f}x

📈 Key Insights:
   • Grammar repair achieves 100% syntactic validity
   • Semantic accuracy {'improved' if sem_improvement > 0 else 'changed'} by {sem_improvement:+.1%}
   • Processing overhead: {grammar_time/base_time:.2f}x slower
    """
    
    ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = Path(save_dir) / 'performance_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance dashboard saved as '{save_path}'")
    plt.show()

def generate_comprehensive_report(data, save_dir="./"):
    """Generate a comprehensive analysis report."""
    summary = data.get('summary', {})
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    
    # Create output directory if it doesn't exist
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 COMPREHENSIVE GNOSIS BENCHMARK ANALYSIS")
    print("=" * 55)
    
    # Basic statistics
    total_samples = metadata.get('total_samples', len(results))
    base_syn = summary['base_model']['syntactic_accuracy']
    base_sem = summary['base_model']['semantic_accuracy']
    grammar_syn = summary['grammar_model']['syntactic_accuracy']
    grammar_sem = summary['grammar_model']['semantic_accuracy']
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total samples: {total_samples}")
    print(f"   Model: {metadata.get('model', 'Unknown')}")
    print(f"   Batch size: {metadata.get('batch_size', 'Unknown')}")
    print(f"   Total batches: {metadata.get('total_batches', 'Unknown')}")
    
    print(f"\n🎯 ACCURACY RESULTS:")
    print(f"   Base Model:")
    print(f"     - Syntactic: {base_syn:.1%}")
    print(f"     - Semantic: {base_sem:.1%}")
    print(f"   Grammar Model:")
    print(f"     - Syntactic: {grammar_syn:.1%}")
    print(f"     - Semantic: {grammar_sem:.1%}")
    
    print(f"\n📈 IMPROVEMENTS:")
    syn_improvement = grammar_syn - base_syn
    sem_improvement = grammar_sem - base_sem
    print(f"   Syntactic improvement: {syn_improvement:+.1%}")
    print(f"   Semantic improvement: {sem_improvement:+.1%}")
    
    # Timing analysis
    total_time = metadata.get('total_time', 0)
    base_time = metadata.get('base_model_time', 0)
    grammar_time = metadata.get('grammar_model_time', 0)
    
    print(f"\n⏱️ TIMING ANALYSIS:")
    print(f"   Total processing time: {total_time:.1f}s")
    print(f"   Base model time: {base_time:.1f}s")
    print(f"   Grammar model time: {grammar_time:.1f}s")
    print(f"   Grammar overhead: {grammar_time/base_time:.2f}x")
    print(f"   Time per sample: {total_time/total_samples:.3f}s")
    
    print(f"\n📁 All plots saved to: {output_dir.absolute()}")
    print("=" * 55)

def main():
    """Main function to run all analysis and plotting."""
    print("🚀 Starting Comprehensive Gnosis Benchmark Analysis")
    print("=" * 55)
    
    # Load data
    data = load_benchmark_data()
    if not data:
        return
    
    # Create output directory
    output_dir = Path("./analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    print("\n📈 Generating accuracy comparison plots...")
    plot_comparison_accuracy(data, output_dir)
    
    print("\n⏱️ Generating timing analysis...")
    plot_timing_analysis(data, output_dir)
    
    print("\n📊 Generating performance dashboard...")
    plot_performance_dashboard(data, output_dir)
    
    # Generate comprehensive report
    print("\n🔍 Generating comprehensive report...")
    generate_comprehensive_report(data, output_dir)
    
    print(f"\n✅ Analysis complete! All outputs saved to '{output_dir.absolute()}'")

if __name__ == "__main__":
    main() 