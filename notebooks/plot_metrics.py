import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_metrics():
    """
    Plot reward metrics from CSV files in the current directory.
    Looks for CSV files starting with 'rewards' and plots step vs value.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Find all CSV files starting with 'rewards'
    csv_files = []
    for file in script_dir.glob('rewards*.csv'):
        csv_files.append(file)
    
    if not csv_files:
        print("No CSV files starting with 'rewards' found in the directory.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each CSV file
    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Extract the metric name from the filename for the label
        metric_name = csv_file.stem.replace('rewards_', '').replace('_mean', '')
        
        # Plot step vs value
        plt.plot(df['step'], df['value'], marker='o', linewidth=2, label=metric_name)
    
    # Customize the plot
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Reward Value', fontsize=12)
    plt.title('Reward Metrics Over Training Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Improve layout
    plt.tight_layout()
    
    # Optionally save the plot
    output_path = script_dir / 'reward_metrics_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_metrics() 