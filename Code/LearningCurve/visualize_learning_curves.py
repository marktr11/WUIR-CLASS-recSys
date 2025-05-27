import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml

def load_evaluation_data(file_path):
    """Load evaluation data from a text file.
    
    Args:
        file_path (str): Path to the evaluation results file
        
    Returns:
        tuple: (steps, metrics) arrays
    """
    data = np.loadtxt(file_path)
    steps = data[:, 0]
    metrics = data[:, 1]
    return steps, metrics

def load_config():
    """Load configuration from run.yaml file.
    
    Returns:
        dict: Configuration parameters
    """
    config_path = os.path.join("Code", "config", "run.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_experiment_title(model_name, feature, k):
    """Generate a descriptive title for the experiment.
    
    Args:
        model_name (str): Name of the model (e.g., 'dqn', 'ppo')
        feature (str): Feature configuration (e.g., 'Usefulness-as-Rwd', 'Weighted-Usefulness-as-Rwd')
        k (str): Number of recommendations
        
    Returns:
        str: Formatted experiment title
    """
    # Load config to check clustering
    config = load_config()
    use_clustering = config.get('use_clustering', False)
    
    # Map model names to full names
    model_map = {
        'dqn': 'DQN',
        'ppo': 'PPO'
    }
    
    # Map feature names to readable format
    feature_map = {
        'Usefulness-as-Rwd': 'UIR',
        'Weighted-Usefulness-as-Rwd': 'WUIR',
        'baseline': 'Baseline'
    }
    
    # Get formatted names
    model = model_map.get(model_name.lower(), model_name.upper())
    feature_name = feature_map.get(feature, feature)
    
    # Add clustering information
    clustering_info = "with Clustering" if use_clustering else "without Clustering"
    
    return f"{model} - {clustering_info} - {feature_name}-No-Mastery-Levels (k={k})"

def plot_learning_curves(results_dir, output_dir):
    """Plot learning curves for all evaluation files in the results directory.
    
    Args:
        results_dir (str): Directory containing evaluation result files
        output_dir (str): Directory to save the plots
    """
    # Get all evaluation result files
    result_files = glob.glob(os.path.join(results_dir, "all_*.txt"))
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Clear existing plots
    existing_plots = glob.glob(os.path.join(output_dir, "lc_*.png"))
    for plot_file in existing_plots:
        try:
            os.remove(plot_file)
            print(f"Deleted: {os.path.basename(plot_file)}")
        except Exception as e:
            print(f"Error deleting {plot_file}: {e}")
    
    print(f"\nFound {len(result_files)} result files to plot...")
    
    for file_path in result_files:
        # Extract model info from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        # Parse model information
        model_name = parts[1]
        feature = parts[2] if len(parts) > 2 else "baseline"
        k = parts[4].replace('k', '') if len(parts) > 4 else "unknown"
        # run = parts[6].replace('run', '') if len(parts) > 6 else "unknown"
        
        # Load config to check clustering
        config = load_config()
        use_clustering = config.get('use_clustering', False)
        
        # Create output filename with new format
        clustering_suffix = "clustered" if use_clustering else "no_clustering"
        output_filename = f"{model_name}_{feature}_{clustering_suffix}_k{k}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Creating plot: {output_filename}...")
        
        # Load data
        steps, metrics = load_evaluation_data(file_path)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot metrics
        plt.plot(steps, metrics, 'b-', linewidth=2, label='Average Applicable Jobs')
        plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
        
        # Set title using the new format
        title = get_experiment_title(model_name, feature, k)
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Make y-axis ticks more readable
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Define directories
    results_dir = os.path.join("Code", "results")
    output_dir = os.path.join("Code", "LearningCurve")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot learning curves
    plot_learning_curves(results_dir, output_dir)
    print("\nAll learning curves have been regenerated from the latest results.") 