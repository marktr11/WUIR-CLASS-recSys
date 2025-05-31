"""
Learning Curves Visualization Module

This module provides functions to visualize and compare learning curves from different experiments.
It handles plotting of individual learning curves and various comparison plots.

Configuration:
    To change the target branch for plots, modify the BRANCH_NAME variable below.
    Example:
        BRANCH_NAME = "clusteringRL-no-mastery-levels"  # Current branch
        BRANCH_NAME = "clusteringRL-mastery-levels"   # Different branch

Directory Structure:
    Code/results/
    ├── [BRANCH_NAME]/
    │   ├── plots/     # All plots will be saved here
    │   └── data/      # Data files
    └── backups/       # Backup directories

Usage:
    1. Set BRANCH_NAME to your target branch
    2. Run the script:
       python visualize_learning_curves.py

Note:
    - All plots will be saved in Code/results/[BRANCH_NAME]/plots/
    - Existing plots will be skipped unless deleted
    - Each comparison plot has a unique name based on model, feature, and k value
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml

# Configuration
BRANCH_NAME = "clusteringRL-no-mastery-levels"  # Change this to your target branch

def get_plots_directory(results_dir):
    """Get the plots directory for the current branch.
    
    Args:
        results_dir (str): Base results directory path
        
    Returns:
        str: Path to the plots directory for current branch
        
    Note:
        Creates the directory if it doesn't exist
    """
    plots_dir = os.path.join(results_dir, BRANCH_NAME, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def load_evaluation_data(file_path):
    """Load evaluation data from a text file.
    
    Args:
        file_path (str): Path to the evaluation data file
        
    Returns:
        tuple: (steps, metrics) arrays from the data file
    """
    data = np.loadtxt(file_path)
    steps = data[:, 0]
    metrics = data[:, 1]
    return steps, metrics

def get_experiment_title(model_name, feature, k, is_clustered):
    """Generate a descriptive title for the experiment.
    
    Args:
        model_name (str): Name of the model (e.g., 'dqn', 'ppo')
        feature (str): Feature name (e.g., 'baseline', 'UIR')
        k (str): Number of clusters
        is_clustered (bool): Whether clustering is used
        
    Returns:
        str: Formatted title for the plot
    """
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
    clustering_info = "with Clustering" if is_clustered else "without Clustering"
    
    return f"{model} - {clustering_info} - {feature_name}-No-Mastery-Levels (k={k})"

def plot_learning_curves(results_dir):
    """Plot learning curves for all evaluation files in the results directory.
    
    Args:
        results_dir (str): Base results directory path
        
    Note:
        - Plots are saved in Code/results/[BRANCH_NAME]/plots/
        - Existing plots are skipped
        - Each plot shows learning curve for a specific model/feature/k combination
    """
    # Get all evaluation result files
    result_files = glob.glob(os.path.join(results_dir, "**", "all_*.txt"), recursive=True)
    
    if not result_files:
        print(f"\nWarning: No result files found in {results_dir}")
        print("Please check if:")
        print("1. The results directory exists")
        print("2. The result files follow the naming pattern: all_*.txt")
        print("3. You are looking in the correct branch")
        return
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Count existing plots
    existing_plots = glob.glob(os.path.join(plots_dir, "*.png"))
    print(f"\nFound {len(existing_plots)} existing plots in {plots_dir}")
    
    print(f"\nFound {len(result_files)} result files to process...")
    print("Files found:")
    for file in result_files:
        print(f"- {os.path.basename(file)}")
    
    # Keep track of new plots
    new_plots = 0
    
    for file_path in result_files:
        # Extract model info from filename
        filename = os.path.basename(file_path)
        
        # Parse model information more carefully
        parts = filename.split('_')
        if len(parts) < 5:  # Skip invalid filenames
            print(f"Skipping invalid filename format: {filename}")
            continue
            
        model_name = parts[1]
        feature = parts[2]
        k = parts[4].replace('k', '')
        
        # Check if file has clustering info from filename
        has_clustering = any('cluster' in part.lower() for part in parts)
        
        # Create output filename based on result file's clustering status
        clustering_suffix = "clustered" if has_clustering else "no_clustering"
        output_filename = f"{model_name}_{feature}_{clustering_suffix}_k{k}.png"
        output_path = os.path.join(plots_dir, output_filename)
        
        # Skip if plot already exists
        if os.path.exists(output_path):
            print(f"Skipping existing plot: {output_filename}")
            continue
            
        print(f"Creating new plot: {output_filename}...")
        new_plots += 1
        
        # Load data
        steps, metrics = load_evaluation_data(file_path)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot metrics
        plt.plot(steps, metrics, 'b-', linewidth=2, label='Average Applicable Jobs')
        plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
        
        # Set title using the new format
        title = get_experiment_title(model_name, feature, k, has_clustering)
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        
        # Set legend position to bottom right
        plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        # Make y-axis ticks more readable
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nSummary:")
    print(f"- Existing plots: {len(existing_plots)}")
    print(f"- New plots created: {new_plots}")
    print(f"- Total plots after update: {len(existing_plots) + new_plots}")

def compare_clustering_effect(results_dir, model_name, k, feature='baseline'):
    """Compare learning curves with and without clustering for specific model and k."""
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Find result files for both cases
    clustered_pattern = f"all_{model_name}_{feature}_k_{k}_*clusters_auto*.txt"
    no_cluster_pattern = f"all_{model_name}_{feature}_k_{k}_run_*.txt"
    
    clustered_files = glob.glob(os.path.join(results_dir, "**", clustered_pattern), recursive=True)
    no_cluster_files = glob.glob(os.path.join(results_dir, "**", no_cluster_pattern), recursive=True)
    
    if not clustered_files or not no_cluster_files:
        print(f"\nWarning: Could not find both clustered and non-clustered files for {model_name} k={k}")
        print("Looking for:")
        print(f"- Clustered pattern: {clustered_pattern}")
        print(f"- No cluster pattern: {no_cluster_pattern}")
        print("\nFiles found:")
        if clustered_files:
            print("Clustered files:")
            for f in clustered_files:
                print(f"- {os.path.basename(f)}")
        if no_cluster_files:
            print("Non-clustered files:")
            for f in no_cluster_files:
                print(f"- {os.path.basename(f)}")
        return
        
    # Load data
    clustered_steps, clustered_metrics = load_evaluation_data(clustered_files[0])
    no_cluster_steps, no_cluster_metrics = load_evaluation_data(no_cluster_files[0])
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot both curves
    plt.plot(clustered_steps, clustered_metrics, 'b-', linewidth=2, label='With Clustering')
    plt.plot(no_cluster_steps, no_cluster_metrics, 'r--', linewidth=2, label='Without Clustering')
    
    plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
    
    # Set title
    title = f"{model_name.upper()} {feature} k={k} - Clustering Comparison"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    
    # Set legend position to bottom right
    plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    
    # Make y-axis ticks more readable
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Create output filename
    output_filename = f"{model_name}_{feature}_k{k}_clustering_comparison.png"
    output_path = os.path.join(plots_dir, output_filename)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created comparison plot: {output_filename}")

def compare_dqn_baseline_clustering(results_dir):
    """Compare DQN baseline with and without clustering for each k value."""
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Process each k value
    for k in ['2', '3']:
        # Find result files for both cases
        clustered_pattern = f"all_dqn_baseline_k_{k}_total_steps_500000_clusters_auto_run_*.txt"
        no_cluster_pattern = f"all_dqn_baseline_k_{k}_total_steps_500000_run_*.txt"
        
        clustered_files = glob.glob(os.path.join(results_dir, "**", clustered_pattern), recursive=True)
        no_cluster_files = glob.glob(os.path.join(results_dir, "**", no_cluster_pattern), recursive=True)
        
        if not clustered_files or not no_cluster_files:
            print(f"\nWarning: Could not find both clustered and non-clustered files for DQN baseline k={k}")
            print("Looking for:")
            print(f"- Clustered pattern: {clustered_pattern}")
            print(f"- No cluster pattern: {no_cluster_pattern}")
            print("\nFiles found:")
            if clustered_files:
                print("Clustered files:")
                for f in clustered_files:
                    print(f"- {os.path.basename(f)}")
            if no_cluster_files:
                print("Non-clustered files:")
                for f in no_cluster_files:
                    print(f"- {os.path.basename(f)}")
            continue
            
        # Load data
        clustered_steps, clustered_metrics = load_evaluation_data(clustered_files[0])
        no_cluster_steps, no_cluster_metrics = load_evaluation_data(no_cluster_files[0])
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot both curves
        plt.plot(clustered_steps, clustered_metrics, 'b-', linewidth=2, label='CL-Baseline')
        plt.plot(no_cluster_steps, no_cluster_metrics, 'r--', linewidth=2, label='Baseline')
        
        plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
        
        # Set title
        title = f"DQN Baseline k={k} - Clustering Comparison"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        
        # Set legend position to bottom right
        plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        # Make y-axis ticks more readable
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Create output filename
        output_filename = f"dqn_baseline_k{k}_clustering_comparison.png"
        output_path = os.path.join(plots_dir, output_filename)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created comparison plot: {output_filename}")

def compare_dqn_models(results_dir):
    """Compare DQN models with and without clustering for k=2 and k=3 separately."""
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Define features to compare
    features = ['baseline', 'Usefulness-as-Rwd', 'Weighted-Usefulness-as-Rwd']
    
    # Define color scheme and styles
    colors = {
        'baseline': {
            'clustered': '#1f77b4',     # Dark blue
            'non_clustered': '#7fb3d5'  # Light blue
        },
        'Usefulness-as-Rwd': {
            'clustered': '#d62728',     # Dark red
            'non_clustered': '#ff9999'  # Light red
        },
        'Weighted-Usefulness-as-Rwd': {
            'clustered': '#006400',     # Dark green
            'non_clustered': '#4daf4a'  # Medium green instead of light green
        }
    }
    
    # Create separate plots for k=2 and k=3
    for k in ['2', '3']:
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Store last points for text positioning
        last_points = []
        
        # First pass: collect all last points
        for feature in features:
            for is_clustered in [True, False]:
                if is_clustered:
                    pattern = f"all_dqn_{feature}_k_{k}_total_steps_500000_clusters_auto_run_*.txt"
                else:
                    pattern = f"all_dqn_{feature}_k_{k}_total_steps_500000_run_*.txt"
                
                files = glob.glob(os.path.join(results_dir, "**", pattern), recursive=True)
                if files:
                    steps, metrics = load_evaluation_data(files[0])
                    last_points.append({
                        'step': steps[-1],
                        'metric': metrics[-1],
                        'feature': feature,
                        'is_clustered': is_clustered
                    })
        
        # Sort points by metric value
        last_points.sort(key=lambda x: x['metric'])
        
        # Second pass: plot lines and adjust text positions
        for feature in features:
            for is_clustered in [True, False]:
                if is_clustered:
                    pattern = f"all_dqn_{feature}_k_{k}_total_steps_500000_clusters_auto_run_*.txt"
                else:
                    pattern = f"all_dqn_{feature}_k_{k}_total_steps_500000_run_*.txt"
                
                files = glob.glob(os.path.join(results_dir, "**", pattern), recursive=True)
                if files:
                    # Load and plot data
                    steps, metrics = load_evaluation_data(files[0])
                    
                    # Define line style and label
                    linestyle = '-' if is_clustered else '--'
                    color = colors[feature]['clustered' if is_clustered else 'non_clustered']
                    
                    # Format feature name for label
                    feature_name = feature.replace('-', ' ').title()
                    if feature == 'Usefulness-as-Rwd':
                        feature_name = 'UIR'
                    elif feature == 'Weighted-Usefulness-as-Rwd':
                        feature_name = 'WUIR'
                    
                    # Add clustering indicator to label
                    clustering_indicator = 'CL-' if is_clustered else ''
                    label = f"{clustering_indicator}{feature_name}"
                    
                    # Plot the line
                    plt.plot(steps, metrics, color=color, linestyle=linestyle, linewidth=2, label=label)
                    
                    # Add marker for the last point
                    last_step = steps[-1]
                    last_metric = metrics[-1]
                    plt.plot(last_step, last_metric, 'o', color=color, markersize=8)
                    
                    # Find index of current point in sorted list
                    current_idx = next(i for i, p in enumerate(last_points) 
                                     if p['feature'] == feature and p['is_clustered'] == is_clustered)
                    
                    # Calculate base offset
                    x_offset = (max(steps) - min(steps)) * 0.015
                    y_offset = (max(metrics) - min(metrics)) * 0.015
                    
                    # Adjust y_offset based on position in sorted list
                    if current_idx > 0:
                        prev_metric = last_points[current_idx - 1]['metric']
                        # Increase threshold for considering points as "close"
                        if abs(last_metric - prev_metric) < (max(metrics) - min(metrics)) * 0.08:  # Increased from 0.05 to 0.08
                            # Increase vertical spacing between close values
                            y_offset = (current_idx + 1) * y_offset  
                    
                    plt.text(last_step + x_offset, last_metric + y_offset, f'{last_metric:.2f}', 
                            color=color, fontsize=14, fontweight='bold',
                            ha='left', va='bottom')
        
        # Set labels and title
        plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
        
        # Set title
        title = f"DQN Models Comparison (k={k})"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        
        # Increase padding for x and y axis
        plt.xlim(min(steps), max(steps) * 1.15)  # Increase x-axis padding to 15%
        plt.ylim(min(metrics) * 0.95, max(metrics) * 1.15)  # Increase y-axis padding to 15%
   
        # Set legend position to bottom right
        plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        # Make y-axis ticks more readable
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Create output filename
        output_filename = f"dqn_models_k{k}_comparison.png"
        output_path = os.path.join(plots_dir, output_filename)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created comparison plot: {output_filename}")

def compare_k_values_all_models(results_dir):
    """Compare k=2 and k=3 for all models with and without clustering."""
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Define models and features to compare
    models = ['dqn', 'ppo']
    features = ['baseline', 'Usefulness-as-Rwd', 'Weighted-Usefulness-as-Rwd']
    
    for model in models:
        for feature in features:
            # Create figure
            plt.figure(figsize=(15, 8))
            
            # Plot for each k value and clustering status
            for k in ['2', '3']:
                for is_clustered in [True, False]:
                    # Define pattern based on clustering status
                    if is_clustered:
                        pattern = f"all_{model}_{feature}_k_{k}_total_steps_500000_clusters_auto_run_*.txt"
                    else:
                        pattern = f"all_{model}_{feature}_k_{k}_total_steps_500000_run_*.txt"
                    
                    # Find matching files
                    files = glob.glob(os.path.join(results_dir, pattern))
                    
                    if files:
                        # Load and plot data
                        steps, metrics = load_evaluation_data(files[0])
                        
                        # Define line style and label
                        line_style = '-' if is_clustered else '--'
                        color = 'b' if k == '2' else 'r'
                        label = f"k={k} {'with' if is_clustered else 'without'} clustering"
                        
                        plt.plot(steps, metrics, f'{color}{line_style}', linewidth=2, label=label)
            
            # Set labels and title
            plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
            plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
            
            # Format feature name for title
            feature_name = feature.replace('-', ' ').title()
            title = f"{model.upper()} {feature_name} - k=2 vs k=3 Comparison"
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            plt.grid(True, alpha=0.3)
            
            # Set legend position to bottom right
            plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
            
            # Make y-axis ticks more readable
            plt.tick_params(axis='both', which='major', labelsize=12)
            
            # Create output filename
            output_filename = f"{model}_{feature}_k2_k3_comparison.png"
            output_path = os.path.join(plots_dir, output_filename)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created comparison plot: {output_filename}")

def compare_ppo_models(results_dir):
    """Compare PPO models with and without clustering for k=2 and k=3 separately."""
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Define features to compare
    features = ['baseline', 'Usefulness-as-Rwd', 'Weighted-Usefulness-as-Rwd']
    
    # Define color scheme and styles
    colors = {
        'baseline': {
            'clustered': '#1f77b4',     # Dark blue
            'non_clustered': '#7fb3d5'  # Light blue
        },
        'Usefulness-as-Rwd': {
            'clustered': '#d62728',     # Dark red
            'non_clustered': '#ff9999'  # Light red
        },
        'Weighted-Usefulness-as-Rwd': {
            'clustered': '#006400',     # Dark green
            'non_clustered': '#4daf4a'  # Medium green instead of light green
        }
    }
    
    # Create separate plots for k=2 and k=3
    for k in ['2', '3']:
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Store last points for text positioning
        last_points = []
        
        # First pass: collect all last points
        for feature in features:
            for is_clustered in [True, False]:
                if is_clustered:
                    pattern = f"all_ppo_{feature}_k_{k}_total_steps_500000_clusters_auto_run_*.txt"
                else:
                    pattern = f"all_ppo_{feature}_k_{k}_total_steps_500000_run_*.txt"
                
                files = glob.glob(os.path.join(results_dir, "**", pattern), recursive=True)
                if files:
                    steps, metrics = load_evaluation_data(files[0])
                    last_points.append({
                        'step': steps[-1],
                        'metric': metrics[-1],
                        'feature': feature,
                        'is_clustered': is_clustered
                    })
        
        # Sort points by metric value
        last_points.sort(key=lambda x: x['metric'])
        
        # Second pass: plot lines and adjust text positions
        for feature in features:
            for is_clustered in [True, False]:
                if is_clustered:
                    pattern = f"all_ppo_{feature}_k_{k}_total_steps_500000_clusters_auto_run_*.txt"
                else:
                    pattern = f"all_ppo_{feature}_k_{k}_total_steps_500000_run_*.txt"
                
                files = glob.glob(os.path.join(results_dir, "**", pattern), recursive=True)
                if files:
                    # Load and plot data
                    steps, metrics = load_evaluation_data(files[0])
                    
                    # Define line style and label
                    linestyle = '-' if is_clustered else '--'
                    color = colors[feature]['clustered' if is_clustered else 'non_clustered']
                    
                    # Format feature name for label
                    feature_name = feature.replace('-', ' ').title()
                    if feature == 'Usefulness-as-Rwd':
                        feature_name = 'UIR'
                    elif feature == 'Weighted-Usefulness-as-Rwd':
                        feature_name = 'WUIR'
                    
                    # Add clustering indicator to label
                    clustering_indicator = 'CL-' if is_clustered else ''
                    label = f"{clustering_indicator}{feature_name}"
                    
                    # Plot the line
                    plt.plot(steps, metrics, color=color, linestyle=linestyle, linewidth=2, label=label)
                    
                    # Add marker for the last point
                    last_step = steps[-1]
                    last_metric = metrics[-1]
                    plt.plot(last_step, last_metric, 'o', color=color, markersize=8)
                    
                    # Find index of current point in sorted list
                    current_idx = next(i for i, p in enumerate(last_points) 
                                     if p['feature'] == feature and p['is_clustered'] == is_clustered)
                    
                    # Calculate base offset
                    x_offset = (max(steps) - min(steps)) * 0.015
                    y_offset = (max(metrics) - min(metrics)) * 0.015
                    
                    # Adjust y_offset based on position in sorted list
                    if current_idx > 0:
                        prev_metric = last_points[current_idx - 1]['metric']
                        # Increase threshold for considering points as "close"
                        if abs(last_metric - prev_metric) < (max(metrics) - min(metrics)) * 0.1:  # Increased from 0.05 to 0.08
                            # Increase vertical spacing between close values
                            y_offset = (current_idx + 1) * y_offset *1.5
                    
                    plt.text(last_step + x_offset, last_metric + y_offset, f'{last_metric:.2f}', 
                            color=color, fontsize=15, fontweight='bold',
                            ha='left', va='bottom')
        
        # Set labels and title
        plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
        
        # Set title
        title = f"PPO Models Comparison (k={k})"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        
        # Increase padding for x and y axis
        plt.xlim(min(steps), max(steps) * 1.11)  # Increase x-axis padding to 11%
        plt.ylim(min(metrics) * 0.95, max(metrics) * 1.15)  # Increase y-axis padding to 15%
   
        # Set legend position to bottom right
        plt.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        # Make y-axis ticks more readable
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Create output filename
        output_filename = f"ppo_models_k{k}_comparison.png"
        output_path = os.path.join(plots_dir, output_filename)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        

        
        print(f"Created comparison plot: {output_filename}")

def show_menu():
    """Display the main menu and get user choice."""
    print("\nLearning Curves Visualization Menu:")
    print("1. Generate all plots")
    print("2. Generate only comparison plots")
    print("3. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    # Define directories
    results_dir = os.path.join("Code", "results")
    
    print(f"Generating plots for branch: {BRANCH_NAME}")
    print(f"Plots will be saved in: {os.path.join(results_dir, BRANCH_NAME, 'plots')}")
    print(f"Looking for result files in: {results_dir}")
    
    while True:
        choice = show_menu()
        
        if choice == 1:
            # Plot learning curves
            plot_learning_curves(results_dir)
            
            # Compare DQN models for k=2 and k=3
            compare_dqn_models(results_dir)
            
            # Compare PPO models for k=2 and k=3
            compare_ppo_models(results_dir)
            
        elif choice == 2:
            # Compare DQN models for k=2 and k=3
            compare_dqn_models(results_dir)
            
            # Compare PPO models for k=2 and k=3
            compare_ppo_models(results_dir)
            
        else:  # choice == 3
            break
    
    print("\nLearning curves update completed.") 