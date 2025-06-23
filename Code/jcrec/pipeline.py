import os
import argparse
import yaml
import numpy as np

from Dataset import Dataset
from Reinforce import Reinforce


def create_and_print_dataset(config):
    """Create and initialize the dataset for the recommendation system.
    
    This function creates a Dataset instance using the provided configuration
    and prints its summary information.
    
    Args:
        config (dict): Configuration dictionary containing dataset parameters
        
    Returns:
        Dataset: Initialized dataset object containing learners, jobs, and courses
    """
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main():
    """Main entry point for the recommendation system pipeline.
    
    This function orchestrates the entire recommendation process:
    1. Parses command line arguments to get the configuration file path
    2. Loads the configuration from YAML file
    3. Runs the specified recommendation model for configured iterations
    
    Command line arguments:
        --config: Path to the configuration file (default: "Code/config/run.yaml")
    """
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument(
        "--config", help="Path to the configuration file", default=r"Code/config/run.yaml"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for run in range(config["nb_runs"]):
        run_name = f"{config['model']}_k_{config['k']}_total_steps_{config['total_steps']}"

        # Add clustering info to run name if enabled
        if config["use_clustering"]:
            run_name = f"{run_name}_clusters_auto"

        print(f"\n--- Starting Run: {run_name}---")
        
        dataset = create_and_print_dataset(config)
        
        # Use the Reinforce class for all models
        recommender = Reinforce(
            dataset,
            config["model"],
            config["k"],
            config["threshold"],
            run,
            config["total_steps"],
            config["eval_freq"]
        )
        recommender.reinforce_recommendation()

        # Handle clustering metrics if enabled
        if config["use_clustering"]:
            # Get clusterer from the environment
            clusterer = recommender.train_env.clusterer
            
            # Update run name with optimal_k if available
            if hasattr(clusterer, 'optimal_k'):
                optimal_k = clusterer.optimal_k
                print(f"Optimal number of clusters: {optimal_k}")
                # Update run name with optimal_k
                new_run_name = f"{run_name}_k{optimal_k}"
            
            # Print clustering metrics
            if hasattr(clusterer, 'inertia_'):
                print(f"Clustering inertia: {clusterer.inertia_}")

        print(f"\n--- Finished Run: {run_name} ---")


if __name__ == "__main__":
    main()
