import os
import argparse
import mlflow
import yaml
import numpy as np


from Dataset import Dataset
from Greedy import Greedy
from Optimal import Optimal
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
    3. Sets up MLflow experiment tracking
    4. Runs the specified recommendation model for configured iterations
    5. Logs parameters, metrics, and artifacts to MLflow
    
    For each run:
    - Creates a new MLflow run with appropriate naming
    - Logs all relevant parameters and configuration
    - Initializes the dataset
    - Runs the selected recommendation model
    - Logs results and artifacts
    
    For clustering-enabled runs:
    - Logs clustering parameters and metrics
    - Updates run name with optimal number of clusters
    - Tracks clustering performance metrics
    
    Command line arguments:
        --config: Path to the configuration file (default: "Code/config/run.yaml")
    """
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument(
        "--config", help="Path to the configuration file", default="Code/config/run.yaml"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_classes = {
        "greedy": Greedy,
        "optimal": Optimal,
        "reinforce": Reinforce,
    }

    # --- MLflow: experiment and run ---
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("CLUSTERING-EXP3")

    for run in range(config["nb_runs"]):
        run_name = f"{config['model']}_k_{config['k']}_total_steps_{config['total_steps']}"

        # Add clustering info to run name if enabled
        if config["use_clustering"]:
            # We'll update this after getting optimal_k
            run_name = f"{run_name}_clusters_auto"

        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Starting MLflow Run: {run_name}---")
        
            # Log parameters
            print("Logging selected parameters...")
            mlflow.log_param("model", config["model"])
            mlflow.log_param("k_recommendations", config["k"])
            mlflow.log_param("threshold", config["threshold"])
            mlflow.log_param("level_3_taxonomy", config["level_3"])
            mlflow.log_param("seed", config["seed"])
            if config.get("model") in ["ppo", "dqn"]:
                mlflow.log_param("total_steps", config.get("total_steps"))
                mlflow.log_param("eval_freq", config.get("eval_freq"))

            # Log clustering parameters if enabled
            if config["use_clustering"]:
                mlflow.log_param("use_clustering", True)
                if config.get("auto_clusters", False):
                    mlflow.log_param("auto_clusters", True)
                    mlflow.log_param("max_clusters", config.get("max_clusters", 10))
                else:
                    mlflow.log_param("auto_clusters", False)
                    mlflow.log_param("n_clusters", config["n_clusters"])
                mlflow.log_param("clustering_random_state", config["seed"])

            mlflow.log_param("nb_cvs", config["nb_cvs"])
            mlflow.log_param("nb_jobs", config["nb_jobs"])
            mlflow.log_param("nb_courses", config["nb_courses"])
            mlflow.log_param("run_iteration", run) 
            mlflow.log_param("config_file_path", args.config)

            # --- MLflow: Log Artifact 
            print("Logging config file artifact...\n")
            print("-------------------------------------------\n")
            mlflow.log_artifact(args.config, artifact_path="config")

            dataset = create_and_print_dataset(config)
            # If the model is greedy or optimal, we use the corresponding class defined in Greedy.py and Optimal.py
            if config["model"] in ["greedy", "optimal"]:
                recommender = model_classes[config["model"]](dataset, config["threshold"])
                recommendation_method = getattr(
                    recommender, f'{config["model"]}_recommendation'
                )
                recommendation_method(config["k"], run)
            # Otherwise, we use the Reinforce class, described in Reinforce.py
            else:
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

                # Log clustering metrics after recommender is initialized
                if config["use_clustering"]:
                    # Get clusterer from the environment
                    clusterer = recommender.train_env.clusterer
                    
                    # Update run name with optimal_k if available
                    if hasattr(clusterer, 'optimal_k'):
                        optimal_k = clusterer.optimal_k
                        mlflow.log_param("optimal_k", optimal_k)  # Log as parameter instead of tag
                        # Update run name with optimal_k
                        new_run_name = f"{run_name}_k{optimal_k}"
                        mlflow.set_tag("mlflow.runName", new_run_name)
                    
                    # Log clustering metrics
                    if hasattr(clusterer, 'inertia_'):
                        mlflow.log_metric("clustering_inertia", clusterer.inertia_)

                # --- MLflow: Log Tags 
                mlflow.set_tag("model_class", config["model"])

                print(f"\n--- Finished MLflow Run: {run_name} ---")


if __name__ == "__main__":
    main()
