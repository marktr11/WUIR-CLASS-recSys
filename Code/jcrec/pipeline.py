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
    
    This function:
    1. Parses command line arguments to get the configuration file path
    2. Loads the configuration from YAML file
    3. Sets up MLflow experiment tracking
    4. Runs the specified recommendation model for the configured number of iterations
    5. Logs parameters, metrics, and artifacts to MLflow
    
    The pipeline supports three types of recommendation models:
    - Greedy: Simple greedy approach for course recommendations
    - Optimal: Optimal solution using mathematical optimization
    - Reinforce: Reinforcement learning-based approach (DQN, A2C, or PPO)
    
    For each run, it:
    - Creates a new MLflow run with appropriate naming
    - Logs all relevant parameters and configuration
    - Initializes the dataset
    - Runs the selected recommendation model
    - Logs results and artifacts
    
    Command line arguments:
        --config: Path to the configuration file (default: "Code/config/run.yaml")
    """
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument(
        "--config", help="Path to the configuration file", default="Code/config/run.yaml"
    )

    args = parser.parse_args()

    # First load initial config
    with open(args.config, "r") as f:
        initial_config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize beta1 and beta2 as None
    beta1 = None
    beta2 = None

    # Run weight optimization if using weighted reward and weights are not in config
    if initial_config.get("feature") == "Weighted-Usefulness-as-Rwd":
        model_weights = initial_config.get("model_weights", {})
        if initial_config["model"] not in model_weights:
            print(f"\nOptimizing weights for {initial_config['model'].upper()}...")
            from weight_optimization import optimize_weights
            optimize_weights(args.config)
            
            # Reload config after weight optimization
            with open(args.config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = initial_config
            weights = model_weights[initial_config["model"]]
            print(f"\nUsing existing weights for {initial_config['model'].upper()}: beta1={weights['beta1']}, beta2={weights['beta2']}")

        # Get beta values for current model
        model_weights = config.get("model_weights", {})
        current_weights = model_weights.get(config["model"], {})
        beta1 = current_weights.get("beta1")
        beta2 = current_weights.get("beta2")
    else:
        config = initial_config

    model_classes = {
        "greedy": Greedy,
        "optimal": Optimal,
        "reinforce": Reinforce,
    }

    # --- MLflow: experiment and run ---
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    mlflow.set_experiment("CLUSTERING-AJUSTED-RWD-EXP2")

    for run in range(config["nb_runs"]):
        if config["baseline"]:
            run_name = f"{config['model']}_baseline_k_{config['k']}_total_steps_{config['total_steps']}"
        else:
            run_name = f"{config['model']}_{config['feature']}_k_{config['k']}_total_steps_{config['total_steps']}"

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
            if config["baseline"]:
                mlflow.log_param("feature", "baseline") 
            else:
                mlflow.log_param("feature", config["feature"]) 
            mlflow.log_param("level_3_taxonomy", config["level_3"])
            mlflow.log_param("seed", config["seed"])
            if config.get("model") in ["ppo", "dqn"]:
                mlflow.log_param("total_steps", config.get("total_steps"))
                mlflow.log_param("eval_freq", config.get("eval_freq"))

            # Log clustering parameters if enabled
            if config["use_clustering"]:
                mlflow.log_param("use_clustering", True)
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
                if config["baseline"]: 
                    print("feature: baseline")
                    print("-------------------------------------------")
                else: 
                    print(f"feature: {config['feature']}")
                    print("-------------------------------------------")
                recommender = Reinforce(
                    dataset,
                    config["model"],
                    config["k"],
                    config["threshold"],
                    run,
                    config["total_steps"],
                    config["eval_freq"],
                    config["feature"],
                    config["baseline"],
                    beta1,
                    beta2
                )
                recommender.reinforce_recommendation()

                # Log clustering metrics after recommender is initialized
                if config["use_clustering"]:
                    # Get clusterer from the environment
                    clusterer = recommender.train_env.clusterer
                    
                    # Update run name with optimal_k if available
                    if hasattr(clusterer, 'optimal_k'):
                        optimal_k = clusterer.optimal_k
                        mlflow.set_tag("optimal_k", optimal_k)
                        # Update run name with optimal_k
                        new_run_name = f"{run_name}_k{optimal_k}"
                        mlflow.set_tag("mlflow.runName", new_run_name)
                    
                    # Log clustering metrics
                    if hasattr(clusterer, 'inertia_'):
                        mlflow.log_metric("clustering_inertia", clusterer.inertia_)

                # --- MLflow: Log Tags 
                mlflow.set_tag("model_class", config["model"])
                mlflow.set_tag("feature_mode", config["feature"])

                print(f"\n--- Finished MLflow Run: {run_name} ---")


if __name__ == "__main__":
    main()
