import os
import argparse
import yaml

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
    
    This function:
    1. Parses command line arguments to get the configuration file path
    2. Loads the configuration from YAML file
    3. Handles weight optimization if needed
    4. Runs the specified recommendation model for the configured number of iterations
    
    Command line arguments:
        --config: Path to the configuration file (default: "UIR/config/run.yaml")
    """
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument(
        "--config", help="Path to the configuration file", default="UIR/config/run.yaml"
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

    for run in range(config["nb_runs"]):
        
        
        dataset = create_and_print_dataset(config)
        
        # Use the Reinforce class for all models
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

        


if __name__ == "__main__":
    main()