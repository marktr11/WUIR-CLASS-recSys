import os
import argparse
import mlflow
import yaml


from Dataset import Dataset
from Greedy import Greedy
from Optimal import Optimal
from Reinforce import Reinforce


def create_and_print_dataset(config):
    """Create and print the dataset."""
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main():
    """Run the recommender system based on the provided model and parameters."""
    parser = argparse.ArgumentParser(description="Run recommender models.")

    parser.add_argument(
        "--config", help="Path to the configuration file", default="Code/config/run.yaml"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_classes = {
        "greedy": Greedy,
        "optimal": Optimal,
        "reinforce": Reinforce,
    }

    # --- MLflow: experiment and run ---
    mlflow.set_tracking_uri("http://127.0.0.1:8080")


    if config["feature2"] != "None":
       mlflow.set_experiment(f"{config['feature']}_{config['feature2']}")
    else:
       mlflow.set_experiment(f"{config['feature']}")

    for run in range(config["nb_runs"]):
      
        if config["feature2"] != "None":
           run_name = f"{config['model']}_{config['feature']}_{config['feature2']}_k_{config['k']}_total_steps_{config['total_steps']}"
        else:
           run_name = f"{config['model']}_{config['feature']}_k_{config['k']}_total_steps_{config['total_steps']}"


        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Starting MLflow Run: {run_name}---")
        
            # Log parameters
            print("Logging selected parameters...")
            mlflow.log_param("model", config["model"])
            mlflow.log_param("k_recommendations", config["k"])
            mlflow.log_param("threshold", config["threshold"])
            mlflow.log_param("feature_1", config["feature"]) 
            mlflow.log_param("feature_2", config["feature2"]) 
            mlflow.log_param("level_3_taxonomy", config["level_3"])
            mlflow.log_param("seed", config["seed"])
            if config.get("model") in ["ppo", "dqn"]:
                mlflow.log_param("total_steps", config.get("total_steps"))
                mlflow.log_param("eval_freq", config.get("eval_freq"))

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
                if config["feature"] == "skip-expertise":
                    print("feature: skip-expertise")
                    print("-------------------------------------------")
                else: 
                    print("feature: Original")
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
                    
                )
                recommender.reinforce_recommendation()



                # --- MLflow: Log Tags 
                mlflow.set_tag("model_class", config["model"])
                mlflow.set_tag("feature_mode", config["feature"])

                print(f"\n--- Finished MLflow Run: {run_name} ---")


if __name__ == "__main__":
    main()
