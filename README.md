# Course Recommendation System

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs using mastery levels and clustering techniques.

## Overview

This system uses reinforcement learning to recommend courses to learners based on their current skills and job market requirements. It operates in multiple modes:

1. **Baseline Model**: Uses number of applicable jobs as reward (skip-expertise only)
2. **Enhanced Model**: Uses mastery levels with clustering and utility-based rewards
   - **Usefulness-as-Rwd**: Uses utility function as reward
   - **Weighted-Usefulness-as-Rwd**: Combines number of applicable jobs with utility function

## Features

- **Skill Mastery Levels**: Supports beginner, intermediate, and expert skill levels
- **Clustering**: Groups learners based on skill profiles for personalized recommendations
- **Binary and Multi-level Skill Representation**: Flexible skill matching approaches
- Support for multiple RL algorithms (DQN, A2C, PPO)
- Course recommendation based on:
  - Learner's current skills and mastery levels
  - Job market requirements
  - Course prerequisites and outcomes
  - Clustering-based personalization
- Evaluation metrics:
  - Learner attractiveness
  - Number of applicable jobs
  - Course recommendation utility
  - Clustering performance

## Project Structure

```
Code/
├── UIR/                          # Main implementation with mastery levels and clustering
│   ├── CourseRecEnv.py           # RL environment for course recommendations
│   ├── Dataset.py                # Data loading and processing with mastery levels
│   ├── matchings.py              # Skill matching functions
│   ├── pipeline.py               # Main execution pipeline
│   ├── Reinforce.py              # RL model implementation
│   └── weight_optimization.py    # Weight optimization for reward functions
├── config/
│   └── run.yaml                  # Configuration file
├── results/                      # Experiment results and outputs
└── backups/                      # Backup of previous experiments
```

## Requirements

- Python 3.x
- NumPy
- Pandas
- Gymnasium
- Stable-baselines3
- MLflow
- Scikit-learn (for clustering)

## Usage

1. Configure the system in `UIR/config/run.yaml`
2. Run the pipeline:
```bash
python Code/UIR/pipeline.py --config UIR/config/run.yaml
```

## Configuration

The system can be configured through `run.yaml` with parameters like:
- Model type (baseline vs enhanced with mastery levels)
- Number of recommendations (k)
- Matching threshold
- Training steps
- Evaluation frequency
- Feature type:
  - "Usefulness-as-Rwd": Uses utility function as reward
  - "Weighted-Usefulness-as-Rwd": Combines number of applicable jobs with utility

## Reward Mechanisms

The system supports multiple reward mechanisms:

1. **Baseline Mode**:
   - Reward = Number of applicable jobs
   - No mastery levels or clustering

2. **Usefulness-as-Rwd**:
   - Reward = Utility function value
   - Utility considers skill acquisition and job applicability
   - Includes mastery levels and clustering

3. **Weighted-Usefulness-as-Rwd**:
   - Reward = β₁ × Number of applicable jobs + β₂ × Utility function value
   - Combines immediate job eligibility with long-term skill development
   - Configurable weights for different RL algorithms

