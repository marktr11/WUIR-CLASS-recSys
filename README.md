# Course Recommendation System

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs.

## Overview

This system uses reinforcement learning to recommend courses to learners based on their current skills and job market requirements. It operates in three modes:

1. **Baseline Model**: Uses number of applicable jobs as reward
2. **Usefulness-as-Rwd (UIR)**: Uses utility function as reward
3. **Weighted-Usefulness-as-Rwd (WUIR)**: Combines number of applicable jobs with utility function

## Key Features

- Binary skill representation (0/1) for simplified skill matching
- Support for multiple RL algorithms (DQN, A2C, PPO)
- Course recommendation based on:
  - Learner's current skills
  - Job market requirements
  - Course outcomes
- K-means clustering for course grouping and reward adjustment
- Comprehensive evaluation metrics

## Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Configure the system in `Code/config/run.yaml`

3. Run the pipeline:
```bash
python Code/jcrec/pipeline.py --config Code/config/run.yaml
```

## Project Structure

```
Code/
├── jcrec/              # Core recommendation system
├── config/             # Configuration files
├── results/            # Training results and plots
└── README_DEVELOPMENT.md  # Detailed development guide
```

## Requirements

- Python 3.x
- NumPy
- Pandas
- Gymnasium
- Stable-baselines3
- MLflow
- scikit-learn

## Documentation

For detailed information about:
- Development setup and guidelines
- Configuration options
- Results management
- Clustering implementation
- Model training and evaluation

Please refer to `Code/README_DEVELOPMENT.md` 