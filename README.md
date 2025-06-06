# Course Recommendation System

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs using mastery levels and clustering-based reward adjustment.

## Overview

This system uses reinforcement learning to recommend courses to learners based on their current skills and job market requirements. It operates with mastery levels and clustering:

1. **Mastery Levels**: Skills are represented with different levels of proficiency (1-3)
   - Level 1: Basic mastery
   - Level 2: Intermediate mastery
   - Level 3: Advanced mastery

2. **Clustering-based Reward Adjustment**:
   - Groups similar courses using K-means clustering
   - Adjusts rewards based on cluster transitions
   - Encourages stable learning patterns

## Key Features

- Mastery level representation (0-3) for detailed skill matching
- Support for multiple RL algorithms (DQN, A2C, PPO)
- Course recommendation based on:
  - Learner's current skill levels
  - Job market requirements
  - Course outcomes
- K-means clustering for course grouping and reward adjustment
- Comprehensive evaluation metrics
- Automatic cluster optimization using elbow method

## Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Configure the system in `Code/config/run.yaml`:
   - Set mastery levels and clustering parameters
   - Choose RL algorithm and training settings
   - Configure evaluation metrics

3. Run the pipeline:
```bash
python Code/jcrec/pipeline.py --config Code/config/run.yaml
```

## Project Structure

```
Code/
├── jcrec/              # Core recommendation system
│   ├── CourseRecEnv.py # RL environment with mastery levels
│   ├── Reinforce.py    # RL implementation
│   ├── Dataset.py      # Data management
│   └── clustering.py   # Clustering implementation
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
- Mastery levels system

Please refer to `Code/README_DEVELOPMENT.md` 