# Course Recommendation System

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs.

## Overview

This system uses reinforcement learning to recommend courses to learners based on their current skills and job market requirements. It operates in two modes:

1. **Baseline Mode**: Uses number of applicable jobs as reward
2. **No-Mastery-Levels Mode**: Uses a utility function that considers both skill acquisition and job applicability

## Features

- Binary skill representation (0/1) for simplified skill matching
- Support for multiple RL algorithms (DQN, A2C, PPO)
- Course recommendation based on:
  - Learner's current skills
  - Job market requirements
  - Course prerequisites and outcomes
- Evaluation metrics:
  - Learner attractiveness
  - Number of applicable jobs
  - Course recommendation utility

## Project Structure

```
Code/
├── jcrec/
│   ├── CourseRecEnv.py    # RL environment for course recommendations
│   ├── Dataset.py         # Data loading and processing
│   ├── matchings.py       # Skill matching functions
│   ├── pipeline.py        # Main execution pipeline
│   └── Reinforce.py       # RL model implementation
└── config/
    └── run.yaml          # Configuration file
```

## Requirements

- Python 3.x
- NumPy
- Pandas
- Gymnasium
- Stable-baselines3
- MLflow

## Usage

1. Configure the system in `Code/config/run.yaml`
2. Run the pipeline:
```bash
python Code/jcrec/pipeline.py --config Code/config/run.yaml
```

## Configuration

The system can be configured through `run.yaml` with parameters like:
- Model type (greedy, optimal, reinforce)
- Number of recommendations (k)
- Matching threshold
- Training steps
- Evaluation frequency

## Evaluation

The system tracks and logs:
- Original and new learner attractiveness
- Number of applicable jobs
- Recommendation time
- Course recommendations for each learner 