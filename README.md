# Course Recommendation System

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs using clustering-based reward adjustment and usefulness-based approaches.

## Overview

This system uses reinforcement learning to recommend courses to learners based on their current skills and job market requirements. It operates with two different approaches:

1. **CLASS**: Clustering-based approach
   - Groups similar courses using K-means clustering
   - Adjusts rewards based on cluster transitions
   - Encourages stable learning patterns within course clusters
   - Uses mastery levels (0-3) for skill representation

2. **UIR**: Usefulness-based approach
   - Uses information usefulness as the primary metric
   - Implements weighted reward functions
   - Focuses on maximizing the utility of course recommendations
   - Uses binary skill representation (0-1)

## Key Features

- Support for multiple RL algorithms (DQN, A2C, PPO)
- Course recommendation based on:
  - Learner's current skill levels
  - Job market requirements
  - Course outcomes
- Two distinct approaches:
  - **Clustering-based**: K-means clustering for course grouping and reward adjustment
  - **Usefulness-based**: Information utility maximization with weighted rewards
- Comprehensive evaluation metrics
- Automatic cluster optimization using elbow method (CLASS)
- Weight optimization for usefulness-based rewards (UIR)

## Project Structure

```
Project/
├── CLASS/                      # Clustering-based approach
│   ├── Scripts/              # Core recommendation system
│   │   ├── CourseRecEnv.py   # RL environment with clustering
│   │   ├── Reinforce.py      # RL implementation
│   │   ├── Dataset.py        # Data management
│   │   └── clustering.py     # Clustering implementation
│   ├── config/               # Configuration files
│   ├── results/              # Training results and plots
│   └── README_DEVELOPMENT.md # Detailed development guide
├── UIR/                      # Usefulness-based approach
│   ├── Scripts/              # Core recommendation system
│   │   ├── CourseRecEnv.py   # RL environment with usefulness
│   │   ├── Reinforce.py      # RL implementation
│   │   ├── Dataset.py        # Data management
│   │   └── weight_optimization.py # Weight optimization
│   ├── config/               # Configuration files
│   ├── results/              # Training results and plots
│   └── README_DEVELOPMENT.md # Detailed development guide
└── Data - Collection/        # Dataset files
    └── Final/
        ├── courses.json      # Course data
        ├── jobs.json         # Job listings
        ├── resumes.json      # Learner profiles
        ├── taxonomy.csv      # Skill taxonomy
        └── mastery_levels.json # Skill mastery definitions
```

## Quick Start

### CLASS (Clustering-based)

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Configure the system in `config/run.yaml`:
   - Set clustering parameters
   - Choose RL algorithm and training settings
   - Configure evaluation metrics

3. Run the pipeline:
```bash
cd UIR
python Scripts/pipeline.py --config config/run.yaml
```

### UIR (Usefulness-based)

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Configure the system in `config/run.yaml`:
   - Set usefulness parameters and weights
   - Choose RL algorithm and training settings
   - Configure evaluation metrics

3. Run the pipeline:
```bash
cd UIR
python Scripts/pipeline.py --config config/run.yaml
```

## Requirements

### Dependencies

The project requires the following Python packages:

**Core ML/RL Libraries:**
- `stable-baselines3==2.2.1` - Reinforcement learning algorithms (DQN, PPO, A2C)
- `gymnasium>=0.28.0` - RL environment interface (compatible with stable-baselines3)
- `scikit-learn>=1.0.0` - Machine learning utilities (K-means clustering, PCA)

**Data Processing:**
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation and analysis

**Visualization:**
- `matplotlib>=3.5.0` - Plotting and visualization
- `seaborn>=0.11.0` - Statistical data visualization

**Configuration:**
- `PyYAML==6.0.1` - YAML configuration files

**Utilities:**
- `tqdm>=4.62.0` - Progress bars for weight optimization and long-running operations

## Documentation

For detailed information about:
- Development setup and guidelines
- Configuration options
- Results management
- Clustering implementation (CLASS)
- Usefulness-based approach (UIR)
- Model training and evaluation

Please refer to:
- `CLASS/README_DEVELOPMENT.md` for clustering-based approach
- `UIR/README_DEVELOPMENT.md` for usefulness-based approach

## Acknowledgements

This project is developed based on the repository [JCRec](https://github.com/Jibril-Frej/JCRec) by [Jibril Frej](https://github.com/Jibril-Frej).

