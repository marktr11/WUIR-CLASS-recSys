# Course Recommendation System

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs using clustering-based reward adjustment and usefulness-based approaches.

## Overview

This system uses reinforcement learning to recommend courses to learners based on their current skills and job market requirements. It operates with two different approaches:

1. **JCREC-CourseCLF**: Clustering-based approach
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
- Automatic cluster optimization using elbow method (JCREC-CourseCLF)
- Weight optimization for usefulness-based rewards (UIR)

## Project Structure

```
Project/
├── JCREC-CourseCLF/          # Clustering-based approach
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

### JCREC-CourseCLF (Clustering-based)

1. Install requirements:
```bash
cd JCREC-CourseCLF
pip install -r requirements.txt
```

2. Configure the system in `config/run.yaml`:
   - Set clustering parameters
   - Choose RL algorithm and training settings
   - Configure evaluation metrics

3. Run the pipeline:
```bash
python Scripts/pipeline.py --config config/run.yaml
```

### UIR (Usefulness-based)

1. Install requirements:
```bash
cd UIR
pip install -r requirements.txt
```

2. Configure the system in `config/run.yaml`:
   - Set usefulness parameters and weights
   - Choose RL algorithm and training settings
   - Configure evaluation metrics

3. Run the pipeline:
```bash
python Scripts/pipeline.py --config config/run.yaml
```

## Requirements

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```


## Documentation

For detailed information about:
- Development setup and guidelines
- Configuration options
- Results management
- Clustering implementation (JCREC-CourseCLF)
- Usefulness-based approach (UIR)
- Model training and evaluation

Please refer to:
- `JCREC-CourseCLF/README_DEVELOPMENT.md` for clustering-based approach
- `UIR/README_DEVELOPMENT.md` for usefulness-based approach 

## Acknowledgements

This project is developed based on the repository [JCRec](https://github.com/Jibril-Frej/JCRec) by [Jibril Frej](https://github.com/Jibril-Frej).

