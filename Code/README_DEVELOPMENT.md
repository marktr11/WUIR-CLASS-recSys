# Course Recommendation System - Development Guide

This document provides detailed information for developers working on the course recommendation system with mastery levels and clustering-based reward adjustment.

## System Architecture

### Core Components

1. **RL Environment** (`CourseRecEnv.py`):
   - Implements Gymnasium environment for course recommendations
   - Handles state representation with mastery levels (0-3)
   - Supports clustering-based reward adjustment:
     - Same cluster & reward increase: x1.1
     - Same cluster & reward decrease: x0.9
     - Different cluster & reward increase: x1.3
     - Different cluster & reward decrease: x0.8

2. **Data Management** (`Dataset.py`):
   - Handles data loading and preprocessing
   - Manages learner, job, and course data with mastery levels
   - Provides skill matching functionality considering mastery levels

3. **RL Implementation** (`Reinforce.py`):
   - Implements DQN, A2C, and PPO algorithms
   - Manages model training and evaluation
   - Handles hyperparameter tuning
   - Supports clustering-based reward adjustment

4. **Pipeline** (`pipeline.py`):
   - Orchestrates the training process
   - Manages configuration and logging
   - Handles results storage and visualization
   - Supports multiple k values (1,2,3,...)

## Clustering Implementation

The system uses K-means clustering to group similar courses based on their skill profiles. This helps improve the RL performance by adjusting rewards based on course cluster membership.

### Clustering Features
1. **Skill Coverage**: Percentage of total skills that a course provides
2. **Skill Diversity**: Entropy-based measure of how evenly distributed the course's skills are
3. **Mastery Level Distribution**: Distribution of skill mastery levels in courses

### Reward Adjustment Rules
The clustering mechanism modifies rewards based on cluster transitions:
1. **Same Cluster & Reward Increase**: Moderate encouragement (x1.1)
   - Encourages continued exploration within successful clusters
   - Reduced from 1.2 to make it easier for k=3 to overcome
2. **Same Cluster & Reward Decrease**: Light penalty (x0.9)
   - Slightly discourages actions that decrease reward within same cluster
3. **Different Cluster & Reward Increase**: Strong encouragement (x1.3)
   - Encourages exploration of new clusters when improvements are found
   - Reduced from 1.5 to prevent over-exploration
4. **Different Cluster & Reward Decrease**: Moderate penalty (x0.8)
   - Discourages actions that decrease reward when switching clusters
   - Increased from 0.7 to reduce penalty severity

### Clustering Process
1. **Feature Extraction**:
   - Calculate skill coverage for each course
   - Compute skill diversity using entropy
   - Analyze mastery level distribution
2. **Clustering**:
   - Normalize features using StandardScaler
   - Apply K-means clustering
   - Optionally use elbow method to determine optimal k
3. **Reward Adjustment**:
   - Track cluster transitions during training
   - Apply reward multipliers based on transition rules
   - Store cluster information for analysis

## Configuration Guide

The system is configured through `config/run.yaml` with the following parameters:

### Model Configuration
```yaml
model: "ppo"  # or "dqn", "a2c"
total_steps: 500000
eval_freq: 1000
```

### Environment Configuration
```yaml
threshold: 0.8  # Matching threshold
k: 4  # Number of recommendations
use_clustering: true  # Enable clustering
```

### Clustering Configuration
```yaml
use_clustering: true
n_clusters: 4
auto_clusters: true
max_clusters: 10
```

## Results Management

### Directory Structure
```
Code/results/
├── [branch_name]/      # Results for specific branch
│   ├── plots/          # Plot files
│   │   ├── individual/ # Individual learning curves
│   │   ├── comparison/ # Model comparison plots
│   │   └── clustering/ # Clustering effect plots
│   └── data/           # Training data
└── backups/            # Backup directories
```

### Management Commands
1. List results:
```bash
python Code/manage_results.py list
```

2. Backup results:
```bash
python Code/manage_results.py backup [branch_name]
```

3. Clean up old results:
```bash
python Code/manage_results.py clean [branch_name]
```

## Development Guidelines

1. **Code Organization**:
   - Keep related functionality in the same module
   - Use clear and descriptive function/class names
   - Document all public interfaces
   - Maintain consistent mastery level handling

2. **Testing**:
   - Write unit tests for new functionality
   - Test edge cases and error conditions
   - Maintain test coverage
   - Verify mastery level calculations

3. **Documentation**:
   - Update docstrings when modifying code
   - Keep README files up to date
   - Document configuration changes
   - Document mastery level system

4. **Version Control**:
   - Use meaningful commit messages
   - Create feature branches for new development
   - Review code before merging
   - Track changes to mastery levels

## Important Notes

1. **Results Management**:
   - Always backup results before deleting or switching branches
   - Results are organized by branch structure
   - Do not delete main branch results
   - Keep track of k values in filenames

2. **Model Training**:
   - Each model should be trained separately
   - Compare results between versions
   - Save used hyperparameters
   - Monitor clustering performance

3. **Results Analysis**:
   - Compare performance between models
   - Analyze by cluster
   - Plot learning curves for evaluation
   - Consider mastery level impact

4. **Backup and Version Control**:
   - Backups are stored in `Code/backups/` with timestamp
   - Each branch has its own results directory
   - Do not commit results to git
   - Document mastery level changes 