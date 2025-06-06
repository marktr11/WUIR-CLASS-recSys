"""
Clustering Module for Course Recommendation System

This module implements clustering functionality to improve RL performance by adjusting rewards
based on course cluster membership. The clustering helps identify similar courses based on their
provided skills, which is then used to modify the reward signal to encourage more stable learning.

The reward adjustment follows these rules:
1. Same cluster & reward increase: Moderate encouragement (x1.1)
   - Encourages the agent to continue exploring within the same cluster when it's working well
   - Reduced from 1.2 to make it easier for k=3 to overcome
2. Same cluster & reward decrease: Light penalty (x0.9)
   - Slightly discourages actions that decrease reward within the same cluster
3. Different cluster & reward increase: Strong encouragement (x1.3)
   - Encourages the agent to explore new clusters when it finds improvements
   - Reduced from 1.5 to prevent over-exploration
4. Different cluster & reward decrease: Moderate penalty (x0.8)
   - Discourages actions that decrease reward when switching clusters
   - Increased from 0.7 to reduce penalty severity

The clustering is based on two key metrics for each course:
1. Skill Coverage: The percentage of total skills that the course provides
2. Skill Diversity: An entropy-based measure of how evenly distributed the course's skills are

Example:
    ```python
    # Initialize clusterer
    clusterer = CourseClusterer(n_clusters=5, random_state=42, auto_clusters=True)
    
    # Fit clusters to courses
    clusterer.fit_course_clusters(courses)
    
    # Adjust reward based on clustering
    adjusted_reward = clusterer.adjust_reward(
        course_idx=0,
        original_reward=1.0,
        prev_reward=0.8
    )
    ```
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class CourseClusterer:
    """Class for clustering courses and adjusting rewards based on cluster membership.
    
    This class implements a clustering-based reward adjustment mechanism that helps
    stabilize and improve the learning process in the RL environment. It clusters
    courses based on their provided skills and uses this information to modify
    rewards according to predefined rules.
    
    Attributes:
        n_clusters (int): Number of clusters to create
        course_clusters (np.ndarray): Array of cluster assignments for each course
        scaler (StandardScaler): Scaler for normalizing features before clustering
        prev_cluster (int): Cluster of the previous course recommendation
        features (np.ndarray): Original features used for clustering
        random_state (int): Random seed for reproducibility
        auto_clusters (bool): Whether to automatically determine optimal number of clusters
        max_clusters (int): Maximum number of clusters to try when using elbow method
        optimal_k (int): Optimal number of clusters determined by elbow method
        clustering_dir (str): Directory to save clustering results
        reward_multipliers (dict): Dictionary of reward adjustment multipliers
    """
    
    def __init__(self, n_clusters=5, random_state=42, auto_clusters=False, max_clusters=10, config=None):
        """Initialize the clusterer.
        
        Args:
            n_clusters (int): Number of clusters to create
            random_state (int): Random seed for reproducibility
            auto_clusters (bool): Whether to automatically determine optimal number of clusters
            max_clusters (int): Maximum number of clusters to try when using elbow method
            config (dict): Configuration dictionary containing reward multipliers
        """
        self.n_clusters = n_clusters
        self.course_clusters = None
        self.scaler = StandardScaler()
        self.prev_cluster = None
        self.features = None
        self.random_state = random_state
        self.auto_clusters = auto_clusters
        self.max_clusters = max_clusters
        self.optimal_k = None
        
        # Set reward multipliers from config or use defaults
        self.reward_multipliers = {
            'same_cluster_increase': config.get('same_cluster_increase', 1.1) if config else 1.1,
            'same_cluster_decrease': config.get('same_cluster_decrease', 0.9) if config else 0.9,
            'diff_cluster_increase': config.get('diff_cluster_increase', 1.3) if config else 1.3,
            'diff_cluster_decrease': config.get('diff_cluster_decrease', 0.8) if config else 0.8
        }
        
        # Create Clustering directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.clustering_dir = os.path.join(current_dir, "..", "Clustering")
        os.makedirs(self.clustering_dir, exist_ok=True)
        
    def find_optimal_clusters(self, features_scaled):
        """Find optimal number of clusters using elbow method."""
        plot_path = os.path.join(self.clustering_dir, 'elbow_curve_skillslevels.png')
        print("\nFinding optimal number of clusters...")
        
        inertias = []
        K = range(1, self.max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Calculate the rate of change of inertia
        changes = np.diff(inertias)
        changes_r = np.diff(changes)
        
        # Find the elbow point (where the rate of change changes most)
        optimal_k = np.argmax(changes_r) + 2  # +2 because we took two diffs
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
        plt.legend()
        
        # Save plot
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
        
    def fit_course_clusters(self, courses):
        """Fit clusters for courses based on their required and provided skills."""
        print("\nStarting course clustering...")
        required_skills = courses[:, 0]  # Get required skills
        provided_skills = courses[:, 1]  # Get provided skills
        
        # Calculate skill coverage and diversity metrics
        n_skills = required_skills.shape[1]
        
        # Calculate coverage considering both required and provided skills
        # Coverage is now the average of required and provided skill levels
        required_coverage = np.sum(required_skills, axis=1) / n_skills
        provided_coverage = np.sum(provided_skills, axis=1) / n_skills
        coverage = (required_coverage + provided_coverage) / 2
        
        # Calculate skill diversity using entropy for both required and provided skills
        # For required skills
        required_distribution = required_skills / (np.sum(required_skills, axis=1, keepdims=True) + 1e-10)
        required_entropy = -np.sum(required_distribution * np.log2(required_distribution + 1e-10), axis=1)
        
        # For provided skills
        provided_distribution = provided_skills / (np.sum(provided_skills, axis=1, keepdims=True) + 1e-10)
        provided_entropy = -np.sum(provided_distribution * np.log2(provided_distribution + 1e-10), axis=1)
        
        # Combine metrics into features for clustering
        # Now we have 4 features: coverage, required_entropy, provided_entropy
        self.features = np.column_stack([coverage, required_entropy, provided_entropy])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(self.features)
        
        # Find optimal number of clusters if auto_clusters is enabled
        if self.auto_clusters:
            self.optimal_k = self.find_optimal_clusters(features_scaled)
            self.n_clusters = self.optimal_k
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.course_clusters = kmeans.fit_predict(features_scaled)
        
        # Store cluster centers and inertia
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        
        # Print cluster information
        print("\nCluster Information:")
        for i in range(self.n_clusters):
            n_formations = np.sum(self.course_clusters == i)
            print(f"Cluster {i}: {n_formations} formations")
        
        # Visualize clusters
        self.visualize_clusters(features_scaled)
        
    def visualize_clusters(self, features_scaled):
        """Visualize the clusters using multiple 2D plots."""
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Coverage vs Required Entropy
        for i in range(self.n_clusters):
            cluster_points = features_scaled[self.course_clusters == i]
            n_formations = len(cluster_points)
            
            # Plot points
            ax1.scatter(
                cluster_points[:, 0],  # Coverage
                cluster_points[:, 1],  # Required Entropy
                label=f'Cluster {i} ({n_formations} courses)',
                alpha=0.7,
                s=100
            )
            
            # Plot cluster center
            center = self.cluster_centers_[i]
            ax1.scatter(
                center[0],
                center[1],
                c='black',
                marker='x',
                s=200,
                linewidths=3
            )
        
        ax1.set_xlabel('Skill Coverage (scaled)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Required Skills Entropy (scaled)', fontsize=12, fontweight='bold')
        ax1.set_title('Coverage vs Required Entropy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Coverage vs Provided Entropy
        for i in range(self.n_clusters):
            cluster_points = features_scaled[self.course_clusters == i]
            
            # Plot points
            ax2.scatter(
                cluster_points[:, 0],  # Coverage
                cluster_points[:, 2],  # Provided Entropy
                label=f'Cluster {i} ({len(cluster_points)} courses)',
                alpha=0.7,
                s=100
            )
            
            # Plot cluster center
            center = self.cluster_centers_[i]
            ax2.scatter(
                center[0],
                center[2],
                c='black',
                marker='x',
                s=200,
                linewidths=3
            )
        
        ax2.set_xlabel('Skill Coverage (scaled)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Provided Skills Entropy (scaled)', fontsize=12, fontweight='bold')
        ax2.set_title('Coverage vs Provided Entropy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Required vs Provided Entropy
        for i in range(self.n_clusters):
            cluster_points = features_scaled[self.course_clusters == i]
            
            # Plot points
            ax3.scatter(
                cluster_points[:, 1],  # Required Entropy
                cluster_points[:, 2],  # Provided Entropy
                label=f'Cluster {i} ({len(cluster_points)} courses)',
                alpha=0.7,
                s=100
            )
            
            # Plot cluster center
            center = self.cluster_centers_[i]
            ax3.scatter(
                center[1],
                center[2],
                c='black',
                marker='x',
                s=200,
                linewidths=3
            )
        
        ax3.set_xlabel('Required Skills Entropy (scaled)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Provided Skills Entropy (scaled)', fontsize=12, fontweight='bold')
        ax3.set_title('Required vs Provided Entropy', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Create a single legend for all subplots
        handles, labels = ax1.get_legend_handles_labels()
        # Add total courses to the legend
        total_formations = len(self.course_clusters)
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=0))
        labels.append(f'Total Courses: {total_formations}')
        
        fig.legend(
            handles, labels,
            loc='center',
            bbox_to_anchor=(0.5,   0.03),
            ncol=self.n_clusters + 1,  # +1 for total courses
            prop={'weight': 'bold', 'size': 12},
            frameon=True,
            framealpha=0.9,
        )
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for the legend
        
        # Save plot with high DPI for better quality
        plot_path = os.path.join(self.clustering_dir, f'cluster_visualization_skillslevels.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def adjust_reward(self, course_idx, original_reward, prev_reward):
        """Adjust reward based on cluster membership and reward change.
        
        This method implements the reward adjustment rules based on whether the
        course is in the same cluster as the previous course and whether the
        reward has increased or decreased.
        
        For first recommendation in any sequence (k=1,2,3), no reward adjustment is applied.
        
        For subsequent recommendations (k>1), the reward is adjusted based on:
        - Whether the course is in the same cluster as the previous course
        - Whether the reward has increased or decreased
        
        Reward adjustment rules:
        1. Same cluster & reward increase: x{same_cluster_increase}
        2. Same cluster & reward decrease: x{same_cluster_decrease}
        3. Different cluster & reward increase: x{diff_cluster_increase}
        4. Different cluster & reward decrease: x{diff_cluster_decrease}
        
        Args:
            course_idx (int): Index of the current course
            original_reward (float): Original reward value from the environment
            prev_reward (float): Reward value from the previous step
            
        Returns:
            float: Adjusted reward value based on clustering rules
        """
        if self.course_clusters is None:
            return original_reward
            
        # Get current course's cluster
        current_cluster = self.course_clusters[course_idx]
        
        # For first recommendation in any sequence (k=1,2,3)
        # Check if this is the first recommendation (prev_reward is None or 0)
        if prev_reward is None or prev_reward == 0:
            # Store current cluster for next comparison
            self.prev_cluster = current_cluster
            # Return original reward without adjustment
            return original_reward
            
        # For subsequent recommendations in sequence (k>1)
        # Calculate reward change
        reward_change = original_reward - prev_reward
        
        # Store current cluster for next comparison
        self.prev_cluster = current_cluster
        
        # Apply reward adjustment rules using multipliers from config
        if reward_change > 0:  # Reward increased
            if current_cluster == self.prev_cluster:  # Same cluster
                return original_reward * self.reward_multipliers['same_cluster_increase']
            else:  # Different cluster
                return original_reward * self.reward_multipliers['diff_cluster_increase']
        else:  # Reward decreased
            if current_cluster == self.prev_cluster:  # Same cluster
                return original_reward * self.reward_multipliers['same_cluster_decrease']
            else:  # Different cluster
                return original_reward * self.reward_multipliers['diff_cluster_decrease'] 