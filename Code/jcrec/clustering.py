"""
Clustering Module for Course Recommendation System

This module implements clustering functionality to improve RL performance by adjusting rewards
based on course cluster membership. The clustering helps identify similar courses based on their
provided skills, which is then used to modify the reward signal to encourage more stable learning.

The reward adjustment follows these rules:
1. Same cluster & reward increase: Strong encouragement (x1.2)
   - Encourages the agent to continue exploring within the same cluster when it's working well
2. Same cluster & reward decrease: Light penalty (x0.9)
   - Slightly discourages actions that decrease reward within the same cluster
3. Different cluster & reward increase: Strong encouragement (x1.5)
   - Strongly encourages the agent to explore new clusters when it finds improvements
4. Different cluster & reward decrease: Heavy penalty (x0.7)
   - Heavily discourages actions that decrease reward when switching clusters

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
    """
    
    def __init__(self, n_clusters=5, random_state=42, auto_clusters=False, max_clusters=10):
        """Initialize the clusterer."""
        self.n_clusters = n_clusters
        self.course_clusters = None
        self.scaler = StandardScaler()
        self.prev_cluster = None
        self.features = None
        self.random_state = random_state
        self.auto_clusters = auto_clusters
        self.max_clusters = max_clusters
        self.optimal_k = None
        
        # Create Clustering directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
        self.clustering_dir = os.path.join(current_dir, "..", "Clustering")
        print(f"\nCurrent directory: {current_dir}")
        print(f"Creating clustering directory at: {os.path.abspath(self.clustering_dir)}")
        
        if not os.path.exists(self.clustering_dir):
            os.makedirs(self.clustering_dir)
            print("Created new clustering directory")
        else:
            print("Clustering directory already exists")
            
        # Verify directory exists and is writable
        if os.path.exists(self.clustering_dir):
            print(f"Clustering directory exists at: {self.clustering_dir}")
            if os.access(self.clustering_dir, os.W_OK):
                print("Clustering directory is writable")
            else:
                print("WARNING: Clustering directory is not writable!")
        else:
            print("ERROR: Failed to create clustering directory!")
        
    def find_optimal_clusters(self, features_scaled):
        """Find optimal number of clusters using elbow method."""
        # Check if elbow curve already exists
        plot_path = os.path.join(self.clustering_dir, 'elbow_curve.png')
        print(f"\nChecking for existing elbow curve at: {plot_path}")
        if os.path.exists(plot_path):
            print("Using existing elbow curve from previous run")
            return self.optimal_k if self.optimal_k is not None else self.n_clusters
            
        print("Generating new elbow curve...")
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
        
        # Save plot to Clustering directory
        print(f"Saving elbow curve to: {plot_path}")
        try:
            plt.savefig(plot_path)
            print("Successfully saved elbow curve")
            if os.path.exists(plot_path):
                print(f"Verified elbow curve exists at: {plot_path}")
            else:
                print("ERROR: Elbow curve file was not created!")
        except Exception as e:
            print(f"ERROR saving elbow curve: {str(e)}")
        finally:
            plt.close()
        
        print(f"Generated new elbow curve with optimal k = {optimal_k}")
        return optimal_k
        
    def fit_course_clusters(self, courses):
        """Fit clusters for courses based on their provided skills.
        
        This method performs K-means clustering on courses based on two metrics:
        1. Skill Coverage: Percentage of total skills provided by the course
        2. Skill Diversity: Entropy-based measure of skill distribution
        
        The clustering helps identify groups of similar courses based on their
        skill profiles, which is used to adjust rewards during training.
        
        Args:
            courses (np.ndarray): Array of course data [required_skills, provided_skills]
                Shape: (n_courses, 2, n_skills)
                - required_skills: Binary array of required skills (currently all zeros)
                - provided_skills: Binary array of skills provided by the course
        """
        print("\nStarting course clustering...")
        # Extract provided skills (ignore required skills as they are all zeros)
        provided_skills = courses[:, 1]
        print(f"Number of courses to cluster: {len(provided_skills)}")
        
        # Calculate skill coverage and diversity metrics
        n_skills = provided_skills.shape[1]
        print(f"Number of skills per course: {n_skills}")
        
        # Calculate coverage (percentage of skills provided)
        coverage = np.sum(provided_skills, axis=1) / n_skills
        
        # Calculate skill diversity using entropy
        skill_distribution = provided_skills / (np.sum(provided_skills, axis=1, keepdims=True) + 1e-10)
        entropy = -np.sum(skill_distribution * np.log2(skill_distribution + 1e-10), axis=1)
        
        # Combine metrics into features for clustering
        self.features = np.column_stack([
            coverage,  # How many skills the course provides
            entropy    # How diverse the skills are
        ])
        
        # Scale features to have zero mean and unit variance
        features_scaled = self.scaler.fit_transform(self.features)
        
        # Find optimal number of clusters if auto_clusters is enabled
        if self.auto_clusters:
            print("\nFinding optimal number of clusters...")
            self.optimal_k = self.find_optimal_clusters(features_scaled)
            self.n_clusters = self.optimal_k
            print(f"Optimal number of clusters: {self.optimal_k}")
        
        # Perform K-means clustering with fixed random state
        print(f"\nPerforming K-means clustering with {self.n_clusters} clusters...")
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.course_clusters = kmeans.fit_predict(features_scaled)
        
        # Store cluster centers and inertia for reproducibility check
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        print(f"Clustering inertia: {self.inertia_:.2f}")
        
        # Visualize clusters
        print("\nGenerating cluster visualization...")
        self.visualize_clusters(features_scaled)
        
    def visualize_clusters(self, features_scaled):
        """Visualize the clusters in 2D space."""
        plt.figure(figsize=(10, 8))
        
        # Plot each cluster with different color
        for i in range(self.n_clusters):
            # Get points in this cluster
            cluster_points = features_scaled[self.course_clusters == i]
            print(f"Cluster {i} has {len(cluster_points)} courses")
            
            # Plot points
            plt.scatter(
                cluster_points[:, 0],  # Coverage
                cluster_points[:, 1],  # Diversity
                label=f'Cluster {i}',
                alpha=0.6
            )
            
            # Plot cluster center
            center = self.cluster_centers_[i]
            plt.scatter(
                center[0],
                center[1],
                c='black',
                marker='x',
                s=100,
                linewidths=2
            )
        
        plt.xlabel('Skill Coverage (scaled)')
        plt.ylabel('Skill Diversity (scaled)')
        plt.title('Course Clusters Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.clustering_dir, 'cluster_visualization.png')
        print(f"\nSaving cluster visualization to: {plot_path}")
        try:
            plt.savefig(plot_path)
            print("Successfully saved cluster visualization")
            if os.path.exists(plot_path):
                print(f"Verified cluster visualization exists at: {plot_path}")
            else:
                print("ERROR: Cluster visualization file was not created!")
        except Exception as e:
            print(f"ERROR saving cluster visualization: {str(e)}")
        finally:
            plt.close()
        
        print(f"Cluster visualization saved to {plot_path}")
        
    def adjust_reward(self, course_idx, original_reward, prev_reward):
        """Adjust reward based on cluster membership and reward change.
        
        This method implements the reward adjustment rules based on whether the
        course is in the same cluster as the previous course and whether the
        reward has increased or decreased.
        
        For first recommendation in any sequence (k=1,2,3), no reward adjustment is applied.
        
        For subsequent recommendations (k>1), the reward is adjusted based on:
        - Whether the course is in the same cluster as the previous course
        - Whether the reward has increased or decreased
        
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
        
        # Apply reward adjustment rules
        if reward_change > 0:  # Reward increased
            if current_cluster == self.prev_cluster:  # Same cluster
                return original_reward * 1.2  # Strong encouragement
            else:  # Different cluster
                return original_reward * 1.5  # Strong encouragement
        else:  # Reward decreased
            if current_cluster == self.prev_cluster:  # Same cluster
                return original_reward * 0.9  # Light penalty
            else:  # Different cluster
                return original_reward * 0.7  # Heavy penalty 