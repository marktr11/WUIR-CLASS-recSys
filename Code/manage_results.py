"""
Results Management Module for Course Recommendation System

This module provides functionality to manage experiment results across different branches
or experiment configurations. It handles:
- Creating and maintaining result directories
- Backing up results with timestamps
- Listing available results
- Cleaning up old results

IMPORTANT: This script MUST be executed from the Code directory to work correctly.
Example:
    cd Code
    python manage_results.py list

Directory Structure:
    Code/
    ├── results/
    │   ├── [branch_name]/
    │   │   ├── plots/     # Contains all plot files
    │   │   └── data/      # Contains all data files
    │   └── .gitignore
    └── backups/
        └── [branch_name]_[timestamp]/  # Backup directories

Important Notes:
1. Branch names in results/ are independent of git branches
2. Each backup creates a new directory with timestamp
3. Main branch results cannot be deleted
4. Script must be run from Code directory to ensure correct path resolution

Warnings:
1. Always backup important results before cleaning
   # Cleaning is destructive – data cannot be recovered once deleted.

2. Check available space before creating backups
   # Backups duplicate entire result directories – large files may fill disk quickly.

3. Verify branch name before cleaning results
   # Cleaning the wrong branch (due to typo or misunderstanding) will result in permanent data loss.

4. Keep track of backup timestamps for important experiments
   # Backup folders are timestamped – it may be difficult to identify the right one without notes.

Usage:
    # Make sure you are in the Code directory
    cd Code
    
    # List all available results
    python manage_results.py list
    
    # Backup results for a specific branch
    python manage_results.py backup experiment1
    
    # Clean up old results for a branch
    python manage_results.py clean old_branch
"""

import os
import shutil
import subprocess
from datetime import datetime


def get_current_branch():
    """Get current branch name from git or return default.
    
    This function attempts to get the current git branch name. If git is not
    available or there's an error, it returns "main" as default.
    
    Returns:
        str: Current git branch name or "main" if not available
        
    Note:
        This is only used as a default. The results directory structure
        is independent of git branches.
    """
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        return branch.decode('utf-8').strip()
    except:
        return "main"


def ensure_branch_directories():
    """Create and ensure existence of result directories for current branch.
    
    This function creates the necessary directory structure for storing results:
    - results/[branch_name]/
    - results/[branch_name]/plots/
    - results/[branch_name]/data/
    
    Returns:
        tuple: (branch_dir, plots_dir, data_dir) paths
        
    Note:
        Directories are created if they don't exist. Existing directories
        are not modified.
    """
    current_branch = get_current_branch()
    
    # Define paths
    base_dir = "results"  # Changed from "Code/results"
    branch_dir = os.path.join(base_dir, current_branch)
    plots_dir = os.path.join(branch_dir, "plots")
    data_dir = os.path.join(branch_dir, "data")
    
    # Create directories
    for directory in [branch_dir, plots_dir, data_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return branch_dir, plots_dir, data_dir


def list_branch_results():
    """List all available results across branches.
    
    This function scans the results directory and displays:
    - Number of plot files in each branch
    - Number of data files in each branch
    
    Note:
        Only counts files in plots/ and data/ subdirectories.
        Other files in branch directory are ignored.
    """
    base_dir = "results"  # Changed from "Code/results"
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
        
    branches = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not branches:
        print("Warning: No results found in any branch!")
        return
    
    print("\nAvailable results:")
    for branch in branches:
        branch_dir = os.path.join(base_dir, branch)
        plots_dir = os.path.join(branch_dir, "plots")
        data_dir = os.path.join(branch_dir, "data")
        
        # Create plots and data directories if they don't exist
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        plots = len(os.listdir(plots_dir)) if os.path.exists(plots_dir) else 0
        data = len(os.listdir(data_dir)) if os.path.exists(data_dir) else 0
        
        print(f"- {branch}:")
        print(f"  + Plots: {plots} files")
        print(f"  + Data: {data} files")


def create_directories(base_dir):
    """Create necessary directories for results.
    
    Args:
        base_dir (str): Base directory path
        
    Returns:
        tuple: (plots_dir, data_dir) paths
        
    Note:
        Creates plots and data directories if they don't exist
    """
    # Convert to absolute path
    base_dir = os.path.abspath(base_dir)
    plots_dir = os.path.join(base_dir, "plots")
    data_dir = os.path.join(base_dir, "data")
    
    try:
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        print(f"Attempted to create:")
        print(f"- {plots_dir}")
        print(f"- {data_dir}")
        raise
        
    return plots_dir, data_dir


def backup_results(source_dir, backup_dir):
    """Backup results to a timestamped directory.
    
    Args:
        source_dir (str): Source directory path
        backup_dir (str): Backup directory path
        
    Note:
        Creates backup directory if it doesn't exist
    """
    # Convert to absolute paths
    source_dir = os.path.abspath(source_dir)
    backup_dir = os.path.abspath(backup_dir)
    
    try:
        # Create backup directory structure
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(os.path.join(source_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(source_dir, "data"), exist_ok=True)
        
        # Create parent directory for backup
        os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
        
        # Copy files
        shutil.copytree(source_dir, backup_dir)
        print(f"Backup created at: {backup_dir}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        print(f"Source: {source_dir}")
        print(f"Backup: {backup_dir}")
        raise


def backup_branch_results(branch_name):
    """Create a timestamped backup of a branch's results.
    
    This function creates a backup of all results in the specified branch
    directory. The backup is stored in Code/backups/ with a timestamp.
    
    Args:
        branch_name (str): Name of the branch to backup
        
    Note:
        - Creates a new backup directory each time
        - Backup format: Code/backups/[branch_name]_[YYYYMMDD_HHMMSS]/
        - Existing backups are not modified
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backups", f"{branch_name}_{timestamp}")  # Changed from "Code/backups"
    
    source_dir = os.path.join("results", branch_name)  # Changed from "Code/results"
    # Create source directory if it doesn't exist
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(os.path.join(source_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(source_dir, "data"), exist_ok=True)
    
    if os.path.exists(source_dir):
        os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
        shutil.copytree(source_dir, backup_dir)
        print(f"Backed up {branch_name} results to {backup_dir}")
    else:
        print(f"Warning: No results found for branch {branch_name}")


def clean_branch_results(branch_name):
    """Remove all results for a specified branch.
    
    This function deletes ONLY the specified branch directory and its contents.
    It will NOT affect other branches or the main results directory.
    
    Args:
        branch_name (str): Name of the branch to clean
        
    Warning:
        - Cannot clean main branch results
        - This operation cannot be undone
        - Always backup important results before cleaning
    """
    if branch_name == "main":
        print("Error: Cannot delete main branch results!")
        return
        
    # Only clean the specific branch directory
    branch_dir = os.path.join("results", branch_name)
    if not os.path.exists(branch_dir):
        print(f"Warning: No results found for branch {branch_name}")
        return
        
    # Double check that we're only deleting the specific branch
    if not os.path.basename(branch_dir) == branch_name:
        print(f"Error: Safety check failed. Aborting clean operation.")
        return
        
    print(f"\nWARNING: You are about to delete ALL results for branch: {branch_name}")
    print("This operation cannot be undone!")
    confirm = input("Are you sure you want to continue? (yes/no): ")
    
    if confirm.lower() != "yes":
        print("Clean operation cancelled.")
        return
        
    try:
        # Try to remove files first
        for root, dirs, files in os.walk(branch_dir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except PermissionError:
                    print(f"Warning: Could not remove file {os.path.join(root, name)}")
                    print("Please close any applications that might be using this file.")
                    return
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except PermissionError:
                    print(f"Warning: Could not remove directory {os.path.join(root, name)}")
                    print("Please close any applications that might be using this directory.")
                    return
        # Finally remove the branch directory
        os.rmdir(branch_dir)
        print(f"Successfully cleaned up results for branch {branch_name}")
    except PermissionError:
        print(f"Error: Access denied when trying to clean {branch_name}")
        print("Please close any applications that might be using these files.")
        return


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python manage_results.py [command] [branch_name]")
        print("\nCommands:")
        print("  list              - List results for all branches")
        print("  backup [branch]   - Backup results for a branch")
        print("  clean [branch]    - Clean up old results for a branch")
        print("\nExamples:")
        print("  python manage_results.py list")
        print("  python manage_results.py backup experiment1")
        print("  python manage_results.py clean old_branch")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "list":
        list_branch_results()
    elif command in ["backup", "clean"]:
        if len(sys.argv) < 3:
            print(f"Error: Missing branch name for command {command}")
            sys.exit(1)
            
        branch_name = sys.argv[2]
        if command == "backup":
            backup_branch_results(branch_name)
        else:
            clean_branch_results(branch_name)
    else:
        print(f"Error: Invalid command: {command}")
        print("Use 'list', 'backup', or 'clean'") 