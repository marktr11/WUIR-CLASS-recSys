"""
Results Management Module for Course Recommendation System

This module provides functionality to manage experiment results across different branches
or experiment configurations. It handles:
- Creating and maintaining result directories
- Backing up results with timestamps
- Listing available results
- Cleaning up old results

Important Notes:
1. Branch names in results/ are independent of git branches
2. Each backup creates a new directory with timestamp
3. Main branch results cannot be deleted
4. Results are organized as:
   Code/results/
   ├── [branch_name]/
   │   ├── plots/     # Contains all plot files
   │   └── data/      # Contains all data files
   └── backups/
       └── [branch_name]_[timestamp]/  # Backup directories


Warnings:
1. Always backup important results before cleaning
   # Cleaning is destructive – data cannot be recovered once deleted.

2. Check available space before creating backups
   # Backups duplicate entire result directories – large files may fill disk quickly.

3. Verify branch name before cleaning results
   # Cleaning the wrong branch (due to typo or misunderstanding) will result in permanent data loss.

4. Keep track of backup timestamps for important experiments
   # Backup folders are timestamped – it may be difficult to identify the right one without notes.
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
    - Code/results/[branch_name]/
    - Code/results/[branch_name]/plots/
    - Code/results/[branch_name]/data/
    
    Returns:
        tuple: (branch_dir, plots_dir, data_dir) paths
        
    Note:
        Directories are created if they don't exist. Existing directories
        are not modified.
    """
    current_branch = get_current_branch()
    
    # Define paths
    base_dir = os.path.join("Code", "results")
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
    base_dir = os.path.join("Code", "results")
    if not os.path.exists(base_dir):
        print("Warning: Results directory does not exist!")
        return
        
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
        
        plots = len(os.listdir(plots_dir)) if os.path.exists(plots_dir) else 0
        data = len(os.listdir(data_dir)) if os.path.exists(data_dir) else 0
        
        print(f"- {branch}:")
        print(f"  + Plots: {plots} files")
        print(f"  + Data: {data} files")


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
    backup_dir = os.path.join("Code", "backups", f"{branch_name}_{timestamp}")
    
    source_dir = os.path.join("Code", "results", branch_name)
    if os.path.exists(source_dir):
        os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
        shutil.copytree(source_dir, backup_dir)
        print(f"Backed up {branch_name} results to {backup_dir}")
    else:
        print(f"Warning: No results found for branch {branch_name}")


def clean_branch_results(branch_name):
    """Remove all results for a specified branch.
    
    This function deletes the entire results directory for the specified branch.
    It includes all plots and data files.
    
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
        
    branch_dir = os.path.join("Code", "results", branch_name)
    if os.path.exists(branch_dir):
        shutil.rmtree(branch_dir)
        print(f"Cleaned up old results for branch {branch_name}")
    else:
        print(f"Warning: No results found for branch {branch_name}")


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