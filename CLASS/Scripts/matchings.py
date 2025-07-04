"""
Matching Score Calculation Module

This module implements various matching score calculations between different entities
in the course recommendation system, taking into account mastery levels (1-3) for skills.

Key Functions:
    - matching: Base function for computing matching scores between skill vectors
    - learner_job_matching: Computes matching between learner skills and job requirements
    - learner_course_required_matching: Computes matching between learner skills and course prerequisites
    - learner_course_provided_matching: Computes matching between learner skills and course outcomes
    - learner_course_matching: Computes overall course relevance score

Matching Score Calculation:
    For each skill:
    1. Take minimum of learner's level and required/provided level
    2. Divide by required/provided level
    3. Average across all required skills
    
    Example:
        Learner skills:    [3, 2, 0, 1]
        Job requirements:  [2, 3, 1, 2]
        Matching:         [2/2, 2/3, 0/1, 1/2] = [1.0, 0.67, 0.0, 0.5]
        Final score:      (1.0 + 0.67 + 0.0 + 0.5) / 4 = 0.54

Course Matching Rules:
    1. Required Skills (Prerequisites):
       - Score = 1.0 if no prerequisites
       - Score = 0.0 if missing any required skill
       - Score = average of skill level matches if all prerequisites met
    
    2. Provided Skills (Learning Outcomes):
       - Score = 0.0 if learner has no skills
       - Score = 1.0 if learner has all skills at required levels
       - Score = average of skill level matches otherwise
    
    3. Overall Course Relevance:
       - Score = required_matching * (1 - provided_matching)
       - Higher score means better course recommendation
       - Balances prerequisites and learning potential
"""

import numpy as np
from typing import Tuple, Union, Dict, Set


def matching(level1: np.ndarray, level2: np.ndarray) -> float:
    """
    Compute the matching score between two skill vectors taking into account mastery levels.
    This is the original version that considers mastery levels (1-3).

    Args:
        level1 (np.ndarray): An array of skills for the first entity (e.g., learner).
                            Values represent mastery levels (1-3).
        level2 (np.ndarray): An array of skills for the second entity (e.g., job or course).
                            Values represent required/provided levels (1-3).

    Returns:
        float: A matching score between 0 and 1, where:
            - 1.0 means perfect match (all required skills at required levels)
            - 0.0 means no match
            - Values in between represent partial matches based on skill levels
    """
    # get the minimum of the two arrays
    minimum_skill = np.minimum(level1, level2)

    # get the indices of the non zero elements of the job skill levels
    nonzero_indices = np.nonzero(level2)[0]

    # divide the minimum by the job skill levels on the non zero indices
    matching = minimum_skill[nonzero_indices] / level2[nonzero_indices]

    # sum the result and divide by the number of non zero job skill levels
    matching = np.sum(matching) / np.count_nonzero(level2)

    return matching


def learner_job_matching(learner: np.ndarray, job: np.ndarray) -> float:
    """
    Compute the matching score between a learner and a job.
    Takes into account the mastery levels of skills.

    Args:
        learner (np.ndarray): Learner's skill vector where values (1-3) indicate
                            mastery levels of skills.
        job (np.ndarray): Job's required skills vector where values (1-3) indicate
                         required mastery levels.

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means the learner has all required skills at required levels
            - 0.0 means the learner has none of the required skills
            - Values in between represent partial matches based on skill levels
    """
    # Check if one of the arrays is empty
    if not (np.any(job) and np.any(learner)):
        return 0.0

    return matching(learner, job)


def learner_course_required_matching(learner: np.ndarray, course: np.ndarray) -> float:
    """
    Compute the matching score between a learner and a course's required skills.
    Takes into account the mastery levels of skills.

    Args:
        learner (np.ndarray): Learner's skill vector where values (1-3) indicate
                            mastery levels of skills.
        course (np.ndarray): Course's skills array [required, provided] where
                            required skills are in the first dimension.

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means the learner has all required skills at required levels
            - 0.0 means the learner has none of the required skills
            - Values in between represent partial matches based on skill levels
    """
    required_course = course[0]  # required skills

    # check if the course has no required skills and return 1
    if not np.any(required_course):
        return 1.0

    return matching(learner, required_course)


def learner_course_provided_matching(learner: np.ndarray, course: np.ndarray) -> float:
    """
    Compute the matching score between a learner and a course's provided skills.
    Takes into account the mastery levels of skills.

    Args:
        learner (np.ndarray): Learner's skill vector where values (1-3) indicate
                            mastery levels of skills.
        course (np.ndarray): Course's skills array [required, provided] where
                            provided skills are in the second dimension.

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means the learner already has all skills at required levels
            - 0.0 means the learner has none of the skills
            - Values in between represent partial matches based on skill levels
    """
    provided_course = course[1]  # provided skills
    return matching(learner, provided_course)


def learner_course_matching(learner: np.ndarray, course: np.ndarray) -> float:
    """
    Compute the overall matching score between a learner and a course.
    This is used to measure user-course relevantness.
    Takes into account the mastery levels of skills.

    The score is calculated as: required_matching * (1 - provided_matching)
    This formula ensures that:
    - Courses that provide new skills (low provided_matching) are preferred
    - Courses that the learner is qualified for (high required_matching) are preferred

    Args:
        learner (np.ndarray): Learner's skill vector where values (1-3) indicate
                            mastery levels of skills.
        course (np.ndarray): Course's skills array [required, provided] where
                            required skills are in the first dimension and
                            provided skills are in the second dimension.

    Returns:
        float: Overall matching score between 0 and 1, where higher values
               indicate better course recommendations for the learner.
    """
    # Get the required and provided matchings
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching)  # user-course relevantness

