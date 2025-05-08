import numpy as np


def matching(level1, level2):
    """
    Compute the matching score between two skill vectors in binary case.
    Only checks if skills exist (non-zero) in both vectors, regardless of their levels.

    Args:
        level1 (np.ndarray): An array of skills for the first entity (e.g., learner).
        level2 (np.ndarray): An array of skills for the second entity (e.g., job or course).

    Returns:
        float: A matching score between 0 and 1, where:
            - 1 means all required skills (non-zero in level2) exist in level1
            - 0 means none of the required skills exist in level1
            - Values in between represent the proportion of required skills that exist
    """
    # Get indices of non-zero elements in both arrays
    skills1 = set(np.nonzero(level1)[0])
    skills2 = set(np.nonzero(level2)[0])
    
    # If no skills are required, return 1
    if not skills2:
        return 1.0
        
    # Count how many skills exist in level1
    matching_skills = len(skills1.intersection(skills2))
    
    # Return the proportion of required skills that exist
    return matching_skills / len(skills2)



def learner_job_matching(learner, job):

    # check if one of the arrays is empty
    if not (np.any(job) and np.any(learner)):
        return 0

    return matching(learner, job)


def learner_course_required_matching(learner, course):

    required_course = course[0] #required skills

    # check if the course has no required skills and return 1
    if not np.any(required_course): # not( true if at least one element is not 0 )
        return 1.0

    return matching(learner, required_course)


def learner_course_provided_matching(learner, course):

    provided_course = course[1] #provided skills

    return matching(learner, provided_course)


def learner_course_matching(learner, course):

    # get the required and provided matchings
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching) # user-course relevantness
