import numpy as np


def matching(level1, level2): # use for binary case : 0 or 1
    """
    Compute the matching score between two skill vectors in binary case.
    Only checks if skills exist (non-zero) in both vectors, regardless of their levels.

    Args:
        level1 (np.ndarray): An array of skills for the first entity (e.g., learner).
        level2 (np.ndarray): An array of skills for the second entity (e.g., job or course).

    Returns:
        float: A matching score between 0 and 1, where:
            - 1.0 means all required skills exist in level1 (perfect match)
            - 0.0 means none of the required skills exist in level1
            - Values in between represent the proportion of required skills that exist
            - If level2 has no skills (all zeros), returns -1.0 (invalid case)
    """
    # Get indices of non-zero elements in both arrays
    skills1 = set(np.nonzero(level1)[0])
    skills2 = set(np.nonzero(level2)[0])
    
    # If no skills are required/provided, return -1.0 (invalid case)
    if not skills2:
        return -1.0
        
    # Count how many skills exist in level1
    matching_skills = len(skills1.intersection(skills2))
    
    # Return the proportion of required skills that exist
    return matching_skills / len(skills2)

# def matching_ori(level1, level2): # use for original case : mastery level
#     """
#     Compute the matching score between two skill vectors in original case.
#     Takes into account the mastery levels of skills.

#     Args:
#         level1 (np.ndarray): An array of skills for the first entity (e.g., learner).
#         level2 (np.ndarray): An array of skills for the second entity (e.g., job or course).

#     Returns:
#         float: A matching score between 0 and 1, where:
#             - 1.0 means perfect match (all required skills at required levels)
#             - 0.0 means no match
#             - Values in between represent partial matches based on skill levels
#     """
#     # get the minimum of the two arrays
#     minimum_skill = np.minimum(level1, level2)

#     # get the indices of the non zero elements of the job skill levels
#     nonzero_indices = np.nonzero(level2)[0]

#     # divide the minimum by the job skill levels on the non zero indices
#     matching = minimum_skill[nonzero_indices] / level2[nonzero_indices]

#     # sum the result and divide by the number of non zero job skill levels
#     matching = np.sum(matching) / np.count_nonzero(level2)

#     return matching

def learner_job_matching(learner, job):
    """
    Compute the matching score between a learner and a job.

    Args:
        learner (np.ndarray): Learner's skill vector
        job (np.ndarray): Job's required skills vector

    Returns:
        float: Matching score between 0 and 1
    """
    # check if one of the arrays is empty
    if not (np.any(job) and np.any(learner)):
        return 0

    return matching(learner, job)

# def learner_course_required_matching(learner, course):
#     """
#     Compute the matching score between a learner and a course's required skills.
#     Always uses original matching as required skills have mastery levels.

#     Args:
#         learner (np.ndarray): Learner's skill vector
#         course (np.ndarray): Course's skills array [required, provided]

#     Returns:
#         float: Matching score between 0 and 1
#     """
#     required_course = course[0] #required skills

#     # check if the course has no required skills and return 1
#     if not np.any(required_course): # not( true if at least one element is not 0 )
#         return 1.0

#     return matching_ori(learner, required_course)

def learner_course_provided_matching(learner, course):
    """
    Compute the matching score between a learner and a course's provided skills.

    Args:
        learner (np.ndarray): Learner's skill vector
        course (np.ndarray): Course's skills array [required, provided]

    Returns:
        float: Matching score between 0 and 1
    """
    provided_course = course[1] #provided skills
    

    return matching(learner, provided_course)



def learner_course_matching(learner, course):
    """
    Compute the overall matching score between a learner and a course.
    This is used to measure user-course relevantness.

    Args:
        learner (np.ndarray): Learner's skill vector
        course (np.ndarray): Course's skills array [required, provided]

    Returns:
        float: Overall matching score
    """
    # get the required and provided matchings
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching) # user-course relevantness
