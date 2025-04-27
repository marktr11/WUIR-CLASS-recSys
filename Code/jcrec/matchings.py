import numpy as np


def matching(level1, level2):
    """
    Compute the matching score between two skill level vectors.

    This function compares two arrays representing skill levels (e.g., from a learner and a job/course)
    and calculates how well `level1` matches `level2`.
    Only skills that are required (non-zero in `level2`) are considered.

    The matching score is computed as:
    - For each required skill (non-zero in `level2`), take the ratio of the learner's skill to the required level,
      capped at 1 (since minimum is taken).
    - Sum the ratios and normalize by the number of required skills.

    Args:
        level1 (np.ndarray): An array of skill levels for the first entity (e.g., learner).
        level2 (np.ndarray): An array of required skill levels for the second entity (e.g., job or course).

    Returns:
        float: A matching score between 0 and 1, where 1 indicates a perfect match.
    """
    # get the minimum of the two arrays
    minimum_skill = np.minimum(level1, level2) #ouput : length of nb of skills (46)

    # get the indices of the non-zero elements of the job/course skill levels
    nonzero_indices = np.nonzero(level2)[0]

    # divide the minimum by the job/course skill levels at non-zero indices
    matching = minimum_skill[nonzero_indices] / level2[nonzero_indices]

    # sum the result and divide by the number of non-zero job/course skill levels
    matching = np.sum(matching) / np.count_nonzero(level2)

    return matching



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
