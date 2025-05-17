import os
import random

from time import process_time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

import matchings


class CourseRecEnv(gym.Env):
    # The CourseRecEnv class is a gym environment that simulates the recommendation of courses to learners. It is used to train the Reinforce model.
    def __init__(self, dataset, threshold=0.8, k=3, baseline=False):
        self.baseline = baseline
        self.dataset = dataset 
        self.nb_skills = len(dataset.skills) # 46 skills
        self.mastery_levels = [
            elem for elem in list(dataset.mastery_levels.values()) if elem > 0 # mastery level: [1,2,3,-1]
        ]
        self.max_level = max(self.mastery_levels)
        self.nb_courses = len(dataset.courses) #100 courses
        # get the minimum and maximum number of skills of the learners using np.nonzero
        self.min_skills = min(np.count_nonzero(self.dataset.learners, axis=1)) # 1
        self.max_skills = max(np.count_nonzero(self.dataset.learners, axis=1)) # 15
        self.threshold = threshold 
        self.k = k
        # The observation space is a vector of length nb_skills that represents the learner's skills.
        # The vector contains skill levels, where the minimum level is 0 and the maximum level is max_level (e.g., 3).
        # We cannot set the lower bound to -1 because negative values are not allowed in this Box space.
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.nb_skills,), dtype=np.int32)

        # Define the action space for the environment.
        # This is a discrete space where each action corresponds to recommending a specific course.
        # The total number of possible actions is equal to the number of available courses (nb_courses = 100).
        # The agent will select an integer in [0, nb_courses - 1], representing the index of the recommended course.
        self.action_space = gym.spaces.Discrete(self.nb_courses)

    def _get_obs(self):
        """Method required by the gym environment. It returns the current observation of the environment.

        Returns:
            np.array: the current observation of the environment, that is the learner's skills
        """
        return self._agent_skills

    def _get_info(self):
        """Method required by the gym environment. It returns the current info of the environment.

        Returns:
            dict: the current info of the environment, that is the number of applicable jobs
        """

        return {
            "nb_applicable_jobs": self.dataset.get_nb_applicable_jobs(
                self._agent_skills, threshold=self.threshold
            )
        }

    def get_random_learner(self):
        """Creates a random learner with a random number of skills and levels. This method is used to initialize the environment.

        Returns:
            np.array: the initial observation of the environment, that is the learner's initial skills
        """
        # Randomly choose the number of skills the agent has randomly
        n_skills = random.randint(self.min_skills, self.max_skills)

        # Initialize the skills array with zeros
        initial_skills = np.zeros(self.nb_skills, dtype=np.int32)

        # Choose unique skill indices without replacement
        skill_indices = np.random.choice(self.nb_skills, size=n_skills, replace=False)

        # Assign random mastery levels to these skills, levels can repeat
        initial_skills[skill_indices] = np.random.choice(
            self.mastery_levels, size=n_skills, replace=True
        )
        return initial_skills

    def reset(self, seed=None, learner=None):
        """Method required by the gym environment. It resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            learner (list, optional): Learner to initialize the environment with, if None, the environment is initialized with a random learner. Defaults to None.

        Returns:
            _type_: _description_
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if learner is not None:
            self._agent_skills = learner
        else:
            self._agent_skills = self.get_random_learner()
        self.nb_recommendations = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def calculate_course_metrics(self, learner, course):
        """Calculate N1, N2, N3 metrics for a course recommendation.
        
        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            
        Returns:
            tuple: (N1, N2, N3) where:
                - N1: number of missing skills resolved by this course
                - N2: remaining missing skills after learning this course
                - N3: number of skills provided by the course that are not in missing skills
        """
        # Get missing skills before updating
        missing_skills_before = self.dataset.get_learner_missing_skills(learner)
        
        # Get skills provided by the course
        course_provided_skills = set(np.nonzero(course[1])[0])
        
        # Calculate skills after learning the course, update state
        updated_skills = np.maximum(learner, course[1])
        
        # Get missing skills after updating
        missing_skills_after = self.dataset.get_learner_missing_skills(updated_skills)
        
        # Calculate N1: number of missing skills resolved by this course
        N1 = len(missing_skills_before - missing_skills_after)
        
        # Calculate N2: remaining missing skills after learning this course
        N2 = len(missing_skills_after)
        
        # Calculate N3: number of skills provided by the course that are not in missing skills
        N3 = len(course_provided_skills - missing_skills_before)
        
        return N1, N2, N3

    def calculate_achievable_goals(self, learner, course):
        """Calculate the set of goals that a course allows the learner to achieve.
        
        Args:
            learner (np.ndarray): Current learner's skill vector (B)
            course (np.ndarray): Course's skills array [required, provided] (φ)
            
        Returns:
            tuple: (initial_goals, new_goals) where:
                - initial_goals: number of jobs applicable with initial skills (B |= g)
                - new_goals: number of new jobs that become applicable after learning the course (B ∪ φ |= g)
        """
        # Calculate initial goals (jobs applicable with current skills)
        initial_goals = self.dataset.get_nb_applicable_jobs(learner, threshold=self.threshold)
        
        # Calculate skills after learning the course
        updated_skills = np.maximum(learner, course[1])
        
        # Calculate new goals (jobs applicable after learning the course)
        new_goals = self.dataset.get_nb_applicable_jobs(updated_skills, threshold=self.threshold)
        
        return initial_goals, new_goals

    def calculate_utility(self, learner, course):
        """Calculate U(φ) = 1/(|G|+1) * [|E(φ)| + N1(φ)/(N1(φ)+N2(φ)+(N3(φ)/ (N3(φ)+1)))]
        
        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            
        Returns:
            float: The utility value U(φ)
        """
        # Calculate N1, N2, N3 metrics
        N1, N2, N3 = self.calculate_course_metrics(learner, course)
        
        # Calculate achievable goals
        initial_goals, new_goals = self.calculate_achievable_goals(learner, course)
        
        # Calculate |G|: number of jobs not applicable with initial skills
        total_jobs = len(self.dataset.jobs)
        Ga = total_jobs - initial_goals
        
        # Calculate |E(φ)|: number of new jobs that become applicable
        E_phi = new_goals - initial_goals
        
        # Calculate denominator for N1 fraction
        denominator = N1 + N2 + (N3/N3+1)
        if denominator == 0:  # Avoid division by zero
            N1_fraction = 0
        else:
            N1_fraction = N1 / denominator
        
        # Calculate U(φ)
        utility = (1 / (Ga + 1)) * (E_phi + N1_fraction)
        
        return utility

    def step(self, action):
        course = self.dataset.courses[action]
        learner = self._agent_skills

        # Skip-expertise case: use new metrics and utility
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        if provided_matching == 1.0:
            observation = self._get_obs()
            reward = -1
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info
        
        if self.baseline : #baseline model
            self._agent_skills = np.maximum(self._agent_skills, course[1])
            observation = self._get_obs()
            info = self._get_info()
            reward = info["nb_applicable_jobs"]
        else: # skip-expertise-Usefulness-as-Rwd model
            # Calculate metrics for skip-expertise-Usefulness
            utility = self.calculate_utility(learner, course)
            
            self._agent_skills = np.maximum(self._agent_skills, course[1])
            observation = self._get_obs()
            info = self._get_info()
            info["utility"] = utility
            reward = info["utility"]  #1sp exp : substitute for nb_applicable_jobs
                                      #2sp exp : add nb_applicable_jobs with utility

        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k


        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    # The EvaluateCallback class is a callback that evaluates the model at regular intervals during the training.
    def __init__(self, eval_env, eval_freq, all_results_filename, verbose=1):
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.all_results_filename = all_results_filename
        self.mode = "w"

    def _on_step(self):
        """
        Custom evaluation method called at every step of training by the callback system.

        This method performs model evaluation every `eval_freq` steps using the current policy on a fixed evaluation environment (`self.eval_env`).
        It calculates the average number of applicable jobs across all learners in the dataset, prints it, and logs the result to a file.
        Note: This always uses nb_applicable_jobs for evaluation, regardless of the reward type used in training.

        Returns:
            bool: Always returns True to indicate that training should continue.
        """
        # Only evaluate every 'eval_freq' training steps
        if self.n_calls % self.eval_freq == 0:
            time_start = process_time()  # Start timing the evaluation
            avg_jobs = 0  # Accumulator for average jobs across learners

            # Loop through each learner in the evaluation dataset
            for learner in self.eval_env.dataset.learners:
                self.eval_env.reset(learner=learner)  # Reset environment with current learner
                done = False  # Flag to control evaluation episode
                tmp_avg_jobs = self.eval_env._get_info()["nb_applicable_jobs"]  # Initial jobs applicable without any recommendations

                # Run one full evaluation episode for the learner
                while not done:
                    obs = self.eval_env._get_obs()  # Get current observation (learner's skills)
                    action, _state = self.model.predict(obs, deterministic=True)  # Predict action using current policy
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)  # Step in environment
                    done = terminated or truncated  # Properly compute done flag

                    # Only update if the recommendation was valid and use nb_applicable_jobs
                    if reward != -1:
                        tmp_avg_jobs = info["nb_applicable_jobs"]

                avg_jobs += tmp_avg_jobs  # Add learner's result to total

            time_end = process_time()  # End timing the evaluation

            # Log the result to the console
            print(
                f"Iteration {self.n_calls}. "
                f"Average jobs: {avg_jobs / len(self.eval_env.dataset.learners)} "
                f"Time: {time_end - time_start}"
            )

            # Write evaluation result to file
            with open(
                os.path.join(
                    self.eval_env.dataset.config["results_path"],
                    self.all_results_filename,
                ),
                self.mode,  # 'w' for first time, 'a' for append afterward
            ) as f:
                f.write(
                    f"{self.n_calls} "
                    f"{avg_jobs / len(self.eval_env.dataset.learners)} "
                    f"{time_end - time_start}\n"
                )

            # After first write, switch mode to append for future evaluations
            if self.mode == "w":
                self.mode = "a"

        return True  # Returning True continues training

