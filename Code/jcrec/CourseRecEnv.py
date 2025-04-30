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
    def __init__(self, dataset, threshold=0.8, k=3):
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

    def step(self, action):
        """Method required by the gym environment. It performs the action in the environment and returns the new observation, the reward, whether the episode is terminated and additional information.

        Args:
            action (int): the course to be recommended

        Returns:
            tuple: the new observation, the reward, whether the episode is terminated, additional information
        """
        # Update the agent's skills with the course provided_skills

        course = self.dataset.courses[action] # 2d array, [0]:required skills; [1]:provided skills
        learner = self._agent_skills # Current learner skill vector (agent state)

        required_matching = matchings.learner_course_required_matching(learner, course)
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        if required_matching < self.threshold or provided_matching >= 1.0: # The case where the system needs to strongly detect recommendations that fall outside the scope of C_u
            observation = self._get_obs()
            reward = -1
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info
        

        # Accept the course: update learner's skills using element-wise max between current and provided skills
        self._agent_skills = np.maximum(self._agent_skills, course[1]) #else learn the recommended course and update state

        observation = self._get_obs()
        info = self._get_info()
        reward = info["nb_applicable_jobs"] # nb_applicable_jobs
        # Track number of recommended courses and check if max (k) is reached
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
        It calculates the average number of applicable jobs (reward signal) across all learners in the dataset, prints it, and logs the result to a file.

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

                    # Only update the reward if the recommendation was valid
                    if reward != -1:
                        tmp_avg_jobs = reward

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

