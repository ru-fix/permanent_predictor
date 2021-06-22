import gym
import numpy as np

from synthetic_bandit_environments.spaces.norm_distribution_space import NormDistSpace


class StaticNormLinearEnvironment(gym.Env):

    def __init__(
            self, amount_arms=4, context_size=10, amount_contexts=100,
            arm_range=(0, 1), context_range=(0, 1), arm_error_params=(0, 0.05), seed=None
    ):
        super(StaticNormLinearEnvironment, self).__init__()

        self.amount_contexts = amount_contexts
        self.action_space = gym.spaces.Discrete(amount_arms)
        self.reward_range = self.__get_reward_ragne(arm_range, context_range)
        self.observation_space = gym.spaces.Box(self.reward_range[0], self.reward_range[1], (1,))
        self.arm_space = gym.spaces.Box(arm_range[0], arm_range[1], (context_size,))
        self.context_space = gym.spaces.Box(context_range[0], context_range[1], (context_size,))
        self.arm_error_space = NormDistSpace(
            arm_error_params[0],
            arm_error_params[1],
            (1,)
        )

        self.__set_seed(seed)
        self.__generate_environment()

    def __get_reward_ragne(self, arm_range, context_range):
        borders = [arm_border * context_border
                   for arm_border in arm_range
                   for context_border in context_range]
        return (min(borders), max(borders))

    def __set_seed(self, seed):
        self.arm_space.seed(seed)
        self.context_space.seed(seed)
        self.arm_error_space.seed(seed)

    @staticmethod
    def __normalize_by_norm(vectors):
        return vectors / np.expand_dims(np.linalg.norm(vectors, axis=1), axis=1)

    def __generate_environment(self):
        self.arms = self.__normalize_by_norm(
            np.stack([self.arm_space.sample() for _ in range(self.action_space.n)])
        )
        self.contexts = self.__normalize_by_norm(
            np.stack([self.context_space.sample() for _ in range(self.amount_contexts)])
        )
        self.arm_errors = np.stack(
            [self.arm_error_space.sample() for _ in range(self.action_space.n)],
            axis = 1
        )
        self.rewards = np.dot(self.contexts, self.arms.T)\
                       + self.arm_errors.repeat(self.amount_contexts, axis = 0)

    def step(self, action):
        reward = self.rewards[self.index, action]
        desc = {
            'optimal': np.argmax(self.rewards[self.index]),
            'max_reward': np.max(self.rewards[self.index])
        }
        self.index += 1
        done = self.index == self.amount_contexts

        return None, reward, done, desc

    def reset(self):
        self.index = 0

    def render(self, mode='human'):
        return self.contexts[self.index: self.index + 1]

    def seed(self, seed=None):
        self.__set_seed(seed)
        self.__generate_environment()
        return [seed]
