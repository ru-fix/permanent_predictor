import numpy as np
import gym
import scipy
from sklearn.preprocessing import PolynomialFeatures
from synthetic_bandit_environments.spaces.norm_distribution_space import NormDistSpace

class MovingPolynomialEnvironment(gym.Env):

    def __init__(
            self, amount_arms=4, context_size=10, amount_contexts=100, degree=1,
            arm_range=(0, 1), context_range=(0, 1),
            arm_error_params=(0, 1), smoof_steps=1, seed=None
    ):
        super(MovingPolynomialEnvironment, self).__init__()

        self.polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

        self.amount_contexts = amount_contexts
        self.action_space = gym.spaces.Discrete(amount_arms)
        
        poly_context_size = int(scipy.special.comb(context_size + degree, degree)) - 1
        
        self.reward_range = self.__get_reward_range(
            arm_range,
            context_range,
            arm_error_params,
            poly_context_size,
            degree
        )
        
        self.observation_space = gym.spaces.Box(self.reward_range[0], self.reward_range[1], (1,))
        self.arm_space = gym.spaces.Box(arm_range[0], arm_range[1], (poly_context_size,))
        self.context_space = gym.spaces.Box(context_range[0], context_range[1], (context_size,))
        self.arm_error_space = NormDistSpace(
            arm_error_params[0],
            arm_error_params[1],
            (1,)
        )

        self.smoof_iterations = np.linspace(
            start=0,
            stop=amount_contexts,
            num=min(smoof_steps + 2, amount_contexts + 2),
            dtype=np.int
        )[1:]
        self.lambdas = np.linspace(start=0, stop=1, num=min(smoof_steps + 1, amount_contexts + 1))

        self.__set_seed(seed)
        self.__generate_environment()

    def __get_reward_range(self, arm_range, context_range, arm_error_params, context_size, degree):
        borders = [arm_border * np.power(context_border, degree) * context_size + variance_border
                   for arm_border in arm_range
                   for context_border in context_range
                   for variance_border in [arm_error_params[0] - arm_error_params[1],
                                           arm_error_params[0] + arm_error_params[1]]]
        return (min(borders), max(borders))

    def __set_seed(self, seed):
        self.arm_space.seed(seed)
        self.context_space.seed(seed)
        self.arm_error_space.seed(seed)

    def __generate_environment(self):
        self.start_arms = np.stack([self.arm_space.sample() for _ in range(self.action_space.n)])
        self.finish_arms = np.stack([self.arm_space.sample() for _ in range(self.action_space.n)])
        self.contexts = np.stack([self.context_space.sample() for _ in range(self.amount_contexts)])

        
        self.arm_errors = np.reshape(
            [self.arm_error_space.sample() for _ in range(self.amount_contexts * self.action_space.n)],
            (self.amount_contexts, self.action_space.n)
        )

        self.poly_contexts = self.polynomial_features.fit_transform(self.contexts)
        
        self.mean_context_norm = np.mean(np.linalg.norm(self.poly_contexts, axis=1))
        self.max_context_norm = np.max(np.linalg.norm(self.poly_contexts, axis=1))
        
    def __calculate_rewards(self):
        self.arms = (1 - self.lambdas[self.lambda_index]) * self.start_arms\
                       + self.lambdas[self.lambda_index] * self.finish_arms

        rewards = np.dot(self.poly_contexts[self.index], self.arms.T) + self.arm_errors[self.index]
        rewards = np.squeeze(rewards)
        return rewards

    def step(self, action):
        rewards = self.__calculate_rewards()
        desc = {
            'optimal': np.argmax(rewards),
            'max_reward': np.max(rewards)
        }
        self.index += 1
        if self.index >= self.smoof_iterations[self.lambda_index]:
            self.lambda_index += 1
        done = self.index == self.amount_contexts

        return None, rewards[action], done, desc

    def reset(self):
        self.index = 0
        self.lambda_index = 0

    def render(self, mode='human'):
        return self.contexts[self.index: self.index + 1]

    def seed(self, seed=None):
        self.__set_seed(seed)
        self.__generate_environment()
        return [seed]
