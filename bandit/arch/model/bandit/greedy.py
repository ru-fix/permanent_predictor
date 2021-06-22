import numpy as np

from arch.model.bandit.base_bandit import BaseBandit


class SimpleGreedy(BaseBandit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reinitialize_weights()

    def reinitialize_weights(self):
        self.exprected_rewards = np.zeros(self.amount_arms)
        self.amount_pulls = np.zeros(self.amount_arms)

    def get_arm(self):
        played_arm = np.argmax(self.exprected_rewards)
        expected_reward = self.exprected_rewards[played_arm]
        return played_arm, expected_reward

    def get_arm_expected_reward(self, arm):
        return self.exprected_rewards[arm]

    def update(self, arm, reward):
        if self.amount_pulls.sum() < self.exploration_trials:
            self.amount_pulls[arm] += 1
            self.exprected_rewards[arm] += 1 / self.amount_pulls[arm] \
                                           * (reward - self.exprected_rewards[arm])


class EGreedy(BaseBandit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__set_seed(self.seed)
        self.reinitialize_weights()

    def reinitialize_weights(self):
        self.exprected_rewards = np.full(
            self.amount_arms,
            self.optimistic_initialization,
            dtype=np.float
        )
        self.amount_pulls = np.zeros(self.amount_arms)

    def __set_seed(self, seed):
        np.random.seed(seed)

    def get_arm(self):
        exploration_probability = np.random.random()
        if exploration_probability < self.epsilon:
            played_arm = np.random.choice(self.amount_arms)
        else:
            played_arm = np.argmax(self.exprected_rewards)
        expected_reward = self.exprected_rewards[played_arm]
        return played_arm, expected_reward

    def update(self, arm, reward):
        self.amount_pulls[arm] += 1
        self.exprected_rewards[arm] += 1. / self.amount_pulls[arm] * (reward - self.exprected_rewards[arm])

class NSEGreedy(BaseBandit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__set_seed(self.seed)
        self.reinitialize_weights()

    def reinitialize_weights(self):
        self.exprected_rewards = np.full(
            self.amount_arms,
            self.optimistic_initialization,
            dtype = np.float
        )

    def __set_seed(self, seed):
        np.random.seed(seed)

    def get_arm(self):
        exploration_probability = np.random.random()
        if exploration_probability < self.epsilon:
            played_arm = np.random.choice(self.amount_arms)
        else:
            played_arm = np.argmax(self.exprected_rewards)
        expected_reward = self.exprected_rewards[played_arm]
        return played_arm, expected_reward

    def update(self, arm, reward):
        self.exprected_rewards[arm] += self.alpha * (reward - self.exprected_rewards[arm])
