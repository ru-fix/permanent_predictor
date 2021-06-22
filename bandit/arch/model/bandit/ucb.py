import numpy as np

from arch.model.bandit.base_bandit import BaseBandit


class UCB(BaseBandit):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.reinitialize_weights()

    def reinitialize_weights(self):
        self.exprected_rewards = np.zeros(self.amount_arms)
        self.amount_pulls = np.zeros(self.amount_arms)
        self.confidences = np.zeros(self.amount_arms)

    def get_arm(self):
        played_arm = np.argmax(self.exprected_rewards + self.alpha * self.confidences)
        expected_reward = self.exprected_rewards[played_arm]
        return played_arm, expected_reward

    def get_arm_expected_reward(self, arm):
        return self.exprected_rewards[arm]

    def get_arm_confidence(self, arm):
        return self.confidences[arm]

    def update(self, arm, reward):
        self.amount_pulls[arm] += 1
        self.exprected_rewards[arm] += 1 / self.amount_pulls[arm] * (reward - self.exprected_rewards[arm])
        self.confidences[arm] = np.sqrt(2 * np.log(np.sum(self.amount_pulls)) / self.amount_pulls[arm])


class NSUCB(UCB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reinitialize_weights()

    def reinitialize_weights(self):
        super().reinitialize_weights()
        self.amount_total_pulls = 0
        self.discount_factors = []
        self.arm_discount = np.zeros(self.amount_arms)
        self.pulled_arms = []
        self.rewards = []

    def update(self, arm, reward):
        self.pulled_arms.append(arm)
        self.amount_pulls[arm] += 1
        self.amount_total_pulls += 1
        self.discount_factors.insert(0, self.gamma * self.discount_factors[0]
                                        if self.amount_total_pulls > 1 else 1)
        self.rewards.append(reward)

        if (self.amount_pulls > 0).all():
            self.arm_discount.fill(0)
            self.exprected_rewards.fill(0)

            for index in range(self.amount_total_pulls):
                self.arm_discount[self.pulled_arms[index]] += self.discount_factors[index]
                self.exprected_rewards[self.pulled_arms[index]] += self.discount_factors[index] \
                                                                  * self.rewards[index]
            self.exprected_rewards /= self.arm_discount

            self.confidences = np.sqrt(2 * np.log(np.sum(self.arm_discount)) / self.arm_discount)
