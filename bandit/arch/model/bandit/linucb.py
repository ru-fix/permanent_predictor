import torch

from arch.model.bandit.base_bandit import BaseContextBandit


class LinUCB(BaseContextBandit):

    """
    LinUCB realization.
    See: https://arxiv.org/pdf/1003.0146.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reinitialize_weights()

    def reinitialize_weights(self):
        # Initialize A matrices (amount_arms, context_size, context_size)
        self.identity_matrices = \
            torch.eye(self.context_size).repeat(self.amount_arms, 1, 1) * self.dzeta
        # Initialize inv_A matrices (amount_arms, context_size, context_size)
        self.inv_identity_matrices = self.identity_matrices.clone()
        # Initialize b vectors (amount arms, context_size)
        self.biases = torch.zeros(self.amount_arms, self.context_size)
        # Calculate first theta for all arms (amount arms, context_size)
        # Formula for one arm: inv_A * b
        self.thetas = torch.einsum('kld,kd->kl', self.inv_identity_matrices, self.biases)

    def get_arm(self, contexts):
        """
        Get best arm based on "expected reward + confidence"
        Formula for one expected reward: theta.T * x
        Formula for one confidence: sqrt(x.T * inv_A * x)
        :param contexts: Contexts for which need choose best arm for each batch. Have two shapes:
            1. (batch_size, context_size) - one batch = one context
            2. (batch_size, amount_arms, context_size) - one batch = contexts by arms
        :return:
        """
        # If shape size is equal two then is first shape type (one batch = one context)
        # For one context will have amount arms "expected reward + confidence" values
        if len(contexts.shape) == 2:
            # Calculate expected rewards (batch_size, amount_arms)
            expected_rewards = torch.mm(contexts, self.thetas.T)
            # Calculate confidences (batch_size, amount_arms)
            confidences = torch.einsum('nd,kdl,nl->nk', contexts, self.inv_identity_matrices, contexts)
        # Else shape size is equal three then is second shape type (one batch = amount arm contexts)
        # For one context will have one "expected reward + confidence" value
        else:
            # Calculate expected rewards (batch_size, )
            expected_rewards = torch.einsum('nkd,kd->nk', contexts, self.thetas)
            # Calculate confidences (batch_size, amount_arms)
            confidences = torch.einsum('nkd,kdl,nkl->nk', contexts, self.inv_identity_matrices, contexts)
        # Get best arms by "expected reward + confidence" value (batch_size)
        played_arms = torch.argmax(expected_rewards + self.alpha * confidences, axis=1)
        # Best arm by "expected reward + confidence" value for all contexts
        return played_arms, expected_rewards[:,played_arms], confidences[:,played_arms]

    def get_arm_expected_reward(self, contexts, played_arms):
        # If shape size is equal two then is first shape type (one batch = one context)
        # For one context will have amount arms "expected reward + confidence" values
        if len(contexts.shape) == 2:
            # Calculate expected rewards (batch_size, amount_arms)
            expected_rewards = torch.mm(contexts, self.thetas.T)
        # Else shape size is equal three then is second shape type (one batch = amount arm contexts)
        # For one context will have one "expected reward + confidence" value
        else:
            # Calculate expected rewards (batch_size, amount_arms)
            expected_rewards = torch.einsum('nkd,kd->nk', contexts, self.thetas)
        return expected_rewards[:, played_arms]

    def update(self, arms, contexts, rewards):
        possible_arms = torch.unique(arms).tolist()
        for arm_index in possible_arms:
            arm_mask = (arms == arm_index)[:, 0]
            self.identity_matrices[arm_index] += torch.mm(contexts[arm_mask].T, contexts[arm_mask])
            self.inv_identity_matrices[arm_index] = torch.inverse(self.identity_matrices[arm_index])
            self.biases[arm_index: arm_index+1] += torch.mm(rewards[arm_mask].T, contexts[arm_mask])
            self.thetas[arm_index: arm_index+1] = torch.mm(
                self.inv_identity_matrices[arm_index],
                self.biases[arm_index: arm_index+1].T
            ).T


class NSLinUCB(LinUCB):

    """
    NSLinUCB realization
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reinitialize_weights()

    def reinitialize_weights(self):
        super().reinitialize_weights()
        self.history = []

    @staticmethod
    def __add(first, second):
        first += second

    @staticmethod
    def __subtract(first, second):
        first -= second

    def __update(self, arms, contexts, rewards, update_func):
        possible_arms = torch.unique(arms).tolist()
        for arm_index in possible_arms:
            arm_mask = (arms == arm_index)[:, 0]
            update_func(
                self.identity_matrices[arm_index],
                torch.mm(contexts[arm_mask].T, contexts[arm_mask])
            )
            self.inv_identity_matrices[arm_index] = torch.inverse(self.identity_matrices[arm_index])
            update_func(
                self.biases[arm_index: arm_index + 1],
                torch.mm(rewards[arm_mask].T,  contexts[arm_mask])
            )
            self.thetas[arm_index: arm_index + 1] = torch.mm(
                self.inv_identity_matrices[arm_index],
                self.biases[arm_index: arm_index + 1].T
            ).T

    def update(self, arms, contexts, rewards):
        self.history.insert(0, (arms, contexts, rewards))
        if len(self.history) > self.history_size:
            self.__update(*self.history.pop(), update_func=self.__subtract)
        self.__update(arms, contexts, rewards, update_func=self.__add)