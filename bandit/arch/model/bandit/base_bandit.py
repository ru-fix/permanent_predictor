"""
Base class for all bandits
"""

class BaseBandit:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_arm(self, *args, **kwargs):
        """
        Get expected best arm index function
        :return Int. Expected best arm index.
        """
        raise NotImplementedError('Not implement method \'get_arm\'')

    def get_arm_expected_reward(self, *args, **kwargs):
        """
        Get current expected reward arm
        :param arm: Int. Arm index.
        :return: Float. Expected arm reward
        """
        raise NotImplementedError('Not implement method \'get_arm_expected_reward\'')

    def get_arm_confidence(self, *args, **kwargs):
        """
        Get current arm confidence
        :param arm: Int. Arm index.
        :return: Float. Arm confidence
        """
        raise NotImplementedError('Not implement method \'get_arm_confidence\'')

    def update(self, *args, **kwargs):
        """ Update bandit weights by input reward """
        raise NotImplementedError('Not implement method \'update\'')

    def state_dict(self):
        """ Return current bandit state """
        return self.__dict__

    def load(self, state_dict):
        """
        Load bandit by state
        :param state: Bandit state
        :return None, loaded bandit
        """
        self.__dict__ = state_dict

    def reinitialize_weights(self):
        """ Initialize bandit weights """
        raise NotImplementedError('Not implement method \'initialize_weights\'')


class BaseContextBandit(BaseBandit):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_arm(self, contexts):
        """
        Get expected best arm index function
        :param contexts: Tensor (batch_size, context_size), batch_size of user contexts
        :return: Tensor (batch_size), arms with best expected reward value
        """
        raise NotImplementedError('Not implement method \'get_arm\'')

    def update(self, arms, contexts, rewards):
        """
        Update bandit weights by input contexts and rewards
        :param arms: Tensor (batch_size), batch_size of arms which need update
        :param contexts: Tensor (batch_size, context_size), batch_size of user contexts
        :param rewards: Tensor (batch_size), batch_size of rewards
        :return: None, update bandit weights
        """
        raise NotImplementedError('Not implement method \'update\'')
