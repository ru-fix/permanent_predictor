import torch

from arch import model_archs
from utils import useful_functions


class NNAndContextBanditComplex:
    def __init__(self, nn_config, bandit_config, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.nn = useful_functions.init_object(model_archs, nn_config)

        bandit_config.kwargs.context_size = self.nn.get_embedding_size()
        self.bandit = useful_functions.init_object(model_archs, bandit_config)

    def state_dict(self):
        return {'nn': self.nn.state_dict(), 'bandit': self.bandit.state_dict()}

    def load(self, state_dict):
        self.nn.load(state_dict["nn"])
        self.bandit.load(state_dict['bandit'])

    def reinitialize_weights(self):
        self.bandit.reinitialize_weights()

    def get_arm(self, contexts):
        embeddings = self.nn.get_embeddings(contexts).transpose(1, 0)
        return self.bandit.get_arm(embeddings)

    def get_arm_expected_reward(self, context, played_arm):
        embedding = self.nn.get_embeddings(context).transpose(1, 0)
        return self.bandit.get_arm_expected_reward(embedding, played_arm)

    def update(self, arms, contexts, rewards):
        embeddings = torch.stack([self.nn.get_embedding(contexts, arm) for arm in arms]).squeeze(1)
        self.bandit.update(arms, embeddings, rewards)
