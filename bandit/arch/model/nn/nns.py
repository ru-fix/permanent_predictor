import torch

from arch.model.nn.base_nn import BaseNN
from utils import useful_functions

class MultiLayer(BaseNN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc0 = torch.nn.Linear(
            self.context_size,
            useful_functions.find_nearest_power_of_two(self.context_size),
            bias = False
        )
        for i in range(1, self.amount_layers + 1):
            last_fc = getattr(self, f"fc{i - 1}")
            setattr(
                self,
                f"fc{i}",
                torch.nn.Linear(
                    last_fc.out_features,
                    2 * last_fc.out_features,
                    bias = False
                )
            )
        setattr(
            self,
            f"fc{self.amount_layers + 1}",
            torch.nn.Linear(
                getattr(self, f"fc{self.amount_layers}").out_features,
                1,
                bias = False
            )
        )

    def reinitialize_weights(self):
        for i in range(self.amount_layers + 1):
            getattr(self, f"fc{i}").reset_parameters()
        getattr(self, f"fc{self.amount_layers + 1}").reset_parameters()

    def get_embedding_size(self):
        return getattr(self, f"fc{self.amount_layers}").out_features

    def forward(self, contexts):
        for i in range(self.amount_layers + 1):
            contexts = torch.relu(
                getattr(
                    self,
                    f"fc{i}"
                )(contexts)
            )
        embeddings = contexts
        rewards = \
            getattr(
                    self,
                    f"fc{self.amount_layers+1}"
            )(embeddings)

        return embeddings, rewards


class MultiLayerGroup:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.multiple_arms:
            self.nns = [MultiLayer(**self.multilayer_kwargs) for _ in range(self.amount_arms)]
        else:
            self.nns = [MultiLayer(**self.multilayer_kwargs)]

    def reset_weights(self, index):
        self.nns[index].reinitialize_weights()

    def state_dict(self):
        state_dict = {'nns': [nn.state_dict() for nn in self.nns]}
        return state_dict

    def load(self, state_dict):
        for nn_index, nn_state in enumerate(state_dict['nns']):
            self.nns[nn_index].load(nn_state)

    def eval(self, nn_index):
        self.nns[nn_index].eval()

    def train(self, nn_index):
        self.nns[nn_index].train()

    def parameters(self):
        return sum([list(nn.parameters()) for nn in self.nns], [])

    def get_embeddings(self, contexts):
        return torch.stack([nn(contexts)[0] for nn in self.nns])

    def get_embedding(self, contexts, nn_index):
        return self.nns[nn_index](contexts)[0]

    def get_embedding_size(self):
        return self.nns[0].get_embedding_size()