import torch


class BaseNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self, state_dict):
        self.load_state_dict(state_dict)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'forward\'')

    def reinitialize_weights(self):
        raise NotImplementedError('Not implement method \'initialize_weights\'')

    def get_embedding_size(self):
        raise NotImplementedError('Not implement method \'get_embedding_size\'')
