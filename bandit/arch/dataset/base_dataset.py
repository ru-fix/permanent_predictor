import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def add_sample(self, *sample_data):
        format_sample_data = []
        for value in sample_data:
            value = torch.tensor(value)
            shape_size = len(value.shape)
            if shape_size == 0:
                value = value.unsqueeze(dim=0)
            elif shape_size == 2:
                value = value.squeeze()
            format_sample_data.append(value)
        self.samples.append(format_sample_data)

    def __getitem__(self, index):
        return self.samples[index]
