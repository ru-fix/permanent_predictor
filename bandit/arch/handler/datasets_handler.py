from torch.utils.data import SubsetRandomSampler, DataLoader, ConcatDataset, Subset

from arch.dataset.base_dataset import BaseDataset


class DatasetsHandler:
    def __init__(self, batch_size, amount_datasets, proportions, amount_retrain_samples=0):
        """
        :param batch_size: Int, amount sample in batch
        :param amount_datasets: Int, amount arms (one arm by dataset)
        :param proportions: List, list of subdataset proportions
        :param amount_retrain_samples: Int, amount samples to retrain
        """
        self.batch_size = batch_size
        self.amount_datasets = amount_datasets
        self.proportions = proportions
        self.amount_retrain_samples = amount_retrain_samples


        self.datasets = [BaseDataset() for _ in range(self.amount_datasets)]
        self.indexes = [{key: [] for key in self.proportions} for _ in range(self.amount_datasets)]
        self.updated_datasets = [False for _ in range(self.amount_datasets)]

    def add(self, dataset_index, *sample_data):
        self.datasets[dataset_index].add_sample(dataset_index, *sample_data)

    def __update_indexes(self):
        self.updated_datasets = [False for _ in range(self.amount_datasets)]

        for dataset_index in range(self.amount_datasets):
            new_indexes = set(range(len(self.datasets[dataset_index])))
            for subdatset_indexes in self.indexes[dataset_index].values():
                for trial_indexes in subdatset_indexes:
                    new_indexes -= set(trial_indexes)
            if len(new_indexes) > self.amount_retrain_samples:
                self.updated_datasets[dataset_index] = True
                amount_new_indexes = len(new_indexes)
                for subdataset_name, proportion in self.proportions.items():
                    amount_new_subdataset_indexes = int(round(amount_new_indexes * proportion))
                    self.indexes[dataset_index][subdataset_name].append(
                        [new_indexes.pop() for _ in range(amount_new_subdataset_indexes)]
                    )

    def __not_split_datasets_and_split_subdatasets(self, last_update, last_trials):
        dataloader = {}
        for subdataset_name in self.proportions:
            subsets = []
            for dataset_index, dataset in enumerate(self.datasets):

                if self.updated_datasets[dataset_index]:
                    indexes_by_trials = self.indexes[dataset_index][subdataset_name][-1 if last_update else 0:]
                else:
                    indexes_by_trials = []

                all_indexes = sum(indexes_by_trials, [])[-last_trials:]
                subsets.append(Subset(dataset, all_indexes))

            dataloader[subdataset_name] = DataLoader(ConcatDataset(subsets), self.batch_size)

        return [dataloader]

    def __not_split_datasets_and_subdatasets(self, last_update, last_trials):
        subsets = []
        for subdataset_name in self.proportions:
            for dataset_index, dataset in enumerate(self.datasets):

                if self.updated_datasets[dataset_index]:
                    indexes_by_trials = self.indexes[dataset_index][subdataset_name][-1 if last_update else 0:]
                else:
                    indexes_by_trials = []

                all_indexes = sum(indexes_by_trials, [])[-last_trials:]
                subsets.append(Subset(dataset, all_indexes))

        dataloader = DataLoader(ConcatDataset(subsets), self.batch_size)
        return [dataloader]

    def __split_datasets_and_subdatasets(self, last_update, last_trials):
        dataloaders = []
        for dataset_index, dataset in enumerate(self.datasets):
            subdataset_dataloaders = {}
            for subdataset_name in self.proportions:
                if self.updated_datasets[dataset_index]:
                    indexes_by_trials = self.indexes[dataset_index][subdataset_name][-1 if last_update else 0:]
                else:
                    indexes_by_trials = []

                all_indexes = sum(indexes_by_trials, [])[-last_trials:]
                dataloader = DataLoader(
                    self.datasets[dataset_index],
                    self.batch_size,
                    sampler=SubsetRandomSampler(all_indexes))

                subdataset_dataloaders[subdataset_name] = dataloader
            dataloaders.append(subdataset_dataloaders)
        return dataloaders

    def __split_datasets_and_not_split_subdatasets(self, last_update, last_trials):
        dataloaders = []
        for dataset_index in range(self.amount_datasets):
            subdatasets_indexes_sums = []
            for subdataset_name in self.proportions:
                if self.updated_datasets[dataset_index]:
                    indexes_by_trials = self.indexes[dataset_index][subdataset_name][-1 if last_update else 0:]
                else:
                    indexes_by_trials = []

                subdataset_all_indexes = sum(indexes_by_trials, [])[-last_trials:]
                subdatasets_indexes_sums.append(subdataset_all_indexes)
            subdatasets_indexes_sums = sum(subdatasets_indexes_sums, [])

            dataloader = DataLoader(
                self.datasets[dataset_index],
                self.batch_size,
                sampler=SubsetRandomSampler(subdatasets_indexes_sums))

            dataloaders.append(dataloader)
        return dataloaders

    def get_dataloaders(
            self,
            last_trials=0,
            last_update=False,
            split_datasets=True,
            split_subdatasets=True,
            update_indexes=True
    ):
        """
        last_update >>> last_trials >>> subdataset_split >>> concatenate_datasets
        :param update_indexes:
        :param last_trials: return dataloaders with last_trials elems from end
        :param last_update: return dataloader with last_update elems count
        :param concatenate_datasets: concat all spearate nns dataset in one
        :param subdataset_split: concat all subdataset (train, valid, test)
        :return:

        """
        if update_indexes:
            self.__update_indexes()

        if split_datasets:
            if split_subdatasets:
                return self.__split_datasets_and_subdatasets(last_update,last_trials)
            else:
                return self.__split_datasets_and_not_split_subdatasets(last_update,last_trials)
        else:
            if split_subdatasets:
                return self.__not_split_datasets_and_split_subdatasets(last_update, last_trials)
            else:
                return self.__not_split_datasets_and_subdatasets(last_update, last_trials)

    def get_data(self):
        return self.__dict__

    def load(self, datasets_data):
        self.__dict__ = datasets_data
