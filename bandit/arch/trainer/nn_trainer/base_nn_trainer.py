import tqdm

import torch

from arch.trainer.base_trainer import BaseTrainer
from arch.model import losses


class BaseNNTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = getattr(losses, self.criterion_name)
        self.optimizer = getattr(
            torch.optim,
            self.optimizer_name
        )(self.model.parameters(), lr = self.learning_rate)


    def _train(self, round_index, update_indexes=True):
        dataloaders = self.datasets_handler.get_dataloaders(
            self.last_trials,
            self.last_update,
            split_datasets=self.model.multiple_arms,
            update_indexes=update_indexes
        )

        for nn_index, nn_dataloaders in enumerate(dataloaders):

            if self.reset and (len(nn_dataloaders['train']) > 0):
                self.model.reset_weights(nn_index)

            for epoch_index in range(self.amount_epochs):
                if len(nn_dataloaders['train']) > 0:
                    self.model.train(nn_index)

                    train_epoch_size = len(nn_dataloaders['train'])
                    for sample_data in nn_dataloaders['train']:
                        _, contexts, rewards = sample_data
                        train_iteration_desc = dict(
                            train_epoch_index=epoch_index,
                            train_round_index=round_index,
                            train_epoch_size=train_epoch_size
                        )
                        self.train_iteration(
                            nn_index,
                            contexts,
                            rewards,
                            iteration_desc=train_iteration_desc
                        )

                    self.model.eval(nn_index)
                    validation_epoch_size = len(nn_dataloaders['validation'])
                    for sample_data in nn_dataloaders['validation']:
                        _, contexts, rewards = sample_data
                        validation_iteration_desc = dict(
                            validation_epoch_index = epoch_index,
                            validation_round_index = round_index,
                            validation_epoch_size = validation_epoch_size
                        )
                        self.validation_iteration(
                            nn_index,
                            contexts,
                            rewards,
                            iteration_desc=validation_iteration_desc
                        )

    def train(self):
        self.environment.reset()

        bar = tqdm.tqdm(
            total=self.environment.amount_contexts,
            desc=f"Train instance of {self.__class__.__name__} on {self.experiment_name}"
        )

        round_index = 0
        last_calculate_metrics_round = 0
        for iteration in range(self.environment.amount_contexts):
            if iteration == self.round_iterations[round_index]:
                self._train(round_index)
                if round_index % self.metrics_save_period == 0:
                    last_calculate_metrics_round = round_index
                    self.calculate_metrics()
                    self.save(f"Round_{round_index}")
                round_index += 1
            self.get_sample(iteration)
            bar.update()
        bar.close()

        if last_calculate_metrics_round != round_index:
            self._train(round_index)
            self.calculate_metrics()
            self.save(f"Round_{round_index}")

        return self.get_last_save_description()

    def get_sample(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'get_sample\'')

    def train_iteration(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'train_iteration\'')

    def validation_iteration(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'validation_iteration\'')
