"""
Base trainer for all trainers
"""

import tqdm

from arch.trainer.base_trainer import BaseTrainer


class BaseBanditTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train(self, update_indexes=True):
        dataloader = self.datasets_handler.get_dataloaders(
            self.last_trials,
            self.last_update,
            split_datasets=False,
            split_subdatasets=False,
            update_indexes=update_indexes
        )[0]

        if len(dataloader) > 0:
            if self.reset:
                self.model.reinitialize_weights()
            for sample_data in dataloader:
                self.train_iteration(*sample_data)

    def train(self):
        self.environment.reset()

        bar = tqdm.tqdm(
            total=self.environment.amount_contexts,
            desc=f"Train instance of {self.__class__.__name__} on {self.experiment_name}"
        )

        round_index = 0
        last_calculate_metrics_round = 0
        for iteration in range(self.environment.amount_contexts):
            if self.exploration_exist and (iteration < self.round_iterations[0]):
                self.exploration_iteration(iteration, iteration_desc=dict(round_index=round_index))
            else:
                if iteration == self.round_iterations[round_index]:
                    self._train()
                    if round_index % self.metrics_save_period == 0:
                        last_calculate_metrics_round = round_index
                        self.calculate_metrics()
                        self.save(f"Round_{round_index}")
                    round_index += 1
                self.inference_iteration(iteration_desc=dict(round_index=round_index))
            bar.update()
        bar.close()

        if last_calculate_metrics_round != round_index:
            self.calculate_metrics()
            self.save(f"Round_{round_index}")

        return self.get_last_save_description()

    def train_iteration(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'train_iteration\'')

    def exploration_iteration(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'exploration_iteration\'')

    def inference_iteration(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'inference_iteration\'')
