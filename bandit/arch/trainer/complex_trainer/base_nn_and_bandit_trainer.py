"""
Simple neural network trainer
"""

import tqdm

from arch.trainer.base_trainer import BaseTrainer
from arch import trainer_archs
from utils import useful_functions



class BaseNNAndBanditTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bandit_trainer = useful_functions.init_object(
            trainer_archs,
            self.bandit_trainer_config,
            model=self.model,
            environment=self.environment,
            metrics_handler=self.metrics_handler,
            datasets_handler=self.datasets_handler
        )

        self.nn_trainer = useful_functions.init_object(
            trainer_archs,
            self.nn_trainer_config,
            model=self.model.nn,
            metrics_handler=self.metrics_handler,
            datasets_handler=self.datasets_handler
        )


    def _train(self, round_index):
        self.nn_trainer._train(round_index)
        self.bandit_trainer._train(update_indexes=False)

    def train(self):
        self.environment.reset()

        bar = tqdm.tqdm(
            total = self.environment.amount_contexts,
            desc = f"Train instance of {self.__class__.__name__} on {self.experiment_name}"
        )

        round_index = 0
        last_calculate_metrics_round = 0
        for iteration in range(self.environment.amount_contexts):
            if self.exploration_exist and (iteration < self.round_iterations[0]):
                self.bandit_trainer.exploration_iteration(
                    iteration,
                    iteration_desc=dict(round_index=round_index)
                )
            else:
                if iteration == self.round_iterations[round_index]:
                    self._train(round_index)
                    if (round_index % self.metrics_save_period == 0) \
                            or (self.exploration_exist and (round_index == 0)):
                        last_calculate_metrics_round = round_index
                        self.bandit_trainer.calculate_metrics()
                        self.nn_trainer.calculate_metrics()
                        self.save(f"Round_{round_index}")
                    round_index += 1
                self.bandit_trainer.inference_iteration(
                    iteration_desc=dict(round_index=round_index)
                )
            bar.update()
        bar.close()

        if last_calculate_metrics_round != round_index:
            self.bandit_trainer.calculate_metrics()
            self.nn_trainer._train(round_index)
            self.nn_trainer.calculate_metrics()
            self.save(f"Round_{round_index}")

        return self.get_last_save_description()
