"""
Simple bandit trainer (ex. A/B test)
"""

import torch

from arch.trainer.bandit_trainer.base_bandit_trainer import BaseBanditTrainer


class ContextBanditTrainer(BaseBanditTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BaseBanditTrainer._iteration_decorator()
    def exploration_iteration(self, iteration):
        context = torch.tensor(self.environment.render())
        played_arm = iteration % self.amount_arms
        _, obtained_reward, done, description = self.environment.step(played_arm)
        expected_reward = self.model.get_arm_expected_reward(context, played_arm)
        locals().update(description)
        return locals()

    @BaseBanditTrainer._iteration_decorator()
    def inference_iteration(self):
        context = torch.tensor(self.environment.render())
        played_arm, expected_reward, confidence = self.model.get_arm(context)
        _, obtained_reward, done, description = self.environment.step(played_arm.item())
        locals().update(description)
        return locals()

    @BaseBanditTrainer._iteration_decorator(
        save_samples=False,
        save_metrics=False
    )
    def train_iteration(self, *sample_data):
        self.model.update(*sample_data)
