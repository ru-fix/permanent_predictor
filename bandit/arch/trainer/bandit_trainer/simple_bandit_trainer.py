"""
Simple bandit trainer (ex. A/B test)
"""

from arch.trainer.bandit_trainer.base_bandit_trainer import BaseBanditTrainer


class SimpleBanditTrainer(BaseBanditTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BaseBanditTrainer._iteration_decorator()
    def exploration_iteration(self, iteration):
        self.environment.render()
        played_arm = iteration % self.amount_arms
        _, obtained_reward, done, description = self.environment.step(played_arm)
        expected_reward = self.model.get_arm_expected_reward(played_arm)
        locals().update(description)
        return locals()

    @BaseBanditTrainer._iteration_decorator()
    def inference_iteration(self):
        self.environment.render()
        played_arm, expected_reward = self.model.get_arm()
        _, obtained_reward, done, description = self.environment.step(played_arm)
        locals().update(description)
        return locals()

    @BaseBanditTrainer._iteration_decorator(
        formate_args=True,
        save_samples=False,
        save_metrics=False
    )
    def train_iteration(self, *sample_data):
        self.model.update(*sample_data)

