from arch.trainer.nn_trainer.base_nn_trainer import BaseNNTrainer

class SimpleNNTrainer(BaseNNTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @BaseNNTrainer._iteration_decorator(
        save_metrics=False
    )
    def get_sample(self, iteration):
        context = self.environment.render()
        played_arm = iteration % self.amount_arms
        _, obtained_reward, done, description = self.environment.step(played_arm)
        locals().update(description)
        return locals()

    @BaseNNTrainer._iteration_decorator(
        save_samples = False
    )
    def train_iteration(
        self,
        train_nn_index,
        contexts,
        train_gt_rewards
    ):
        _, train_prediction_rewards = self.model.nns[train_nn_index](contexts)
        self.optimizer.zero_grad()
        train_loss = self.criterion(train_prediction_rewards, train_gt_rewards)
        train_loss.backward()
        self.optimizer.step()
        return locals()

    @BaseNNTrainer._iteration_decorator(
        save_samples = False
    )
    def validation_iteration(
        self,
        validation_nn_index,
        contexts,
        validation_gt_rewards,
    ):
        _, validation_prediction_rewards = self.model.nns[validation_nn_index](contexts)
        validation_loss = self.criterion(validation_prediction_rewards, validation_gt_rewards)
        return locals()