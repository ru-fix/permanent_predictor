import numpy as np

def optimal_choice_percentage(played_arms, optimal_arms, round_indexes, last_trials=0):
    """
    Calculates percentage of optimal arms among all choosen
    :param played_arms:
    :param optimal_arms:
    :return:
    """
    played_arms = np.array(played_arms)[-last_trials:]
    optimal_arms = np.array(optimal_arms)[-last_trials:]
    total_played = played_arms.size

    last_round_index = round_indexes[-1]
    optimal_played =  np.count_nonzero(played_arms==optimal_arms)
    return [last_round_index, optimal_played * 100 / total_played]


def mean_rewards_per_arm(rewards, played_arms, round_indexes):
    """
    Calculates mean reward for each arm separately.
    :param rewards:
    :param played_arms:
    :return:
    """
    rewards = np.array(rewards)
    played_arms = np.array(played_arms)
    possible_arms = np.unique(played_arms)

    last_round_index = round_indexes[-1]
    mean_rewards = {
        arm: [last_round_index, rewards[played_arms==arm].mean()]
        for arm in possible_arms
    }
    return mean_rewards


def cumulative_regret(max_rewards, obtained_rewards, round_indexes):
    max_rewards = np.array(max_rewards)
    last_round_index = round_indexes[-1]
    obtained_rewards = np.array(obtained_rewards)
    return [last_round_index, np.sum(max_rewards - obtained_rewards)]


def cumulative_reward(rewards, round_indexes):
    rewards = np.array(rewards)
    last_round_index = round_indexes[-1]
    return [last_round_index, np.sum(rewards)]


def mean_exprected_reward_deviation_per_arm(expected_rewards, obtained_rewards,
                                            played_arms, round_indexes):
    played_arms = np.array(played_arms)
    expected_rewards = np.array(expected_rewards)
    obtained_rewards = np.array(obtained_rewards)
    last_round_index = round_indexes[-1]

    mean_deviation = {}
    possible_arms = np.unique(played_arms)
    for arm in possible_arms:
        mean_expected = expected_rewards[played_arms == arm].mean()
        mean_obtained = obtained_rewards[played_arms == arm].mean()
        mean_deviation[arm] = [last_round_index, np.abs(mean_expected - mean_obtained)]
    return mean_deviation


def optimal_choice_percentage_per_arm(played_arms, optimal_arms, amount_arms):
    played_arms = np.array(played_arms)
    optimal_arms = np.array(optimal_arms)
    dict_optimal = {}
    dict_played = {}
    total_played = played_arms.size
    for arm in range(amount_arms):
        optimal_arm_sum = np.sum(optimal_arms == arm)
        played_arm_sum = np.sum(played_arms == arm)
        dict_optimal[arm] = (optimal_arm_sum, optimal_arm_sum / total_played)
        dict_played[arm] = (played_arm_sum, played_arm_sum / total_played)
    return [dict_optimal, dict_played]


def mean_loss_per_nn(losses, epoch_indexes, nn_indexes, epoch_sizes, round_indexes,
                     amount_epochs, amount_save_epochs):
    losses = np.array(losses)
    epoch_indexes = np.array(epoch_indexes)
    nn_indexes = np.array(nn_indexes)
    epoch_sizes = np.array(epoch_sizes)
    round_indexes = np.array(round_indexes)

    unique_nn_indexes = np.unique(nn_indexes)
    mean_losses = []
    for nn_index in unique_nn_indexes:
        all_nn_mask = nn_indexes == nn_index
        last_nn_round_index = round_indexes[-1]
        last_nn_round_mask = round_indexes == last_nn_round_index
        nn_mask = all_nn_mask & last_nn_round_mask
        if np.any(nn_mask):
            nn_losses = losses[nn_mask]
            nn_epoch_indexes = epoch_indexes[nn_mask]
            epoch_size = epoch_sizes[nn_mask][-1]

            last_epoch_index = nn_epoch_indexes[-1]
            complete_coefficient = last_epoch_index / (amount_epochs - 1)
            amount_iterations = len(nn_epoch_indexes)
            iteration_save_period = \
                max(1, int(amount_iterations / (amount_save_epochs * complete_coefficient)))

            for save_iteration in range(iteration_save_period, amount_iterations, iteration_save_period):
                mean_losses.append({
                    nn_index: [save_iteration / epoch_size, np.mean(nn_losses[:save_iteration])]
                })

    return mean_losses


def gt_vs_predicted_per_nn_for_last_epoch(
    prediction_rewards,
    gt_rewards,
    epoch_indexes,
    nn_indexes,
    round_indexes
):
    prediction_rewards = np.array(prediction_rewards)
    gt_rewards = np.array(gt_rewards)
    nn_indexes = np.array(nn_indexes)
    epoch_indexes = np.array(epoch_indexes)
    round_indexes = np.array(round_indexes)

    unique_nn_indexes = np.unique(nn_indexes)
    gt_and_predicted_for_last_nn = []
    for nn_index in unique_nn_indexes:
        all_nn_mask = nn_indexes == nn_index
        last_nn_round_index = round_indexes[all_nn_mask][-1]
        last_nn_round_mask = round_indexes == last_nn_round_index
        max_epochs = epoch_indexes[all_nn_mask][-1]
        last_epoch_mask = epoch_indexes == max_epochs
        last_epoch_nn_indexes = all_nn_mask & last_nn_round_mask & last_epoch_mask
        nn_predictions = prediction_rewards[last_epoch_nn_indexes]
        nn_targets = gt_rewards[last_epoch_nn_indexes]
        for prediction, target in zip(nn_predictions, nn_targets):
            gt_and_predicted_for_last_nn.append({nn_index: [prediction, target]})

    return gt_and_predicted_for_last_nn
