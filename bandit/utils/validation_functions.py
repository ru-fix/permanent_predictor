import numpy as np

def square_under_curve(data):
    return np.trapz(data)

def mean_square_under_curve_by_arms(data, amount_arms):
    arms_dic = {arm: [] for arm in range(amount_arms)}
    for trial_dic in data:
        for arm in trial_dic:
            arms_dic[arm].append(trial_dic[arm])
    return np.mean([np.trapz(arms_dic[arm]) for arm in range(arms_dic)])

def maximum_score(data, last_trials=0):
    return np.max(data[-last_trials:])

def mean_maximum_score_by_arms(data, amount_arms, last_trials=0):
    arms_dic = {arm: [] for arm in range(amount_arms)}
    for trial_dic in data:
        for arm in trial_dic:
            arms_dic[arm].append(trial_dic[arm])
    return np.mean([np.max(arms_dic[arm][-last_trials:]) for arm in range(arms_dic)])

def mean_score(data, last_trials=0):
    return np.mean(data[-last_trials:])

def mean_score_by_arms(data, amount_arms, last_trials=0):
    arms_dic = {arm: [] for arm in range(amount_arms)}
    for trial_dic in data:
        for arm in trial_dic:
            arms_dic[arm].append(trial_dic[arm])
    return np.mean([np.mean(arms_dic[arm][-last_trials:]) for arm in range(arms_dic)])