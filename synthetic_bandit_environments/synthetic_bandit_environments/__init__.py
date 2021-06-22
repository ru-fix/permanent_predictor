from gym.envs.registration import register

register(
    id="slinenv-v0",
    entry_point='synthetic_bandit_environments.envs:StaticLinearEnvironment'
)
register(
    id="snlinenv-v0",
    entry_point='synthetic_bandit_environments.envs:StaticNormLinearEnvironment'
)
register(
    id="spolyenv-v0",
    entry_point='synthetic_bandit_environments.envs:StaticPolynomialEnvironment'
)
register(
    id="mlinenv-v0",
    entry_point='synthetic_bandit_environments.envs:MovingLinearEnvironment'
)
register(
    id="mpolyenv-v0",
    entry_point='synthetic_bandit_environments.envs:MovingPolynomialEnvironment'
)
register(
    id="mnlinenv-v0",
    entry_point='synthetic_bandit_environments.envs:MovingNormLinearEnvironment'
)