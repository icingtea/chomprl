from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import environment

env = environment.ChompEnv()
env.opponent_mode = True
check_env(env)

# training loop
model = DQN(
    "MlpPolicy", env, verbose=1, buffer_size=50000, learning_rate=1e-3, batch_size=64
)
model.learn(total_timesteps=100000)

model.save("model/chomp_dqn")  # weird that it's a zip. wonder what's stored in it.
