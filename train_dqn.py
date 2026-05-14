from stable_baselines3 import DQN
from env.malmoenv_wrapper import MalmoEnvWrapper

env = MalmoEnvWrapper("missions/gold_mission.xml")

model = DQN(
    policy="MlpPolicy",
    env=env,

    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,

    gamma=0.99,

    train_freq=4,
    target_update_interval=1000,

    exploration_fraction=0.2,
    exploration_final_eps=0.05,

    verbose=1,
)

print('Learning...')
model.learn(total_timesteps=100000)

print('Saving...')
model.save("output/dqn_malmo")
print("DONE TRAINING")