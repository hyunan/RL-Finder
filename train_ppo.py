from stable_baselines3 import PPO
from env.malmoenv_wrapper import MalmoEnvWrapper

env = MalmoEnvWrapper("missions/gold_mission.xml")

model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=1e-4,
    verbose=1,
)

print('Learning...')
model.learn(total_timesteps=100)

print('Saving...')
model.save("output/dqn_malmo")
print("DONE TRAINING")