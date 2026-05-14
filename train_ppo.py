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
model.learn(total_timesteps=100000)

print('Saving...')
model.save("output/ppo_malmo")
print("DONE TRAINING")