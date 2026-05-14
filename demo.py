from stable_baselines3 import DQN
from env.malmoenv_wrapper import MalmoEnvWrapper
import time

env = MalmoEnvWrapper("missions/gold_mission.xml")
model = DQN.load("output/dqn_malmo", env=env)

obs = env.reset()

done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    print(f"Action taken: {env.action_map[int(action)]}")
    time.sleep(0.5)

print("DONE")