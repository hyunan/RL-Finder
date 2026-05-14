from env.malmoenv_wrapper import MalmoEnvWrapper

env = MalmoEnvWrapper("missions/gold_mission.xml")
info = env.reset()
# actions = [0, 4, 2, 4, 3, 1]

# for a in actions:
    # obs, r, d, i = env.step(a)
# print("OBS SHAPE:", obs.shape)  # Should be (75,)
# print("OBS SAMPLE:", obs[:10])  # Should be floats like [0., 1., ...]
# print("INFO TYPE:", type(i))  # Should be <class 'dict'>
# print(i)