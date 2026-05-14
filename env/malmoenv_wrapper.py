import gym
import numpy as np
from pathlib import Path
import malmoenv
import json
import math

DEBUG = False

BLOCK_MAP = {
    'bedrock': 0,
    'air': 1,
    'grass': 2,
    'stone': 3,
    'gold_block': 4
}

class MalmoEnvWrapper(gym.Env):

    def __init__(self, xml_path, port=9000):
        super().__init__()
        self.env = malmoenv.make()
        xml = Path(xml_path).read_text()
        self.env.init(xml=xml, port=port)
        self.action_space = gym.spaces.Discrete(4) # w, a, s, d, turn left, turn right
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(125,),
            dtype=np.float32
        )
        self.action_map = {
            0: "move forwards",
            1: "move backwards",
            2: "turn left",
            3: "turn right"
        }

        self.last_action = None
        self.old_grid = None

    def reset(self):
        _ = self.env.reset()
        _, _, _, info = self.env.step(0)
        if not info:
            return np.zeros(125, dtype=np.float32)
        info_dict = json.loads(info)
        grid = info_dict['grid']
        encoded = self._process_grid(grid)
        
        return encoded
    
    def step(self, action):
        if DEBUG:
            print(self.action_map[action])
        _, reward, done, info = self.env.step(action)
        if not info:
            return np.zeros(125), -0.1, False, {}
        info_dict = json.loads(info)
        grid = info_dict['grid']
        x, z = info_dict['XPos'], info_dict['ZPos']
        encoded = self._process_grid(grid)

        if grid != self.old_grid:
            reward += 0.1
        if action == self.last_action:
            reward -= 0.05
        self.last_action = action
        reward -= -0.01

        if _gold_adjacent(grid, 2, 2, 2):
            reward += 10
            done = True

        return encoded, reward, done, info_dict

    def _process_grid(self, grid):
        return np.array([BLOCK_MAP[block] for block in grid], dtype=np.float32)

def _gold_adjacent(grid, x, y, z):
    size = 5
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    for dx, dy, dz in directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        if not (0 <= nx < size and 0 <= ny < size and 0 <= nz < size):
            continue
        
        i = ny * size * size + nz * size + nx
        if grid[i] == "gold_block":
            return True

    return False