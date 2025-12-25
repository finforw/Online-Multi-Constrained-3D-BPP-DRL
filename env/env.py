import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import cutter

class BinPackingEnv(gym.Env):
    def __init__(self, bin_size=(10, 10, 10)):
        super(BinPackingEnv, self).__init__()
        self.bin_size = bin_size
        
        # Action space: (x, y) coordinates for the front-bottom-left corner of the item
        self.action_space = spaces.Discrete(bin_size[0] * bin_size[1])
        
        # Observation: Current heightmap + Current item dimensions (l, w, h, arrival_time, weight)
        self.observation_space = spaces.Dict({
            "heightmap": spaces.Box(low=0, high=bin_size[2], shape=(bin_size[0], bin_size[1]), dtype=np.int32),
            "item": spaces.Box(low=1, high=max(bin_size), shape=(5,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.heightmap = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.int32)
        self._generate_items()
        return self._get_obs(), {}

    def step(self, action):
        # This function should be implemented to handle the following:
        # 1. Apply the action and update the observation space
        # 2. Calculate reward and whether the episode is done
        
        # Action will be bottom-left-front corner of the item placement.
        x, y = divmod(action, self.bin_size[1])
        raw_data = self.items[self.current_item_index]
        item_l, item_w, item_h = map(int, raw_data[:3])
        arrival_time, weight = raw_data[3:]

        # Update heightmap and next item index.
        current_max_height = np.max(self.heightmap[x:x+item_l, y:y+item_w])
        self.heightmap[x:x+item_l, y:y+item_w] = current_max_height + item_h
        self.current_item_index += 1
        # Calculate reward and termination state.
        reward = 10 * (item_l * item_w * item_h) / np.prod(self.bin_size) # volume utilization reward
        terminated = self.current_item_index >= len(self.items)
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return {"heightmap": self.heightmap.copy(), "item": self.items[self.current_item_index]}

    def _generate_items(self):
        box_generator = cutter.Cutter(self.bin_size[0], self.bin_size[1], self.bin_size[2])
        self.items = box_generator.generate_boxes()
        # Sort by arrival time.
        self.items.sort(key=lambda x: x[3])
        self.current_item_index = 0

# if __name__ == "__main__":
#     env = BinPackingEnv()
#     obs = env.reset()
#     print("initial: " + str(obs))
#     action = 0
#     print("action: " + divmod(action, env.bin_size[1]).__str__())
#     obs, reward, terminated, truncated, info = env.step(action)
#     print("------------------------------------")
#     print(obs, reward, terminated, truncated, info)
#     print("------------------------------------")
#     action = 50
#     print("action: " + divmod(action, env.bin_size[1]).__str__())
#     obs, reward, terminated, truncated, info = env.step(action)
#     print("------------------------------------")
#     print(obs, reward, terminated, truncated, info)
#     print("------------------------------------")