import random
from .cog import calculate_bin_cog
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import cutter

# Hyperparameters for reward calculation.
ALPHA = 10.0
BETA = 0.0
PENALTY = -2.0

class BinPackingEnv(gym.Env):
    def __init__(self, bin_size=(10, 10, 10)):
        super(BinPackingEnv, self).__init__()
        self.bin_size = bin_size
        
        # Action space: (x, y) coordinates for the front-bottom-left corner of the item
        self.action_space = spaces.Discrete(bin_size[0] * bin_size[1])
        
        # Observation: Current heightmap + Current item dimensions (l, w, h, arrival_time, weight)
        self.observation_space = spaces.Dict({
            "heightmap": spaces.Box(low=0, high=bin_size[2], shape=(bin_size[0], bin_size[1]), dtype=np.int32),
            "weightmap": spaces.Box(low=0, high=bin_size[2], shape=(bin_size[0], bin_size[1]), dtype=np.float32),
            "item": spaces.Box(low=1, high=max(bin_size), shape=(5,), dtype=np.float32)
        })

        self.heightmap = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.int32)
        self.weightmap = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.float32)
        self.placed_items = []
        self.cog_distance_to_center = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.heightmap = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.int32)
        self.weightmap = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.float32)
        self.items = None
        self._generate_items(seed=seed)
        self.placed_items.clear()
        return self.get_obs(), {}

    def step(self, action):
        # This function should be implemented to handle the following:
        # 1. Apply the action and update the observation space
        # 2. Calculate reward and whether the episode is done
        
        # Action will be bottom-left-front corner of the item placement.
        x, y = divmod(action, self.bin_size[1])
        raw_data = self.items[self.current_item_index]
        item_l, item_w, item_h = map(int, raw_data[:3])
        _, weight = raw_data[3:]

        # Update heightmap and next item index.
        # Put the item into the bin.
        current_max_height = np.max(self.heightmap[x:x+item_l, y:y+item_w])
        self.heightmap[x:x+item_l, y:y+item_w] = current_max_height + item_h
        current_per_cell_weight = weight / (item_l * item_w)
        self.weightmap[x:x+item_l, y:y+item_w] += current_per_cell_weight
        self.current_item_index += 1
        self.placed_items.append({'pos': (x, y, current_max_height), 'size': (item_l, item_w, item_h), 'weight': weight})
        # Calculate reward and termination state.
        box_reward = (item_l * item_w * item_h) / np.prod(self.bin_size) # volume utilization reward
        cog_reward = 0
        cog = calculate_bin_cog(self.placed_items)
        if self.cog_distance_to_center == -1:
            self.cog_distance_to_center = np.linalg.norm(cog - np.array(self.bin_size) / 2)
        else:
            new_distance = np.linalg.norm(cog - np.array(self.bin_size) / 2)
            cog_reward = (self.cog_distance_to_center - new_distance) / (np.linalg.norm(np.array(self.bin_size) / 2)) # normalize by max possible distance
            self.cog_distance_to_center = new_distance
        reward = ALPHA * box_reward + BETA * cog_reward
        next_obs = self.get_obs()
        if self.current_item_index >= len(self.items): # all items have been placed
            return next_obs, reward, True, False, {}
        next_mask = self.get_action_mask(next_obs)
        if np.all(next_mask == 0): # penalty for no valid actions
            return next_obs, PENALTY, True, False, {}
        return next_obs, reward, False, False, {}

    def get_obs(self):
        if self.current_item_index >= len(self.items):
            return {"heightmap": self.heightmap.copy(), "weightmap": self.weightmap.copy(), "item": np.zeros(5, dtype=np.float32)}
        return {"heightmap": self.heightmap.copy(), "weightmap": self.weightmap.copy(), "item": self.items[self.current_item_index]}

    def _generate_items(self, seed=None):
        box_generator = cutter.Cutter(self.bin_size[0], self.bin_size[1], self.bin_size[2])
        self.items = box_generator.generate_boxes(seed=seed)
        # Sort by arrival time.
        # self.items.sort(key=lambda x: x[3])
        self.current_item_index = 0
    
    def get_action_mask(self, obs):
        # All actions are valid by default.
        mask = np.ones(self.bin_size[0] * self.bin_size[1], dtype=np.float32)
        for action in range(len(mask)):
            x, y = divmod(action, self.bin_size[1])
            item_l, item_w, item_h = map(int, obs['item'][:3])
            # 1) boundary check (plus height check)
            if x + item_l > self.bin_size[0] or y + item_w > self.bin_size[1]:
                mask[action] = 0.0
                continue
            current_max_h = np.max(self.heightmap[x:x+item_l, y:y+item_w])
            if current_max_h + item_h > self.bin_size[2]:
                mask[action] = 0.0
                continue
            # 2) physical stability check
            if not self._physical_stability_check(x, y, item_l, item_w, current_max_h):
                mask[action] = 0.0
                continue
            # 3) arrival time check ?    
        return mask
    
    def _physical_stability_check(self, x, y, l, w, support_height):
        support_area = 0
        for i in range(x, x + l):
            for j in range(y, y + w):
                if self.heightmap[i, j] >= support_height:
                    support_area += 1
        item_area = l * w
        
        if support_area > 0.95 * item_area:
            return True
        
        corners = [
            (x, y),
            (x + l - 1, y),
            (x, y + w - 1),
            (x + l - 1, y + w - 1)
        ]

        corner_supports = 0
        for corner in corners:
            if self.heightmap[corner] >= support_height:
                corner_supports += 1
        
        if support_area > 0.8 * item_area and corner_supports >= 3:
            return True

        if support_area > 0.6 * item_area and corner_supports == 4:
            return True

        return False 