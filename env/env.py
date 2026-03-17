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
BETA = 2.5
GAMMA = 0.25
PENALTY = 0.0

class BinPackingEnv(gym.Env):
    def __init__(self, bin_size=(10, 10, 10), exclude_eta=False, exclude_cog=False):
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
        self.etamap = np.full((self.bin_size[0], self.bin_size[1]), 1e3, dtype=np.float32) # Initialize with a high value (infinity)
        self.placed_items = []
        self.cog_distance_to_center = -1
        self.beta = BETA
        self.gamma = GAMMA
        if exclude_cog:
            self.beta = 0.0 # Ignore COG reward
        if exclude_eta:
            self.gamma = 0.0 # Ignore ETA reward
        self.enable_eta_check = not exclude_eta

    def reset(self, seed=None, options=None, test_sequence=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.heightmap = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.int32)
        self.weightmap = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.float32)
        self.etamap = np.full((self.bin_size[0], self.bin_size[1]), 1e3, dtype=np.float32)
        self.items = None
        self._generate_items(seed=seed, test_sequence=test_sequence)
        self.placed_items.clear()
        self.cog_distance_to_center = -1
        return self.get_obs(), {}

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
        # Put the item into the bin.
        current_max_height = np.max(self.heightmap[x:x+item_l, y:y+item_w])
        self.heightmap[x:x+item_l, y:y+item_w] = current_max_height + item_h
        current_per_cell_weight = weight / (item_l * item_w)
        self.weightmap[x:x+item_l, y:y+item_w] += current_per_cell_weight
        self.etamap[x:x+item_l, y:y+item_w] = np.minimum(self.etamap[x:x+item_l, y:y+item_w], arrival_time)
        self.current_item_index += 1
        self.placed_items.append({'pos': (x, y, current_max_height), 'size': (item_l, item_w, item_h), 'weight': weight, 'eta': arrival_time})
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
        
        # Calculate ETA alignment reward
        max_eta = 42.0
        normalized_eta = arrival_time / max_eta
        normalized_y = y / self.bin_size[1] 
        # The closer the normalized ETA matches the normalized Depth, the higher the reward.
        # E.g., item with ETA 42.0 placed at Y=10 gives distance 0 (Max reward)
        eta_alignment_reward = 1.0 - abs(normalized_eta - normalized_y)
        reward = ALPHA * box_reward + self.beta * cog_reward + self.gamma * eta_alignment_reward
        next_obs = self.get_obs()
        if self.current_item_index >= len(self.items): # all items have been placed
            return next_obs, reward, True, False, {'cog_distance': self.cog_distance_to_center}
        next_mask = self.get_action_mask(next_obs)
        if np.all(next_mask == 1e-3): # penalty=0 for no valid actions; no more viable actions
            return next_obs, reward + PENALTY, True, False, {'cog_distance': self.cog_distance_to_center}
        return next_obs, reward, False, False, {}

    def get_obs(self):
        if self.current_item_index >= len(self.items):
            return {"heightmap": self.heightmap.copy(), "weightmap": self.weightmap.copy(), "etamap": self.etamap.copy(), "item": np.zeros(5, dtype=np.float32)}
        return {"heightmap": self.heightmap.copy(), "weightmap": self.weightmap.copy(), "etamap": self.etamap.copy(), "item": self.items[self.current_item_index]}

    def _generate_items(self, seed=None, test_sequence=None):
        if test_sequence is not None:
            self.items = test_sequence
        else:
            box_generator = cutter.Cutter(self.bin_size[0], self.bin_size[1], self.bin_size[2])
            self.items = box_generator.generate_boxes(seed=seed)

        # Sort by arrival time.
        # self.items.sort(key=lambda x: x[3])
        self.current_item_index = 0
    
    def get_action_mask(self, obs):
        # Initialize with the "Magic Number" (0.001) instead of 0.0
        # This prevents the Dead ReLU problem in the Mask Head.
        mask = np.full(self.bin_size[0] * self.bin_size[1], 1e-3, dtype=np.float32)
        current_item_eta = obs['item'][3] # arrival_time is index 3
        
        for action in range(len(mask)):
            x, y = divmod(action, self.bin_size[1])
            item_l, item_w, item_h = map(int, obs['item'][:3])
            
            # 1) Boundary Check
            if x + item_l > self.bin_size[0] or y + item_w > self.bin_size[1]:
                continue # Stays 1e-3

            current_max_h = np.max(self.heightmap[x:x+item_l, y:y+item_w])
            if current_max_h + item_h > self.bin_size[2]:
                continue # Stays 1e-3
            
            # 2) Physical Stability Check (Using your 50% rule)
            if not self._physical_stability_check(x, y, item_l, item_w, current_max_h):
                continue # Stays 1e-3

            # 3) ETA Blocking Check
            if self.enable_eta_check and not self._eta_blocking_check(x, y, item_l, item_w, current_item_eta):
                continue
            
            # If we reach here, it's Valid!
            mask[action] = 1.0

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
        
        if support_area > 0.80 * item_area and corner_supports >= 3:
            return True

        if support_area > 0.50 * item_area and corner_supports == 4:
            return True

        return False 
    
    def _eta_blocking_check(self, x, y, l, w, current_eta):
        """
        Ensures that this item (current_eta) does not block any already placed 
        items that have an EARLIER ETA. An item is blocked if it's behind 
        (greater Y) the new item.
        """
        for placed in self.placed_items:
            p_x, p_y, _ = placed['pos']
            p_l, _, _ = placed['size']
            p_eta = placed['eta']

            # If placed item is behind the new item and they overlap on the X-axis
            is_behind = p_y >= (y + w)
            overlap_x = not (x + l <= p_x or x >= p_x + p_l)

            if is_behind and overlap_x and p_eta < current_eta:
                return False # Blocking an earlier package
        return True