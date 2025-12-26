import unittest
import numpy as np
from env import BinPackingEnv

class TestBinPackingEnv(unittest.TestCase):
    def setUp(self):
        # Initialize with a small bin for easy manual calculation
        self.bin_size = (10, 10, 10)
        self.env = BinPackingEnv(bin_size=self.bin_size)
        
        # Manually override generated items for predictable testing
        # Format: (l, w, h, arrival_time, weight)
        self.env.items = [
            np.array([2, 2, 2, 0, 10.0]),  # Item 0
            np.array([2, 2, 2, 1, 10.0]),  # Item 1
        ]
        self.env.current_item_index = 0

    def test_reset(self):
        obs, _ = self.env.reset()
        self.assertEqual(self.env.heightmap.sum(), 0)
        self.assertEqual(len(self.env.placed_items), 0)
        self.assertTrue(len(self.env.items) > 2)
        self.assertTrue("heightmap" in obs)
        self.assertTrue("item" in obs)

    def test_first_step_placement(self):
        # Put first item at position (0, 0)
        self.env.step(0)
        
        # Check heightmap: 2x2 area should now be height 2
        np.testing.assert_array_equal(self.env.heightmap[0:2, 0:2], 2)
        # Rest of the heightmap should be 0
        self.assertEqual(np.sum(self.env.heightmap), 2 * 4)

    def test_stacking_logic(self):
        # Place first item at (0,0)
        self.env.step(0)
        # Place second item exactly on top of the first (action 0 again)
        self.env.step(0)
        
        # Check heightmap: 2x2 area should now be height 2 (first item) + 2 (second item) = 4
        np.testing.assert_array_equal(self.env.heightmap[0:2, 0:2], 4)
        self.assertEqual(len(self.env.placed_items), 2)

    def test_reward_calculation(self):
        _, reward, _, _, _ = self.env.step(0)

        # Since it's the first item, cog_reward is 0.
        self.assertEqual(reward, 0.08)

        # Place 2nd item at (0,0) again
        _, reward, _, _, _ = self.env.step(0)
        self.assertAlmostEqual(reward, 0.1406, places=4)

if __name__ == '__main__':
    unittest.main()