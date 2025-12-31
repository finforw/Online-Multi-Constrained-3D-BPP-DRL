import unittest
import numpy as np
from .env import BinPackingEnv

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
            np.array([2, 2, 8, 2, 20.0]),  # Item 2
        ]
        self.env.current_item_index = 0

    def test_reset(self):
        obs, _ = self.env.reset()
        self.assertEqual(self.env.heightmap.sum(), 0)
        self.assertEqual(self.env.weightmap.sum(), 0)
        self.assertEqual(len(self.env.placed_items), 0)
        self.assertTrue(len(self.env.items) > 2)
        self.assertTrue("heightmap" in obs)
        self.assertTrue("weightmap" in obs)
        self.assertTrue("item" in obs)

    def test_first_step_placement(self):
        # Put first item at position (0, 0)
        self.env.step(0)
        
        # Check heightmap: 2x2 area should now be height 2
        np.testing.assert_array_equal(self.env.heightmap[0:2, 0:2], 2)
        # Check weightmap: 2x2 area should now be weight 10.0 / (2*2) = 2.5
        np.testing.assert_array_equal(self.env.weightmap[0:2, 0:2], 2.5)
        # Rest of the heightmap should be 0
        self.assertEqual(np.sum(self.env.heightmap), 2 * 4)

    def test_stacking_logic(self):
        # Place first item at (0,0)
        self.env.step(0)
        # Place second item exactly on top of the first (action 0 again)
        self.env.step(0)
        
        # Check heightmap: 2x2 area should now be height 2 (first item) + 2 (second item) = 4
        np.testing.assert_array_equal(self.env.heightmap[0:2, 0:2], 4)
        # Check weightmap: 2x2 area should now be weight 2.5 (first) + 2.5 (second) = 5.0
        np.testing.assert_array_equal(self.env.weightmap[0:2, 0:2], 5)
        # Ensure only two items are placed
        self.assertEqual(len(self.env.placed_items), 2)

    def test_reward_calculation(self):
        _, reward, _, _, _ = self.env.step(0)

        # Since it's the first item, cog_reward is 0.
        self.assertEqual(reward, 0.08)

        # Place 2nd item at (0,0) again
        _, reward, _, _, _ = self.env.step(0)
        self.assertAlmostEqual(reward, 0.1406, places=4)
    
    def test_mask_boundary_check(self):
        mask = self.env.get_action_mask(self.env.get_obs())
        expected = np.ones(self.bin_size[0] * self.bin_size[1], dtype=np.float32)
        expected[90:100] = 0.0  # Last row should be invalid for 2x2 item
        for i in range(10): # Last column should be invalid
            expected[i*10 + 9] = 0.0 
        np.testing.assert_array_equal(mask, expected)
    
    def test_mask_height_check(self):
        self.env.step(0)
        self.env.step(0)
        mask = self.env.get_action_mask(self.env.get_obs())
        expected = np.ones(self.bin_size[0] * self.bin_size[1], dtype=np.float32)
        expected[90:100] = 0.0  # Last row should be invalid for 2x2 item
        for i in range(10): # Last column should be invalid
            expected[i*10 + 9] = 0.0 
        # Cannot place 8-height item on top of 4-height stack
        expected[0: 2] = 0.0
        expected[10: 12] = 0.0
        np.testing.assert_array_equal(mask, expected)
    
    def test_mask_physical_stability(self):
        self.env.items = [
            np.array([2, 2, 2, 0, 10.0]),  # Item 0
            np.array([2, 2, 2, 1, 10.0]),  # Item 1
            np.array([2, 2, 2, 2, 10.0]),  # Item 2
            np.array([2, 2, 2, 3, 10.0]),  # Item 3
            np.array([3, 3, 3, 4, 15.0]),  # Item 4
        ]
        self.env.step(0)
        self.env.step(3)
        self.env.step(30)
        self.env.step(33)
        mask = self.env.get_action_mask(self.env.get_obs())
        self.assertEqual(mask[0], 0.0)
        self.assertEqual(mask[1], 0.0)
        self.assertEqual(mask[11], 0.0)
        # Change item 4
        self.env.items[4] = np.array([2, 3, 2, 5, 10.0])
        mask = self.env.get_action_mask(self.env.get_obs())
        self.assertEqual(mask[1], 1.0)
        self.assertEqual(mask[31], 1.0)
        self.assertEqual(mask[32], 0.0)
        # Change item 4 again
        self.env.items[4] = np.array([5, 5, 5, 4, 15.0])
        mask = self.env.get_action_mask(self.env.get_obs())
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 0.0)


if __name__ == '__main__':
    unittest.main()