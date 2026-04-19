import random
import copy
from . import util
from .box import Box

class Cutter:
    def __init__(self, length, width, height, max_len=5, max_width=5, max_height=5, min_len=2, min_width=2, min_height=2):
        # Store 6 values: (length, width, height, x_pos, y_pos, z_pos)
        self.spaces = [(length, width, height, 0, 0, 0)]
        self.boxes = []
        self.length = length
        self.width = width
        self.height = height
        self.max_len = max_len
        self.max_width = max_width
        self.max_height = max_height
        self.min_len = min_len
        self.min_width = min_width
        self.min_height = min_height

    def cut(self):
        self.reset()
        continue_flag = True
        res = []
        while continue_flag:
            continue_flag = False
            for box in self.spaces:
                mask = self._check_box(box)
                if mask == 0:
                    res.append(box)
                else:
                    continue_flag = True
                    box1, box2 = self._split(box, mask)
                    res.append(box1)
                    res.append(box2)
            self.spaces = copy.deepcopy(res)
            res.clear()

        # ---------------------------------------------------------
        # 1) Sort spaces strictly by Z-coordinate (bottom-to-top)
        # ---------------------------------------------------------
        # We tie-break with X and Y just to be deterministic if Z is identical
        self.spaces.sort(key=lambda b: (b[5], b[4], b[3]))
        
        num_boxes = len(self.spaces)
        time_min, time_max = util.get_time_range()
        
        # Generate exactly `num_boxes` random timestamps and sort them ascending
        timestamps = sorted([random.randint(time_min, time_max) for _ in range(num_boxes)])

        # ---------------------------------------------------------
        # 2) Assign timestamps strictly based on Y-coordinate
        # ---------------------------------------------------------
        # Get the indices that would sort the `self.spaces` list by Y-coordinate
        # self.spaces[i][4] refers to the Y-coordinate of the i-th box.
        y_sorted_indices = sorted(range(num_boxes), key=lambda i: self.spaces[i][4])
        
        # Create an array to hold the mapped timestamps
        final_timestamps = [0] * num_boxes
        
        # The lowest Y gets the lowest timestamp, the highest Y gets the highest timestamp
        for rank, original_index in enumerate(y_sorted_indices):
            final_timestamps[original_index] = timestamps[rank]

        # ---------------------------------------------------------
        # 3) Build final array keeping the Z-order sequence unchanged
        # ---------------------------------------------------------
        for space, t in zip(self.spaces, final_timestamps):
            # space[0, 1, 2] are the dimensions (l, w, h)
            self.boxes.append(Box(space[0], space[1], space[2], (t, t)).to_numpy_array())

    def generate_boxes(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.cut()
        return self.boxes
    
    def get_boxes(self):
        return self.boxes

    def get_box_count(self):
        return len(self.boxes)
    
    def reset(self):
        self.spaces = [(self.length, self.width, self.height, 0, 0, 0)]
        self.boxes.clear()
    
    def _check_box(self, box):
        x_flag = box[0] < self.min_len or box[0] > self.max_len
        y_flag = box[1] < self.min_width or box[1] > self.max_width
        z_flag = box[2] < self.min_height or box[2] > self.max_height
        return x_flag * 1 + y_flag * 2 + z_flag * 4
    
    def _split(self, box, mask):
        # Unpack the 6 values so we can track exact (x, y, z) placement during the cuts
        l, w, h, x, y, z = box
        
        axis_list = []
        if 1 & mask:
            axis_list.append(0)
        if 2 & mask:
            axis_list.append(1)
        if 4 & mask:
            axis_list.append(2)
        axis = random.choice(axis_list)

        # Split and preserve relative 3D FTB coordinates
        if axis == 0:
            pos = random.randint(self.min_len, l - self.min_len)
            box1 = (pos, w, h, x, y, z)
            box2 = (l - pos, w, h, x + pos, y, z)
        elif axis == 1:
            pos = random.randint(self.min_width, w - self.min_width)
            box1 = (l, pos, h, x, y, z)
            box2 = (l, w - pos, h, x, y + pos, z) 
        else: # axis == 2
            pos = random.randint(self.min_height, h - self.min_height)
            box1 = (l, w, pos, x, y, z)
            box2 = (l, w, h - pos, x, y, z + pos)
            
        return box1, box2