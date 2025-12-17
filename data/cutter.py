import random
import copy
import util
from box import Box

class Cutter:
    def __init__(self, length, width, height, max_len, max_width, max_height, min_len, min_width, min_height):
        # List of available spaces to be cut
        self.spaces = [(length, width, height)]
        self.length = length
        self.width = width
        self.height = height
        self.max_len = max_len
        self.max_width = max_width
        self.max_height = max_height
        self.min_len = min_len
        self.min_width = min_width
        self.min_height = min_height
        self.boxes = []

    def cut(self):
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
        
        for space in self.spaces:
            self.boxes.append(Box(space[0], space[1], space[2], util.get_time_range()))
    
    def get_boxes(self):
        return self.boxes

    def get_box_count(self):
        return len(self.boxes)
    
    def reset(self):
        self.spaces = [(self.length, self.width, self.height)]
        self.boxes.clear()
    
    def _check_box(self, box):
        x_flag = box[0] < self.min_len or box[0] > self.max_len
        y_flag = box[1] < self.min_width or box[1] > self.max_width
        z_flag = box[2] < self.min_height or box[2] > self.max_height
        return x_flag * 1 + y_flag * 2 + z_flag * 4
    
    def _split(self, box, mask):
        # Axis that needs to be splitted.
        axis_list = []
        if 1 & mask:
            axis_list.append(0)
        if 2 & mask:
            axis_list.append(1)
        if 4 & mask:
            axis_list.append(2)
        axis = random.choice(axis_list)
        pos_range = ()
        if axis == 0:
            pos_range = (self.min_len, box[0] - self.min_len)
        if axis == 1:
            pos_range = (self.min_width, box[1] - self.min_width)
        if axis == 2:
            pos_range = (self.min_height, box[2] - self.min_height)
        pos = random.randint(pos_range[0], pos_range[1])

        # Split on axis at pos
        if axis == 0:
            box1 = (pos, box[1], box[2])
            box2 = (box[0] - pos, box[1], box[2])
        if axis == 1:
            box1 = (box[0], pos, box[2])
            box2 = (box[0], box[1] - pos, box[2])
        if axis == 2:
            box1 = (box[0], box[1], pos)
            box2 = (box[0], box[1], box[2] - pos)
        return box1, box2

# if __name__ == "__main__":
#     cutter = Cutter(10, 10, 10, 5, 5, 5, 2, 2, 2)
#     cutter.cut()
#     boxes = cutter.get_boxes()
#     for b in boxes:
#         print(b)
#     print("Total boxes:", cutter.get_box_count())
#     cutter.reset()
#     boxes = cutter.get_boxes()
#     for b in boxes:
#         print(b)
#     print("Total boxes:", cutter.get_box_count())
#     cutter.cut()
#     boxes = cutter.get_boxes()
#     for b in boxes:
#         print(b)
#     print("Total boxes:", cutter.get_box_count())
