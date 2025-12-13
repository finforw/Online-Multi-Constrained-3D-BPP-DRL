import random

class Box:
    def __init__(self, length, width, height, time_range):
        self.length = length
        self.width = width
        self.height = height
        self.arrival_time = random.randint(time_range[0], time_range[1])
        self.weight = random.uniform(0, 50)
    
    def __str__(self):
        return f"Box(length={self.length}, width={self.width}, height={self.height}, arrival_time={self.arrival_time}, weight={self.weight})"