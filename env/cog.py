import numpy as np

def calculate_bin_cog(placed_items):
    """
    placed_items: List of dicts, e.g., 
    [{'pos': (x,y,z), 'size': (l,w,h), 'weight': w}, ...]
    """

    total_weight = 0
    weighted_sum_x = 0
    weighted_sum_y = 0
    weighted_sum_z = 0
    
    for item in placed_items:
        x, y, z = item['pos']
        l, w, h = item['size']
        m = item['weight']
        
        # Find center of this box
        cx = x + l/2
        cy = y + w/2
        cz = z + h/2
        
        weighted_sum_x += m * cx
        weighted_sum_y += m * cy
        weighted_sum_z += m * cz
        total_weight += m
        
    cog_x = weighted_sum_x / total_weight
    cog_y = weighted_sum_y / total_weight
    cog_z = weighted_sum_z / total_weight
    
    return np.array([cog_x, cog_y, cog_z])