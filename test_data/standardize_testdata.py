import torch
import random

def modify_dataset(input_path, output_path):
    print(f"--- Loading Original Dataset: {input_path} ---")
    # Load the original data: List of sequences [[(L, W, H), ...], [...]]
    data = torch.load(input_path)
    
    modified_data = []
    
    for sequence in data:
        new_sequence = []
        for box in sequence:
            # 1. Extract original dimensions
            l, w, h = box[0], box[1], box[2]
            
            # 2. Generate random ETA (1 to 42)
            eta = random.randint(1, 42)
            
            # 3. Generate random Weight (1e-9 to 50)
            weight = random.uniform(1e-9, 50.0)
            
            # 4. Create new box format: (L, W, H, ETA, Weight)
            # Keeping it as a list/tuple to match your training expectations
            new_box = [l, w, h, eta, weight]
            new_sequence.append(new_box)
            
        modified_data.append(new_sequence)

    # Save the modified list
    torch.save(modified_data, output_path)
    print(f"--- Successfully saved {len(modified_data)} cases to: {output_path} ---")

if __name__ == "__main__":
    INPUT_FILE = "test_data/cut_1.pt"
    OUTPUT_FILE = "test_data/cut_1_expanded.pt"
    
    modify_dataset(INPUT_FILE, OUTPUT_FILE)