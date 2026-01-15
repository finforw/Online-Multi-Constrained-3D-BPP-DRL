import torch

def inspect_pt_file(file_path):
    try:
        # Load the data
        print(f"Loading {file_path}...")
        data = torch.load(file_path)
        
        # 1. Check the Top-Level Type
        print(f"--- Structure Analysis ---")
        print(f"Top-level Type: {type(data)}")
        
        # 2. Inspect based on type
        if isinstance(data, list):
            print(f"Structure: List of {len(data)} elements")
            if len(data) > 0:
                print(f"First Element Type: {type(data[0])}")
                # Print sample of the first element (e.g., if it's a sequence of boxes)
                print(f"Sample (First Element): {data[0]}")
                
                # Check for nested lists/tensors (common in RL datasets)
                if isinstance(data[0], list):
                     print(f"  -> Inner List Length: {len(data[0])}")
                     print(f"  -> Inner Item Type: {type(data[0][0])}")
                elif isinstance(data[0], torch.Tensor):
                     print(f"  -> Inner Tensor Shape: {data[0].shape}")
                     
        elif isinstance(data, torch.Tensor):
            print(f"Structure: Tensor")
            print(f"Shape: {data.shape}")
            print(f"Data Type (dtype): {data.dtype}")
            
        elif isinstance(data, dict):
            print(f"Structure: Dictionary")
            print(f"Keys: {list(data.keys())}")
            # Print type of the first value
            if data:
                first_key = next(iter(data))
                print(f"Type of value for '{first_key}': {type(data[first_key])}")
        
        else:
            print(f"Content: {data}")

    except Exception as e:
        print(f"Error inspecting file: {e}")

# Run the inspection
if __name__ == "__main__":
    inspect_pt_file('cut_2.pt')