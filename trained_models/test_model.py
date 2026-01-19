import torch
import numpy as np
import random
from model.model import CNNMaskedActorCritic
from env.env import BinPackingEnv
import os

def evaluate_model(model_path, dataset_path, device='cpu'):
    print(f"--- Loading Model: {model_path} ---")
    
    # 1. Load Model Architecture
    model = CNNMaskedActorCritic(bin_size=(10, 10, 10), hidden_size=256, device=device)
    
    # 2. Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval() # Turns off Dropout/BatchNorm math
    
    # 3. Load Test Data
    print(f"--- Loading Dataset: {dataset_path} ---")
    # The .pt file is likely a list of box sequences: [[(l,w,h), (l,w,h)...], [seq2], ...]
    test_data = torch.load(dataset_path)
    print(f"Found {len(test_data)} test cases.")

    env = BinPackingEnv()
    
    total_rewards = []
    utilizations = []
    
    for i, box_sequence in enumerate(test_data):
        for box in box_sequence:
            box.append(random.random())  # Adds 'Arrival Time' (0.0 to 1.0)
            box.append(random.uniform(1.0, 10.0)) # Adds 'Weight' (1.0 to 10.0)
        # 4. Inject Sequence into Environment
        # We need to hack the env slightly to force this specific sequence
        # Assuming env.reset() generates random boxes, we override them immediately after.
        obs, _ = env.reset(test_sequence=box_sequence)
        
        done = False
        ep_reward = 0
        
        while not done:
            mask = env.get_action_mask(obs)
            
            with torch.no_grad():
                # Get action logits from Model (Deterministic)
                logits, _, _ = model(obs)
                
                # Apply Mask
                # explicitly specify dtype=torch.float32 to prevent inference errors
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                logits = logits.masked_fill(mask_tensor < 0.5, float('-inf'))
                
                # GREEDY ACTION (Argmax) for Testing
                action = torch.argmax(logits, dim=1).item()
            
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            
        total_rewards.append(ep_reward)
        utilizations.append(calc_space_utilization(env.placed_items))
        
        if (i+1) % 100 == 0:
            print(f"Test Case {i+1}: Reward {ep_reward:.2f}")

    mean_reward = np.mean(total_rewards)
    mean_utilization = np.mean(utilizations)
    print(f"\nAverage Reward: {mean_reward:.3f}")
    print(f"Average Utilization: {mean_utilization:.3%}")
    return mean_reward, mean_utilization

def test_model(model, dataset_path, device='cpu'):
    # Load Test Data
    print(f"--- Loading Dataset: {dataset_path} ---")
    # The .pt file is likely a list of box sequences: [[(l,w,h), (l,w,h)...], [seq2], ...]
    test_data = torch.load(dataset_path)
    print(f"Found {len(test_data)} test cases.")

    env = BinPackingEnv()
    
    total_rewards = []
    utilizations = []
    
    for i, box_sequence in enumerate(test_data):
        for box in box_sequence:
            box.append(random.random())  # Adds 'Arrival Time' (0.0 to 1.0)
            box.append(random.uniform(1.0, 10.0)) # Adds 'Weight' (1.0 to 10.0)
        # 4. Inject Sequence into Environment
        # We need to hack the env slightly to force this specific sequence
        # Assuming env.reset() generates random boxes, we override them immediately after.
        obs, _ = env.reset(test_sequence=box_sequence)
        
        done = False
        ep_reward = 0
        
        while not done:
            mask = env.get_action_mask(obs)
            
            with torch.no_grad():
                # Get action logits from Model (Deterministic)
                logits, _, _ = model(obs)
                
                # Apply Mask
                # explicitly specify dtype=torch.float32 to prevent inference errors
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                logits = logits.masked_fill(mask_tensor < 0.5, float('-inf'))
                
                # GREEDY ACTION (Argmax) for Testing
                action = torch.argmax(logits, dim=1).item()
            
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            
        total_rewards.append(ep_reward)
        utilizations.append(calc_space_utilization(env.placed_items))
        
        if (i+1) % 100 == 0:
            print(f"Test Case {i+1}: Reward {ep_reward:.2f}")

    mean_reward = np.mean(total_rewards)
    mean_utilization = np.mean(utilizations)
    print(f"\nAverage Reward: {mean_reward:.3f}")
    print(f"Average Utilization: {mean_utilization:.3%}")
    return mean_reward, mean_utilization

def calc_space_utilization(placed_items, bin_size=1000):
    total_volume = 0
    for item in placed_items:
        l, w, h = item['size']
        total_volume += l * w * h
    return total_volume / bin_size

if __name__ == "__main__":
    # Example Usage
    evaluate_model(
        model_path="trained_models/golden/model.pt",
        dataset_path="test_data/cut_1.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )