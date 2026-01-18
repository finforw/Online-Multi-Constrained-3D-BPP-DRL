from .model import CNNMaskedActorCritic, to_tensor
from trained_models.test_model import test_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to files
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import BinPackingEnv

ALPHA = 1.0
BETA = 0.5
OMEGA = 0.01
PSI = 0.01
LEARNING_RATE = 3e-4
MIN_LR = 1e-5
EPISODES = 300000
# mask pred loss
AUX_LOSS_WEIGHT = 0.5


def choose_action_and_evaluate(model, obs, mask):
    # Unpack 3 values
    logits, state_value, mask_pred = model(obs) 
    
    probs = torch.softmax(logits, dim=-1)
    mask_tensor = to_tensor(mask, model.device)
    if mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0)
        
    # Create Binary Mask for RL logic (filter out the 1e-3s)
    binary_mask = (mask_tensor > 0.5).float()
    
    # Calculate infeasible prob using binary mask
    infeasible_mask = (binary_mask == 0).float()
    e_inf = torch.sum(probs * infeasible_mask, dim=-1)
    
    # Mask Logits using binary mask
    masked_logits = logits.masked_fill(binary_mask == 0, float('-inf'))
    
    dist = Categorical(logits=masked_logits)
    action_tensor = dist.sample() 
    action = action_tensor.item()
    log_prob = dist.log_prob(action_tensor)
    
    # Return mask_pred for training
    return int(action), log_prob, state_value, e_inf, dist.entropy(), mask_pred

def get_target_value(model, next_obs, reward, done, truncated, discount_factor):
    with torch.inference_mode():
        _, next_state_value = model(next_obs)

    running = 0.0 if (done or truncated) else 1.0
    target_value = reward + running * discount_factor * next_state_value
    return target_value

def calculate_returns(rewards, next_value, done, gamma=0.95, device='cpu'):
    """Calculates the target returns (R) for the entire rollout."""
    returns = []
    R = 0 if done else next_value
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)

def a2c_training_step(optimizer, values, log_probs, returns, entropies, e_infs, 
                      mask_preds, true_masks):
    # Flatten RL tensors
    values = torch.cat(values).view(-1)
    log_probs = torch.stack(log_probs).view(-1)
    returns = returns.to(values.device).view(-1)
    entropies = torch.stack(entropies).view(-1)
    e_infs = torch.stack(e_infs).view(-1)

    # --- 1. MASK LOSS (MSE) ---
    mask_preds = torch.cat(mask_preds).view(-1, 100)
    true_masks = torch.stack(true_masks).view(-1, 100)
    
    # Use MSE Loss as per Author's code
    mask_loss_func = nn.MSELoss()
    graph_loss = mask_loss_func(mask_preds, true_masks)

    # --- 2. RL Update ---
    advantages = returns - values.detach()
    # Safe Normalization
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()
        
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = F.mse_loss(values, returns)
    
    # --- 3. TOTAL LOSS ---
    loss = (ALPHA * actor_loss + 
            BETA * critic_loss + 
            OMEGA * e_infs.mean() - 
            PSI * entropies.mean() +
            AUX_LOSS_WEIGHT * graph_loss)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.5)
    optimizer.step()

def run_episode_and_train(model, optimizer, criterion, env, discount_factor, seed=None):
    obs, _ = env.reset(seed=seed)
    
    # Buffers
    values, log_probs, rewards, entropies, e_infs = [], [], [], [], []
    mask_preds, true_masks = [], [] # <--- New Buffers
    
    total_rewards = 0
    steps_taken = 0
    
    while True:
        mask = env.get_action_mask(obs) # Soft Mask (1.0 or 0.001)
        
        # Unpack mask_pred
        action, log_prob, state_value, e_inf, e_entropy, current_mask_pred = choose_action_and_evaluate(model, obs, mask)
        
        # Store prediction and ground truth
        mask_preds.append(current_mask_pred)
        true_masks.append(torch.tensor(mask, dtype=torch.float32, device=model.device))

        next_obs, reward, done, truncated, _ = env.step(action)

        values.append(state_value)
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(e_entropy)
        e_infs.append(e_inf)

        total_rewards += reward
        steps_taken += 1
        obs = next_obs

        if done or truncated:
            with torch.no_grad():
                _, final_value, _ = model(next_obs) # Ignore mask_pred here
            
            returns = calculate_returns(rewards, final_value.item(), done, discount_factor, device=model.device)
            
            # Pass new buffers to training step
            a2c_training_step(optimizer, values, log_probs, returns, entropies, e_infs, 
                              mask_preds, true_masks)
            
            avg_ep_entropy = torch.stack(entropies).mean().item()
            return total_rewards, steps_taken, env.placed_items, avg_ep_entropy

def train_actor_critic(model, optimizer, criterion, env, n_episodes=2000,
                       discount_factor=0.95, scheduler=None):
    step_history = []
    reward_history = []
    boxes_history = []     # Number of boxes placed per episode
    utilization_history = [] # Space utilization rate per episode
    entropy_history = []
    total_global_steps = 0
    # totals = []
    model.train()
    best_val_score = -float('inf')

    for episode in range(n_episodes):
        seed = torch.randint(0, 2**32, size=()).item()
        ep_reward, ep_steps, placed_items, ep_entropy = run_episode_and_train(model, optimizer, criterion, env, discount_factor, seed=seed)
        total_global_steps += ep_steps

        # Store data (Convert steps to Millions for the plot)
        step_history.append(total_global_steps / 1e6) 
        reward_history.append(ep_reward)
        boxes_history.append(ep_steps) # This tracks boxes per episode
        utilization_rate = calc_space_utilization(placed_items)
        utilization_history.append(utilization_rate)
        entropy_history.append(ep_entropy) # Store entropy

        if scheduler is not None:
            scheduler.step()

        if (episode + 1) % 10 == 0:
            print(f"\rStep: {total_global_steps} | Episode: {episode + 1} | Reward: {ep_reward:.2f} | Utilization: {utilization_rate:.2f} | Entropy: {ep_entropy:.3f}", end="")
        
        # Checkpoint model every 1000 episodes
        if (episode + 1) % 1000 == 0:
            # 1. Switch to Eval mode (turns off dropout/randomness)
            model.eval()
            
            # 2. Run on fixed test set
            _, utilization_score = test_model(model, 'test_data/cut_1.pt', device=model.device)
            
            # 3. Save if better
            if utilization_score > best_val_score:
                best_val_score = utilization_score
                print("Saving best model with utilization: {:.3%}".format(best_val_score))
                torch.save(model.state_dict(), os.path.join("trained_models", "best_val_model.pt"))
            
            # 4. Switch back to Train mode
            model.train()

    return step_history, reward_history, boxes_history, utilization_history, entropy_history

def plot_results(steps, rewards, boxes, utilizations, entropies, filename="training_results.png"):
    # 1. Increased height (from 20 to 24) to give titles more room
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 24))
    
    # 2. Define a reusable function for moving averages to keep code clean
    def plot_with_ma(ax, data, label, color, ma_color, window=4500):
        ax.plot(steps, data, alpha=0.2, color=color, label=f'Raw {label}')
        if len(data) > window:
            smooth = np.convolve(data, np.ones(window)/window, mode='valid')
            # Ensure X and Y are same length for the plot
            ax.plot(steps[len(steps)-len(smooth):], smooth, color=ma_color, 
                    linewidth=2, label=f'Moving Avg ({window})')
        ax.set_xlabel("Total Steps (Millions)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} per Million Steps", fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # Plot the 4 metrics
    plot_with_ma(ax1, rewards, "Reward", "blue", "red")
    plot_with_ma(ax2, boxes, "Boxes Placed", "green", "darkgreen")
    plot_with_ma(ax3, utilizations, "Utilization Rate", "purple", "darkblue")
    plot_with_ma(ax4, entropies, "Policy Entropy", "orange", "darkorange")

    # 3. Use tight_layout with a top padding to prevent the first title 
    # from hitting the top of the window, and h_pad to separate charts.
    plt.tight_layout(pad=3.0, h_pad=4.0)
    # plt.show()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {filename}")
    plt.close(fig) # Free up memory

def calc_space_utilization(placed_items, bin_size=1000):
    total_volume = 0
    for item in placed_items:
        l, w, h = item['size']
        total_volume += l * w * h
    return total_volume / bin_size

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ac_model = CNNMaskedActorCritic(hidden_size=256, device=device)
    n_episodes = EPISODES
    lr_ratio = MIN_LR / LEARNING_RATE
    optimizer = torch.optim.NAdam(ac_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda ep: 1.0 - (ep / n_episodes) * (1.0 - lr_ratio)
    )
    criterion = nn.MSELoss()
    env = BinPackingEnv()
    # Capture the history
    steps, rewards, boxes, utilizations, entropies = train_actor_critic(ac_model, optimizer, criterion, env, n_episodes=n_episodes, discount_factor=0.99, scheduler=scheduler)
    # Call your plot function
    plot_results(steps, rewards, boxes, utilizations, entropies)