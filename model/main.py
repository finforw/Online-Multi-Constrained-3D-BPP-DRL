from .model import CNNMaskedActorCritic, to_tensor
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
PSI = 0.12
LEARNING_RATE = 3e-4
MIN_LR = 1e-5


def choose_action_and_evaluate(model, obs, mask):
    logits, state_value = model(obs) # Unmasked logits first.
    probs = torch.softmax(logits, dim=-1)
    mask_tensor = to_tensor(mask, model.device)
    if mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0)
    infeasible_mask = (mask_tensor == 0).float()
    e_inf = torch.sum(probs * infeasible_mask, dim=-1)
    masked_logits = logits.masked_fill(mask_tensor == 0, float('-inf'))
    dist = Categorical(logits=masked_logits)
    action_tensor = dist.sample() 
    action = action_tensor.item()
    log_prob = dist.log_prob(action_tensor)
    return int(action), log_prob, state_value, e_inf, dist.entropy()

def ac_training_step(optimizer, criterion, state_value, target_value, log_prob, e_inf, e_entropy):
    td_error = target_value - state_value
    actor_loss = -log_prob * td_error.detach()
    critic_loss = criterion(state_value, target_value)
    # TODO: refactor the loss function to include other factors.
    loss = ALPHA * actor_loss + BETA * critic_loss + OMEGA * e_inf.mean() - PSI * e_entropy.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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

def a2c_training_step(optimizer, values, log_probs, returns, entropies, e_infs, psi=PSI):
    """Performs a batch update on the collected rollout."""
    # Convert lists to tensors
    values = torch.cat(values).view(-1)       # Force 1D: [Batch]
    log_probs = torch.stack(log_probs).view(-1)
    returns = returns.to(values.device).view(-1)
    entropies = torch.stack(entropies).view(-1)
    e_infs = torch.stack(e_infs).view(-1)

    # 1. Advantage = Actual Return - Predicted Value
    # We detach values so the actor loss doesn't affect the critic weights
    advantages = returns - values.detach()

    # 2. A2C Loss Components
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = F.mse_loss(values, returns)
    
    # 3. Combined Loss (using your existing weights)
    loss = (ALPHA * actor_loss + 
            BETA * critic_loss + 
            OMEGA * e_infs.mean() - 
            psi * entropies.mean())

    optimizer.zero_grad()
    loss.backward()
    
    # Essential for A2C stability
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.5)
    
    optimizer.step()

def run_episode_and_train(model, optimizer, criterion, env, discount_factor, seed=None, psi=PSI):
    obs, _ = env.reset(seed=seed)
    total_rewards = 0
    steps_taken = 0

    # Rollout Buffers
    values, log_probs, rewards, entropies, e_infs = [], [], [], [], []
    
    while True:
        mask = env.get_action_mask(obs)
        action, log_prob, state_value, e_inf, e_entropy = choose_action_and_evaluate(model, obs, mask)
        next_obs, reward, done, truncated, _ = env.step(action)

        # Store transition data
        values.append(state_value)
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(e_entropy)
        e_infs.append(e_inf)

        total_rewards += reward
        steps_taken += 1
        obs = next_obs

        if done or truncated:
            # 1. Get the value of the final state (0 if done, V(s') if truncated)
            with torch.no_grad():
                _, final_value = model(next_obs)
            
            # 2. Calculate the returns for the whole episode
            returns = calculate_returns(rewards, final_value.item(), done, discount_factor, device=model.device)
            
            # 3. Update the model using the whole rollout (A2C)
            a2c_training_step(optimizer, values, log_probs, returns, entropies, e_infs, psi=psi)

            # Calculate the average entropy for this entire episode
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
    initial_psi = 0.1  # Start very high
    final_psi = 0.01   # End low to allow convergence
    for episode in range(n_episodes):
        psi = initial_psi - (episode / n_episodes) * (initial_psi - final_psi)
        seed = torch.randint(0, 2**32, size=()).item()
        ep_reward, ep_steps, placed_items, ep_entropy = run_episode_and_train(model, optimizer, criterion, env, discount_factor, seed=seed, psi=psi)
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

    return step_history, reward_history, boxes_history, utilization_history, entropy_history

def plot_results(steps, rewards, boxes, utilizations, entropies, filename="training_results.png"):
    # 1. Increased height (from 20 to 24) to give titles more room
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 24))
    
    # 2. Define a reusable function for moving averages to keep code clean
    def plot_with_ma(ax, data, label, color, ma_color, window=100):
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

# def plot_results(steps, rewards, boxes, utilizations, entropies):
#     # Create three subplots: one for rewards, one for boxes, and one for utilization
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
#     plt.subplots_adjust(hspace=0.4)
    
#     # --- Plot 1: Reward per Episode ---
#     ax1.plot(steps, rewards, alpha=0.3, color='blue', label='Raw Reward')
#     if len(rewards) > 50:
#         smooth_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
#         ax1.plot(steps[len(steps)-len(smooth_rewards):], smooth_rewards, color='red', label='Moving Avg (50)')
#     ax1.set_xlabel("Total Steps (Millions)")
#     ax1.set_ylabel("Reward")
#     ax1.set_title("Reward per Million Steps")
#     ax1.legend()

#     # --- Plot 2: Boxes per Episode ---
#     ax2.plot(steps, boxes, alpha=0.3, color='green', label='Raw Boxes')
#     if len(boxes) > 50:
#         smooth_boxes = np.convolve(boxes, np.ones(50)/50, mode='valid')
#         ax2.plot(steps[len(steps)-len(smooth_boxes):], smooth_boxes, color='darkgreen', label='Moving Avg (50)')
#     ax2.set_xlabel("Total Steps (Millions)")
#     ax2.set_ylabel("Number of Boxes Placed")
#     ax2.set_title("Boxes Placed per Million Steps")
#     ax2.legend()

#     # --- Plot 3: Utilization Rate per Episode ---
#     ax3.plot(steps, utilizations, alpha=0.3, color='purple', label='Raw Utilization')
#     if len(boxes) > 50:
#         smooth_utilization = np.convolve(utilizations, np.ones(50)/50, mode='valid')
#         ax3.plot(steps[len(steps)-len(smooth_utilization):], smooth_utilization, color='darkblue', label='Moving Avg (50)')
#     ax3.set_xlabel("Total Steps (Millions)")
#     ax3.set_ylabel("Utilization rate")
#     ax3.set_title("Space Utilization Rate per Million Steps")
#     ax3.legend()

#     # --- Plot 4: Policy Entropy (Curiosity) ---
#     ax4.plot(steps, entropies, alpha=0.3, color='orange', label='Raw Entropy')
#     if len(entropies) > 50:
#         smooth_entropy = np.convolve(entropies, np.ones(50)/50, mode='valid')
#         ax4.plot(steps[-len(smooth_entropy):], smooth_entropy, color='darkorange', label='Moving Avg (50)')
#     ax4.set_xlabel("Total Steps (Millions)")
#     ax4.set_ylabel("Entropy")
#     ax4.set_title("Policy Entropy (Exploration Level)")
#     ax4.legend()

#     plt.show()

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
    n_episodes = 1000
    lr_ratio = MIN_LR / LEARNING_RATE
    optimizer = torch.optim.NAdam(ac_model.parameters(), lr=LEARNING_RATE)
    # Linear Decay
    # The lambda goes from 1.0 down to lr_ratio
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