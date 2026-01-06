from .model import CNNMaskedActorCritic, to_tensor
import torch
import torch.nn as nn
from torch.distributions import Categorical
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

def run_episode_and_train(model, optimizer, criterion, env, discount_factor, seed=None):
    obs, _ = env.reset(seed=seed)
    total_rewards = 0
    steps_taken = 0
    
    while True:
        mask = env.get_action_mask(obs)
        action, log_prob, state_value, e_inf, e_entropy = choose_action_and_evaluate(model, obs, mask)
        next_obs, reward, done, truncated, _ = env.step(action)

        steps_taken += 1

        target_value = get_target_value(model, next_obs, reward, done,
                                        truncated, discount_factor)
        ac_training_step(optimizer, criterion, state_value, target_value, log_prob, e_inf, e_entropy)
        total_rewards += reward
        if done or truncated:
            return total_rewards, steps_taken
        obs = next_obs

def train_actor_critic(model, optimizer, criterion, env, n_episodes=2000,
                       discount_factor=0.95):
    step_history = []
    reward_history = []
    boxes_history = []     # Number of boxes placed per episode
    total_global_steps = 0
    # totals = []
    model.train()
    for episode in range(n_episodes):
        seed = torch.randint(0, 2**32, size=()).item()
        ep_reward, ep_steps = run_episode_and_train(model, optimizer, criterion, env, discount_factor, seed=seed)
        total_global_steps += ep_steps

        # Store data (Convert steps to Millions for the plot)
        step_history.append(total_global_steps / 1e6) 
        reward_history.append(ep_reward)
        boxes_history.append(ep_steps) # This tracks boxes per episode
        if (episode + 1) % 10 == 0:
            print(f"\rStep: {total_global_steps} | Episode: {episode + 1} | Reward: {ep_reward:.2f}", end="")

    return step_history, reward_history, boxes_history

def plot_results(steps, rewards, boxes):
    # Create two subplots: one for rewards, one for boxes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)
    
    # --- Plot 1: Reward per Episode ---
    ax1.plot(steps, rewards, alpha=0.3, color='blue', label='Raw Reward')
    if len(rewards) > 50:
        smooth_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax1.plot(steps[len(steps)-len(smooth_rewards):], smooth_rewards, color='red', label='Moving Avg (50)')
    ax1.set_xlabel("Total Steps (Millions)")
    ax1.set_ylabel("Reward")
    ax1.set_title("Reward per Million Steps")
    ax1.legend()

    # --- Plot 2: Boxes per Episode ---
    ax2.plot(steps, boxes, alpha=0.3, color='green', label='Raw Boxes')
    if len(boxes) > 50:
        smooth_boxes = np.convolve(boxes, np.ones(50)/50, mode='valid')
        ax2.plot(steps[len(steps)-len(smooth_boxes):], smooth_boxes, color='darkgreen', label='Moving Avg (50)')
    ax2.set_xlabel("Total Steps (Millions)")
    ax2.set_ylabel("Number of Boxes Placed")
    ax2.set_title("Boxes Placed per Million Steps")
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    torch.manual_seed(36)
    ac_model = CNNMaskedActorCritic()
    optimizer = torch.optim.NAdam(ac_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    env = BinPackingEnv()
    # Capture the history
    steps, rewards, boxes = train_actor_critic(ac_model, optimizer, criterion, env)
    # Call your plot function
    plot_results(steps, rewards, boxes)