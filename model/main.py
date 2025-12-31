from .model import CNNMaskedActorCritic, to_tensor
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import BinPackingEnv

ALPHA = 1.0
BETA = 0.5
OMEGA = 0.01
PSI = 0.01


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
    while True:
        mask = env.get_action_mask(obs)
        if np.all(mask == 0):
            return total_rewards
        action, log_prob, state_value, e_inf, e_entropy = choose_action_and_evaluate(model, obs, mask)
        next_obs, reward, done, truncated, _ = env.step(action)
        target_value = get_target_value(model, next_obs, reward, done,
                                        truncated, discount_factor)
        ac_training_step(optimizer, criterion, state_value, target_value, log_prob, e_inf, e_entropy)
        total_rewards += reward
        if done or truncated:
            return total_rewards
        obs = next_obs

def train_actor_critic(model, optimizer, criterion, env, n_episodes=400,
                       discount_factor=0.95):
    totals = []
    model.train()
    for episode in range(n_episodes):
        seed = torch.randint(0, 2**32, size=()).item()
        total_rewards = run_episode_and_train(model, optimizer, criterion, env,
                                              discount_factor,
                                              seed=seed)
        totals.append(total_rewards)
        print(f"\rEpisode: {episode + 1}, Rewards: {total_rewards}", end=" ")

    return totals

if __name__ == "__main__":
    torch.manual_seed(42)
    ac_model = CNNMaskedActorCritic()
    optimizer = torch.optim.NAdam(ac_model.parameters(), lr=1.1e-3)
    criterion = nn.MSELoss()
    env = BinPackingEnv()
    totals = train_actor_critic(ac_model, optimizer, criterion, env)