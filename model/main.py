from model import CNNMaskedActorCritic
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import BinPackingEnv


def choose_action_and_evaluate(model, obs, mask):
    logits, state_value = model(obs, mask)
    dist = Categorical(logits=logits)
    action_tensor = dist.sample() 
    action = action_tensor.item()
    log_prob = dist.log_prob(action_tensor)
    return int(action), log_prob, state_value

def ac_training_step(optimizer, criterion, state_value, target_value, log_prob,
                     critic_weight=0.5):
    td_error = target_value - state_value
    actor_loss = -log_prob * td_error.detach()
    critic_loss = criterion(state_value, target_value)
    # TODO: refactor the loss function to include other factors.
    loss = actor_loss + critic_weight * critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_target_value(model, next_obs, reward, done, truncated, discount_factor):
    with torch.inference_mode():
        _, next_state_value = model(next_obs)

    running = 0.0 if (done or truncated) else 1.0
    target_value = reward + running * discount_factor * next_state_value
    return target_value

def run_episode_and_train(model, optimizer, criterion, env, discount_factor,
                          critic_weight, seed=None):
    obs, _ = env.reset(seed=seed)
    total_rewards = 0    
    while True:
        mask = env.get_action_mask()
        action, log_prob, state_value = choose_action_and_evaluate(model, obs, mask)
        next_obs, reward, done, truncated, _ = env.step(action)
        target_value = get_target_value(model, next_obs, reward, done,
                                        truncated, discount_factor)
        ac_training_step(optimizer, criterion, state_value, target_value,
                         log_prob, critic_weight)
        total_rewards += reward
        if done or truncated:
            return total_rewards
        obs = next_obs

def train_actor_critic(model, optimizer, criterion, env, n_episodes=400,
                       discount_factor=0.95, critic_weight=0.5):
    totals = []
    model.train()
    for episode in range(n_episodes):
        seed = torch.randint(0, 2**32, size=()).item()
        total_rewards = run_episode_and_train(model, optimizer, criterion, env,
                                              discount_factor, critic_weight,
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