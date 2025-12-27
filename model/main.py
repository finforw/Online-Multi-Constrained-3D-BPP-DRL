from torch.distributions import Categorical

def choose_action_and_evaluate(model, obs, mask):
    logits, state_value = model(obs['heightmap'], obs['weightmap'], obs['item'], mask)
    dist = Categorical(logits=logits)
    action_tensor = dist.sample() 
    action = action_tensor.item()
    log_prob = dist.log_prob(action_tensor)
    return int(action), log_prob, state_value