import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_WEIGHT = 50.0

def to_tensor(data, device):
    """Safely converts data to float32 tensor on the correct device."""
    if torch.is_tensor(data):
        return data.float().to(device)
    return torch.tensor(data, dtype=torch.float32, device=device)

# Helper for Orthogonal Initialization (Crucial for RL)
def init_layer(m, gain=1.0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return m

class CNNMaskedActorCritic(nn.Module):
    def __init__(self, bin_size=(10, 10, 10), hidden_size=256, device='cpu'):
        super(CNNMaskedActorCritic, self).__init__()
        self.bin_size = bin_size
        self.device = device
        
        # 1. SHARED BACKBONE
        self.backbone = nn.Sequential(
            init_layer(nn.Conv2d(6, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
        )
        
        # 2. ACTOR HEAD
        self.actor_features = nn.Sequential(
            init_layer(nn.Conv2d(64, 8, kernel_size=1)), nn.ReLU(), # Bottleneck
            nn.Flatten(),
            init_layer(nn.Linear(8 * bin_size[0] * bin_size[1], hidden_size), gain=nn.init.calculate_gain('relu')), 
            nn.ReLU()
        )
        self.actor_linear = init_layer(nn.Linear(hidden_size, bin_size[0] * bin_size[1]), gain=0.01) # Small gain for action logits
        
        # 3. CRITIC HEAD
        self.critic_features = nn.Sequential(
            init_layer(nn.Conv2d(64, 4, kernel_size=1)), nn.ReLU(), # Bottleneck to 4
            nn.Flatten(),
            init_layer(nn.Linear(4 * bin_size[0] * bin_size[1], hidden_size), gain=nn.init.calculate_gain('relu')),
            nn.ReLU()
        )
        self.critic_linear = init_layer(nn.Linear(hidden_size, 1), gain=1.0)

        # 4. MASK HEAD
        # Conv(64->8) -> ReLU -> Flatten -> Linear -> ReLU -> Linear -> ReLU
        self.mask_head = nn.Sequential(
            init_layer(nn.Conv2d(64, 8, kernel_size=1)), nn.ReLU(), 
            nn.Flatten(),
            init_layer(nn.Linear(8 * bin_size[0] * bin_size[1], hidden_size), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_size, bin_size[0] * bin_size[1]), gain=1.0),
            nn.ReLU() 
        )

        self.to(self.device)

    def forward(self, obs, mask=None):
        # --- Input Processing ---
        heightmap = to_tensor(obs['heightmap'], self.device) / self.bin_size[2]
        weightmap = to_tensor(obs['weightmap'], self.device) / MAX_WEIGHT

        item_raw = to_tensor(obs['item'], self.device)
        norm_scale = torch.tensor([self.bin_size[0], self.bin_size[1], self.bin_size[2]], device=self.device)
        item_dims = item_raw[..., :3] / norm_scale
        item_weight = item_raw[..., 4:5] / MAX_WEIGHT
        
        if heightmap.dim() == 2: heightmap = heightmap.unsqueeze(0)
        if weightmap.dim() == 2: weightmap = weightmap.unsqueeze(0)
        if item_dims.dim() == 1: item_dims = item_dims.unsqueeze(0)
        if item_weight.dim() == 1: item_weight = item_weight.unsqueeze(0)

        batch_size = heightmap.shape[0]
        l, w = self.bin_size[0], self.bin_size[1]
        
        item_channels = item_dims.view(batch_size, 3, 1, 1).expand(batch_size, 3, l, w)
        weight_channels = item_weight.view(batch_size, 1, 1, 1).expand(batch_size, 1, l, w)
        x = torch.cat([heightmap.unsqueeze(1), weightmap.unsqueeze(1), item_channels, weight_channels], dim=1)
        
        # --- Shared Backbone ---
        features = self.backbone(x)
        
        # --- Heads ---
        actor_emb = self.actor_features(features)
        logits = self.actor_linear(actor_emb)
        
        # Apply mask to Actor (Logic: If passed mask < 0.5, it's invalid)
        if mask is not None:
            mask_tensor = to_tensor(mask, self.device)
            if mask_tensor.dim() == 1: mask_tensor = mask_tensor.unsqueeze(0)
            # Threshold at 0.5 because our Soft Mask uses 1e-3 for invalid
            logits = logits.masked_fill(mask_tensor < 0.5, float('-inf'))
        
        critic_emb = self.critic_features(features)
        value = self.critic_linear(critic_emb)
        
        mask_pred = self.mask_head(features) 
        
        return logits, value, mask_pred