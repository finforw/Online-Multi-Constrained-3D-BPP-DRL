import torch
import torch.nn as nn
import torch.nn.functional as F

def to_tensor(data, device):
    """Safely converts data to float32 tensor on the correct device."""
    if torch.is_tensor(data):
        return data.float().to(device)
    return torch.tensor(data, dtype=torch.float32, device=device)

class CNNMaskedActorCritic(nn.Module):
    def __init__(self, bin_size=(10, 10), hidden_size=128, device='cpu'):
        super(CNNMaskedActorCritic, self).__init__()
        self.bin_size = bin_size
        self.device = device
        
        # 6 channels: Heightmap(1) + Weightmap(1) + L,W,H,Weight(4)
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_dim = 32 * bin_size[0] * bin_size[1]
        
        # Actor Head
        self.actor_hidden = nn.Linear(cnn_out_dim, hidden_size)
        self.actor_logits = nn.Linear(hidden_size, bin_size[0] * bin_size[1])
        
        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # MOVE THE MODEL TO THE SPECIFIED DEVICE
        self.to(self.device)

    def forward(self, obs, mask=None):
        # Use the safe conversion for all inputs
        heightmap = to_tensor(obs['heightmap'], self.device)
        weightmap = to_tensor(obs['weightmap'], self.device)
        item_dims = to_tensor(obs['item'], self.device)

        # Ensure batch dimension
        if heightmap.dim() == 2:
            heightmap = heightmap.unsqueeze(0)  # [1, 10, 10]
            weightmap = weightmap.unsqueeze(0)
        
        if item_dims.dim() == 1:
            item_dims = item_dims.unsqueeze(0)   # [1, 5]

        batch_size = heightmap.shape[0]
        l, w = self.bin_size

        # 0. Select L, W, H, and Weight (Indices 0, 1, 2, and 4)
        # We skip Index 3 (Arrival Time)
        selected_item_features = item_dims[:, [0, 1, 2, 4]] # Shape: [Batch, 4]
        
        # 1. Broadcast to [Batch, 4, L, W]
        item_channels = selected_item_features.view(batch_size, 4, 1, 1).expand(batch_size, 4, l, w)
        x = torch.cat([heightmap.unsqueeze(1), weightmap.unsqueeze(1), item_channels], dim=1)
        
        # 2. Extract Features
        features = self.cnn(x)
        
        # 3. Actor - Apply Masking
        actor_feat = F.relu(self.actor_hidden(features))
        logits = self.actor_logits(actor_feat)
        
        if mask is not None:
            mask_tensor = to_tensor(mask, self.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            # Use float('-inf') so softmax results in 0 probability
            logits = logits.masked_fill(mask_tensor == 0, float('-inf'))
        
        # 4. Critic
        value = self.critic(features)
        
        return logits, value