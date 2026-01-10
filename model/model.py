import torch
import torch.nn as nn
import torch.nn.functional as F

def to_tensor(data, device):
    if torch.is_tensor(data):
        return data.float().to(device)
    return torch.tensor(data, dtype=torch.float32, device=device)

class CNNMaskedActorCritic(nn.Module):
    def __init__(self, bin_size=(10, 10), hidden_size=128, device='cpu'):
        super(CNNMaskedActorCritic, self).__init__()
        self.bin_size = bin_size
        self.device = device
        
        # CHANGED: Input channels reduced from 6 to 4
        # 1 Channel (Heightmap) + 3 Channels (Item Length, Width, Height)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), nn.ReLU(), # Changed 6 -> 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_dim = 32 * bin_size[0] * bin_size[1]
        
        self.actor_hidden = nn.Linear(cnn_out_dim, hidden_size)
        self.actor_logits = nn.Linear(hidden_size, bin_size[0] * bin_size[1])
        
        self.critic = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.to(self.device)

    def forward(self, obs, mask=None):
        # 1. Normalize Heightmap
        # Divide by bin height (e.g., 10.0)
        heightmap = to_tensor(obs['heightmap'], self.device) / self.bin_size[1]

        # 2. Prepare Item Dimensions (Normalize & Select L,W,H)
        item_raw = to_tensor(obs['item'], self.device)
        
        # Normalize item dims by bin sizes (L, W, H)
        # Note: Assuming bin is 10x10x10. If dimensions differ, divide separately.
        norm_scale = torch.tensor(
            [self.bin_size[0], self.bin_size[1], self.bin_size[1]], 
            device=self.device
        )
        
        # Select indices 0, 1, 2 (L, W, H). Ignore 3 (Time) and 4 (Weight).
        item_dims = item_raw[..., :3] / norm_scale 

        # Ensure batch dimension
        if heightmap.dim() == 2:
            heightmap = heightmap.unsqueeze(0)  # [1, 10, 10]
        
        if item_dims.dim() == 1:
            item_dims = item_dims.unsqueeze(0)   # [1, 3]

        batch_size = heightmap.shape[0]
        l, w = self.bin_size

        # 3. Broadcast Item features to spatial dimensions
        # Shape: [Batch, 3, 1, 1] -> [Batch, 3, 10, 10]
        item_channels = item_dims.view(batch_size, 3, 1, 1).expand(batch_size, 3, l, w)
        
        # 4. Concatenate: 1 Heightmap + 3 Item Channels = 4 Channels
        x = torch.cat([heightmap.unsqueeze(1), item_channels], dim=1)
        
        # Standard processing
        features = self.cnn(x)
        actor_feat = F.relu(self.actor_hidden(features))
        logits = self.actor_logits(actor_feat)
        
        if mask is not None:
            mask_tensor = to_tensor(mask, self.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            logits = logits.masked_fill(mask_tensor == 0, float('-inf'))
        
        value = self.critic(features)
        
        return logits, value