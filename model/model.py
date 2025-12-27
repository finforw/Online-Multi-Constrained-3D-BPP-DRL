import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCNNOnlyActorCritic(nn.Module):
    def __init__(self, bin_size=(10, 10), hidden_size=128):
        super(MaskedCNNOnlyActorCritic, self).__init__()
        self.bin_size = bin_size
        
        # 5-layer CNN for 4 channels (Heightmap + L, W, H)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), nn.ReLU(),
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

    def forward(self, heightmap, item_dims, mask=None):
        batch_size = heightmap.shape[0]
        h, w = self.bin_size
        
        # 1. Broadcoast Item L, W, H to channels
        item_channels = item_dims.view(batch_size, 3, 1, 1).expand(batch_size, 3, h, w)
        x = torch.cat([heightmap.unsqueeze(1), item_channels], dim=1)
        
        # 2. Extract Features
        features = self.cnn(x)
        
        # 3. Actor - Apply Masking
        actor_feat = F.relu(self.actor_hidden(features))
        logits = self.actor_logits(actor_feat)
        
        if mask is not None:
            # Mask is 1 for valid, 0 for invalid. 
            # We set invalid logits to a very large negative number.
            logits = logits.masked_fill(mask == 0, -1e9)
        
        # 4. Critic
        value = self.critic(features)
        
        return logits, value