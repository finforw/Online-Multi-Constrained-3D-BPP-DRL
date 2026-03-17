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

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scale parameter

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Flatten spatial dimensions: (B, C, W*H)
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        
        # Calculate Attention Map: (B, W*H, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Residual connection
        return self.gamma * out + x

class CNNMaskedActorCritic(nn.Module):
    def __init__(self, bin_size=(10, 10, 10), hidden_size=256, device='cpu', exclude_eta=False, exclude_cog=False, use_sota=False):
        super(CNNMaskedActorCritic, self).__init__()
        self.bin_size = bin_size
        self.device = device
        self.use_sota = use_sota
        self.exclude_eta = exclude_eta
        self.exclude_cog = exclude_cog
        self.in_channels = 8 # Default to all channels
        if exclude_eta and exclude_cog:
            self.in_channels = 4 # Heightmap + Item Dims
        else:
            self.in_channels = 6
        
        # 1. SHARED BACKBONE
        if self.use_sota:
            # SOTA Model: Simpler architecture without attention
            self.backbone = nn.Sequential(
                init_layer(nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
            )
        else:
            self.backbone = nn.Sequential(
                init_layer(nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                SpatialSelfAttention(64), # Self-Attention Layer to capture global spatial dependencies
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
                init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), 
                nn.ReLU(),
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
        etamap = to_tensor(obs['etamap'], self.device) / 42.0 # Normalize by max ETA range

        item_raw = to_tensor(obs['item'], self.device)
        norm_scale = torch.tensor([self.bin_size[0], self.bin_size[1], self.bin_size[2]], device=self.device)
        item_dims = item_raw[..., :3] / norm_scale
        item_weight = item_raw[..., 4:5] / MAX_WEIGHT
        item_eta = item_raw[..., 3:4] / 42.0 # Normalize current item ETA
        
        if heightmap.dim() == 2: heightmap = heightmap.unsqueeze(0)
        if weightmap.dim() == 2: weightmap = weightmap.unsqueeze(0)
        if etamap.dim() == 2: etamap = etamap.unsqueeze(0)
        if item_dims.dim() == 1: item_dims = item_dims.unsqueeze(0)
        if item_weight.dim() == 1: item_weight = item_weight.unsqueeze(0)
        if item_eta.dim() == 1: item_eta = item_eta.unsqueeze(0)

        batch_size = heightmap.shape[0]
        l, w = self.bin_size[0], self.bin_size[1]
        
        item_channels = item_dims.view(batch_size, 3, 1, 1).expand(batch_size, 3, l, w)
        weight_channels = item_weight.view(batch_size, 1, 1, 1).expand(batch_size, 1, l, w)
        eta_channels = item_eta.view(batch_size, 1, 1, 1).expand(batch_size, 1, l, w)
        if self.in_channels == 4:
            # Golden Model: Heightmap (1) + Item Dims (3)
            x = torch.cat([heightmap.unsqueeze(1), item_channels], dim=1)
            
        elif self.in_channels == 6 and self.exclude_eta:
            # COG Constraint Model: Heightmap (1) + Weightmap (1) + Item Dims (3) + Item Weight (1)
            x = torch.cat([
                heightmap.unsqueeze(1), 
                weightmap.unsqueeze(1), 
                item_channels, 
                weight_channels
            ], dim=1)
        
        elif self.in_channels == 6 and self.exclude_cog:
            # ETA Constraint Model: Heightmap (1) + Etamap (1) + Item Dims (3) + Item ETA (1)
            x = torch.cat([
                heightmap.unsqueeze(1), 
                etamap.unsqueeze(1), 
                item_channels, 
                eta_channels
            ], dim=1)
            
        elif self.in_channels == 8:
            # ETA + COG Constraint Model: All channels
            x = torch.cat([
                heightmap.unsqueeze(1), 
                weightmap.unsqueeze(1),
                etamap.unsqueeze(1),
                item_channels,
                eta_channels,
                weight_channels
            ], dim=1)
        else:
            raise ValueError(f"Unrecognized number of input channels: {self.in_channels}")
        
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