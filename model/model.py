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

class BinItemCrossAttention(nn.Module):
    """
    Novel Cross-Attention module where the incoming item acts as the Query,
    and the bin's spatial features act as the Keys and Values.
    """
    def __init__(self, embed_dim, num_heads=4):
        super(BinItemCrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable residual scale

    def forward(self, bin_features, item_embedding):
        B, C, L, W = bin_features.shape
        
        # Keys and Values: Spatial bin features -> (B, L*W, C)
        kv_bin = bin_features.view(B, C, L * W).permute(0, 2, 1) 
        
        # Query: Incoming item embedding -> (B, 1, C)
        q_item = item_embedding.unsqueeze(1)
        
        # Cross Attention: The item "looks" at the bin to find the best spots
        # attn_weights shape: (B, 1, L*W) - This is our spatial heatmap!
        attn_output, attn_weights = self.multihead_attn(q_item, kv_bin, kv_bin)
        
        # Reshape attention weights to form a spatial mask: (B, 1, L, W)
        spatial_attn_map = attn_weights.view(B, 1, L, W)
        
        # Scale by L*W so the expected value is around 1.0 (since softmax sums to 1)
        attended_bin = bin_features * spatial_attn_map * (L * W) 
        
        # Broadcast the aggregated item-bin context back to spatial dimensions
        context_broadcast = attn_output.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, L, W)
        
        # Combine original features, spatial spotlighting, and global context
        out = bin_features + self.gamma * attended_bin + context_broadcast
        return out

class CNNMaskedActorCritic(nn.Module):
    def __init__(self, bin_size=(10, 10, 10), hidden_size=256, device='cpu', exclude_eta=False, exclude_cog=False, use_sota=False):
        super(CNNMaskedActorCritic, self).__init__()
        self.bin_size = bin_size
        self.device = device
        self.use_sota = use_sota
        self.exclude_eta = exclude_eta
        self.exclude_cog = exclude_cog
        
        # Calculate decoupled input channels
        self.bin_channels = 1 # Always heightmap
        self.item_channels = 3 # Always item dimensions
        
        if not exclude_cog:
            self.bin_channels += 1 # weightmap
            self.item_channels += 1 # item weight
        if not exclude_eta:
            self.bin_channels += 1 # etamap
            self.item_channels += 1 # item eta
        
        # 1. BIN SPATIAL BACKBONE (First 3 layers of SOTA)
        self.bin_backbone = nn.Sequential(
            init_layer(nn.Conv2d(self.bin_channels, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
        )
        
        # 2. ITEM FEATURE EXTRACTOR (MLP)
        self.item_backbone = nn.Sequential(
            init_layer(nn.Linear(self.item_channels, 64), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Linear(64, 64), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
        )
        
        # 3. CROSS-ATTENTION MODULE
        self.cross_attention = BinItemCrossAttention(embed_dim=64, num_heads=4)
        
        # 4. POST-ATTENTION CONV (Last 2 layers of SOTA to complete the 5-layer design)
        self.post_attn_conv = nn.Sequential(
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, padding=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
        )
        
        # 5. ACTOR HEAD
        self.actor_features = nn.Sequential(
            init_layer(nn.Conv2d(64, 8, kernel_size=1)), nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(8 * bin_size[0] * bin_size[1], hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU()
        )
        self.actor_linear = init_layer(nn.Linear(hidden_size, bin_size[0] * bin_size[1]), gain=0.01) 
        
        # 6. CRITIC HEAD
        self.critic_features = nn.Sequential(
            init_layer(nn.Conv2d(64, 4, kernel_size=1)), nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(4 * bin_size[0] * bin_size[1], hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU()
        )
        self.critic_linear = init_layer(nn.Linear(hidden_size, 1), gain=1.0)

        # 7. MASK HEAD
        self.mask_head = nn.Sequential(
            init_layer(nn.Conv2d(64, 8, kernel_size=1)), nn.ReLU(), 
            nn.Flatten(),
            init_layer(nn.Linear(8 * bin_size[0] * bin_size[1], hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Linear(hidden_size, bin_size[0] * bin_size[1]), gain=1.0), nn.ReLU() 
        )

        self.to(self.device)

    def forward(self, obs, mask=None):
        # --- Data Parsing & Normalization ---
        heightmap = to_tensor(obs['heightmap'], self.device) / self.bin_size[2]
        if heightmap.dim() == 2: heightmap = heightmap.unsqueeze(0)
        batch_size = heightmap.shape[0]

        item_raw = to_tensor(obs['item'], self.device)
        norm_scale = torch.tensor([self.bin_size[0], self.bin_size[1], self.bin_size[2]], device=self.device)
        item_dims = item_raw[..., :3] / norm_scale
        if item_dims.dim() == 1: item_dims = item_dims.unsqueeze(0)

        # Build dynamic inputs based on constraints
        bin_inputs = [heightmap.unsqueeze(1)]
        item_inputs = [item_dims]

        if not self.exclude_cog:
            weightmap = to_tensor(obs['weightmap'], self.device) / MAX_WEIGHT
            if weightmap.dim() == 2: weightmap = weightmap.unsqueeze(0)
            bin_inputs.append(weightmap.unsqueeze(1))
            
            item_weight = item_raw[..., 4:5] / MAX_WEIGHT
            if item_weight.dim() == 1: item_weight = item_weight.unsqueeze(0)
            item_inputs.append(item_weight)

        if not self.exclude_eta:
            etamap = to_tensor(obs['etamap'], self.device) / 42.0 
            if etamap.dim() == 2: etamap = etamap.unsqueeze(0)
            bin_inputs.append(etamap.unsqueeze(1))
            
            item_eta = item_raw[..., 3:4] / 42.0 
            if item_eta.dim() == 1: item_eta = item_eta.unsqueeze(0)
            item_inputs.append(item_eta)

        # --- Forward Pass ---
        bin_x = torch.cat(bin_inputs, dim=1)     # Shape: (B, C_bin, L, W)
        item_x = torch.cat(item_inputs, dim=1)   # Shape: (B, C_item)

        # Independent extraction
        bin_feats = self.bin_backbone(bin_x)     # Shape: (B, 64, L, W)
        item_feats = self.item_backbone(item_x)  # Shape: (B, 64)

        # Cross-Attention Core
        attended_feats = self.cross_attention(bin_feats, item_feats) # Shape: (B, 64, L, W)
        
        # Post-processing fusion
        features = self.post_attn_conv(attended_feats)

        # Heads
        actor_emb = self.actor_features(features)
        logits = self.actor_linear(actor_emb)
        
        if mask is not None:
            mask_tensor = to_tensor(mask, self.device)
            if mask_tensor.dim() == 1: mask_tensor = mask_tensor.unsqueeze(0)
            logits = logits.masked_fill(mask_tensor < 0.5, float('-inf'))
        
        critic_emb = self.critic_features(features)
        value = self.critic_linear(critic_emb)
        mask_pred = self.mask_head(features) 
        
        return logits, value, mask_pred