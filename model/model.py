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

class TemporalGraphAttention(nn.Module):
    def __init__(self, node_features=8, hidden_dim=64, num_heads=4):
        super(TemporalGraphAttention, self).__init__()
        self.node_embedder = nn.Sequential(
            init_layer(nn.Linear(node_features, hidden_dim)), nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()
        )
        self.gat = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes, adj_mask):
        x = self.node_embedder(nodes)
        
        # Expand the float mask for the Multihead Attention layer
        # adj_mask shape is (Batch, Nodes, Nodes). We repeat it for each head.
        mask_for_heads = adj_mask.repeat_interleave(self.gat.num_heads, dim=0)
        
        attn_out, _ = self.gat(query=x, key=x, value=x, attn_mask=mask_for_heads)
        x = self.layer_norm(x + attn_out)
        
        return x.mean(dim=1) # Global Graph Pooling

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
        
        # Dynamically Calculate CNN Input Channels
        self.in_channels = 4 # Base: Heightmap (1) + Item Dims (3)
        if not exclude_cog:
            self.in_channels += 2 # Weightmap (1) + Item Weight (1)
        if not exclude_eta:
            self.in_channels += 2 # Etamap (1) + Item ETA (1)
            
        # Initialize the TS-GNN stream
        if not exclude_eta:
            self.temporal_gnn = TemporalGraphAttention(node_features=8, hidden_dim=64)
        
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
        
        # Calculate dimension sizes for concatenation
        spatial_actor_dim = 8 * bin_size[0] * bin_size[1]
        spatial_critic_dim = 4 * bin_size[0] * bin_size[1]
        temporal_dim = 0 if exclude_eta else 64
        
        # 2. ACTOR HEAD (Split to allow mid-stream fusion)
        self.actor_conv = nn.Sequential(
            init_layer(nn.Conv2d(64, 8, kernel_size=1)), nn.ReLU(), nn.Flatten()
        )
        self.actor_linear = nn.Sequential(
            init_layer(nn.Linear(spatial_actor_dim + temporal_dim, hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Linear(hidden_size, bin_size[0] * bin_size[1] * 2), gain=0.01) # 200 actions
        )
        
        # 3. CRITIC HEAD (Split to allow mid-stream fusion)
        self.critic_conv = nn.Sequential(
            init_layer(nn.Conv2d(64, 4, kernel_size=1)), nn.ReLU(), nn.Flatten()
        )
        self.critic_linear = nn.Sequential(
            init_layer(nn.Linear(spatial_critic_dim + temporal_dim, hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Linear(hidden_size, 1), gain=1.0)
        )

        # 4. DECOUPLED AUXILIARY HEADS
        # CNN Head: Predicts physical geometry
        self.spatial_mask_head = nn.Sequential(
            init_layer(nn.Conv2d(64, 8, kernel_size=1)), nn.ReLU(), 
            nn.Flatten(),
            init_layer(nn.Linear(8 * bin_size[0] * bin_size[1], hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            init_layer(nn.Linear(hidden_size, bin_size[0] * bin_size[1] * 2)) # Output 200 raw logits
        )
        
        # GNN Head: Predicts temporal/ETA logistics
        if not exclude_eta:
            self.temporal_mask_head = nn.Sequential(
                init_layer(nn.Linear(64, hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
                init_layer(nn.Linear(hidden_size, bin_size[0] * bin_size[1] * 2)) # Output 200 raw logits
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
        # Dynamically stack channels to ensure backwards compatibility
        channels_to_cat = [heightmap.unsqueeze(1)]
        
        if not self.exclude_cog:
            channels_to_cat.append(weightmap.unsqueeze(1))
            
        channels_to_cat.append(item_channels)
        
        if not self.exclude_cog:
            weight_channels = item_weight.view(batch_size, 1, 1, 1).expand(batch_size, 1, l, w)
            channels_to_cat.append(weight_channels)
            
        if not self.exclude_eta:
            eta_channels = item_eta.view(batch_size, 1, 1, 1).expand(batch_size, 1, l, w)
            channels_to_cat.append(etamap.unsqueeze(1))
            channels_to_cat.append(eta_channels)
            
        x = torch.cat(channels_to_cat, dim=1)

        # --- Shared Geometry Backbone ---
        features = self.backbone(x)
        
        actor_spatial = self.actor_conv(features)
        critic_spatial = self.critic_conv(features)
        
        # --- Temporal Graph Stream ---
        if not self.exclude_eta:
            nodes = to_tensor(obs['graph_nodes'], self.device)
            if nodes.dim() == 2: nodes = nodes.unsqueeze(0) 
            
            # THE FIX: Cast as float32 instead of bool
            adj_mask = torch.tensor(obs['graph_adj'], dtype=torch.float32, device=self.device)
            if adj_mask.dim() == 2: adj_mask = adj_mask.unsqueeze(0) 
            
            temporal_vector = self.temporal_gnn(nodes, adj_mask)
            
            # Fuse Spatial and Temporal Streams
            actor_in = torch.cat([actor_spatial, temporal_vector], dim=1)
            critic_in = torch.cat([critic_spatial, temporal_vector], dim=1)
        else:
            actor_in = actor_spatial
            critic_in = critic_spatial
            
        # --- Final Linear Heads ---
        logits = self.actor_linear(actor_in)
        value = self.critic_linear(critic_in)
        
        if mask is not None:
            mask_tensor = to_tensor(mask, self.device)
            if mask_tensor.dim() == 1: mask_tensor = mask_tensor.unsqueeze(0)
            logits = logits.masked_fill(mask_tensor < 0.5, float('-inf'))
            
        # --- DECOUPLED PREDICTIONS ---
        spatial_mask_pred = self.spatial_mask_head(features)
        
        if not self.exclude_eta:
            temporal_mask_pred = self.temporal_mask_head(temporal_vector)
        else:
            temporal_mask_pred = torch.zeros_like(spatial_mask_pred)
        
        # Return 4 items now
        return logits, value, spatial_mask_pred, temporal_mask_pred