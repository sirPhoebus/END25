import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: (b, seq, c) or (b, c, h, w) ??
        # The external lib handles various shapes, we need to be careful.
        # In our code 'x_emb' is passed to RoPE.
        # Check usage in 'forward':
        # x_emb = self.embedding(x.long()).permute(0, 3, 1, 2)
        # z_local = x_emb + self.pos_emb 
        # Wait, we aren't using RoPE in the current `HybridTRM` code I wrote!
        # I only imported it but used 'pos_emb' (learned absolute) instead!
        # So I can just remove the import!
        return x



class DualStreamBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # --- Local Stream (System 1: Physics/Grid) ---
        # ResNet-style convolution block
        self.local_norm1 = nn.InstanceNorm2d(dim)
        self.local_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.local_act = nn.GELU()
        self.local_norm2 = nn.InstanceNorm2d(dim)
        self.local_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
        # --- Global Stream (System 2: Logic/Abstract) ---
        # Transformer Block
        self.global_norm1 = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.global_norm2 = nn.LayerNorm(dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # --- Interaction (Fusion) ---
        # Bidirectional communication
        # Local -> Global: Pool grid info to global
        # Fusion Projections - Initialize small to prevent signal explosion in deep recurrence
        self.l2g_proj = nn.Linear(dim, dim)
        nn.init.uniform_(self.l2g_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.l2g_proj.bias)
        
        self.g2l_proj = nn.Linear(dim, dim)
        nn.init.uniform_(self.g2l_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.g2l_proj.bias)
        
        # Gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, z_local, z_global, update_global=True):
        """
        z_local: (B, C, H, W)
        z_global: (B, N, C) - N latent tokens
        update_global: bool - hard frequency control
        """
        B, C, H, W = z_local.shape
        
        # --- 1. Local Update (Fast / System 1) ---
        # Always updates
        res = z_local
        x = self.local_norm1(z_local)
        x = self.local_conv1(x)
        x = self.local_act(x)
        x = self.local_norm2(x)
        x = self.local_conv2(x)
        z_local_next = res + x
        
        # --- 2. Global Update (Slow / System 2) ---
        if update_global:
            # Add interaction from Local before Attn
            local_flat = rearrange(z_local, 'b c h w -> b (h w) c')
            local_ctx = local_flat.mean(dim=1, keepdim=True) # (B, 1, C)
            
            # Inject Local Context into Global
            z_global = z_global + self.l2g_proj(local_ctx)
            
            # Self Attention
            res = z_global
            x = self.global_norm1(z_global)
            x, _ = self.global_attn(x, x, x)
            z_global = res + x
            
            # MLP
            res = z_global
            x = self.global_norm2(z_global)
            x = self.global_mlp(x)
            
        # Nested Learning: Sluggish update (0.1) combined with hard frequency (if flag is True)
            # FIX: Removed 0.1 factor as per critique (double dampening was too much)
            z_global_next = res + x
        else:
            # No update this step (Identity)
            z_global_next = z_global
        
        # --- 3. Additional Cross: Global -> Local ---
        # Broadcast global (summarized) back to local
        # We ALWAYS inject global context into local, even if global didn't change this step.
        # This keeps the local stream guided by the persistent global state.
        
        global_ctx = z_global_next.mean(dim=1) # (B, C)
        # global_ctx_grid = repeat(global_ctx, 'b c -> b c h w', h=H, w=W) # unused?
        
        z_local_next = z_local_next + self.g2l_proj(global_ctx).view(B, C, 1, 1)
        
        return z_local_next, z_global_next

class HybridTRM(nn.Module):
    def __init__(self, grid_size=30, colors=10, dim=256, layers=4, global_tokens=16):
        super().__init__()
        self.grid_size = grid_size
        self.colors = colors
        self.dim = dim
        self.max_steps = 12 # Recurrence depth (Increased for logic)

        
        self.embedding = nn.Embedding(colors, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, dim, grid_size, grid_size))
        
        # Support Encoder Projection
        self.support_proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        
        # Latent Initialization
        self.global_tokens = global_tokens
        self.global_init = nn.Parameter(torch.randn(1, global_tokens, dim))
        
        # Core Recursion
        self.blocks = nn.ModuleList([
            DualStreamBlock(dim) for _ in range(layers)
        ])
        
        # Heads
        self.y_head = nn.Conv2d(dim, colors, kernel_size=1)
        self.critic_head = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1) # Sigmoid applied in loss/inference
        )

    def encode_support(self, support, steps=None):
        """
        support: (B, S, 2, H, W) - Batch of S support items, each having Input(0) and Output(1)
        Returns: (B, N, C) - Aggregated global state
        """
        B, S, _, H, W = support.shape
        steps = steps or self.max_steps
        
        # We want to process each support pair to get a global 'rule' vector.
        # Check if S > 0
        
        # Flatten Batch and Support dims to process in parallel
        # (B*S, 2, H, W)
        flat_support = rearrange(support, 'b s c h w -> (b s) c h w')
        
        inp = flat_support[:, 0] # (B*S, H, W)
        out = flat_support[:, 1] # (B*S, H, W)
        
        # Embed
        x_emb = self.embedding(inp.long()).permute(0, 3, 1, 2)
        y_emb = self.embedding(out.long()).permute(0, 3, 1, 2)
        
        # Combine Input and Output for support processing
        # FIX: Concatenate + Project instead of Sum
        cat_emb = torch.cat([x_emb, y_emb], dim=1) # (B*S, 2*Dim, H, W)
        z_local = self.support_proj(cat_emb) + self.pos_emb[:, :, :H, :W]
        
        # Initialize global
        bs = B * S
        z_global = repeat(self.global_init, '1 n c -> b n c', b=bs)
        
        # Run TRM
        for step in range(steps):
            # Slow Stream: Update every 2nd step
            update_global = (step % 2 == 0)
            
            for block in self.blocks:
                z_local, z_global = block(z_local, z_global, update_global=update_global)
                
        # Now z_global holds the "program" for each support example
        # Reshape back to (B, S, N, C)
        z_global = rearrange(z_global, '(b s) n c -> b s n c', b=B)
        
        # Aggregate over Support set (Mean Pooling)
        z_global_agg = z_global.mean(dim=1) # (B, N, C)
        return z_global_agg

    def forward(self, x, initial_state=None, steps=None):
        """
        x: (B, H, W) input grid
        initial_state: (B, N, C) optional primed global state
        steps: recurrence iterations
        """
        B, H, W = x.shape
        steps = steps or self.max_steps
        
        # Embed
        x_emb = self.embedding(x.long()).permute(0, 3, 1, 2) # (B, C, H, W)
        z_local = x_emb + self.pos_emb[:, :, :H, :W]
        
        # Initialize Global State
        if initial_state is not None:
            z_global = initial_state
        else:
            z_global = repeat(self.global_init, '1 n c -> b n c', b=B)
        
        outputs = []
        critic_scores = []
        
        for step in range(steps):
            # Slow Stream: Update every 2nd step
            update_global = (step % 2 == 0)
            
            # Run through all layers (Deep Processing per step)
            for block in self.blocks:
                z_local, z_global = block(z_local, z_global, update_global=update_global)
            
            # Predict
            logits = self.y_head(z_local) # (B, Colors, H, W)
            
            # Critic (Logic Check) based on Global state
            # Mean pool global tokens for critique
            g_state = z_global.mean(dim=1)
            score = self.critic_head(g_state) # Logits
            
            outputs.append(logits)
            critic_scores.append(score)
            
        return outputs, critic_scores
