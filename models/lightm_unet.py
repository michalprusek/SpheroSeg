# lightm_unet.py
# LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation
# Paper: https://arxiv.org/abs/2403.05246
# Original implementation based on the paper architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Optional, Tuple

# Note: This is a simplified implementation without the actual mamba-ssm library
# For production use, install: pip install causal-conv1d mamba-ssm


# ===========================
# Normalization Helper
# ===========================
def get_norm_layer(num_features, use_instance_norm=True):
    """Get normalization layer (Instance or Batch)"""
    if use_instance_norm:
        return nn.InstanceNorm2d(num_features, affine=True)
    else:
        return nn.BatchNorm2d(num_features)


# ===========================
# Depthwise Convolution for efficiency
# ===========================
class DWConv(nn.Module):
    """Depthwise convolution as used in the paper"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, 
            padding=padding, groups=in_channels, bias=False
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        return x


# ===========================
# Simplified Mamba Block (State Space Model)
# ===========================
class MambaBlock(nn.Module):
    """
    Simplified Mamba block implementation
    In the actual implementation, this would use the mamba-ssm library
    """
    def __init__(
        self, 
        d_model,           # Model dimension
        d_state=16,        # SSM state expansion factor  
        d_conv=4,          # Local convolution width
        expand=2,          # Block expansion factor
        dt_rank="auto",    # Rank of dt projection
        dropout=0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == "auto":
            dt_rank = math.ceil(self.d_model / 16)
        self.dt_rank = dt_rank
        
        # Linear projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize special dt projection
        dt_init_std = self.dt_rank**-0.5 * 2
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # S4D real initialization
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (B, L, D) where L is sequence length, D is model dimension
        """
        batch, seqlen, dim = x.shape
        
        # Linear projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seqlen]
        x = rearrange(x, 'b d l -> b l d')
        
        # SSM step
        x = F.silu(x)
        
        # Apply SSM (simplified version)
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        return output
    
    def ssm(self, x):
        """Simplified SSM step - in practice uses selective scan from mamba-ssm"""
        # This is a simplified placeholder
        # Real implementation would use selective_scan_cuda from mamba-ssm
        batch, seqlen, dim = x.shape
        
        # Project x to get dt, B, C
        x_proj = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        
        dt, BC = x_proj.split([self.dt_rank, 2 * self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)
        
        B, C = BC.split(self.d_state, dim=-1)
        
        # Simplified state space computation
        # In reality, this would be a proper SSM scan
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Very simplified SSM (not the actual selective scan)
        y = x * dt.sigmoid()  # Placeholder computation
        
        return y


# ===========================
# Vision State Space Module (VSS)
# ===========================
class VSSModule(nn.Module):
    """
    Vision State Space Module for processing visual features
    Combines SSM with visual-specific operations
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
        # Two parallel branches as described in the paper
        # Branch 1: Linear -> SiLU -> SSM
        self.branch1_proj = nn.Linear(dim, dim * expand)
        self.mamba = MambaBlock(
            d_model=dim * expand,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,  # Already expanded
            dropout=dropout
        )
        
        # Branch 2: Linear -> SiLU
        self.branch2_proj = nn.Linear(dim, dim * expand)
        
        # Output projection
        self.out_proj = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence format
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)
        
        # Normalize
        x_norm = self.norm(x_seq)
        
        # Branch 1: SSM path
        branch1 = F.silu(self.branch1_proj(x_norm))
        branch1 = self.mamba(branch1)
        
        # Branch 2: Gate path
        branch2 = F.silu(self.branch2_proj(x_norm))
        
        # Hadamard product (element-wise multiplication)
        out = branch1 * branch2
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Add residual
        out = out + x_seq
        
        # Reshape back to image format
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return out


# ===========================
# Residual Vision Mamba Layer (RVM)
# ===========================
class RVMLayer(nn.Module):
    """
    Residual Vision Mamba Layer - core building block of LightM-UNet
    Combines VSS with residual connections and adjustment factors
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        
        # First LayerNorm and VSS
        self.norm1 = nn.LayerNorm(dim)
        self.vss = VSSModule(dim, d_state, d_conv, expand, dropout)
        
        # Second LayerNorm and projection
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        
        # Adjustment factor for residual connection
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # First path through VSS
        identity = x
        
        # Apply VSS module
        x = self.vss(x)
        
        # Residual connection with adjustment factor
        x = identity + self.alpha * x
        
        # Second normalization and projection
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_seq = self.norm2(x_seq)
        x = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = self.proj(x)
        
        return x


# ===========================
# Encoder Block
# ===========================
class EncoderBlock(nn.Module):
    """
    Encoder block consisting of RVM layers
    """
    def __init__(self, in_channels, out_channels, num_layers=1, d_state=16, dropout=0.0):
        super().__init__()
        
        # Channel adjustment if needed
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Stack of RVM layers
        self.layers = nn.ModuleList([
            RVMLayer(out_channels, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Downsampling
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        # Adjust channels
        x = self.proj(x)
        
        # Apply RVM layers
        for layer in self.layers:
            x = layer(x)
        
        # Store for skip connection
        skip = x
        
        # Downsample
        x_down = self.downsample(x)
        
        return x_down, skip


# ===========================
# Decoder Block  
# ===========================
class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connections
    """
    def __init__(self, in_channels, skip_channels, out_channels, num_layers=1, d_state=16, dropout=0.0):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # Combine skip connection
        self.combine = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)
        
        # RVM layers
        self.layers = nn.ModuleList([
            RVMLayer(out_channels, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Combine channels
        x = self.combine(x)
        
        # Apply RVM layers
        for layer in self.layers:
            x = layer(x)
        
        return x


# ===========================
# LightM-UNet Main Architecture
# ===========================
class LightMUNet(nn.Module):
    """
    LightM-UNet: Mamba-assisted lightweight UNet for medical image segmentation
    Paper: https://arxiv.org/abs/2403.05246
    """
    def __init__(
        self, 
        in_channels=3, 
        out_channels=1,
        base_channels=32,
        encoder_layers=[1, 2, 2],     # Number of RVM layers per encoder block
        decoder_layers=[2, 2, 1],     # Number of RVM layers per decoder block
        bottleneck_layers=4,           # Number of RVM layers in bottleneck
        d_state=16,                    # State dimension for SSM
        dropout_rate=0.1,
        use_instance_norm=True
    ):
        super().__init__()
        
        # Channel progression
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        # Initial shallow feature extraction with DWConv
        self.init_conv = DWConv(in_channels, channels[0])
        self.init_norm = get_norm_layer(channels[0], use_instance_norm)
        self.init_act = nn.ReLU(inplace=True)
        
        # Encoder blocks (3 blocks as per paper)
        self.encoder1 = EncoderBlock(
            channels[0], channels[0], 
            num_layers=encoder_layers[0], 
            d_state=d_state, 
            dropout=dropout_rate * 0.5
        )
        
        self.encoder2 = EncoderBlock(
            channels[0], channels[1], 
            num_layers=encoder_layers[1], 
            d_state=d_state, 
            dropout=dropout_rate * 0.75
        )
        
        self.encoder3 = EncoderBlock(
            channels[1], channels[2], 
            num_layers=encoder_layers[2], 
            d_state=d_state, 
            dropout=dropout_rate
        )
        
        # Bottleneck (4 successive RVM layers as per paper)
        self.bottleneck_proj = nn.Conv2d(channels[2], channels[3], kernel_size=1)
        self.bottleneck = nn.ModuleList([
            RVMLayer(channels[3], d_state=d_state, dropout=dropout_rate * 1.2)
            for _ in range(bottleneck_layers)
        ])
        
        # Decoder blocks (3 blocks)
        self.decoder3 = DecoderBlock(
            channels[3], channels[2], channels[2],
            num_layers=decoder_layers[0],
            d_state=d_state,
            dropout=dropout_rate
        )
        
        self.decoder2 = DecoderBlock(
            channels[2], channels[1], channels[1],
            num_layers=decoder_layers[1],
            d_state=d_state,
            dropout=dropout_rate * 0.75
        )
        
        self.decoder1 = DecoderBlock(
            channels[1], channels[0], channels[0],
            num_layers=decoder_layers[2],
            d_state=d_state,
            dropout=dropout_rate * 0.5
        )
        
        # Final output head
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] // 2, kernel_size=3, padding=1),
            get_norm_layer(channels[0] // 2, use_instance_norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(channels[0] // 2, out_channels, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = self.init_act(x)
        
        # Encoder path with skip connections
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        
        # Bottleneck
        x = self.bottleneck_proj(x3)
        for layer in self.bottleneck:
            x = layer(x)
        
        # Decoder path with skip connections
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Final output
        out = self.final_conv(x)
        
        return out
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# ===========================
# Test the model
# ===========================
if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with original paper configuration
    model = LightMUNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
        encoder_layers=[1, 2, 2],
        decoder_layers=[2, 2, 1],
        bottleneck_layers=4
    ).to(device)
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    print("=" * 60)
    print("LightM-UNet Model Summary (Original Architecture)")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    print()
    
    # Test forward pass
    print("Testing forward pass with different input sizes:")
    for size in [128, 256, 512, 1024]:
        x = torch.randn(1, 3, size, size).to(device)
        with torch.no_grad():
            y = model(x)
        print(f"  Input {size}x{size} -> Output {y.shape}")
    
    print("=" * 60)