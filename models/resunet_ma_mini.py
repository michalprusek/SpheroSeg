# Advanced ResUNet with state-of-the-art attention mechanisms
# Optimized for spheroid segmentation with ~66M parameters (similar to HRNet)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


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
# SimAM: Parameter-Free Attention Module
# ===========================
class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module
    Paper: https://proceedings.mlr.press/v139/yang21o.html
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Calculate channel mean and variance
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = ((x - mean) ** 2).sum(dim=[2, 3], keepdim=True) / (h * w)
        
        # Calculate importance of each neuron
        y = (x - mean) / (var + self.e_lambda).sqrt()
        
        # Calculate attention weight
        sigmoid = torch.sigmoid(-y)
        
        return x * sigmoid


# ===========================
# NAM: Normalization-based Attention Module
# ===========================
class NAM(nn.Module):
    """
    NAM: Normalization-based Attention Module
    Uses batch norm weights for attention
    """
    def __init__(self, channels, use_instance_norm=True):
        super(NAM, self).__init__()
        self.norm = get_norm_layer(channels, use_instance_norm)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Get normalized features
        norm_out = self.norm(x)
        
        # Use normalization statistics as attention
        weight = self.sigmoid(norm_out)
        
        return x * weight


# ===========================
# Triplet Attention
# ===========================
class TripletAttention(nn.Module):
    """
    Triplet Attention: Learning Triple Attention
    Captures cross-dimension interaction between C-H-W
    """
    def __init__(self, channels, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        
        # Three branches for C-H, C-W, and H-W interactions
        self.conv_ch = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), 
                                 padding=(0, padding), bias=False)
        self.conv_cw = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), 
                                 padding=(padding, 0), bias=False)
        self.conv_hw = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                                padding=padding, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Branch 1: C-H interaction (pool over width)
        x_ch = x.mean(dim=3, keepdim=True)  # [B, C, H, 1]
        avg_ch = torch.mean(x_ch, dim=1, keepdim=True)  # [B, 1, H, 1]
        max_ch, _ = torch.max(x_ch, dim=1, keepdim=True)  # [B, 1, H, 1]
        ch_att = torch.cat([avg_ch, max_ch], dim=1)  # [B, 2, H, 1]
        ch_att = self.conv_ch(ch_att).sigmoid()  # [B, 1, H, 1]
        
        # Branch 2: C-W interaction (pool over height)
        x_cw = x.mean(dim=2, keepdim=True)  # [B, C, 1, W]
        avg_cw = torch.mean(x_cw, dim=1, keepdim=True)  # [B, 1, 1, W]
        max_cw, _ = torch.max(x_cw, dim=1, keepdim=True)  # [B, 1, 1, W]
        cw_att = torch.cat([avg_cw, max_cw], dim=1)  # [B, 2, 1, W]
        cw_att = self.conv_cw(cw_att).sigmoid()  # [B, 1, 1, W]
        
        # Branch 3: H-W interaction (spatial attention)
        avg_hw = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_hw, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        hw_att = torch.cat([avg_hw, max_hw], dim=1)  # [B, 2, H, W]
        hw_att = self.conv_hw(hw_att).sigmoid()  # [B, 1, H, W]
        
        # Combine all three attentions
        out = x * ch_att * cw_att * hw_att
        
        return out


# ===========================
# Lightweight Self-Attention for Bottleneck
# ===========================
class LightweightSelfAttention(nn.Module):
    """
    Efficient self-attention module for bottleneck
    Uses depthwise convolutions for efficiency
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(LightweightSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Use depthwise separable convolutions for efficiency
        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * 7 - 1) * (2 * 7 - 1), num_heads)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# ===========================
# Multi-Stage Attention Block
# ===========================
class MultiStageAttention(nn.Module):
    """
    Combines SimAM/NAM + Triplet Attention for comprehensive feature enhancement
    """
    def __init__(self, channels, use_simam=True, use_instance_norm=True):
        super(MultiStageAttention, self).__init__()
        
        # Stage 1: Parameter-free attention
        if use_simam:
            self.stage1 = SimAM()
        else:
            self.stage1 = NAM(channels, use_instance_norm)
            
        # Stage 2: Cross-dimension attention
        self.stage2 = TripletAttention(channels)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            get_norm_layer(channels, use_instance_norm),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Apply parameter-free attention
        out1 = self.stage1(x)
        
        # Apply cross-dimension attention
        out2 = self.stage2(out1)
        
        # Residual connection with fusion
        out = self.fusion(out2) + x
        
        return out


# ===========================
# Advanced Residual Block
# ===========================
class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, 
                 use_attention=True, use_simam=True, dropout=0.1, use_instance_norm=True):
        super(AdvancedResidualBlock, self).__init__()
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = get_norm_layer(out_channels, use_instance_norm)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.norm2 = get_norm_layer(out_channels, use_instance_norm)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
        # Attention mechanism
        if use_attention:
            self.attention = MultiStageAttention(out_channels, use_simam, use_instance_norm)
        else:
            self.attention = nn.Identity()
        
        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm_layer(out_channels, use_instance_norm)
            )
            
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Apply attention before adding residual
        out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out


# ===========================
# Enhanced Attention Gate
# ===========================
class AdvancedAttentionGate(nn.Module):
    """
    Advanced attention gate with multi-scale features and triplet attention
    """
    def __init__(self, F_g, F_l, F_int, use_instance_norm=True):
        super(AdvancedAttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            get_norm_layer(F_int, use_instance_norm)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            get_norm_layer(F_int, use_instance_norm)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            get_norm_layer(1, use_instance_norm),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Additional triplet attention for refined features
        self.refine = TripletAttention(F_l)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention and refine with triplet attention
        out = x * psi
        out = self.refine(out)
        
        return out


# ===========================
# Advanced ResUNet Architecture
# ===========================
class AdvancedResUNet(nn.Module):
    """
    State-of-the-art ResUNet with SimAM/NAM + Triplet Attention + Lightweight Self-Attention
    Optimized for ~66M parameters to match HRNet capacity
    """
    def __init__(self, in_channels=3, out_channels=1,
                 features=[20, 40, 80, 160],  # Further reduced capacity for better generalization
                 use_simam=True, use_instance_norm=True, dropout_rate=0.2):
        super(AdvancedResUNet, self).__init__()
        
        self.use_instance_norm = use_instance_norm
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=1, padding=3, bias=False),
            get_norm_layer(features[0], use_instance_norm),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder1 = self._make_encoder_block(features[0], features[0], num_blocks=2, 
                                                use_simam=use_simam)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = self._make_encoder_block(features[0], features[1], num_blocks=2, 
                                                use_simam=use_simam)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = self._make_encoder_block(features[1], features[2], num_blocks=3, 
                                                use_simam=use_simam)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = self._make_encoder_block(features[2], features[3], num_blocks=3, 
                                                use_simam=use_simam)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with lightweight self-attention
        bottleneck_channels = features[3] * 2
        self.bottleneck = nn.Sequential(
            AdvancedResidualBlock(features[3], bottleneck_channels, 
                                use_simam=use_simam, use_instance_norm=use_instance_norm),
            LightweightSelfAttention(bottleneck_channels),
            AdvancedResidualBlock(bottleneck_channels, bottleneck_channels, 
                                use_simam=use_simam, use_instance_norm=use_instance_norm)
        )
        
        # Decoder with attention gates
        self.upconv4 = nn.ConvTranspose2d(bottleneck_channels, features[3], kernel_size=2, stride=2)
        self.att4 = AdvancedAttentionGate(features[3], features[3], features[3]//2, use_instance_norm)
        self.decoder4 = self._make_decoder_block(features[3]*2, features[3], use_simam=use_simam)
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AdvancedAttentionGate(features[2], features[2], features[2]//2, use_instance_norm)
        self.decoder3 = self._make_decoder_block(features[2]*2, features[2], use_simam=use_simam)
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AdvancedAttentionGate(features[1], features[1], features[1]//2, use_instance_norm)
        self.decoder2 = self._make_decoder_block(features[1]*2, features[1], use_simam=use_simam)
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.att1 = AdvancedAttentionGate(features[0], features[0], features[0]//2, use_instance_norm)
        self.decoder1 = self._make_decoder_block(features[0]*2, features[0], use_simam=use_simam)
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0]//2, kernel_size=3, padding=1, bias=False),
            get_norm_layer(features[0]//2, use_instance_norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0]//2, out_channels, kernel_size=1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _make_encoder_block(self, in_channels, out_channels, num_blocks=2, use_simam=True):
        layers = []
        layers.append(AdvancedResidualBlock(in_channels, out_channels, 
                                          use_simam=use_simam, 
                                          use_instance_norm=self.use_instance_norm))
        for _ in range(1, num_blocks):
            layers.append(AdvancedResidualBlock(out_channels, out_channels, 
                                              use_simam=use_simam,
                                              use_instance_norm=self.use_instance_norm))
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_channels, out_channels, use_simam=True):
        return nn.Sequential(
            AdvancedResidualBlock(in_channels, out_channels, 
                                use_simam=use_simam,
                                use_instance_norm=self.use_instance_norm),
            AdvancedResidualBlock(out_channels, out_channels, 
                                use_simam=use_simam,
                                use_instance_norm=self.use_instance_norm)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # Encoder
        x1 = self.init_conv(x)
        e1 = self.encoder1(x1)
        
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with attention gates
        d4 = self.upconv4(b)
        d4 = torch.cat([self.att4(d4, e4), d4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([self.att3(d3, e3), d3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([self.att2(d2, e2), d2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([self.att1(d1, e1), d1], dim=1)
        d1 = self.decoder1(d1)
        
        # Final output
        out = self.final_conv(d1)
        
        return out


# Alias for compatibility
ResUNet = AdvancedResUNet