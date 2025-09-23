# resunet_small.py
"""
ResUNet Small - Optimized architecture with ~60M parameters for 1024x1024 resolution
Enhanced with modern regularization techniques for better generalization
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F


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
# Squeeze-and-Excitation Block with Regularization
# ===========================
class SEBlock(nn.Module):
    """Enhanced SE Block with dropout for regularization"""
    
    def __init__(self, in_channels, reduction=16, dropout=0.1):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(max(in_channels // reduction, 8), in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ===========================
# Spatial Attention Module
# ===========================
class SpatialAttention(nn.Module):
    """Spatial attention for better feature focus"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


# ===========================
# Enhanced Attention Gate
# ===========================
class EnhancedAttentionGate(nn.Module):
    """Enhanced attention gate with spatial attention"""
    
    def __init__(self, F_g, F_l, F_int, use_instance_norm=True):
        super(EnhancedAttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            get_norm_layer(F_int, use_instance_norm)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            get_norm_layer(F_int, use_instance_norm)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            get_norm_layer(1, use_instance_norm),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.spatial_attention = SpatialAttention()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention and spatial refinement
        out = x * psi
        out = self.spatial_attention(out)
        return out


# ===========================
# Enhanced Residual Block
# ===========================
class EnhancedResidualBlock(nn.Module):
    """Enhanced residual block with multiple regularization techniques"""
    
    def __init__(self, in_channels, out_channels, dilation=1, reduction=16, 
                 dropout=0.15, use_instance_norm=True, use_spatial_dropout=True):
        super(EnhancedResidualBlock, self).__init__()
        
        # First convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.norm1 = get_norm_layer(out_channels, use_instance_norm)
        self.relu = nn.ReLU(inplace=True)
        
        # Regularization after first conv
        if use_spatial_dropout:
            self.dropout1 = nn.Dropout2d(p=dropout * 0.5)
        else:
            self.dropout1 = nn.Dropout(p=dropout * 0.5)
        
        # Second convolution path
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, bias=False)
        self.norm2 = get_norm_layer(out_channels, use_instance_norm)
        
        # Regularization after second conv
        if use_spatial_dropout:
            self.dropout2 = nn.Dropout2d(p=dropout)
        else:
            self.dropout2 = nn.Dropout(p=dropout)

        # Residual connection adjustment
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                       padding=0, bias=False) if in_channels != out_channels else None
        
        # Channel and spatial attention
        self.se = SEBlock(out_channels, reduction, dropout=dropout * 0.5)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.adjust_channels:
            residual = self.adjust_channels(x)

        # Forward path with regularization
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        # Add residual connection
        out += residual
        out = self.relu(out)
        
        # Apply attention mechanisms
        out = self.se(out)
        out = self.spatial_attention(out)
        
        return out


# ===========================
# ResUNet Small Architecture
# ===========================
class ResUNetSmall(nn.Module):
    """
    ResUNet Small with ~60M parameters optimized for 1024x1024 resolution
    Enhanced with modern regularization and attention mechanisms
    """
    
    def __init__(self, in_channels=3, out_channels=1, 
                 features=[48, 96, 192, 384, 512],  # Optimized for ~60M params
                 use_instance_norm=True, dropout_rate=0.15, 
                 use_deep_supervision=False, use_spatial_dropout=True):
        super(ResUNetSmall, self).__init__()
        
        self.use_instance_norm = use_instance_norm
        self.use_deep_supervision = use_deep_supervision
        self.dropout_rate = dropout_rate
        
        # Initial convolution with regularization
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=1, padding=3, bias=False),
            get_norm_layer(features[0], use_instance_norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5) if use_spatial_dropout else nn.Dropout(p=dropout_rate * 0.5)
        )
        
        # Encoder path
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_features = features[0]
        for feature in features:
            self.downs.append(EnhancedResidualBlock(
                in_features, feature, dropout=dropout_rate,
                use_instance_norm=use_instance_norm,
                use_spatial_dropout=use_spatial_dropout
            ))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_features = feature
        
        # Enhanced bottleneck with multiple blocks
        self.bottleneck = nn.Sequential(
            EnhancedResidualBlock(features[-1], features[-1] * 2, dropout=dropout_rate * 1.2,
                                use_instance_norm=use_instance_norm, use_spatial_dropout=use_spatial_dropout),
            EnhancedResidualBlock(features[-1] * 2, features[-1] * 2, dropout=dropout_rate * 1.2,
                                use_instance_norm=use_instance_norm, use_spatial_dropout=use_spatial_dropout)
        )
        
        # Decoder path
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        reversed_features = list(reversed(features))
        for i, feature in enumerate(reversed_features):
            # Upsampling layer
            if i == 0:
                # From bottleneck
                self.ups.append(nn.ConvTranspose2d(features[-1] * 2, feature, kernel_size=2, stride=2))
            else:
                # From previous decoder layer
                prev_feature = reversed_features[i-1]
                self.ups.append(nn.ConvTranspose2d(prev_feature, feature, kernel_size=2, stride=2))
            
            # Decoder block
            self.ups.append(EnhancedResidualBlock(
                feature * 2, feature, dropout=dropout_rate,
                use_instance_norm=use_instance_norm,
                use_spatial_dropout=use_spatial_dropout
            ))
            
            # Attention gate
            self.attentions.append(EnhancedAttentionGate(
                F_g=feature, F_l=feature, F_int=max(feature // 2, 8),
                use_instance_norm=use_instance_norm
            ))
        
        # Final output layers with regularization
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1, bias=False),
            get_norm_layer(features[0] // 2, use_instance_norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5) if use_spatial_dropout else nn.Dropout(p=dropout_rate * 0.5),
            nn.Conv2d(features[0] // 2, out_channels, kernel_size=1)
        )
        
        # Deep supervision outputs (optional)
        if use_deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(feature, out_channels, kernel_size=1) for feature in reversed(features[:-1])
            ])
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization"""
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store original input size
        input_size = x.shape[2:]
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder path
        skip_connections = []
        for i, (down, pool) in enumerate(zip(self.downs, self.pools)):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        deep_outputs = []
        for idx in range(0, len(self.ups), 2):
            # Upsampling
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            # Apply attention gate
            attention = self.attentions[idx // 2](g=x, x=skip_connection)
            concat_skip = torch.cat((attention, x), dim=1)
            
            # Decoder block
            x = self.ups[idx + 1](concat_skip)
            
            # Store for deep supervision if enabled
            if self.use_deep_supervision and idx // 2 < len(self.deep_outputs):
                deep_out = self.deep_outputs[idx // 2](x)
                deep_out = F.interpolate(deep_out, size=input_size, mode='bilinear', align_corners=False)
                deep_outputs.append(deep_out)
        
        # Final output
        x = self.final_conv(x)
        
        # Ensure output matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        if self.use_deep_supervision and self.training:
            return x, deep_outputs
        return x


# Alias for compatibility
ResUNet = ResUNetSmall
