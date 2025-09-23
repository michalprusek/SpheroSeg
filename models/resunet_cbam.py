# resunet_cbam.py
# ResUNet with CBAM (Convolutional Block Attention Module)

import torch
import torch.nn as nn
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
# Channel Attention Module
# ===========================
class ChannelAttention(nn.Module):
    """
    Channel Attention Module of CBAM
    Uses both average and max pooling for better feature representation
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 8), in_channels, bias=False)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.shared_mlp(avg_out).view(b, c, 1, 1)
        
        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.shared_mlp(max_out).view(b, c, 1, 1)
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return x * self.sigmoid(out)


# ===========================
# Spatial Attention Module
# ===========================
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module of CBAM
    Focuses on 'where' is an informative part
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)
        
        return x * self.sigmoid(attention)


# ===========================
# CBAM Block
# ===========================
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Sequential application of channel and spatial attention
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = self.channel_attention(x)
        # Apply spatial attention
        x = self.spatial_attention(x)
        return x


# ===========================
# Residual Block with CBAM
# ===========================
class ResidualBlockCBAM(nn.Module):
    """
    Residual block enhanced with CBAM attention
    """
    def __init__(self, in_channels, out_channels, dilation=1, reduction=16,
                 dropout=0.15, use_instance_norm=True):
        super(ResidualBlockCBAM, self).__init__()
        
        # First convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.norm1 = get_norm_layer(out_channels, use_instance_norm)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout * 0.5)
        
        # Second convolution path
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.norm2 = get_norm_layer(out_channels, use_instance_norm)
        self.dropout2 = nn.Dropout2d(p=dropout)
        
        # Residual connection adjustment
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_layer(out_channels, use_instance_norm)
            )
        
        # CBAM attention
        self.cbam = CBAM(out_channels, reduction)
    
    def forward(self, x):
        # Store residual
        residual = x
        if self.adjust_channels:
            residual = self.adjust_channels(x)
        
        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        
        # Add residual
        out += residual
        out = self.relu(out)
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        return out


# ===========================
# Attention Gate for Skip Connections
# ===========================
class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections
    """
    def __init__(self, F_g, F_l, F_int, use_instance_norm=True):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            get_norm_layer(F_int, use_instance_norm)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            get_norm_layer(F_int, use_instance_norm)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            get_norm_layer(1, use_instance_norm),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ===========================
# ResUNet with CBAM
# ===========================
class ResUNetCBAM(nn.Module):
    """
    ResUNet enhanced with CBAM (Convolutional Block Attention Module)
    Combines residual learning with channel and spatial attention
    """
    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512],
                 use_instance_norm=True, dropout_rate=0.15,
                 use_deep_supervision=False):
        super(ResUNetCBAM, self).__init__()
        
        self.use_instance_norm = use_instance_norm
        self.use_deep_supervision = use_deep_supervision
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1, bias=False),
            get_norm_layer(features[0], use_instance_norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5)
        )
        
        # Encoder path with CBAM
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        in_features = features[0]
        for feature in features:
            self.downs.append(
                ResidualBlockCBAM(in_features, feature, dropout=dropout_rate,
                                use_instance_norm=use_instance_norm)
            )
            in_features = feature
        
        # Bottleneck with CBAM (1024 channels)
        self.bottleneck = nn.Sequential(
            ResidualBlockCBAM(features[-1], features[-1] * 2, dropout=dropout_rate * 1.2,
                            use_instance_norm=use_instance_norm),
            ResidualBlockCBAM(features[-1] * 2, features[-1] * 2, dropout=dropout_rate * 1.2,
                            use_instance_norm=use_instance_norm)
        )
        
        # Decoder path with attention gates
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        reversed_features = list(reversed(features))
        for i, feature in enumerate(reversed_features):
            # Upsampling layer
            if i == 0:
                # From bottleneck (1024 channels)
                self.ups.append(
                    nn.ConvTranspose2d(features[-1] * 2, feature, kernel_size=2, stride=2)
                )
            else:
                # From previous decoder layer
                prev_feature = reversed_features[i-1]
                self.ups.append(
                    nn.ConvTranspose2d(prev_feature, feature, kernel_size=2, stride=2)
                )
            
            # Decoder block with CBAM (takes concatenated features)
            self.ups.append(
                ResidualBlockCBAM(feature * 2, feature, dropout=dropout_rate,
                                use_instance_norm=use_instance_norm)
            )
            
            # Attention gate for skip connection
            self.attentions.append(
                AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2,
                            use_instance_norm=use_instance_norm)
            )
        
        # Final output layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1, bias=False),
            get_norm_layer(features[0] // 2, use_instance_norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5),
            nn.Conv2d(features[0] // 2, out_channels, kernel_size=1)
        )
        
        # Deep supervision outputs (optional)
        if use_deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(feature, out_channels, kernel_size=1)
                for feature in reversed(features[:-1])
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
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder path with attention gates
        deep_outputs = []
        for idx in range(0, len(self.ups), 2):
            # Upsample
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            # Apply attention gate to skip connection
            attention = self.attentions[idx // 2](g=x, x=skip_connection)
            
            # Concatenate with attended skip connection
            concat_skip = torch.cat((attention, x), dim=1)
            
            # Apply decoder block with CBAM
            x = self.ups[idx + 1](concat_skip)
            
            # Store for deep supervision if enabled
            if self.use_deep_supervision and self.training and idx // 2 < len(self.deep_outputs):
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
ResUNet = ResUNetCBAM