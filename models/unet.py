# unet.py
# Basic UNet architecture for spheroid segmentation

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
# Double Convolution Block
# ===========================
class DoubleConv(nn.Module):
    """
    Basic UNet building block: (Conv -> Norm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, use_instance_norm=True, dropout=0.0):
        super(DoubleConv, self).__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(out_channels, use_instance_norm),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(out_channels, use_instance_norm),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# ===========================
# Basic UNet Architecture
# ===========================
class UNet(nn.Module):
    """
    Standard UNet architecture for biomedical image segmentation
    Paper: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_channels=3, out_channels=1, 
                 features=[64, 128, 256, 512, 1024],
                 use_instance_norm=True, dropout_rate=0.1,
                 use_deep_supervision=False):
        super(UNet, self).__init__()
        
        self.use_instance_norm = use_instance_norm
        self.use_deep_supervision = use_deep_supervision
        
        # Initial convolution
        self.init_conv = DoubleConv(in_channels, features[0], use_instance_norm, dropout=0)
        
        # Encoder path
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(len(features) - 1):
            self.downs.append(DoubleConv(features[i], features[i+1], use_instance_norm, 
                                        dropout=dropout_rate if i > 0 else 0))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-2], features[-1], use_instance_norm, 
                                    dropout=dropout_rate * 1.5)
        
        # Decoder path
        self.ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features) - 1):
            # Upsampling layer
            self.ups.append(nn.ConvTranspose2d(reversed_features[i], reversed_features[i+1], 
                                              kernel_size=2, stride=2))
            # Decoder block (takes concatenated features)
            self.decoder_blocks.append(DoubleConv(reversed_features[i], reversed_features[i+1], 
                                                 use_instance_norm, dropout=dropout_rate))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Deep supervision outputs (optional)
        if use_deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(features[i], out_channels, kernel_size=1) 
                for i in range(len(features) - 1)
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
    
    def forward(self, x):
        # Store original input size
        input_size = x.shape[2:]
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder path with skip connections
        skip_connections = [x]
        
        for i, (pool, down) in enumerate(zip(self.pools[:-1], self.downs[:-1])):
            x = pool(skip_connections[-1])
            x = down(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.pools[-1](skip_connections[-1])
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        deep_outputs = []
        for i, (up, decoder_block) in enumerate(zip(self.ups, self.decoder_blocks)):
            # Upsample
            x = up(x)
            
            # Get skip connection
            skip = skip_connections[i]
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([skip, x], dim=1)
            
            # Apply decoder block
            x = decoder_block(x)
            
            # Store for deep supervision if enabled
            if self.use_deep_supervision and self.training and i < len(self.deep_outputs):
                deep_out = self.deep_outputs[len(self.deep_outputs) - 1 - i](x)
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