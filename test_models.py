#!/usr/bin/env python3
"""Test script to verify all models can be loaded and initialized"""

import sys
import torch
sys.path.insert(0, 'src/training')

# Import all models as in CNN_main_spheroid.py
from models.hrnet import HRNetV2
from models.pspnet import PSPNet
from models.unet import UNet
from models.resunet_cbam import ResUNetCBAM
from models.lightm_unet import LightMUNet
from models.resunet_ma import AdvancedResUNet as MAResUNet
from models.resunet_ma_mini import AdvancedResUNet as MAMiniResUNet
from models.resunet_lc import ResUNetSmall as LCResUNet

def test_model(model_name, model_class, **kwargs):
    """Test if a model can be initialized and forward pass works"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}...")
    print(f"{'='*60}")

    try:
        # Initialize model
        model = model_class(**kwargs)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✅ Model initialized successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
            # Handle models that return tuple (PSPNet with aux output)
            if isinstance(output, tuple):
                output = output[0]

        print(f"✅ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Test all available models"""
    print("TESTING ALL MODELS FROM /models DIRECTORY")
    print("="*60)

    results = []

    # Test each model
    models_to_test = [
        ("HRNetV2", HRNetV2, {"n_class": 1, "pretrained": False, "use_instance_norm": True}),
        ("PSPNet", PSPNet, {"n_class": 1, "backbone": "resnet101", "pretrained": False, "use_instance_norm": True}),
        ("UNet", UNet, {"in_channels": 3, "out_channels": 1, "use_instance_norm": True, "dropout_rate": 0.1}),
        ("ResUNetCBAM", ResUNetCBAM, {"in_channels": 3, "out_channels": 1, "use_instance_norm": True, "dropout_rate": 0.15}),
        ("LightMUNet", LightMUNet, {
            "in_channels": 3, "out_channels": 1, "base_channels": 32,
            "use_instance_norm": True, "encoder_layers": [1, 2, 2],
            "decoder_layers": [2, 2, 1], "bottleneck_layers": 4, "dropout_rate": 0.1
        }),
        ("MAResUNet", MAResUNet, {"in_channels": 3, "out_channels": 1, "use_instance_norm": True}),
        ("MAMiniResUNet", MAMiniResUNet, {"in_channels": 3, "out_channels": 1, "use_instance_norm": True}),
        ("LCResUNet", LCResUNet, {"in_channels": 3, "out_channels": 1, "use_instance_norm": True, "dropout_rate": 0.15}),
    ]

    for model_name, model_class, kwargs in models_to_test:
        success = test_model(model_name, model_class, **kwargs)
        results.append((model_name, success))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name:20} {status}")

    # Check if all models passed
    all_passed = all(success for _, success in results)

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL MODELS TESTED SUCCESSFULLY!")
    else:
        print("⚠️  SOME MODELS FAILED - Please check the errors above")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())