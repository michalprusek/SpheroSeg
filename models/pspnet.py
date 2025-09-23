import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) z původního článku PSPNet.
    """
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = in_channels // len(pool_sizes)

        self.stages = nn.ModuleList([
            self._make_stage(pool_size) for pool_size in self.pool_sizes
        ])

        # Finální konvoluce po zřetězení
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),  # Použití GroupNorm místo BatchNorm pro stabilitu
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def _make_stage(self, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, self.out_channels), self.out_channels),  # Adaptivní GroupNorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        
        # Zřetězení výstupů z pyramidového poolingu s původními rysy
        pyramid_features = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            pyramid_features.append(upsampled)
        
        output = torch.cat(pyramid_features, dim=1)
        output = self.bottleneck(output)
        return output

class PSPNet(nn.Module):
    """
    Kompletní implementace PSPNet s páteří ResNet a pomocnou větví.
    
    Během tréninku vrací (main_output, auxiliary_output).
    Během evaluace vrací pouze main_output.
    """
    def __init__(self, n_class=21, backbone='resnet101', pretrained=True, use_instance_norm=False):
        super(PSPNet, self).__init__()
        self.n_class = n_class
        self.use_instance_norm = use_instance_norm
        
        # 1. Načtení páteře (ResNet)
        if backbone == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            deep_features_size = 2048
            aux_features_size = 1024
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            deep_features_size = 2048
            aux_features_size = 1024
        else:
            raise ValueError(f"Neznámá páteř: {backbone}")

        # Extrahování vrstev ResNetu
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 2. Nahrazení kroku (stride) dilatací pro zachování rozlišení (output stride = 8)
        # To je klíčové pro segmentaci ve vysokém rozlišení
        for n, m in self.layer3.named_modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif isinstance(m, nn.Sequential) and len(m) > 0 and isinstance(m[0], nn.Conv2d) and m[0].stride == (2,2):
                # Pro downsample bloky
                 m[0].stride = (1,1)

        for n, m in self.layer4.named_modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.dilation = (4, 4)
                m.padding = (4, 4)
                m.stride = (1, 1)
            elif isinstance(m, nn.Sequential) and len(m) > 0 and isinstance(m[0], nn.Conv2d) and m[0].stride == (2,2):
                 # Pro downsample bloky
                 m[0].stride = (1,1)

        # 3. Definice modulů specifických pro PSPNet
        self.ppm = PyramidPoolingModule(deep_features_size)
        
        # Použití Instance Norm místo Batch Norm pokud je požadováno
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        
        self.cls = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, self.n_class, kernel_size=1)
        )
        
        # 4. Pomocná větev (Auxiliary Branch) pro "deep supervision"
        self.aux = nn.Sequential(
            nn.Conv2d(aux_features_size, 256, kernel_size=3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, self.n_class, kernel_size=1)
        )
        
        # 5. Inicializace vah pro nově přidané vrstvy
        self._init_custom_weights()

    def _init_custom_weights(self):
        for m in self.ppm.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.cls.modules():
             if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.aux.modules():
             if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Průchod páteří
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        
        # Hlavní větev
        x = self.ppm(x)
        main_output = self.cls(x)
        main_output = F.interpolate(main_output, size=input_size, mode='bilinear', align_corners=True)
        
        # Chování závisí na režimu (trénink vs. evaluace)
        if self.training:
            # Pomocná větev
            aux_output = self.aux(x_aux)
            aux_output = F.interpolate(aux_output, size=input_size, mode='bilinear', align_corners=True)
            return main_output, aux_output
        else:
            return main_output

# --- Ukázka použití a ověření ---
if __name__ == '__main__':
    # Počet tříd pro segmentaci (např. Pascal VOC má 21 tříd)
    NUM_CLASSES = 21

    # Vytvoření modelu s páteří ResNet-101 a předtrénovanými vahami
    model = PSPNet(n_class=NUM_CLASSES, backbone='resnet101', pretrained=True)
    
    # Výpočet celkového počtu parametrů
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architektura: PSPNet s ResNet-101")
    print(f"Celkový počet trénovatelných parametrů: {total_params / 1_000_000:.2f} M")
    print("-" * 30)

    # Vytvoření náhodného vstupního obrázku o velikosti 1024x1024
    # (batch_size=1, kanály=3, výška=1024, šířka=1024)
    dummy_input = torch.randn(1, 3, 1024, 1024)

    # 1. Test v trénovacím režimu
    model.train()
    print("Test v režimu model.train():")
    main_out, aux_out = model(dummy_input)
    print(f"  Tvar hlavního výstupu: {main_out.shape}")
    print(f"  Tvar pomocného výstupu: {aux_out.shape}")
    print("  Model vrací dva výstupy pro výpočet kombinované ztráty.")
    print("-" * 30)

    # 2. Test v evaluačním režimu
    model.eval()
    print("Test v režimu model.eval():")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Tvar výstupu: {output.shape}")
    print("  Model vrací pouze finální predikci.")
    print("-" * 30)

    # Ověření výstupních rozměrů
    assert main_out.shape == (1, NUM_CLASSES, 1024, 1024)
    assert aux_out.shape == (1, NUM_CLASSES, 1024, 1024)
    assert output.shape == (1, NUM_CLASSES, 1024, 1024)
    print("Testy úspěšně prošly!")
