import torch
import torch.nn as nn
import torchvision.models as tv_models


class CNNEncoder(nn.Module):
    """
    2-D CNN frame encoder (ResNet-18 / ResNet-50) for CSLR.

    Interface:
        input : (B, T, C, H, W)
        output: (B, T, out_features)  — LayerNorm applied

    Fix log:
        - Added output LayerNorm for stable downstream BiLSTM/Transformer.
        - Removed VGG (not needed for ablation, adds complexity).
        - ResNet50 uses V2 weights (better than V1).
    """

    SUPPORTED = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

    def __init__(
        self,
        backbone:     str   = "resnet18",
        pretrained:   bool  = True,
        out_features: int   = 512,
        freeze_bn:    bool  = False,
    ):
        super().__init__()
        assert backbone in self.SUPPORTED, \
            f"backbone must be one of {list(self.SUPPORTED)}, got '{backbone}'"

        self.backbone_name = backbone
        self.raw_dim       = self.SUPPORTED[backbone]

        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base    = tv_models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base    = tv_models.resnet34(weights=weights)
        else:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base    = tv_models.resnet50(weights=weights)

        # Remove classification head; keep avgpool → (B, raw_dim, 1, 1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        # Optional projection
        if out_features != self.raw_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.raw_dim, out_features),
                nn.GELU(),
                nn.Dropout(p=0.1),
            )
        else:
            self.proj = nn.Identity()

        # Output norm — stabilises BiLSTM / TransformerCTC input scale
        self.out_norm     = nn.LayerNorm(out_features)
        self.out_features = out_features

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def _extract_single(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        return feat.flatten(1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) → (B, T, out_features)"""
        B, T, C, H, W = frames.shape
        flat     = frames.view(B * T, C, H, W)
        features = self._extract_single(flat)             # (B*T, raw_dim)
        features = self.proj(features)                    # (B*T, out_features)
        features = features.view(B, T, self.out_features)
        return self.out_norm(features)
