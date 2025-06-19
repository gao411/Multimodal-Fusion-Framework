import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18



class AuxWeightLayer(nn.Module):
    def __init__(self, in_channels=12, reduction=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.global_pool(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w



def modify_resnet(base_model, in_channels):
    model = base_model(pretrained=False)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )


class DualBranchAttentionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, feat1, feat2):
        b, c, _, _ = feat1.shape
        f1 = self.avg_pool(feat1).view(b, -1)
        f2 = self.avg_pool(feat2).view(b, -1)
        weights = self.fc(torch.cat([f1, f2], dim=1))  # B×2
        w1 = weights[:, 0].view(b, 1, 1, 1)
        w2 = weights[:, 1].view(b, 1, 1, 1)
        fused = feat1 * w1 + feat2 * w2
        return fused


class MultiscaleFeatureModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.branch5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.branch7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.project = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        b7 = self.branch7x7(x)
        out = torch.cat([b3, b5, b7], dim=1)
        return self.project(out)



class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=6, num_layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, N, C]
        return self.encoder(x)



class AuxiliaryBranch(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.awl = AuxWeightLayer(in_channels=in_channels)
        self.backbone = modify_resnet(resnet18, in_channels=in_channels)

    def forward(self, x):
        x = self.awl(x)
        x = self.backbone(x)  # 输出 [B, 512, H/32, W/32]
        return x



class UrbanFunctionZoneModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # 主分支（遥感影像）
        self.rsi_branch = modify_resnet(resnet50, in_channels=3)  # 输出 [B, 2048, H/32, W/32]

        # 辅助分支（POI + DEM）
        self.aux_branch = AuxiliaryBranch(in_channels=12)  # 输出 [B, 512, H/32, W/32]

        # 对辅助分支升维，使其与主分支对齐
        self.aux_proj = nn.Conv2d(512, 2048, kernel_size=1)

        # 融合模块
        self.dbam = DualBranchAttentionModule(in_channels=2048)
        self.mfm = MultiscaleFeatureModule(in_channels=2048)

        # Transformer
        self.project = nn.Conv2d(2048, 768, kernel_size=1)
        self.transformer = TransformerEncoder(embed_dim=768, num_heads=6, num_layers=3)

        # 分类器
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x_rsi, x_aux):
        rsi_feat = self.rsi_branch(x_rsi)  # [B, 2048, H, W]
        aux_feat = self.aux_branch(x_aux)  # [B, 512, H, W]
        aux_feat = self.aux_proj(aux_feat)  # → [B, 2048, H, W]
        aux_feat = F.interpolate(aux_feat, size=rsi_feat.shape[2:], mode='bilinear', align_corners=False)

        fused = self.dbam(rsi_feat, aux_feat)
        fused = self.mfm(fused)  # [B, 2048, H, W]
        fused = self.project(fused)  # [B, 768, H, W]

        x = fused.flatten(2).transpose(1, 2)  # [B, HW, 768]
        x = self.transformer(x)  # [B, HW, 768]
        x = x.mean(dim=1)  # [B, 768]
        out = self.classifier(x)  # [B, num_classes]

        return out
