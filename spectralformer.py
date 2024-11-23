import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Reshape and permute for attention
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        
        # Apply Layer Norm and Attention
        identity = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + identity
        
        # MLP
        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + identity
        
        # Reshape back
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

class SpectralFormer(nn.Module):
    def __init__(self, num_spectral_bands, num_classes, patch_size=9):
        super(SpectralFormer, self).__init__()
        
        # Parameters
        self.num_spectral_bands = num_spectral_bands
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        # Initial spectral embedding
        self.spectral_embed = nn.Sequential(
            nn.Conv2d(num_spectral_bands, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SpectralAttention(64)
        )
        
        # Multi-scale spatial feature extraction
        self.spatial_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, 1),  # 384 = 128 * 3 (from three spatial blocks)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Transformer blocks for joint spectral-spatial learning
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(256),
            TransformerBlock(256),
            TransformerBlock(256)
        )
        
        # Pyramid pooling for multi-scale context
        self.pyramid_pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                nn.Conv2d(256, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ) for output_size in [(1, 1), (2, 2), (3, 3), (6, 6)]
        ])
        
        # Final classification head
        pyramid_dim = 256 + (64 * 4)  # Original features + pyramid features
        self.classifier = nn.Sequential(
            nn.Conv2d(pyramid_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial spectral embedding
        x = self.spectral_embed(x)
        
        # Multi-scale spatial feature extraction
        spatial_features = []
        for block in self.spatial_blocks:
            spatial_features.append(block(x))
        
        # Feature fusion
        x = torch.cat(spatial_features, dim=1)
        x = self.fusion(x)
        
        # Apply transformer blocks
        x = self.transformer_blocks(x)
        
        # Pyramid pooling
        pyramid_features = [x]
        h, w = x.shape[2:]
        for pool in self.pyramid_pools:
            pyramid_feat = pool(x)
            pyramid_feat = F.interpolate(pyramid_feat, size=(h, w), 
                                      mode='bilinear', align_corners=True)
            pyramid_features.append(pyramid_feat)
        
        # Concatenate all features
        x = torch.cat(pyramid_features, dim=1)
        
        # Final classification
        x = self.classifier(x)
        
        return x

    def training_step(self, batch, device):
        """Single training step."""
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        outputs = self(data)
        
        # Calculate loss (only for non-zero labels)
        mask = labels != 0
        if mask.sum() > 0:
            loss = F.cross_entropy(outputs[mask], labels[mask], label_smoothing=0.1)
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def validation_step(self, batch, device):
        """Single validation step."""
        self.eval()
        with torch.no_grad():
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = self(data)
            
            # Calculate metrics (only for non-zero labels)
            mask = labels != 0
            if mask.sum() > 0:
                loss = F.cross_entropy(outputs[mask], labels[mask])
                predictions = outputs.argmax(dim=1)
                correct = (predictions[mask] == labels[mask]).float().mean()
            else:
                loss = torch.tensor(0.0, device=device)
                correct = torch.tensor(0.0, device=device)
        
        self.train()
        return loss.item(), correct.item()