import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import HSIAdvLoss

class FeatureDenoiser(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.gaussian = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
    def forward(self, x):
        query = self.conv1(x)
        key = self.conv2(x)
        value = self.conv3(x)
        attention = F.softmax(torch.matmul(query.flatten(2), key.flatten(2).transpose(1, 2)), dim=-1)
        denoised = torch.matmul(attention, value.flatten(2)).reshape_as(x)
        local_features = self.gaussian(x)
        return x + 0.1 * denoised + 0.1 * local_features
    
class SpectralExtractor(nn.Module):
    """Enhanced spectral feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        out = torch.cat([feat1, feat3], dim=1)
        return self.relu(self.bn(out))

class SpatialExtractor(nn.Module):
    """Enhanced spatial feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch7(x)
        ], dim=1)

class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling for robustness
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.1)  # Added dropout for regularization

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.dropout(self.fc(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.1, attn_drop=0.1):
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
        self.drop_path = nn.Dropout(0.1)  # Added drop path
        self.feature_denoiser = FeatureDenoiser(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Multi-head attention with residual
        identity = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = identity + self.drop_path(x)
        
        # MLP with residual
        identity = x
        x = self.norm2(x)
        x = identity + self.drop_path(self.mlp(x))
        
        # Reshape and denoise features
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.feature_denoiser(x)
        return x

class SpectralFormer(nn.Module):
    def __init__(self, num_spectral_bands, num_classes, patch_size=9):
        super(SpectralFormer, self).__init__()
        
        # Initial parameters
        self.num_spectral_bands = num_spectral_bands
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        # Input processing
        self.input_norm = nn.BatchNorm2d(num_spectral_bands)
        self.gaussian_noise = nn.Dropout2d(0.1)
        
        # Enhanced spectral feature extraction
        self.spectral_extract = nn.Sequential(
            SpectralExtractor(num_spectral_bands, 64),
            SpectralAttention(64),
            FeatureDenoiser(64)
        )
        
        # Enhanced spatial feature extraction
        self.spatial_extract = nn.Sequential(
            SpatialExtractor(64, 256),
            FeatureDenoiser(256)
        )
        
        # Multi-scale feature refinement
        self.refine_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=k, padding=k//2, groups=8),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                FeatureDenoiser(128)
            ) for k in [3, 5, 7]
        ])
        
        # Feature fusion with attention
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SpectralAttention(256),
            FeatureDenoiser(256)
        )
        
        # Residual projection
        self.residual_proj = nn.Sequential(
            nn.Conv2d(384, 256, 1),
            nn.BatchNorm2d(256)
        )
        
        # Enhanced transformer blocks
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(256),
            TransformerBlock(256),
            TransformerBlock(256)
        )
        
        # Context modeling with pyramid pooling
        self.pyramid_pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                nn.Conv2d(256, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                FeatureDenoiser(64)
            ) for output_size in [(1, 1), (2, 2), (3, 3), (6, 6)]
        ])
        
        # Enhanced classifier
        pyramid_dim = 256 + (64 * 4)
        self.classifier = nn.Sequential(
            nn.Conv2d(pyramid_dim, 512, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SpectralAttention(512),
            FeatureDenoiser(512),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        self.apply(self._init_weights)

    def forward(self, x, training=True):
        # Input preprocessing
        x = self.input_norm(x)
        if training:
            x = self.gaussian_noise(x)
        
        # Enhanced feature extraction
        x = self.spectral_extract(x)  # Spectral features
        spatial_feat = self.spatial_extract(x)  # Spatial features
        
        # Multi-scale refinement
        refined_features = []
        for block in self.refine_blocks:
            refined_features.append(block(spatial_feat))
        
        # Feature fusion with residual
        x = torch.cat(refined_features, dim=1)
        identity = self.residual_proj(x)
        x = self.fusion(x) + 0.1 * identity
        
        # Transformer processing
        transformer_out = self.transformer_blocks(x)
        x = x + 0.1 * transformer_out
        
        # Context modeling
        pyramid_features = [x]
        h, w = x.shape[2:]
        for pool in self.pyramid_pools:
            pyramid_feat = pool(x)
            pyramid_feat = F.interpolate(pyramid_feat, size=(h, w), 
                                      mode='bilinear', align_corners=True)
            pyramid_features.append(pyramid_feat)
        
        # Classification
        x = torch.cat(pyramid_features, dim=1)
        x = self.classifier(x)
        
        return x

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

    def training_step(self, batch, device):
        """Enhanced training step with adversarial defense."""
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        
        # Use the adversarial loss
        criterion = HSIAdvLoss(self, epsilon=0.04, alpha=0.1)
        loss, metrics = criterion(data, labels, training=True)
    
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