import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # Library for state-of-the-art models (like Xception)

# --- CONFIGURATION ---
NUM_CLASSES = 1  # Binary Classification: Real vs Fake
DROPOUT_RATE = 0.5

class SpatialStream(nn.Module):
    """
    Stream 1: Spatial Domain (RGB)
    Uses XceptionNet (pre-trained on ImageNet) to extract visual features.
    """
    def __init__(self):
        super(SpatialStream, self).__init__()
        # Load Xception
        # We remove the final classification layer ('classifier') to get raw features
        self.backbone = timm.create_model('xception', pretrained=True, num_classes=0)
        
        # Xception outputs a 2048-dim vector
        self.feature_dim = 2048

    def forward(self, x):
        # x shape: [batch, 3, 299, 299]
        features = self.backbone(x)
        return features # Shape: [batch, 2048]

class FrequencyStream(nn.Module):
    """
    Stream 2: Frequency Domain (DCT)
    Uses a lightweight ResNet18 to analyze the DCT heatmaps.
    We use a smaller model here because DCT maps have simpler patterns than faces.
    """
    def __init__(self):
        super(FrequencyStream, self).__init__()
        # Load ResNet18
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0)
        
        # ResNet18 outputs a 512-dim vector
        self.feature_dim = 512

    def forward(self, x):
        # x shape: [batch, 3, 299, 299] (DCT Map)
        features = self.backbone(x)
        return features # Shape: [batch, 512]

class AttentionFusion(nn.Module):
    """
    Channel-Wise Gated Fusion Mechanism.
    Instead of 2 coarse scalar weights, this learns a 512-dim gate vector
    via Sigmoid, allowing fine-grained per-dimension control over which
    features come from the spatial vs frequency stream.
    """
    def __init__(self, dim_spatial, dim_freq, fusion_dim=512):
        super(AttentionFusion, self).__init__()
        
        # Project both streams to the same dimension so we can merge them
        self.spatial_proj = nn.Linear(dim_spatial, fusion_dim)
        self.freq_proj = nn.Linear(dim_freq, fusion_dim)
        
        # Channel-wise gating network
        # Input: concatenated projected features [batch, fusion_dim * 2]
        # Output: gate vector [batch, fusion_dim] with values in (0, 1)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()  # Per-dimension gate values between 0 and 1
        )

    def forward(self, spatial_feat, freq_feat):
        # 1. Project to same dimension
        s_emb = F.relu(self.spatial_proj(spatial_feat))  # [batch, 512]
        f_emb = F.relu(self.freq_proj(freq_feat))        # [batch, 512]
        
        # 2. Concatenate
        combined = torch.cat([s_emb, f_emb], dim=1)      # [batch, 1024]
        
        # 3. Compute channel-wise gate
        g = self.gate(combined)                           # [batch, 512]
        
        # 4. Gated fusion: g * spatial + (1-g) * frequency
        fused_embedding = g * s_emb + (1 - g) * f_emb    # [batch, 512]
        
        return fused_embedding, g

class DeepfakeDetector(nn.Module):
    """
    The Full Dual-Stream Architecture
    """
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        
        self.spatial_stream = SpatialStream()
        self.freq_stream = FrequencyStream()
        
        self.fusion = AttentionFusion(
            dim_spatial=self.spatial_stream.feature_dim,
            dim_freq=self.freq_stream.feature_dim
        )
        
        # Final Classifier (2-layer head for more capacity)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, NUM_CLASSES)  # Output 1 logit (for BCEWithLogitsLoss)
        )

    def forward(self, rgb_img, dct_img):
        # 1. Extract Features
        s_feat = self.spatial_stream(rgb_img)
        f_feat = self.freq_stream(dct_img)
        
        # 2. Fuse Features with Attention
        fused_feat, weights = self.fusion(s_feat, f_feat)
        
        # 3. Classify
        logits = self.classifier(fused_feat)
        
        return logits, weights

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    # Create the model
    model = DeepfakeDetector()
    print("Model initialized successfully!")
    
    # Create dummy input (Batch size 2)
    dummy_rgb = torch.randn(2, 3, 299, 299)
    dummy_dct = torch.randn(2, 3, 299, 299)
    
    # Forward pass
    logits, weights = model(dummy_rgb, dummy_dct)
    
    print(f"\nLogits shape: {logits.shape} (Should be [2, 1])")
    print(f"Attention Weights: \n{weights}")
