import torch
import torch.nn as nn

class AECNNEnsemble(nn.Module):
    def __init__(self, input_dim=80, latent_dim=16):
        super(AECNNEnsemble, self).__init__()
        
        # 1. Autoencoder for Feature Compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim), 
            nn.ReLU()
        )
        
        # 2. CNN Heads (Ensemble)
        # Reshaping 16 features into a 4x4 spatial matrix
        self.cnn_heads = nn.ModuleList([
            self._build_cnn_head() for _ in range(3) # 3-model ensemble
        ])

    def _build_cnn_head(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 5) # 5 Classes: Normal, DoS, Probe, R2L, U2R
        )

    def forward(self, x):
        # Stage 1: Compression
        latent = self.encoder(x)
        
        # Stage 2: Spatial Reshaping (Batch, Channel, Height, Width)
        spatial_input = latent.view(-1, 1, 4, 4)
        
        # Stage 3: Ensemble Voting (Averaging Softmax outputs)
        outputs = [torch.softmax(head(spatial_input), dim=1) for head in self.cnn_heads]
        ensemble_avg = torch.mean(torch.stack(outputs), dim=0)
        
        return ensemble_avg