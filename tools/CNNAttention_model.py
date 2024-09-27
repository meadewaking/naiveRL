import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

class CustomCombinedExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        self.grid_extractor = Custom3DCNN(observation_space.spaces["color_grid"])
        self.semantic_extractor = nn.Sequential(
            nn.Linear(observation_space.spaces["features"].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.combined_layer = nn.Sequential(
            nn.Linear(self.grid_extractor.features_dim + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        grid_features = self.grid_extractor(observations["color_grid"])
        semantic_features = self.semantic_extractor(observations["features"])
        combined = torch.cat([grid_features, semantic_features], dim=1)
        return self.combined_layer(combined)


class CNNAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2, name="sa"):
        super(CNNAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size

        self.w_qs = nn.Conv3d(in_channels, out_channels//8, kernel_size, stride, padding)
        self.w_ks = nn.Conv3d(in_channels, out_channels//8, kernel_size, stride, padding)
        self.w_vs = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

        # Add pooling layers
        self.pool = nn.AvgPool3d(pool_size)

        # Initialize weights
        nn.init.orthogonal_(self.w_qs.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.w_ks.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.w_vs.weight, gain=np.sqrt(2))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.name = name

    def forward(self, inputs):
        Batch, _, D, H, W = inputs.shape

        # Apply convolutions and pooling
        q = self.pool(self.pool(self.w_qs(inputs)))
        k = self.pool(self.pool(self.w_ks(inputs)))
        v = self.pool(self.pool(self.w_vs(inputs)))

        # Reshape for attention computation
        _, _, D_p, H_p, W_p = q.shape
        output_size = D_p * H_p * W_p

        q = q.view(Batch, -1, output_size).permute(0, 2, 1)  # N, (D_p*H_p*W_p), C
        k = k.view(Batch, -1, output_size)  # N, C, (D_p*H_p*W_p)
        v = v.view(Batch, -1, output_size)  # N, C, (D_p*H_p*W_p)

        attn = torch.bmm(q, k)  # N, (D_p*H_p*W_p), (D_p*H_p*W_p)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(Batch, self.out_channels, D_p, H_p, W_p)

        # Upsample to original size
        out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=False)

        out = self.gamma * out + self.w_vs(inputs)
        return out

class Custom3DCNN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Custom3DCNN, self).__init__()

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            CNNAttention(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            CNNAttention(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            CNNAttention(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2),  # Global average pooling
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
