# src/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        # Initialize to identity transform
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,
                 upsample=False, style_dim=None):
        super().__init__()

        self.upsample = upsample
        self.use_adain = style_dim is not None

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)

        if self.use_adain:
            self.adain = AdaptiveInstanceNorm(out_channel, style_dim)
        else:
            self.norm = nn.InstanceNorm2d(out_channel)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, input, style=None):
        if self.upsample:
            input = F.interpolate(input, scale_factor=2, mode='nearest')

        out = self.conv(input)

        if self.use_adain:
            out = self.adain(out, style)
        else:
            out = self.norm(out)

        out = self.activation(out)

        return out


class FaceGenerator(nn.Module):
    def __init__(self, embedding_size=256, initial_size=4, channels=512):
        """
        Generator model that creates face images from embeddings.

        Args:
            embedding_size (int): Size of the input embedding vector
            initial_size (int): Size of the initial feature map
            channels (int): Number of channels in the initial feature map
        """
        super(FaceGenerator, self).__init__()

        # Initial constant learned parameter
        self.initial_constant = nn.Parameter(
            torch.randn(1, channels, initial_size, initial_size)
        )

        # Mapping network from embedding to style
        self.mapping = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(0.2),
        )

        # Generator blocks
        self.blocks = nn.ModuleList([
            # 4x4 -> 8x8
            ConvBlock(channels, channels, upsample=True, style_dim=embedding_size),
            # 8x8 -> 16x16
            ConvBlock(channels, channels // 2, upsample=True, style_dim=embedding_size),
            # 16x16 -> 32x32
            ConvBlock(channels // 2, channels // 4, upsample=True, style_dim=embedding_size),
            # 32x32 -> 64x64
            ConvBlock(channels // 4, channels // 8, upsample=True, style_dim=embedding_size),
            # 64x64 -> 128x128
            ConvBlock(channels // 8, channels // 16, upsample=True, style_dim=embedding_size),
        ])

        # Final layer to RGB
        self.to_rgb = nn.Conv2d(channels // 16, 3, 1)

    def forward(self, embedding):
        batch_size = embedding.shape[0]

        # Map embedding to style
        style = self.mapping(embedding)

        # Replicate initial constant for each sample in batch
        x = self.initial_constant.repeat(batch_size, 1, 1, 1)

        # Apply generator blocks
        for block in self.blocks:
            x = block(x, style)

        # Convert to RGB and normalize to [-1, 1]
        x = self.to_rgb(x)
        x = torch.tanh(x)

        return x


def get_generator(embedding_size=256, device='cuda'):
    """
    Returns a face generator model.

    Args:
        embedding_size (int): Size of the embedding vector
        device (str): 'cuda' or 'cpu'
    """
    model = FaceGenerator(embedding_size=embedding_size).to(device)
    return model

