# src/encoder.py
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceEncoder(nn.Module):
    def __init__(self, pretrained='vggface2', embedding_size=512, device='cuda'):
        """
        Face encoder using FaceNet model.

        Args:
            pretrained (str): 'vggface2' or 'casia-webface' for pretrained weights
            embedding_size (int): Size of the embedding vector
            device (str): 'cuda' or 'cpu'
        """
        super(FaceEncoder, self).__init__()

        # Load pretrained FaceNet model
        self.facenet = InceptionResnetV1(pretrained=pretrained).to(device)
        self.facenet.eval()  # Set to evaluation mode

        # Add projection layer if needed (to adjust embedding size)
        self.projection = None
        if embedding_size != 512:  # FaceNet outputs 512-dim embeddings
            self.projection = nn.Linear(512, embedding_size)

        self.device = device

    def forward(self, x):
        """
        Extract face embeddings.

        Args:
            x (torch.Tensor): Batch of face images [B, C, H, W]

        Returns:
            torch.Tensor: Face embeddings [B, embedding_size]
        """
        with torch.no_grad():
            embeddings = self.facenet(x)

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings


def get_encoder(embedding_size=256, device='cuda'):
    """
    Returns a pre-trained face encoder.

    Args:
        embedding_size (int): Size of the embedding vector
        device (str): 'cuda' or 'cpu'
    """
    return FaceEncoder(
        pretrained='vggface2',  # VGGFace2 is more robust than CASIA-WebFace
        embedding_size=embedding_size,
        device=device
    )