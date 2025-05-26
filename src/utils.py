# src/utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils


def tensor_to_image(tensor):
    """Convert a tensor to numpy image."""
    # Convert to CPU, detach from graph, convert to numpy
    img = tensor.cpu().detach().numpy()

    # Convert from [C,H,W] to [H,W,C] and from [-1,1] to [0,255]
    img = np.transpose(img, (1, 2, 0))
    img = ((img + 1) * 127.5).astype(np.uint8)

    return img


def show_tensor_images(tensor_images, num_images=16, figsize=(15, 15)):
    """Display a grid of images from tensor."""
    # Create grid
    grid = vutils.make_grid(
        tensor_images[:num_images],
        nrow=int(np.sqrt(num_images)),
        normalize=True,
        value_range=(-1, 1)
    )

    # Convert to numpy and display
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(grid.cpu().detach().numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def load_image(image_path, img_size=128):
    """Load an image and convert to tensor."""
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def interpolate_embeddings(embedding1, embedding2, steps=10):
    """Linearly interpolate between two embeddings."""
    # Generate interpolation factors
    alphas = np.linspace(0, 1, steps)

    # Generate interpolated embeddings
    embeddings = []
    for alpha in alphas:
        embedding = (1 - alpha) * embedding1 + alpha * embedding2
        embeddings.append(embedding)

    # Stack into batch
    embeddings = torch.stack(embeddings)

    return embeddings