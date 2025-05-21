# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import wandb
import lpips

from src.dataset import FaceDataset, get_transforms
from src.encoder import get_encoder
from src.generator import get_generator


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seed for reproducibility
        torch.manual_seed(config.seed)

        # Initialize directories
        self.checkpoint_dir = os.path.join(config.output_dir, 'checkpoints')
        self.sample_dir = os.path.join(config.output_dir, 'samples')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        # Initialize dataloaders
        self._init_dataloaders()

        # Initialize models, losses, and optimizers
        self._init_models()
        self._init_losses()
        self._init_optimizers()

        # Initialize logging
        if config.use_wandb:
            wandb.init(project=config.project_name, name=config.run_name)
            wandb.config.update(config)

    def _init_dataloaders(self):
        # Get transformations
        train_transform = get_transforms(
            img_size=self.config.img_size,
            is_training=True
        )
        val_transform = get_transforms(
            img_size=self.config.img_size,
            is_training=False
        )

        # Create datasets
        train_dataset = FaceDataset(
            image_dir=self.config.train_dir,
            transform=train_transform
        )

        val_dataset = FaceDataset(
            image_dir=self.config.val_dir,
            transform=val_transform
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def _init_models(self):
        # Initialize encoder
        self.encoder = get_encoder(
            embedding_size=self.config.embedding_size,
            device=self.device
        )

        # Initialize generator
        self.generator = get_generator(
            embedding_size=self.config.embedding_size,
            device=self.device
        )

    def _init_losses(self):
        # L1 loss for pixel-wise reconstruction
        self.l1_loss = nn.L1Loss()

        # LPIPS for perceptual loss
        self.perceptual_loss = lpips.LPIPS(net='alex').to(self.device)

    def _init_optimizers(self):
        # Optimizer for generator
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999)
        )

        # Learning rate schedulers
        self.g_scheduler = optim.lr_scheduler.StepLR(
            self.g_optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma
        )

    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            # Training phase
            self._train_epoch(epoch)

            # Validation phase
            val_metrics = self._validate_epoch(epoch)

            # Learning rate step
            self.g_scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self._save_checkpoint(epoch)

            # Log validation metrics
            if self.config.use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} completed")

    def _train_epoch(self, epoch):
        self.generator.train()

        total_g_loss = 0
        total_l1_loss = 0
        total_perceptual_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training")

        for i, images in enumerate(pbar):
            # Move images to device
            images = images.to(self.device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = self.encoder(images)

            # Generate images from embeddings
            generated_images = self.generator(embeddings)

            # Compute losses
            l1 = self.l1_loss(generated_images, images)
            perceptual = self.perceptual_loss(generated_images, images).mean()

            # Total generator loss
            g_loss = l1 + self.config.perceptual_weight * perceptual

            # Update generator
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Update metrics
            total_g_loss += g_loss.item()
            total_l1_loss += l1.item()
            total_perceptual_loss += perceptual.item()

            # Update progress bar
            pbar.set_postfix({
                'g_loss': g_loss.item(),
                'l1': l1.item(),
                'perceptual': perceptual.item()
            })

            # Log training metrics
            if self.config.use_wandb and i % self.config.log_freq == 0:
                wandb.log({
                    'train/g_loss': g_loss.item(),
                    'train/l1_loss': l1.item(),
                    'train/perceptual_loss': perceptual.item(),
                    'train/step': epoch * len(self.train_loader) + i
                })

            # Save samples
            if i % self.config.sample_freq == 0:
                self._save_samples(epoch, i, images, generated_images)

        # Compute average losses for the epoch
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_l1_loss = total_l1_loss / len(self.train_loader)
        avg_perceptual_loss = total_perceptual_loss / len(self.train_loader)

        # Log epoch metrics
        if self.config.use_wandb:
            wandb.log({
                'train/epoch': epoch,
                'train/avg_g_loss': avg_g_loss,
                'train/avg_l1_loss': avg_l1_loss,
                'train/avg_perceptual_loss': avg_perceptual_loss
            })

    def _validate_epoch(self, epoch):
        self.generator.eval()

        total_g_loss = 0
        total_l1_loss = 0
        total_perceptual_loss = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} Validation")

        with torch.no_grad():
            for i, images in enumerate(pbar):
                # Move images to device
                images = images.to(self.device)

                # Extract embeddings
                embeddings = self.encoder(images)

                # Generate images from embeddings
                generated_images = self.generator(embeddings)

                # Compute losses
                l1 = self.l1_loss(generated_images, images)
                perceptual = self.perceptual_loss(generated_images, images).mean()

                # Total generator loss
                g_loss = l1 + self.config.perceptual_weight * perceptual

                # Update metrics
                total_g_loss += g_loss.item()
                total_l1_loss += l1.item()
                total_perceptual_loss += perceptual.item()

                # Update progress bar
                pbar.set_postfix({
                    'g_loss': g_loss.item(),
                    'l1': l1.item(),
                    'perceptual': perceptual.item()
                })

            # Save validation samples
            if len(self.val_loader) > 0:
                self._save_samples(epoch, 0, images, generated_images, is_validation=True)

        # Compute average losses for the epoch
        avg_g_loss = total_g_loss / len(self.val_loader)
        avg_l1_loss = total_l1_loss / len(self.val_loader)
        avg_perceptual_loss = total_perceptual_loss / len(self.val_loader)

        # Return metrics
        return {
            'avg_g_loss': avg_g_loss,
            'avg_l1_loss': avg_l1_loss,
            'avg_perceptual_loss': avg_perceptual_loss
        }

    def _save_samples(self, epoch, batch_idx, real_images, fake_images, is_validation=False):
        """Save image samples during training."""
        # Create a grid of images
        sample_size = min(8, real_images.size(0))
        real_grid = vutils.make_grid(
            real_images[:sample_size], nrow=4, normalize=True, range=(-1, 1)
        )
        fake_grid = vutils.make_grid(
            fake_images[:sample_size], nrow=4, normalize=True, range=(-1, 1)
        )

        # Combine real and fake grids
        grid = torch.cat((real_grid, fake_grid), dim=1)

        # Save to file
        prefix = 'val' if is_validation else 'train'
        filename = f"{prefix}_samples_epoch_{epoch + 1}_batch_{batch_idx}.png"
        vutils.save_image(
            grid,
            os.path.join(self.sample_dir, filename),
            normalize=False
        )

        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                f"{prefix}/samples": wandb.Image(grid)
            })

    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'config': self.config
        }

        filename = f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(
            checkpoint,
            os.path.join(self.checkpoint_dir, filename)
        )

        print(f"Checkpoint saved at {filename}")


# Configuration class
class Config:
    def __init__(self):
        # Data paths
        self.train_dir = "data/train"
        self.val_dir = "data/val"
        self.test_dir = "data/test"
        self.output_dir = "output"

        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.lr = 2e-4
        self.beta1 = 0.5
        self.lr_step_size = 20
        self.lr_gamma = 0.5
        self.num_workers = 4
        self.seed = 42

        # Model parameters
        self.embedding_size = 256
        self.img_size = 128

        # Loss weights
        self.perceptual_weight = 0.1

        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Logging and checkpoints
        self.use_wandb = True
        self.project_name = "face-generation"
        self.run_name = f"face-gen-{self.embedding_size}-{self.img_size}"
        self.log_freq = 10
        self.sample_freq = 100
        self.save_freq = 5


# Main function to start training
def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

