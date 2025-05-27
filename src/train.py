# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

from src.dataset import FaceDataset, get_transforms
from src.encoder import get_encoder
from src.generator import get_generator
from src.wandb_utils import WandbManager


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

        # Initialize W&B manager (handles all W&B operations)
        self.wandb_manager = WandbManager(config)

        # Initialize dataloaders
        self._init_dataloaders()

        # Initialize models, losses, and optimizers
        self._init_models()
        self._init_losses()
        self._init_optimizers()

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

        print(f"Training dataset: {len(train_dataset)} images")
        print(f"Validation dataset: {len(val_dataset)} images")

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
            train_metrics = self._train_epoch(epoch)

            # Validation phase
            val_metrics = self._validate_epoch(epoch)

            # Learning rate step
            current_lr = self.g_scheduler.get_last_lr()[0]
            self.g_scheduler.step()

            # Log epoch metrics to W&B with proper step and cleaner names
            epoch_step = epoch + 1

            # Log training epoch averages
            train_epoch_metrics = {
                'loss_total': train_metrics['avg_g_loss'],
                'loss_l1': train_metrics['avg_l1_loss'],
                'loss_perceptual': train_metrics['avg_perceptual_loss'],
                'learning_rate': current_lr,
                'epoch': epoch_step
            }
            self.wandb_manager.log_metrics(train_epoch_metrics, step=epoch_step, prefix="train")

            # Log validation epoch averages
            val_epoch_metrics = {
                'loss_total': val_metrics['avg_g_loss'],
                'loss_l1': val_metrics['avg_l1_loss'],
                'loss_perceptual': val_metrics['avg_perceptual_loss'],
                'epoch': epoch_step
            }
            self.wandb_manager.log_metrics(val_epoch_metrics, step=epoch_step, prefix="val")

            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self._save_checkpoint(epoch)

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} completed")
            print(
                f"  Train Loss: {train_metrics['avg_g_loss']:.4f} (L1: {train_metrics['avg_l1_loss']:.4f}, Perceptual: {train_metrics['avg_perceptual_loss']:.4f})")
            print(
                f"  Val Loss: {val_metrics['avg_g_loss']:.4f} (L1: {val_metrics['avg_l1_loss']:.4f}, Perceptual: {val_metrics['avg_perceptual_loss']:.4f})")
            print(f"  Learning Rate: {current_lr:.6f}")

        # Cleanup W&B
        self.wandb_manager.cleanup()

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
                'g_loss': f"{g_loss.item():.4f}",
                'l1': f"{l1.item():.4f}",
                'perceptual': f"{perceptual.item():.4f}"
            })

            # Log training metrics to W&B (batch-level)
            if i % self.config.log_freq == 0:
                batch_step = epoch * len(self.train_loader) + i
                metrics = {
                    'loss_total': g_loss.item(),
                    'loss_l1': l1.item(),
                    'loss_perceptual': perceptual.item(),
                    'batch': i,
                    'epoch': epoch + 1
                }
                self.wandb_manager.log_metrics(metrics, step=batch_step, prefix="train_batch")

            # Save samples with meaningful names - MUCH CLEANER!
            if i % self.config.sample_freq == 0:
                result = self.wandb_manager.save_and_log_samples(
                    real_images=images,
                    generated_images=generated_images,
                    sample_dir=self.sample_dir,
                    epoch=epoch,
                    batch_idx=i,
                    prefix="train",
                    overwrite=self.config.overwrite_samples
                )

                if result['success']:
                    print(f"Saved training samples: {result['filename']}")

        # Return average metrics for the epoch
        return {
            'avg_g_loss': total_g_loss / len(self.train_loader),
            'avg_l1_loss': total_l1_loss / len(self.train_loader),
            'avg_perceptual_loss': total_perceptual_loss / len(self.train_loader)
        }

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
                    'g_loss': f"{g_loss.item():.4f}",
                    'l1': f"{l1.item():.4f}",
                    'perceptual': f"{perceptual.item():.4f}"
                })

            # Save validation samples with meaningful names
            if len(self.val_loader) > 0:
                result = self.wandb_manager.save_and_log_samples(
                    real_images=images,
                    generated_images=generated_images,
                    sample_dir=self.sample_dir,
                    epoch=epoch,
                    batch_idx=None,  # No batch index for validation
                    prefix="val",
                    overwrite=self.config.overwrite_samples
                )

                if result['success']:
                    print(f"Saved validation samples: {result['filename']}")

        # Return average metrics for the epoch
        return {
            'avg_g_loss': total_g_loss / len(self.val_loader),
            'avg_l1_loss': total_l1_loss / len(self.val_loader),
            'avg_perceptual_loss': total_perceptual_loss / len(self.val_loader)
        }

    def _save_checkpoint(self, epoch):
        """Save model checkpoint with meaningful name."""
        try:
            checkpoint = {
                'epoch': epoch,
                'generator': self.generator.state_dict(),
                'g_optimizer': self.g_optimizer.state_dict(),
                'g_scheduler': self.g_scheduler.state_dict(),
                'config': vars(self.config)
            }

            # Use meaningful filename
            filename = f"checkpoint_epoch_{epoch + 1:03d}.pth"
            filepath = os.path.join(self.checkpoint_dir, filename)

            # Overwrite if exists (always keep latest)
            torch.save(checkpoint, filepath)

            print(f"Checkpoint saved: {filename}")

        except Exception as e:
            print(f"Error: Checkpoint saving failed: {e}")


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

        # W&B and logging settings
        self.use_wandb = True
        self.project_name = "face-generation"
        self.run_name = f"face-gen-{self.embedding_size}-{self.img_size}"
        self.wandb_folder = "wandb"  # W&B folder in output directory
        self.wandb_mode = "online"  # online, offline, disabled
        self.log_freq = 10
        self.sample_freq = 100
        self.save_freq = 5

        # File management
        self.overwrite_samples = True  # Overwrite existing sample files


# Main function to start training
def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()


