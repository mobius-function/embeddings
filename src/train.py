# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
import warnings

# Suppress deprecation warnings from LPIPS/torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
# Suppress specific UserWarning from LPIPS about adaptive_avg_pool2d anticausal tensor
warnings.filterwarnings("ignore",
                        message="The default behavior for interpolate/upsample with float scale_factor changed.*",
                        module="torch.nn.functional")

from src.dataset import FaceDataset, get_transforms
from src.encoder import get_encoder
from src.generator import get_generator
from src.wandb_utils import WandbManager  # Ensure this path is correct


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)

        self.checkpoint_dir = os.path.join(config.output_dir, 'checkpoints')
        self.sample_dir = os.path.join(config.output_dir, 'samples')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.wandb_manager = WandbManager(config)  # Initializes W&B and defines metrics

        self._init_dataloaders()
        self._init_models()
        self._init_losses()
        self._init_optimizers()

        # Initialize global step. This will be incremented for each batch.
        # If loading from a checkpoint, this should be restored.
        self.current_global_step = 0

    def _init_dataloaders(self):
        train_transform = get_transforms(img_size=self.config.img_size, is_training=True)
        val_transform = get_transforms(img_size=self.config.img_size, is_training=False)
        train_dataset = FaceDataset(image_dir=self.config.train_dir, transform=train_transform)
        val_dataset = FaceDataset(image_dir=self.config.val_dir, transform=val_transform)

        # drop_last=True for train_loader ensures consistent batch sizes for step counting,
        # otherwise, the last batch could be smaller.
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=True, drop_last=False
        )  # drop_last=False for val is typical

        print(f"Training dataset: {len(train_dataset)} images, Batches: {len(self.train_loader)}")
        print(f"Validation dataset: {len(val_dataset)} images, Batches: {len(self.val_loader)}")

    def _init_models(self):
        self.encoder = get_encoder(embedding_size=self.config.embedding_size, device=self.device)
        self.generator = get_generator(embedding_size=self.config.embedding_size, device=self.device)

    def _init_losses(self):
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = lpips.LPIPS(net='alex').to(self.device)

    def _init_optimizers(self):
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        self.g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=self.config.lr_step_size,
                                                     gamma=self.config.lr_gamma)

    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs on {self.device}")
        # If resuming, self.current_global_step and epoch should be loaded from checkpoint

        for epoch in range(self.config.num_epochs):  # epoch is 0-indexed
            # Training phase
            train_metrics = self._train_epoch(epoch)

            # Validation phase (pass current_global_step for sample logging consistency)
            val_metrics = self._validate_epoch(epoch, self.current_global_step)

            current_lr = self.g_scheduler.get_last_lr()[0]
            # Step the scheduler after validation and logging for the current epoch's state
            self.g_scheduler.step()

            epoch_num_for_axis = epoch + 1  # 1-indexed epoch number for plotting

            train_epoch_log_data = {
                'loss_total': train_metrics['avg_g_loss'],
                'loss_l1': train_metrics['avg_l1_loss'],
                'loss_perceptual': train_metrics['avg_perceptual_loss'],
                'learning_rate': current_lr,
                'epoch': epoch_num_for_axis
            }
            self.wandb_manager.log_metrics(train_epoch_log_data, step=self.current_global_step, prefix="train")

            if self.val_loader and len(self.val_loader) > 0:  # Only log val if val_loader exists
                val_epoch_log_data = {
                    'loss_total': val_metrics['avg_g_loss'],
                    'loss_l1': val_metrics['avg_l1_loss'],
                    'loss_perceptual': val_metrics['avg_perceptual_loss'],
                    'epoch': epoch_num_for_axis
                }
                self.wandb_manager.log_metrics(val_epoch_log_data, step=self.current_global_step, prefix="val")

            if (epoch + 1) % self.config.save_freq == 0:
                self._save_checkpoint(epoch)

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} completed. Global Step: {self.current_global_step}")
            print(
                f"  Train Loss: {train_metrics['avg_g_loss']:.4f} (L1: {train_metrics['avg_l1_loss']:.4f}, Perceptual: {train_metrics['avg_perceptual_loss']:.4f})")
            if self.val_loader and len(self.val_loader) > 0:
                print(
                    f"  Val Loss: {val_metrics['avg_g_loss']:.4f} (L1: {val_metrics['avg_l1_loss']:.4f}, Perceptual: {val_metrics['avg_perceptual_loss']:.4f})")
            print(f"  Learning Rate: {current_lr:.6f}")

        self.wandb_manager.cleanup()

    def _train_epoch(self, epoch):  # epoch is 0-indexed
        self.generator.train()
        total_g_loss, total_l1_loss, total_perceptual_loss = 0.0, 0.0, 0.0

        num_batches = len(self.train_loader)
        if num_batches == 0:
            print(f"Warning: Train loader for epoch {epoch + 1} is empty.")
            return {'avg_g_loss': 0, 'avg_l1_loss': 0, 'avg_perceptual_loss': 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training")
        for i, images in enumerate(pbar):
            self.current_global_step += 1  # Increment global step for each training batch processed.

            images = images.to(self.device)
            with torch.no_grad():  # Encoder should not be trained here
                embeddings = self.encoder(images)
            generated_images = self.generator(embeddings)

            l1 = self.l1_loss(generated_images, images)
            perceptual = self.perceptual_loss(generated_images, images).mean()
            g_loss = l1 + self.config.perceptual_weight * perceptual

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            total_g_loss += g_loss.item()
            total_l1_loss += l1.item()
            total_perceptual_loss += perceptual.item()

            pbar.set_postfix(
                {'g_loss': f"{g_loss.item():.4f}", 'l1': f"{l1.item():.4f}", 'perceptual': f"{perceptual.item():.4f}"})

            if self.config.log_freq > 0 and i % self.config.log_freq == 0:
                batch_metrics = {
                    'loss_total': g_loss.item(),
                    'loss_l1': l1.item(),
                    'loss_perceptual': perceptual.item(),
                }
                self.wandb_manager.log_metrics(batch_metrics, step=self.current_global_step, prefix="batch")

            if self.config.sample_freq > 0 and i % self.config.sample_freq == 0:
                self.wandb_manager.save_and_log_samples(
                    real_images=images, generated_images=generated_images,
                    sample_dir=self.sample_dir, epoch=epoch, batch_idx=i,
                    current_global_step=self.current_global_step,
                    prefix="train", overwrite=self.config.overwrite_samples
                )

        return {
            'avg_g_loss': total_g_loss / num_batches,
            'avg_l1_loss': total_l1_loss / num_batches,
            'avg_perceptual_loss': total_perceptual_loss / num_batches
        }

    def _validate_epoch(self, epoch, current_global_step_for_summary):  # epoch is 0-indexed
        if not self.val_loader or len(self.val_loader) == 0:
            print(f"Warning: Val loader for epoch {epoch + 1} is empty or not defined.")
            return {'avg_g_loss': 0, 'avg_l1_loss': 0, 'avg_perceptual_loss': 0}

        self.generator.eval()
        total_g_loss, total_l1_loss, total_perceptual_loss = 0.0, 0.0, 0.0

        num_batches = len(self.val_loader)
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} Validation")

        last_val_images = None  # To store the last batch for sample logging
        last_val_generated_images = None

        with torch.no_grad():
            for i, images in enumerate(pbar):
                images = images.to(self.device)
                embeddings = self.encoder(images)
                generated_images = self.generator(embeddings)

                l1 = self.l1_loss(generated_images, images)
                perceptual = self.perceptual_loss(generated_images, images).mean()
                g_loss = l1 + self.config.perceptual_weight * perceptual

                total_g_loss += g_loss.item()
                total_l1_loss += l1.item()
                total_perceptual_loss += perceptual.item()
                pbar.set_postfix({'g_loss': f"{g_loss.item():.4f}", 'l1': f"{l1.item():.4f}",
                                  'perceptual': f"{perceptual.item():.4f}"})

                if i == num_batches - 1:  # Store last batch for samples
                    last_val_images = images
                    last_val_generated_images = generated_images

            if last_val_images is not None and last_val_generated_images is not None:
                self.wandb_manager.save_and_log_samples(
                    real_images=last_val_images, generated_images=last_val_generated_images,
                    sample_dir=self.sample_dir, epoch=epoch, batch_idx=None,  # No batch_idx for epoch summary sample
                    current_global_step=current_global_step_for_summary,
                    prefix="val", overwrite=self.config.overwrite_samples
                )

        return {
            'avg_g_loss': total_g_loss / num_batches,
            'avg_l1_loss': total_l1_loss / num_batches,
            'avg_perceptual_loss': total_perceptual_loss / num_batches
        }

    def _save_checkpoint(self, epoch):  # epoch is 0-indexed
        try:
            checkpoint = {
                'epoch': epoch,  # 0-indexed
                'current_global_step': self.current_global_step,
                'generator': self.generator.state_dict(),
                'g_optimizer': self.g_optimizer.state_dict(),
                'g_scheduler': self.g_scheduler.state_dict(),
                'config': vars(self.config)  # Save a copy of the config
            }
            # Filename uses 1-indexed epoch for user readability
            filename = f"checkpoint_epoch_{epoch + 1:03d}.pth"
            filepath = os.path.join(self.checkpoint_dir, filename)
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved: {filename} at global step {self.current_global_step}")
        except Exception as e:
            print(f"Error: Checkpoint saving failed: {e}")


# Configuration class (assuming it's defined as before)
class Config:
    def __init__(self):
        # Data paths
        self.train_dir = "data/train"
        self.val_dir = "data/val"
        self.test_dir = "data/test"  # Not used in trainer, but good to have
        self.output_dir = "output"

        # Training parameters
        self.batch_size = 32  # Make sure this is not larger than dataset size
        self.num_epochs = 100
        self.lr = 2e-4
        self.beta1 = 0.5  # Adam beta1
        self.lr_step_size = 20  # For StepLR scheduler
        self.lr_gamma = 0.5  # For StepLR scheduler
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
        self.use_wandb = True  # Set to False to disable W&B
        self.project_name = "face-generation"  # Your W&B project name
        self.run_name = f"face-gen-emb{self.embedding_size}-img{self.img_size}"  # Example run name
        self.wandb_folder = "wandb"  # W&B specific files within output_dir
        self.wandb_mode = "online"  # "online", "offline", or "disabled"

        self.log_freq = 10  # Log batch metrics every N batches (0 to disable)
        self.sample_freq = 100  # Log sample images every N batches (0 to disable)
        self.save_freq = 5  # Save checkpoint every N epochs

        # File management
        self.overwrite_samples = True  # Overwrite existing sample image files


# Main function to start training
def main():
    config = Config()
    # You can override config parameters here if needed, e.g., from argparse
    # config.batch_size = 16

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

