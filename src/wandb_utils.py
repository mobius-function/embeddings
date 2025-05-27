# src/wandb_utils.py
"""
Dedicated Weights & Biases utilities for face generation training.
Handles W&B setup, logging, and Windows compatibility issues.
"""

import os
import tempfile
import torch
import numpy as np
from PIL import Image
import torchvision.utils as vutils


class WandbManager:
    """Manages all W&B operations with error handling and Windows compatibility."""

    def __init__(self, config):
        self.config = config
        self.wandb_run = None
        self.wandb_dirs = None

        if config.use_wandb:
            self.setup_environment()
            self.initialize()

    def setup_environment(self):
        """Setup W&B directories to avoid Windows temp issues."""
        print("Setting up W&B environment...")

        # Create meaningful directory structure
        self.wandb_dirs = {
            'main': os.path.join(self.config.output_dir, self.config.wandb_folder),
            'cache': os.path.join(self.config.output_dir, self.config.wandb_folder, 'cache'),
            'temp': os.path.join(self.config.output_dir, self.config.wandb_folder, 'temp'),
            'logs': os.path.join(self.config.output_dir, self.config.wandb_folder, 'logs'),
            'media': os.path.join(self.config.output_dir, self.config.wandb_folder, 'media')
        }

        # Create all directories
        for name, path in self.wandb_dirs.items():
            os.makedirs(path, exist_ok=True)
            print(f"Created W&B {name} directory: {path}")

        # Set environment variables
        os.environ['WANDB_DIR'] = self.wandb_dirs['main']
        os.environ['WANDB_CACHE_DIR'] = self.wandb_dirs['cache']
        os.environ['WANDB_DATA_DIR'] = self.wandb_dirs['main']
        os.environ['WANDB_ARTIFACT_DIR'] = self.wandb_dirs['main']
        os.environ['WANDB_CONFIG_DIR'] = self.wandb_dirs['main']

        # Set temp directories
        os.environ['TMPDIR'] = self.wandb_dirs['temp']
        os.environ['TMP'] = self.wandb_dirs['temp']
        os.environ['TEMP'] = self.wandb_dirs['temp']
        tempfile.tempdir = self.wandb_dirs['temp']

        # Additional W&B settings
        os.environ['WANDB_START_METHOD'] = 'thread'
        os.environ['WANDB_CONSOLE'] = 'off'

        print(f"W&B environment configured in: {self.wandb_dirs['main']}")

    def initialize(self):
        """Initialize W&B with error handling."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config),
                mode=getattr(self.config, 'wandb_mode', 'online'),
                dir=self.wandb_dirs['main']
            )

            print(f"W&B initialized successfully!")
            print(f"   Project: {self.config.project_name}")
            print(f"   Run: {self.config.run_name}")
            print(f"   URL: {self.wandb_run.url}")

        except ImportError:
            print("Error: W&B not installed. Run: pip install wandb")
            self.wandb_run = None

        except Exception as e:
            print(f"Error: W&B initialization failed: {e}")
            self.wandb_run = None

    def is_active(self):
        """Check if W&B is active and ready."""
        return self.wandb_run is not None

    def log_metrics(self, metrics_dict, step=None, prefix=""):
        """Log metrics to W&B."""
        if not self.is_active():
            return False

        try:
            # Add prefix to metric keys if provided
            if prefix:
                metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}

            if step is not None:
                self.wandb_run.log(metrics_dict, step=step)
            else:
                self.wandb_run.log(metrics_dict)

            return True

        except Exception as e:
            print(f"Warning: W&B metric logging failed: {e}")
            return False

    def create_meaningful_filename(self, prefix, epoch, batch_idx=None, file_type="png"):
        """Create meaningful filename instead of random names."""
        if batch_idx is not None:
            return f"{prefix}_epoch_{epoch:03d}_batch_{batch_idx:04d}.{file_type}"
        else:
            return f"{prefix}_epoch_{epoch:03d}.{file_type}"

    def save_image_locally(self, grid_tensor, filepath, overwrite=True):
        """Save image grid to local file."""
        try:
            # Check if file exists and handle overwriting
            if os.path.exists(filepath) and not overwrite:
                print(f"File exists, skipping: {filepath}")
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save image
            vutils.save_image(grid_tensor, filepath, normalize=False)
            print(f"Saved: {filepath}")
            return True

        except Exception as e:
            print(f"Error: Failed to save {filepath}: {e}")
            return False

    def log_images_from_file(self, filepath, log_key, caption=None, step=None):
        """Log images to W&B using saved file."""
        if not self.is_active() or not os.path.exists(filepath):
            return False

        try:
            import wandb

            wandb_image = wandb.Image(filepath, caption=caption)
            log_dict = {log_key: wandb_image}

            if step is not None:
                self.wandb_run.log(log_dict, step=step)
            else:
                self.wandb_run.log(log_dict)

            print(f"Logged to W&B: {log_key}")
            return True

        except Exception as e:
            print(f"Warning: W&B image logging failed: {e}")
            return False

    def log_images_from_tensor(self, grid_tensor, log_key, caption=None, step=None):
        """Log images to W&B using tensor (fallback method)."""
        if not self.is_active():
            return False

        try:
            import wandb

            # Convert tensor to PIL image
            grid_np = grid_tensor.permute(1, 2, 0).cpu().detach().numpy()
            grid_np = np.clip(grid_np * 255, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(grid_np)

            wandb_image = wandb.Image(pil_image, caption=caption)
            log_dict = {log_key: wandb_image}

            if step is not None:
                self.wandb_run.log(log_dict, step=step)
            else:
                self.wandb_run.log(log_dict)

            print(f"Logged to W&B (tensor method): {log_key}")
            return True

        except Exception as e:
            print(f"Warning: W&B tensor logging failed: {e}")
            return False

    def create_image_grid(self, real_images, generated_images, nrow=4, max_images=8):
        """Create image grid with version compatibility."""
        # Limit number of images
        sample_size = min(max_images, real_images.size(0), generated_images.size(0))
        real_sample = real_images[:sample_size]
        gen_sample = generated_images[:sample_size]

        try:
            # New torchvision version
            real_grid = vutils.make_grid(
                real_sample, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2
            )
            gen_grid = vutils.make_grid(
                gen_sample, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2
            )
        except TypeError:
            # Older torchvision version
            real_grid = vutils.make_grid(
                real_sample, nrow=nrow, normalize=True, range=(-1, 1), padding=2
            )
            gen_grid = vutils.make_grid(
                gen_sample, nrow=nrow, normalize=True, range=(-1, 1), padding=2
            )

        # Combine grids (real on top, generated on bottom)
        combined_grid = torch.cat([real_grid, gen_grid], dim=1)
        return combined_grid

    def save_and_log_samples(self, real_images, generated_images, sample_dir,
                             epoch, batch_idx=None, prefix="train", overwrite=True):
        """
        Main function to save samples locally and log to W&B.
        Uses meaningful filenames instead of random names.
        """
        try:
            # Create image grid
            grid = self.create_image_grid(real_images, generated_images)

            # Create meaningful filename
            filename = self.create_meaningful_filename(prefix, epoch, batch_idx)
            filepath = os.path.join(sample_dir, filename)

            # Save locally
            save_success = self.save_image_locally(grid, filepath, overwrite=overwrite)

            # Log to W&B if local save was successful
            wandb_success = False
            if save_success and self.is_active():
                if batch_idx is not None:
                    caption = f"Epoch {epoch}, Batch {batch_idx} - Top: Real, Bottom: Generated"
                    step = epoch * 10000 + batch_idx  # Consistent with batch logging
                    log_key = f"samples/{prefix}"
                else:
                    caption = f"Epoch {epoch} - Top: Real, Bottom: Generated"
                    step = (epoch + 1) * 100000  # Use same epoch step offset
                    log_key = f"samples/{prefix}"

                # Try file method first, then tensor method as fallback
                wandb_success = self.log_images_from_file(
                    filepath, log_key, caption, step
                )

                if not wandb_success:
                    wandb_success = self.log_images_from_tensor(
                        grid, log_key, caption, step
                    )

            return {
                'success': save_success,
                'wandb_logged': wandb_success,
                'filepath': filepath if save_success else None,
                'filename': filename
            }

        except Exception as e:
            print(f"Error: Sample saving failed: {e}")
            return {
                'success': False,
                'wandb_logged': False,
                'filepath': None,
                'filename': None
            }

    def cleanup(self):
        """Clean up W&B resources."""
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                print("W&B run finished successfully")
            except:
                pass


# Convenience functions for backward compatibility
def setup_wandb_environment(output_dir, wandb_folder="wandb"):
    """Backward compatibility function."""

    class TempConfig:
        def __init__(self):
            self.output_dir = output_dir
            self.wandb_folder = wandb_folder
            self.use_wandb = True

    manager = WandbManager(TempConfig())
    return manager.wandb_dirs


def initialize_wandb(project_name, run_name, config_dict, wandb_dir, mode="online"):
    """Backward compatibility function."""

    class TempConfig:
        def __init__(self):
            self.project_name = project_name
            self.run_name = run_name
            self.use_wandb = True
            self.wandb_mode = mode
            self.output_dir = os.path.dirname(wandb_dir)
            self.wandb_folder = os.path.basename(wandb_dir)

    manager = WandbManager(TempConfig())
    return manager.wandb_run

