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

# Ensure wandb is imported if not already
try:
    import wandb
except ImportError:
    wandb = None


class WandbManager:
    """Manages all W&B operations with error handling and Windows compatibility."""

    def __init__(self, config):
        self.config = config
        self.wandb_run = None
        self.wandb_dirs = None

        if config.use_wandb:
            if wandb is None:
                print("Error: W&B library not installed. Please run: pip install wandb")
                # Prevent further W&B setup if library is missing
                self.config.use_wandb = False  # Ensure flag is set to false
                return
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
        # Consider commenting out WANDB_CONSOLE='off' during debugging
        # os.environ['WANDB_CONSOLE'] = 'off'
        print(f"W&B environment configured in: {self.wandb_dirs['main']}")

    def initialize(self):
        """Initialize W&B with error handling and define custom metrics."""
        if not self.config.use_wandb or wandb is None:
            return
        try:
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config),
                mode=getattr(self.config, 'wandb_mode', 'online'),
                dir=self.wandb_dirs['main']
            )

            if self.wandb_run:
                print(f"W&B initialized successfully!")
                print(f"   Project: {self.config.project_name}")
                print(f"   Run: {self.config.run_name}")
                print(f"   URL: {self.wandb_run.url}")

                # Define custom x-axes for epoch-level metrics
                self.wandb_run.define_metric("train/epoch")
                self.wandb_run.define_metric("train/*", step_metric="train/epoch")

                self.wandb_run.define_metric("val/epoch")
                self.wandb_run.define_metric("val/*", step_metric="val/epoch")

                # Batch metrics (e.g., "batch/loss_total") and sample images ("samples/*")
                # will use the default global step (wandb.run.step).
            else:
                print("Error: W&B initialization returned None. Check W&B mode and configurations.")
                self.config.use_wandb = False

        except Exception as e:
            print(f"Error: W&B initialization failed: {e}")
            self.wandb_run = None
            self.config.use_wandb = False

    def is_active(self):
        """Check if W&B is active and ready."""
        return self.config.use_wandb and self.wandb_run is not None

    def log_metrics(self, metrics_dict, step=None, prefix=""):
        """Log metrics to W&B."""
        if not self.is_active():
            return False

        try:
            log_payload = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics_dict.items()}

            # --- DEBUG PRINT --- (Remove or comment out after verifying)
            if prefix in ["train", "val"] and "epoch" in metrics_dict:  # Check for epoch summaries
                print(
                    f"DEBUG W&B LOGGING (Epoch Summary): step={step}, prefix='{prefix}', payload_keys={list(log_payload.keys())}")
                print(f"DEBUG Full payload for {prefix}: {log_payload}")
            # --- END DEBUG PRINT ---

            if step is not None:
                self.wandb_run.log(log_payload, step=step)
            else:
                self.wandb_run.log(log_payload)

            return True

        except Exception as e:
            print(f"Warning: W&B metric logging failed for keys {list(log_payload.keys())} at step {step}: {e}")
            return False

    def create_meaningful_filename(self, prefix, epoch, batch_idx=None, file_type="png"):
        """Create meaningful filename. 'epoch' is 0-indexed."""
        if batch_idx is not None:
            return f"{prefix}_epoch_{epoch + 1:03d}_batch_{batch_idx:04d}.{file_type}"
        else:
            return f"{prefix}_epoch_{epoch + 1:03d}.{file_type}"

    def save_image_locally(self, grid_tensor, filepath, overwrite=True):
        """Save image grid to local file. Assumes grid_tensor is 0-1 range."""
        try:
            if os.path.exists(filepath) and not overwrite:
                return True
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            vutils.save_image(grid_tensor, filepath, normalize=False)
            return True
        except Exception as e:
            print(f"Error: Failed to save {filepath}: {e}")
            return False

    def log_images_from_file(self, filepath, log_key, caption=None, step=None):
        """Log images to W&B using saved file."""
        if not self.is_active() or not os.path.exists(filepath) or wandb is None:
            return False
        try:
            wandb_image = wandb.Image(filepath, caption=caption)
            log_dict = {log_key: wandb_image}
            if step is not None:
                self.wandb_run.log(log_dict, step=step)
            else:
                self.wandb_run.log(log_dict)
            return True
        except Exception as e:
            print(f"Warning: W&B image logging from file failed for {log_key}: {e}")
            return False

    def log_images_from_tensor(self, grid_tensor, log_key, caption=None, step=None):
        """Log images to W&B using tensor. Assumes grid_tensor is 0-1 range."""
        if not self.is_active() or wandb is None:
            return False
        try:
            grid_np = grid_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            pil_image = Image.fromarray(grid_np)
            wandb_image = wandb.Image(pil_image, caption=caption)
            log_dict = {log_key: wandb_image}
            if step is not None:
                self.wandb_run.log(log_dict, step=step)
            else:
                self.wandb_run.log(log_dict)
            return True
        except Exception as e:
            print(f"Warning: W&B tensor logging failed for {log_key}: {e}")
            return False

    def create_image_grid(self, real_images, generated_images, nrow=4, max_images=8):
        """Create image grid. Assumes input tensors are in [-1, 1] range."""
        sample_size = min(max_images, real_images.size(0), generated_images.size(0))
        real_sample = real_images[:sample_size].clone().detach().cpu()
        gen_sample = generated_images[:sample_size].clone().detach().cpu()

        real_sample = (real_sample + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        gen_sample = (gen_sample + 1) / 2  # Normalize from [-1, 1] to [0, 1]

        try:
            real_grid = vutils.make_grid(real_sample, nrow=nrow, padding=2, normalize=False)
            gen_grid = vutils.make_grid(gen_sample, nrow=nrow, padding=2, normalize=False)
        except TypeError:
            real_grid = vutils.make_grid(real_sample, nrow=nrow, padding=2, normalize=True)
            gen_grid = vutils.make_grid(gen_sample, nrow=nrow, padding=2, normalize=True)

        combined_grid = torch.cat([real_grid, gen_grid], dim=1)
        return combined_grid

    def save_and_log_samples(self, real_images, generated_images, sample_dir,
                             epoch, current_global_step,
                             batch_idx=None, prefix="train", overwrite=True):
        """'epoch' argument is 0-indexed."""
        try:
            grid = self.create_image_grid(real_images, generated_images)
            filename = self.create_meaningful_filename(prefix, epoch, batch_idx)
            filepath = os.path.join(sample_dir, filename)

            save_success = self.save_image_locally(grid, filepath, overwrite=overwrite)
            wandb_success = False

            if save_success and self.is_active():
                log_key = f"samples/{prefix}"
                if batch_idx is not None:
                    caption = f"Epoch {epoch + 1}, Batch {batch_idx} - Top: Real, Bottom: Generated"
                else:
                    caption = f"Epoch {epoch + 1} - Top: Real, Bottom: Generated"

                step_for_wandb_log = current_global_step

                wandb_success = self.log_images_from_file(
                    filepath, log_key, caption, step=step_for_wandb_log
                )
                if not wandb_success:
                    wandb_success = self.log_images_from_tensor(
                        grid, log_key, caption, step=step_for_wandb_log
                    )

            if save_success or wandb_success:
                if batch_idx is None or (
                        self.config.sample_freq > 0 and batch_idx % (self.config.sample_freq * 5) == 0):
                    print(f"Processed samples: {filename} (Epoch {epoch + 1}) at global step {current_global_step}")

            return {
                'success': save_success,
                'wandb_logged': wandb_success,
                'filepath': filepath if save_success else None,
                'filename': filename
            }
        except Exception as e:
            print(f"Error: Sample processing/logging failed for epoch {epoch + 1}, batch {batch_idx}: {e}")
            return {'success': False, 'wandb_logged': False, 'filepath': None, 'filename': None}

    def cleanup(self):
        """Clean up W&B resources."""
        if self.is_active():
            try:
                print("Attempting to finish W&B run...")
                self.wandb_run.finish()
                print("W&B run finished successfully.")
            except Exception as e:
                print(f"Warning: W&B run finish encountered an issue: {e}")
        self.wandb_run = None


# Convenience functions for backward compatibility (consider refactoring to use WandbManager directly)
def setup_wandb_environment(output_dir, wandb_folder="wandb"):
    class TempConfig:
        def __init__(self):
            self.output_dir = output_dir
            self.wandb_folder = wandb_folder
            self.use_wandb = True

    manager = WandbManager(TempConfig())
    return manager.wandb_dirs


def initialize_wandb(project_name, run_name, config_dict, wandb_dir, mode="online"):
    class TempConfig:
        def __init__(self):
            self.project_name = project_name
            self.run_name = run_name
            self.use_wandb = True
            self.wandb_mode = mode
            self.output_dir = os.path.dirname(wandb_dir) if wandb_dir else "."
            self.wandb_folder = os.path.basename(wandb_dir) if wandb_dir else "wandb"
            for k, v in config_dict.items():
                setattr(self, k, v)

    manager = WandbManager(TempConfig())
    return manager.wandb_run

