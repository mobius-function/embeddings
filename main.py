# main.py
import os
import argparse
import datetime
from src.train import Config, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Generation Model')

    # Data paths
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='data/val',
                        help='Path to validation data')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Path to output directory')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')

    # Model parameters
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of embedding vector')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Size of images')

    # W&B and logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='face-generation',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='W&B mode: online, offline, or disabled')

    # File management
    parser.add_argument('--overwrite_samples', action='store_true', default=True,
                        help='Overwrite existing sample files')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='Save samples every N batches')

    return parser.parse_args()


def main():
    print("=" * 60)
    print("Face Generation Training")
    print("=" * 60)

    # Parse command-line arguments
    args = parse_args()

    # Create configuration
    config = Config()

    # Update config with command-line arguments
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))

    # Set run name if not provided
    if config.run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.run_name = f"face-gen-{config.img_size}x{config.img_size}-{timestamp}"

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)

    # Print configuration
    print(f"Configuration:")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Image size: {config.img_size}x{config.img_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.lr}")
    print(f"  W&B enabled: {config.use_wandb}")
    if config.use_wandb:
        print(f"  W&B project: {config.project_name}")
        print(f"  W&B run: {config.run_name}")
        print(f"  W&B mode: {config.wandb_mode}")

    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    try:
        # Create and run trainer
        # The WandbManager in Trainer will handle all W&B setup automatically!
        trainer = Trainer(config)
        trainer.train()

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"Results saved in: {config.output_dir}")
        print(f"  - Checkpoints: {os.path.join(config.output_dir, 'checkpoints')}")
        print(f"  - Sample images: {os.path.join(config.output_dir, 'samples')}")
        if config.use_wandb:
            print(f"  - W&B logs: {os.path.join(config.output_dir, config.wandb_folder)}")

    except KeyboardInterrupt:
        print("\nWarning: Training interrupted by user")
        print("Partial results may be saved in output directory")

    except Exception as e:
        print(f"\nError: Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


