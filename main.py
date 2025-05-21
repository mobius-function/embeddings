# main.py
import os
import argparse
from src.train import Config, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Generation Model')

    # Data paths
    parser.add_argument('--train_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='data/val', help='Path to validation data')
    parser.add_argument('--test_dir', type=str, default='data/test', help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to output directory')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')

    # Model parameters
    parser.add_argument('--embedding_size', type=int, default=256, help='Size of embedding vector')
    parser.add_argument('--img_size', type=int, default=128, help='Size of images')

    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='face-generation', help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Create configuration
    config = Config()

    # Update config with command-line arguments
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))

    # Set run name if not provided
    if config.run_name is None:
        config.run_name = f"face-gen-{config.embedding_size}-{config.img_size}"

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)

    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()