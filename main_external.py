from main import main_func
import argparse


def parse_arguments():
    """Parses command-line arguments and returns a dictionary of configuration."""
    parser = argparse.ArgumentParser(description="Training configuration.")

    parser.add_argument(
        "--base_learning_rate", type=float, default=0.0001, help="Base learning rate for the optimizer."
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to use as input.")
    parser.add_argument("--val_epochs", type=int, default=1, help="Frequency of validation (in epochs).")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and validation.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer.")
    parser.add_argument("--load_run", type=str, default=None, help="Load a previous run (specify path if needed).")
    parser.add_argument(
        "--init_vit", action="store_true", default=True, help="Initialize the Vision Transformer (ViT) model."
    )
    parser.add_argument(
        "--vit_base_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="Size of the base ViT model.",
    )
    parser.add_argument("--cosine_schedule", action="store_true", help="Use cosine learning rate schedule.")
    parser.add_argument("--augmentations", action="store_true", default=True, help="Apply basic data augmentations.")
    parser.add_argument("--advanced_augmentations", action="store_true", help="Apply advanced data augmentations.")
    parser.add_argument("--mixup", action="store_true", help="Use MixUp data augmentation.")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Alpha parameter for MixUp.")
    parser.add_argument("--dataset", type=str, default="ucf101", help="Name of the dataset to use.")
    parser.add_argument("--subsample", type=float, default=None, help="Subsample the dataset (use a smaller portion).")
    parser.add_argument("--switch", action="store_true", help="Enable the switch mechanism.")
    parser.add_argument(
        "--switch_layers",
        nargs="+",
        type=int,
        default=[0, 1],
        help="List of layer indices where the switch is applied.",
    )
    parser.add_argument("--n_experts", type=int, default=4, help="Number of experts in the switch mechanism.")
    parser.add_argument(
        "--load_balance_alpha", type=float, default=0.01, help="Alpha parameter for the load balancing loss."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="timesformer",
        help="Base model architecture.",
        choices=["timesformer", "vivit", "mvitv2"],
    )

    args = parser.parse_args()
    return vars(args)


config = parse_arguments()
main_func(config)
