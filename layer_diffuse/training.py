from models import DDPMNextTokenV1
from models import DDPMNextTokenV2
from models import DDPMNextTokenV3
from models import DDIMNextTokenV1


from data_loaders import ModularCharatersDataLoader
import argparse


def train_loop():
    val_split = f"train"
    train_split = f"train"
    parser = argparse.ArgumentParser(description="Train DDIMNextTokenV1 model.")
    parser.add_argument(
        "--model_version",
        type=str,
        default="DDIMNextTokenV1",
        choices=["DDPMNextTokenV1", "DDPMNextTokenV2", "DDPMNextTokenV3", "DDIMNextTokenV1"],
        help="Model version to use for training",
    )
    parser.add_argument(
        "--train_size", type=int, default=16000, help="Number of training batches"
    )
    parser.add_argument(
        "--val_size", type=int, default=1600, help="Number of validation batches"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="QLeca/modular_characters_hairs_RGB",
        help="Dataset name to use for training",
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument(
        "--warming_steps",
        type=int,
        default=500,
        help="Learning rate scheduler warming steps",
    )

    args = parser.parse_args()
    # get the training and validation sizes from the command line arguments
    train_size = args.train_size
    val_size = args.val_size
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    num_epochs = args.num_epochs
    lr = args.lr
    warming_steps = args.warming_steps

    # Initialize the DDPMNextTokenV1 pipeline
    if args.model_version == "DDPMNextTokenV1":
        pipeline = DDPMNextTokenV1.DDPMNextTokenV1Pipeline()
    elif args.model_version == "DDPMNextTokenV2":
        pipeline = DDPMNextTokenV2.DDPMNextTokenV2Pipeline()
    elif args.model_version == "DDPMNextTokenV3":
        pipeline = DDPMNextTokenV3.DDPMNextTokenV3Pipeline()
    else:
        pipeline = DDIMNextTokenV1.DDIMNextTokenV1Pipeline()
    # Configure the training parameters (TODO: make this configurable)
    pipeline.train_config.train_batch_size = batch_size
    pipeline.train_config.eval_batch_size = batch_size
    pipeline.train_config.num_epochs = num_epochs
    pipeline.train_config.learning_rate = lr
    pipeline.train_config.lr_warmup_steps = warming_steps

    # Get the dataloaders for training and validation
    train_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(
        dataset_name=dataset_name,
        split=train_split,
        image_size=pipeline.train_config.image_size,
        batch_size=pipeline.train_config.train_batch_size,
        shuffle=True,
    )
    val_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(
        dataset_name=dataset_name,
        split=val_split,
        image_size=pipeline.train_config.image_size,
        batch_size=pipeline.train_config.eval_batch_size,
        shuffle=True,
    )
    # Start the training process
    pipeline.train_accelerate(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_size=train_size,
        val_size=val_size,
    )


if __name__ == "__main__":
    train_loop()
