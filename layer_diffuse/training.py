from models import (
    DDPMNextTokenV1,
    DDPMNextTokenV2,
    DDPMNextTokenV3,
    DDIMNextTokenV1,
    DDIMNextTokenV1_Refactored,
    DDIMNextTokenV2
)


from data_loaders import ModularCharatersDataLoader
import argparse


def train_loop():
    # Parsing
    parser = argparse.ArgumentParser(description="Train DDIMNextTokenV1 model.")
    parser.add_argument("--train_split",
        type=str,
        default="train",
        help="Split to use for training (default: 'train')",
    )
    parser.add_argument("--val_split",
        type=str,
        default="train",
        help="Split to use for validation (default: 'train')",
    )
    parser.add_argument("--model_version",
        type=str,
        default="DDIMNextTokenV1_Refactored",
        choices=[
            "DDPMNextTokenV1",
            "DDPMNextTokenV2",
            "DDPMNextTokenV3",
            "DDIMNextTokenV1",
            "DDIMNextTokenV2",
            "DDIMNextTokenV1_Refactored"
        ],
        help="Model version to use for training",
    )
    parser.add_argument("--train_size",
        type=int,
        default=16000,
        help="Number of training batches",
    )
    parser.add_argument("--val_size",
        type=int,
        default=1600,
        help="Number of validation batches",
    )
    parser.add_argument("--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation",
    )
    parser.add_argument("--dataset_name",
        type=str,
        default="QLeca/modular_characters_hairs_RGB",
        help="Dataset name to use for training",
    )
    parser.add_argument("--num_epochs",
        type=int,
        default=50,
        help="Training epochs",
    )
    parser.add_argument("--lr",
        type=float,
        default=0.0002,
        help="Learning rate",
    )
    parser.add_argument("--warming_steps",
        type=int,
        default=1000,
        help="Learning rate scheduler warming steps",
    )
    parser.add_argument("--num_cycles",
        type=float,
        default=0.5,
        help="Number of cycles for the cosine learning rate scheduler",
    )
    parser.add_argument("--train_tags",
        type=str,
        nargs="*",
        default=None,
        help="Tags for the training run (optional, can be used for wandb tagging)",
    )
    parser.add_argument("--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps (effectively multiplies batch size)",
    )
    parser.add_argument("--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )
    parser.add_argument("--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument("--beta_schedule",
        type=str,
        default="linear",
        choices=["linear", "scaled_linear", "squaredcos_cap_v2"],
        help="Beta schedule for the diffusion scheduler",
    )
    parser.add_argument("--stream_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="To stream the dataset or not (default: False, for large datasets)",
    )
    parser.add_argument("--vocab_file",
        type=str,
        default="",
        help="Path to the vocabulary file (default: 'vocab.json')",
    )
    
    

    args = parser.parse_args()
    # get the training and validation sizes from the command line arguments
    val_split = args.val_split
    train_split = args.train_split
    train_size = args.train_size
    val_size = args.val_size
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    stream_dataset = args.stream_dataset
    vocab = {}
    if args.vocab_file:
        # Load the vocabulary from the specified file
        import json
        try:
            with open(args.vocab_file, 'r') as f:
                vocab = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file '{args.vocab_file}' not found.")
    
    num_epochs = args.num_epochs
    lr = args.lr
    warming_steps = args.warming_steps
    beta_schedule = args.beta_schedule
    num_cycles = args.num_cycles
    
    gradient_accumulation_steps = args.gradient_accumulation_steps
    mixed_precision = args.mixed_precision
    dataloader_num_workers = args.dataloader_num_workers
    
    train_tags = args.train_tags
    
    extra_kwargs = {
        "num_cycles": num_cycles,  # Pass the num_cycles parameter
        "train_tags": train_tags,  # Pass the train_tags parameter
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "dataloader_num_workers": dataloader_num_workers,
    }

    # Initialize the DDPMNextTokenV1 pipeline
    if args.model_version == "DDPMNextTokenV1":
        pipeline = DDPMNextTokenV1.DDPMNextTokenV1Pipeline()
        scheduler_config = DDPMNextTokenV1.SchedulerConfig()
    elif args.model_version == "DDPMNextTokenV2":
        pipeline = DDPMNextTokenV2.DDPMNextTokenV2Pipeline()
    elif args.model_version == "DDPMNextTokenV3":
        pipeline = DDPMNextTokenV3.DDPMNextTokenV3Pipeline()
    elif args.model_version == "DDIMNextTokenV1":
        scheduler_config = DDIMNextTokenV1.SchedulerConfig()
        scheduler_config.config['beta_schedule'] = beta_schedule
        pipeline = DDIMNextTokenV1.DDIMNextTokenV1Pipeline(scheduler_config=scheduler_config)
    elif args.model_version == "DDIMNextTokenV2":
        scheduler_config = DDIMNextTokenV2.DDIMSchedulerConfig()
        scheduler_config.config['beta_schedule'] = beta_schedule
        pipeline = DDIMNextTokenV2.DDIMNextTokenV2Pipeline(scheduler_config=scheduler_config)
    elif args.model_version == "DDIMNextTokenV1_Refactored":
        # Use the refactored version of DDIMNextTokenV1
        scheduler_config = DDIMNextTokenV1_Refactored.DDIMSchedulerConfig()
        scheduler_config.config['beta_schedule'] = beta_schedule        
        pipeline = DDIMNextTokenV1_Refactored.DDIMNextTokenV1PipelineRefactored(scheduler_config=scheduler_config)
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")
    pipeline.train_config.train_batch_size = batch_size
    pipeline.train_config.eval_batch_size = batch_size
    pipeline.train_config.num_epochs = num_epochs
    pipeline.train_config.learning_rate = lr
    pipeline.train_config.lr_warmup_steps = warming_steps

    # Set performance optimizations
    if gradient_accumulation_steps > 1:
        pipeline.train_config.gradient_accumulation_steps = gradient_accumulation_steps
    if mixed_precision in ["fp16", "bf16"]:
        pipeline.train_config.mixed_precision = mixed_precision

    # Get the dataloaders for training and validation
    train_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(
        dataset_name=dataset_name,
        split=train_split,
        image_size=pipeline.train_config.image_size,
        batch_size=pipeline.train_config.train_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True if dataloader_num_workers > 0 else False,
        streaming=stream_dataset,
        conversionRGBA=True,
        vocab=vocab,  # Pass the vocabulary if provided
    )
    val_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(
        dataset_name=dataset_name,
        split=val_split,
        image_size=pipeline.train_config.image_size,
        batch_size=pipeline.train_config.eval_batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True if dataloader_num_workers > 0 else False,
        streaming=stream_dataset,
        conversionRGBA=True,
        vocab=vocab,  # Pass the vocabulary if provided
    )
    # Start the training process
    pipeline.train_accelerate(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_size=train_size,
        val_size=val_size,
        **extra_kwargs,  # Pass all extra parameters through
    )


if __name__ == "__main__":
    train_loop()
