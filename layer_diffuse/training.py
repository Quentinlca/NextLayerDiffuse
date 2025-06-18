from models import DDPMNextTokenV1
from models import DDPMNextTokenV2

from data_loaders import ModularCharatersDataLoader
import argparse

def train_loop():
    val_split = f"train"
    train_split = f"train"
    parser = argparse.ArgumentParser(description="Train DDPMNextTokenV2 model.")
    parser.add_argument('--train_size', type=int, default=16000, help='Number of training batches')
    parser.add_argument('--val_size', type=int, default=1600, help='Number of validation batches')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--dataset_name', type=str, default="QLeca/modular_characters_small", help='Dataset name to use for training')
    
    args = parser.parse_args()
    # get the training and validation sizes from the command line arguments
    train_size = args.train_size
    val_size = args.val_size
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    
    # Initialize the DDPMNextTokenV1 pipeline
    pipeline = DDPMNextTokenV2.DDPMNextTokenV2Pipeline()
    # Configure the training parameters (TODO: make this configurable)
    pipeline.train_config.train_batch_size = batch_size
    pipeline.train_config.eval_batch_size = batch_size
    # Get the dataloaders for training and validation
    train_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(dataset_name=dataset_name,
                                                                            split=train_split,
                                                                            image_size=pipeline.train_config.image_size,
                                                                            batch_size=pipeline.train_config.train_batch_size,
                                                                            shuffle=True)
    val_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(dataset_name=dataset_name,
                                                                            split=val_split,
                                                                            image_size=pipeline.train_config.image_size,
                                                                            batch_size=pipeline.train_config.eval_batch_size,
                                                                            shuffle=True)
    # Start the training process
    pipeline.train_accelerate(train_dataloader=train_dataloader, 
                   val_dataloader=val_dataloader, 
                   train_size=train_size, 
                   val_size=val_size)


if __name__ == "__main__":
    train_loop()
  