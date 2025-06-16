from models import DDPMNextTokenV1
from data_loaders import ModularCharatersDataLoader
from PIL import Image
dataset_name = "QLeca/modular_charactersv2"
val_split = f"train"
train_split = f"train"


from safetensors import safe_open
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPMNextTokenV1 model.")
    parser.add_argument('--train_size', type=int, default=16000, help='Number of training batches')
    parser.add_argument('--val_size', type=int, default=1600, help='Number of validation batches')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of validation batches')
    
    args = parser.parse_args()
    # get the training and validation sizes from the command line arguments
    train_size = args.train_size
    val_size = args.val_size
    batch_size = args.batch_size
    # Initialize the DDPMNextTokenV1 pipeline
    
    pipeline = DDPMNextTokenV1.DDPMNextTokenV1Pipeline()
    pipeline.train_config.train_batch_size = batch_size
    pipeline.train_config.eval_batch_size = batch_size
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

    pipeline.train(train_dataloader, val_dataloader, train_size = train_size, val_size = val_size)

    # pipeline.load_config('/Data/quentin.leca/training_output/')
    # for batch in val_dataloader:
    #     images = pipeline(batch['input'], batch['prompt'])
    #     print(images.shape)
    #     # Image.fromarray(images)
    #     break