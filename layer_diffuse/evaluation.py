from models import DDPMNextTokenV1
from models import DDPMNextTokenV2
from data_loaders import ModularCharatersDataLoader
import argparse

def eval_loop():
    parser = argparse.ArgumentParser(description="Train DDPMNextTokenV2 model.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--dataset_name', type=str, default="QLeca/modular_characters_small", help='Dataset name to use for training')
    parser.add_argument('--run_name', type=str, default='run_2025-06-19_14-12-46', help='Run name for loading the model')
    parser.add_argument('--epoch', type=int, default=49, help='Epoch number to load the model from')
    parser.add_argument('--num_inference_steps', type=int, default=1000, help='Number of inference steps for FID score calculation')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use for evaluation')
    
    args = parser.parse_args()
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    run_name = args.run_name
    epoch = args.epoch
    num_inference_steps = args.num_inference_steps
    split = args.split
    
    # Initialize the DDPMNextTokenV1 pipeline
    pipeline = DDPMNextTokenV2.DDPMNextTokenV2Pipeline()
    # Configure the training parameters (TODO: make this configurable)
    pipeline.train_config.train_batch_size = batch_size
    pipeline.train_config.eval_batch_size = batch_size
    
    dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(dataset_name=dataset_name,
                                                                            split=split,
                                                                            image_size=pipeline.train_config.image_size,
                                                                            batch_size=pipeline.train_config.train_batch_size,
                                                                            shuffle=True)
    
    if pipeline.load_model_from_hub(run=run_name, epoch=epoch):
        print(f"Model loaded successfully from run {run_name} at epoch {epoch}.")
        FID_score = pipeline.get_FID_score(dataloader=dataloader,
                               num_inference_steps=num_inference_steps)
        if FID_score is not None:
            print(f"FID score for run {run_name} at epoch {epoch}: {FID_score}")
            stats = {'FID_score': {'run': run_name,
                                   'epoch': epoch,
                                   'dataset_name': dataset_name,
                                   'split': split,
                                   'dataset_size': len(dataloader.dataset),
                                   'num_inference_steps': num_inference_steps,
                                   'FID_score': FID_score}}
            if pipeline.save_stats(stats=stats, run=run_name, epoch=epoch):
                print(f"Stats saved successfully for run {run_name} at epoch {epoch}.")
            else:
                print(f"Failed to save stats for run {run_name} at epoch {epoch}.")
        else:
            print(f"Failed to compute FID score for run {run_name} at epoch {epoch}.")
            FID_score = None
    else:
        print(f"Failed to load model from run {run_name} at epoch {epoch}.")
        FID_score = None
        
    
    # Get the dataloaders for training and validation


if __name__ == "__main__":
    eval_loop()
  