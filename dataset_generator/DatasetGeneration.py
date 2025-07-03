from src.DatasetGenerators import *

if __name__ == '__main__':
    # Initialize the NextTokenGenerator with the desired parameters
    
    # TokenGenerator = NextTokenGenerator('generated_datasets/modular_characters_medium_RGB',
    #                                     output_image_size=128,
    #                                     save_size=500)
    
    # TokenGenerator.generate_dataset_smarter(assets_dir='assets/kenney_new')
    
    NextTokenGenerator.upload_dataset(dataset_path='generated_datasets/modular_characters_medium_RGB/dataset.csv',
                                  repo_id='QLeca/modular_characters_medium_RGB')
    