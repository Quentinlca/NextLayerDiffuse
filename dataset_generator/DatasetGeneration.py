from src.DatasetGenerators import *

if __name__ == '__main__':
    # Initialize the NextTokenGenerator with the desired parameters
    
    TokenGenerator = NextTokenGenerator('generated_datasets/modular_characters_hairs_RGB',
                                    output_image_size=128,
                                    save_size=500)
    
    TokenGenerator.generate_hair_sequence('characters.json',
                       resume=False)
    