from src.DatasetGenerators import *

if __name__ == '__main__':
    # Initialize the NextTokenGenerator with the desired parameters
    generator = NextTokenGenerator('generated_datasets/modular_characters_v2',
                                    output_image_size=128,
                                    save_size=1000)
    
    generator.generate('characters.json',
                       resume=False)