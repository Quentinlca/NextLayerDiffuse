from src.DatasetGenerators import *
import os

if __name__ == '__main__':
    generator = NextTokenGenerator.NextTokenGenerator('modular_character',
                                                      output_image_size=128,
                                                      save_size=1000)
    
    generator.generate(os.path.abspath('characters.json'),
                       resume=True)