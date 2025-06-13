from src.DatasetGenerators import *
import os

if __name__ == '__main__':
    generator = NextTokenGenerator('modular_characters',
                                                      output_image_size=128,
                                                      save_size=1000)
    
    generator.generate('characters.json',
                       resume=True)