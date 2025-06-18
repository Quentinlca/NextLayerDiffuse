from src.DatasetGenerators import *

if __name__ == '__main__':
    # Initialize the NextTokenGenerator with the desired parameters
    CharGenerator = CharacterGenerator(assets_dir='assets/kenney_new',
                                   output_dir='generated_datasets/kenney_new')
    
    CharGenerator.generate_small_dataset()
    
    TokenGenerator = NextTokenGenerator('generated_datasets/modular_characters_small',
                                    output_image_size=128,
                                    save_size=500)
    
    TokenGenerator.generate('generated_datasets/kenney_new/characters_sequences.json',
                       resume=False)
    