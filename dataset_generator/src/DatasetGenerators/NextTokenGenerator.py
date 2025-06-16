import os
from ..utils.ModularCharactersUtils import *
from tqdm import tqdm
import pandas as pd

MAX_FILES_PER_DIR = 5000
FILE_NUMBER_LENGHT = 7

class NextTokenGenerator:
    def __init__(self, output_dir:str, output_image_size:int = 128, save_size:int=1000) -> None:
        # OUTPUT_DIR
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subdirectories for images
        self.dataset_path = f'{self.output_dir}/dataset.csv'
        self.images_output_dir = f'{self.output_dir}/data'
        if not os.path.exists(self.images_output_dir):
            os.mkdir(self.images_output_dir)
        
        # OUTPUT_IMAGE_SIZE is the size of the output images
        self.image_size = output_image_size
        # SAVE_SIZE is the number of images to save before writing to the CSV file
        self.save_size = save_size
    
    def generate(self, sequence_dataset_path: str, resume=True):
        # Load the path sequence dataset
        dataset = json.load(open(sequence_dataset_path,mode='r'))
        print(f'Loaded dataset with {len(dataset)} sequences of {len(dataset[0])} modules ...')
        
        rows = []
        last_char_generated = 0
        
        # Create the blank image
        blank_image_name = f"char_{'0'*FILE_NUMBER_LENGHT}-layer_0.png"
        blank_image_path = f'{self.images_output_dir}/0/{blank_image_name}'
        if not os.path.exists(os.path.dirname(blank_image_path)):
            os.makedirs(os.path.dirname(blank_image_path))
        Image.new('RGBA', (self.image_size, self.image_size), (255, 255, 255, 0)).save(blank_image_path)

        if resume:
            target = pd.read_csv(self.dataset_path)['Target'].to_list()
            last_char_generated = int(os.path.basename(target[-1]).split('-')[0].split('_')[1])
            print(f"Resuming to character number {last_char_generated} ...")

        for char_id, character in tqdm(enumerate(dataset, start = last_char_generated), initial=last_char_generated, desc="Generating sequence", total=len(dataset),miniters=10):  
            output_paths = NextTokenGenerator.get_output_path_list(char_id=char_id, layer_ids=list(range(1, len(character) + 1)), sequence_lenght=len(character), output_dir=self.images_output_dir)
            rows += generate_sequence(sequence_paths=character, 
                                      blank_path=blank_image_path, 
                                      output_paths=output_paths, 
                                      output_image_size=self.image_size) 
                
            if char_id%self.save_size == 0:
                print(f'Saving the dataset ({char_id}/{len(dataset)})')
                df_batch = pd.DataFrame(rows, columns=['Input', 'Target', 'Prompt'])
                if char_id > 0:
                    existing_df = pd.read_csv(self.dataset_path)
                    df_batch = pd.concat([existing_df,df_batch], ignore_index=True)
                df_batch.to_csv(self.dataset_path,index=False) 
                rows = []
                
        print(f'Saving the dataset (COMPLETE)')
        df_batch = pd.DataFrame(rows, columns=['Input', 'Target', 'Prompt'])
        existing_df = pd.read_csv(self.dataset_path)
        df_batch = pd.concat([existing_df,df_batch], ignore_index=True)
        df_batch.to_csv(self.dataset_path,index=False) 
        rows = []
    
    @staticmethod
    def get_output_path(char_id:int, layer_id:int, sequence_lenght:int, output_dir:str)->str:
        file_number = (char_id)*sequence_lenght + layer_id + 1
        sub_dir_number = file_number//MAX_FILES_PER_DIR
        char_number = '0'*(FILE_NUMBER_LENGHT-len(str(char_id+1)))+str(char_id+1)
        output_name = f'char_{char_number}-layer_{layer_id}.png'
        output_path = f'{output_dir}/{sub_dir_number}/{output_name}'
        return output_path
    
    @staticmethod
    def get_output_path_list(char_id:int, layer_ids:list[int], sequence_lenght:int, output_dir:str)->list[str]:
        output_paths = []
        for layer_id in layer_ids:
            file_number = (char_id)*sequence_lenght + layer_id + 1
            sub_dir_number = file_number//MAX_FILES_PER_DIR
            char_number = '0'*(FILE_NUMBER_LENGHT-len(str(char_id+1)))+str(char_id+1)
            output_name = f'char_{char_number}-layer_{layer_id}.png'
            output_path = f'{output_dir}/{sub_dir_number}/{output_name}'
            output_paths.append(output_path)
        return output_paths

    @staticmethod
    def upload_dataset(dataset_path:str, repo_id:str):
        from datasets import Dataset
        from datasets import Features, Value, Image as HFImage
        
        features = Features({
            'input': HFImage(),
            'target': HFImage(),
            'prompt': Value('string'),
        })
        
        def convert_file_names(inputs, targets, prompts, dataset_dir):
            for input, target, prompt in zip(inputs, targets, prompts):
                input = os.path.join(dataset_dir, input)
                target = os.path.join(dataset_dir, target)
                if not os.path.exists(input) or not os.path.exists(target):
                    continue
                input_image = Image.open(input)
                target_image = Image.open(target)
                row = {'input': input_image,
                       'target': target_image,
                       'prompt': prompt}
                
                yield row
        
        df = pd.read_csv(dataset_path)
        inputs = df['Input'].to_list()
        targets = df['Target'].to_list()
        prompts = df['Prompt'].to_list()
        
        dataset_to_hub = Dataset.from_generator(lambda: convert_file_names(inputs, targets, prompts, os.path.dirname(dataset_path)),
                                                features=features, 
                                                split='train') # type: ignore
        dataset_to_hub.push_to_hub(repo_id=repo_id, split='train') # type: ignore
    
