import os
from ..utils.ModularCharactersUtils import *
from tqdm import tqdm
import pandas as pd

MAX_FILES_PER_DIR = 1000
FILE_NUMBER_LENGHT = 7

class NextTokenGenerator:
    def __init__(self, output_dir:str, output_image_size:int = 128, save_size:int=1000) -> None:
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.dataset_path = f'{self.output_dir}/next_token_dataset.csv'
        self.images_output_dir = f'{self.output_dir}/data'
        if not os.path.exists(self.images_output_dir):
            os.mkdir(self.images_output_dir)
        self.image_size = output_image_size
        self.save_size = save_size
        pass
    
    def generate(self, sequence_dataset_path: str, resume=True):
        dataset = json.load(open(sequence_dataset_path,mode='r'))
        print(f'Loaded dataset with {len(dataset)} sequences of {len(dataset[0])} modules ...')
        rows = []

        resume = True
        blank_image_name = f"char_{'0'*FILE_NUMBER_LENGHT}-layer_0.png"
        blank_image_path = f'{self.images_output_dir}/0/{blank_image_name}'
        if not os.path.exists(blank_image_path):
            _, blank_image_path= merge_composents([], output_path=blank_image_path, output_size=self.image_size, save=True)
        last_one = 0

        if resume:
            target = pd.read_csv(self.dataset_path)['Target'].to_list()
            last_one = int(os.path.basename(target[-1]).split('-')[0].split('_')[1])
            print(f"Resuming to character number {last_one} ...")

        for char_id, character in tqdm(enumerate(dataset, start = last_one), initial=last_one, desc="Generating sequence", total=len(dataset),miniters=10):  
            previous_path = blank_image_path
            for layer_id in range(1, len(character) + 1):
                output_path = self.get_output_path(char_id=char_id,layer_id=layer_id, sequence_lenght=len(character))
                # Check if output_name exists in any subdirectory of self.output_dir
                if not os.path.exists(output_path):
                    _, saved = merge_composents(character[:layer_id], output_path=output_path, save=True, output_size=self.image_size)
                row = [previous_path, output_path, get_class_from_path(character[layer_id-1])]
                rows.append(row)
                previous_path = output_path
                
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
        
    def get_output_path(self, char_id:int, layer_id:int, sequence_lenght:int)->str:
        file_number = (char_id)*sequence_lenght + layer_id + 1
        sub_dir_number = file_number//MAX_FILES_PER_DIR
        char_number = '0'*(FILE_NUMBER_LENGHT-len(str(char_id+1)))+str(char_id+1)
        output_name = f'char_{char_number}-layer_{layer_id}.png'
        output_path = f'{self.output_dir}/{sub_dir_number}/{output_name}'
        return output_path
    