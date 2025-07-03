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
    
    def generate_hair_sequence(self, sequence_dataset_path:str, resume:bool=True):
        # Load the path sequence dataset
        dataset = json.load(open(sequence_dataset_path,mode='r'))
        print(f'Loaded dataset with {len(dataset)} sequences of {len(dataset[0])} modules ...')
        
        rows = [] 
        last_char_generated = 0

        if resume:
            target = pd.read_csv(self.dataset_path)['Target'].to_list()
            last_char_generated = int(os.path.basename(target[-1]).split('-')[0].split('_')[1])
            print(f"Resuming to character number {last_char_generated} ...")

        for char_id, character in tqdm(enumerate(dataset, start = last_char_generated), initial=last_char_generated, desc="Generating sequence", total=len(dataset), miniters=10, unit='char'):  
            sub_dir_input_number = (char_id*2)//MAX_FILES_PER_DIR
            sub_dir_target_number = (char_id*2+1)//MAX_FILES_PER_DIR
            
            input_image_number = '0'*(FILE_NUMBER_LENGHT-len(str(char_id+1)))+str(char_id+1)
            
            input_image_name = f'char_{input_image_number}_input.png'
            target_image_name = f'char_{input_image_number}_target.png'
            
            output_path_input = f'{self.images_output_dir}/{sub_dir_input_number}/{input_image_name}'
            output_path_target = f'{self.images_output_dir}/{sub_dir_target_number}/{target_image_name}'
            
            input_image, _ = merge_composents(character[:-1],
                                           output_path=output_path_input, 
                                           save=True, 
                                           output_size=self.image_size)
            
            target_image = add_component(base_image=input_image,
                                           component_path=character[-1], 
                                           output_path=output_path_target, 
                                           output_image_size=self.image_size)
            
            prompt = get_class_from_path(character[-1])
            
            rows.append([output_path_input, output_path_target, prompt])
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
        
    def generate_dataset_smarter(self, assets_dir:str):
        def get_next_layer(input_image, component_path):
                output_image = add_component(
                    base_image=input_image,
                    component_path=component_path,
                    output_image_size=self.image_size,
                    background=True,
                    save=False
                )
                prompt = get_class_from_path(component_path)
                return input_image, output_image, prompt
            
        rows = []
        blank_image = merge_composents(modules_paths=[],
                                    output_size=self.image_size,
                                    save=False,
                                    background=True)[0]

        arms_L = [f'{assets_dir}/Arm_L/{f}' for f in os.listdir(f'{assets_dir}/Arm_L') if f.endswith('.png')]
        hairs = [f'{assets_dir}/Hair/{f}' for f in os.listdir(f'{assets_dir}/Hair') if f.endswith('.png')][0:15]
        faces = [f'{assets_dir}/Face/{f}' for f in os.listdir(f'{assets_dir}/Face') if f.endswith('.png')][0:1]
        shoes = [f'{assets_dir}/Shoes_L/{f}' for f in os.listdir(f'{assets_dir}/Shoes_L') if f.endswith('.png')][0:1]
        pants_L = [f'{assets_dir}/Pants_L/{f}' for f in os.listdir(f'{assets_dir}/Pants_L') if f.endswith('.png')]
        shirts_L = [f'{assets_dir}/Shirt_L/{f}' for f in os.listdir(f'{assets_dir}/Shirt_L') if f.endswith('.png')]
        shirts = [f'{assets_dir}/Shirt/{f}' for f in os.listdir(f'{assets_dir}/Shirt') if f.endswith('.png') and f.split('.')[0].split('_')[-1] == '1']

        for arm_L in tqdm(arms_L, desc='Body'):
            tint = os.path.basename(arm_L).split('_')[0]
            head = f'{assets_dir}/Head/{tint}_head.png'
            arm_R = f'{assets_dir}/Arm_R/{tint}_arm.png'
            neck = f'{assets_dir}/Neck/{tint}_neck.png'
            hand_L = f'{assets_dir}/Hand_L/{tint}_hand.png'
            hand_R = f'{assets_dir}/Hand_R/{tint}_hand.png'
            leg_L = f'{assets_dir}/Leg_L/{tint}_leg.png'
            leg_R = f'{assets_dir}/Leg_R/{tint}_leg.png'
            # 1 Nothing -> Left Arm
            input_image, output_image, prompt = get_next_layer(blank_image, arm_L)
            rows.append([input_image, output_image, prompt])
            # 2 Left Arm -> Right Arm
            input_image, output_image, prompt = get_next_layer(output_image, arm_R)
            rows.append([input_image, output_image, prompt])
            # 3 Right Arm -> Neck
            input_image, output_image, prompt = get_next_layer(output_image, neck)
            rows.append([input_image, output_image, prompt])
            # 4 Neck -> Head
            input_image, output_image, prompt = get_next_layer(output_image, head)
            rows.append([input_image, output_image, prompt])
            # 5 Head -> Hand_L
            input_image, output_image, prompt = get_next_layer(output_image, hand_L)
            rows.append([input_image, output_image, prompt])
            # 6 Hand_L -> Hand_R
            input_image, output_image, prompt = get_next_layer(output_image, hand_R)
            rows.append([input_image, output_image, prompt])
            for shirt in tqdm(shirts, desc='Shirts'):
                # 7 Hand_R -> Shirt
                input_image, output_image, prompt = get_next_layer(output_image, shirt)
                rows.append([input_image, output_image, prompt])
                shirt_color = os.path.basename(shirt).split('_')[0][:-5]
                shirts_L = [f'{assets_dir}/Shirt_L/{f}' for f in os.listdir(f'{assets_dir}/Shirt_L') if f.endswith('.png') and shirt_color in f]
                for shirt_L in shirts_L:
                    shirt_R = f'{assets_dir}/Shirt_R/{os.path.basename(shirt_L)}'
                    # 8 Shirt -> Shirt L
                    input_image, output_image, prompt = get_next_layer(output_image, shirt_L)
                    rows.append([input_image, output_image, prompt])
                    # 9 Shirt_L -> Shirt R
                    input_image, output_image, prompt = get_next_layer(output_image, shirt_R)
                    rows.append([input_image, output_image, prompt])
                    # 10 Shirt_R -> Leg_L
                    input_image, output_image, prompt = get_next_layer(output_image, leg_L)
                    rows.append([input_image, output_image, prompt])
                    # 11 Leg_L -> Leg_R
                    input_image, output_image, prompt = get_next_layer(output_image, leg_R)
                    rows.append([input_image, output_image, prompt])
                    for shoe_L in shoes:
                        shoe_R = f'{assets_dir}/Shoes_R/{os.path.basename(shoe_L)}'
                        # 12 Leg_R -> Shoe_L
                        input_image, output_image, prompt = get_next_layer(output_image, shoe_L)
                        rows.append([input_image, output_image, prompt])
                        # 13 Shoe_L -> Shoe_R
                        input_image, output_image, prompt = get_next_layer(output_image, shoe_R)
                        rows.append([input_image, output_image, prompt])
                        for pant_L in pants_L:
                            pant_R = f'{assets_dir}/Pants_R/{os.path.basename(pant_L)}'
                            pants_color = os.path.basename(pant_L).split('_')[0]
                            pants = [f'{assets_dir}/Pants/{f}' for f in os.listdir(f'{assets_dir}/Pants') if f.endswith('.png') and pants_color in f]
                            # 14 Shoe_R -> Pant_L
                            input_image, output_image, prompt = get_next_layer(output_image, pant_L)
                            rows.append([input_image, output_image, prompt])
                            # 15 Pant_L -> Pant_R
                            input_image, output_image, prompt = get_next_layer(output_image, pant_R)
                            rows.append([input_image, output_image, prompt])
                            for pant in pants:
                                # 16 Pant_R -> Pant
                                input_image, output_image, prompt = get_next_layer(output_image, pant)
                                rows.append([input_image, output_image, prompt])
                                for hair in hairs:
                                    # 17 Pant -> Hair
                                    input_image, output_image, prompt = get_next_layer(output_image, hair)
                                    rows.append([input_image, output_image, prompt])
                                    for face in faces:
                                        # 18 Hair -> Face
                                        input_image, output_image, prompt = get_next_layer(output_image, face)
                                        rows.append([input_image, output_image, prompt])

            df_batch = pd.DataFrame(rows, columns=['Input', 'Target', 'Prompt'])
            df_batch.to_csv(self.dataset_path, index=False)
            # TOTAL 6 layers * 8 colors tokens + 8*8 shirts + 64 * 3 variations * 2 layers + 384 * 2 legs = 

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
    def upload_dataset(dataset_path:str, repo_id:str, shuffle:bool=True):
        from datasets import Dataset
        from datasets import Features, Value, Image as HFImage
        
        features = Features({
            'input': HFImage(),
            'target': HFImage(),
            'prompt': Value('string'),
        })
        
        def convert_file_names(inputs, targets, prompts, dataset_dir):
            for input, target, prompt in zip(inputs, targets, prompts):
                # input = os.path.join(dataset_dir, input)
                # target = os.path.join(dataset_dir, target)
                if not os.path.exists(input) or not os.path.exists(target):
                    continue
                input_image = Image.open(input)
                target_image = Image.open(target)
                row = {'input': input_image,
                       'target': target_image,
                       'prompt': prompt}
                
                yield row
        
        df = pd.read_csv(dataset_path)
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        inputs = df['Input'].to_list()
        targets = df['Target'].to_list()
        prompts = df['Prompt'].to_list()
        
        dataset_to_hub = Dataset.from_generator(lambda: convert_file_names(inputs, targets, prompts, os.path.dirname(dataset_path)),
                                                features=features, 
                                                split='train') # type: ignore
        dataset_to_hub.push_to_hub(repo_id=repo_id, split='train') # type: ignore
    
