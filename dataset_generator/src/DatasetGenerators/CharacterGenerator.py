import json
import PIL.Image
from tqdm import tqdm
import os
import numpy as np
from ..utils.ModularCharactersUtils import *
from typing import Tuple


class CharacterGenerator:
    def __init__(self, assets_dir:str, output_dir:str) -> None:
        assert os.path.exists(assets_dir)
        self.assets_dir = assets_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        pass
    
    def generate_small_dataset(self, output_path:str|None=None, save=True) -> Tuple[list[list[str]], str|None]:
        if not output_path and save:
            output_path = f'{self.output_dir}/characters_sequences.json'
        
        characters = []
        
        heads = [f'{self.assets_dir}/Head/{f}' for f in os.listdir(f'{self.assets_dir}/Head') if f.endswith('.png')] # Only 1 tint
        hairs = [f'{self.assets_dir}/Hair/{f}' for f in os.listdir(f'{self.assets_dir}/Hair') if f.endswith('Man1.png')] 
        faces = [f'{self.assets_dir}/Face/{f}' for f in os.listdir(f'{self.assets_dir}/Face') if f.endswith('.png')][0:1] # Only 1 face
        shoes = [f'{self.assets_dir}/Shoes_L/{f}' for f in os.listdir(f'{self.assets_dir}/Shoes_L') if f.endswith('.png')][0:1] # Only 1 shoe
        pants_L = [f'{self.assets_dir}/Pants_L/{f}' for f in os.listdir(f'{self.assets_dir}/Pants_L') if f.endswith('.png') and (f.startswith('pantsBlue1') or f.startswith('pantsRed'))] 
        shirts_L = [f'{self.assets_dir}/Shirt_L/{f}' for f in os.listdir(f'{self.assets_dir}/Shirt_L') if f.endswith('.png') and (f.startswith('blue') or f.startswith('red'))]
        
        for head in tqdm(heads, desc='Heads :'):
            tint = os.path.basename(head).split('_')[0]
            arm_L = f'{self.assets_dir}/Arm_L/{tint}_arm.png'
            arm_R = f'{self.assets_dir}/Arm_R/{tint}_arm.png'
            neck = f'{self.assets_dir}/Neck/{tint}_neck.png'
            hand_L = f'{self.assets_dir}/Hand_L/{tint}_hand.png'
            hand_R = f'{self.assets_dir}/Hand_R/{tint}_hand.png'
            leg_L = f'{self.assets_dir}/Leg_L/{tint}_leg.png'
            leg_R = f'{self.assets_dir}/Leg_R/{tint}_leg.png'
            
            for hair in tqdm(hairs, desc='Hairs :'):
                for face in faces:
                    for shoe_L in shoes:
                        shoe_R = f'{self.assets_dir}/Shoes_R/{os.path.basename(shoe_L)}'
                        
                        for pant_L in pants_L:
                            pant_R = f'{self.assets_dir}/Pants_R/{os.path.basename(pant_L)}'
                            pants_color = os.path.basename(pant_L).split('_')[0]
                            pants = [f'{self.assets_dir}/Pants/{f}' for f in os.listdir(f'{self.assets_dir}/Pants') if f.endswith('.png') and pants_color in f][0:1]
                            for pant in pants:
                                for shirt_L in shirts_L:
                                    shirt_R = f'{self.assets_dir}/Shirt_R/{os.path.basename(shirt_L)}'
                                    shirt_color = os.path.basename(shirt_L).split('_')[0][:-3]
                                    shirts = [f'{self.assets_dir}/Shirt/{f}' for f in os.listdir(f'{self.assets_dir}/Shirt') if f.endswith('.png') and shirt_color in f][0:1]
                                    for shirt in shirts:
                                        image_paths = [head, arm_L, arm_R, neck, leg_L, leg_R, hand_L, hand_R, pant, pant_L, pant_R, shirt, shirt_L, shirt_R, shoe_L, shoe_R, hair, face]
                                        image_paths = sort_paths_by_order(image_paths)
                                        characters.append(image_paths)
        if save and output_path:                          
            with open(output_path,mode='w') as f:
                json.dump(characters,f, indent=4)
            f.close()
        return characters, output_path
    
    def generate_random_character(self, output_path: str|None = None, logic=True, save=True) -> Tuple[Image.Image, list[str], str|None]:
        """
        Generates a random character by merging components from the assets directory.
        """
        if not output_path:
            if not os.path.exists(f'{self.output_dir}/random_chars'):
                os.mkdir(f'{self.output_dir}/random_chars')
            output_path = f'{self.output_dir}/random_chars/random_char.png'
            
        char_path_seq = []
        classes = [d for d in os.listdir(self.assets_dir) if os.path.isdir(os.path.join(self.assets_dir,d))]

        if logic:
            head = np.random.choice([f'{self.assets_dir}/Head/{f}' for f in os.listdir(f'{self.assets_dir}/Head') if f.endswith('.png')])
            tint = os.path.basename(head).split('_')[0]
            arm_L = f'{self.assets_dir}/Arm_L/{tint}_arm.png'
            arm_R = f'{self.assets_dir}/Arm_R/{tint}_arm.png'
            neck = f'{self.assets_dir}/Neck/{tint}_neck.png'
            hand_L = f'{self.assets_dir}/Hand_L/{tint}_hand.png'
            hand_R = f'{self.assets_dir}/Hand_R/{tint}_hand.png'
            leg_L = f'{self.assets_dir}/Leg_L/{tint}_leg.png'
            leg_R = f'{self.assets_dir}/Leg_R/{tint}_leg.png'
            
            pants_L = np.random.choice([f'{self.assets_dir}/Pants_L/{f}' for f in os.listdir(f'{self.assets_dir}/Pants_L') if f.endswith('.png')])
            pants_R = f'{self.assets_dir}/Pants_R/{os.path.basename(pants_L)}'
            pants_color = os.path.basename(pants_L).split('_')[0]
            pants = np.random.choice([f'{self.assets_dir}/Pants/{f}' for f in os.listdir(f'{self.assets_dir}/Pants') if f.endswith('.png') and pants_color in f])
            
            shirt_L = np.random.choice([f'{self.assets_dir}/Shirt_L/{f}' for f in os.listdir(f'{self.assets_dir}/Shirt_L') if f.endswith('.png')])
            shirt_R = f'{self.assets_dir}/Shirt_R/{os.path.basename(shirt_L)}'
            shirt_color = os.path.basename(shirt_L).split('_')[0][:-3]
            shirt = np.random.choice([f'{self.assets_dir}/Shirt/{f}' for f in os.listdir(f'{self.assets_dir}/Shirt') if f.endswith('.png') and shirt_color in f])
            
            shoe_L = np.random.choice([f'{self.assets_dir}/Shoes_L/{f}' for f in os.listdir(f'{self.assets_dir}/Shoes_L') if f.endswith('.png')])
            shoe_R = f'{self.assets_dir}/Shoes_R/{os.path.basename(shoe_L)}'
            
            hair = np.random.choice([f'{self.assets_dir}/Hair/{f}' for f in os.listdir(f'{self.assets_dir}/Hair') if f.endswith('.png')])
            face = np.random.choice([f'{self.assets_dir}/Face/{f}' for f in os.listdir(f'{self.assets_dir}/Face') if f.endswith('.png')])
            char_path_seq = [head, arm_L, arm_R, neck, leg_L, leg_R, hand_L, hand_R, pants, pants_L, pants_R, shirt, shirt_L, shirt_R, shoe_L, shoe_R, hair, face]
        else: 
            for class_name in classes:
                class_dir = f'{self.assets_dir}/{class_name}'
                if os.path.isdir(class_dir):
                    images = [f'{class_dir}/{f}' for f in os.listdir(class_dir) if f.endswith('.png')]
                    if images:
                        char_path_seq.append(np.random.choice(images))  # Take the first image from each class directory
                        
        assert len(char_path_seq) == len(classes)                
        merged_image, saved = merge_composents(modules_paths=char_path_seq, output_path=output_path, save=save)
        return merged_image, char_path_seq, output_path
      
    def generate_char_path_seq(self, output_path:str|None=None, save=True) -> Tuple[list[list[str]], str|None]:
        if not output_path and save:
            output_path = f'{self.output_dir}/characters_sequences.json'
        
        characters = []
        
        heads = [f'{self.assets_dir}/Head/{f}' for f in os.listdir(f'{self.assets_dir}/Head') if f.endswith('.png')]
        hairs = [f'{self.assets_dir}/Hair/{f}' for f in os.listdir(f'{self.assets_dir}/Hair') if f.endswith('.png')][0:15]
        faces = [f'{self.assets_dir}/Face/{f}' for f in os.listdir(f'{self.assets_dir}/Face') if f.endswith('.png')][0:1]
        shoes = [f'{self.assets_dir}/Shoes_L/{f}' for f in os.listdir(f'{self.assets_dir}/Shoes_L') if f.endswith('.png')][0:1]
        pants_L = [f'{self.assets_dir}/Pants_L/{f}' for f in os.listdir(f'{self.assets_dir}/Pants_L') if f.endswith('.png')]
        shirts_L = [f'{self.assets_dir}/Shirt_L/{f}' for f in os.listdir(f'{self.assets_dir}/Shirt_L') if f.endswith('.png')]
        
        for head in tqdm(heads, desc='Heads :'):
            tint = os.path.basename(head).split('_')[0]
            arm_L = f'{self.assets_dir}/Arm_L/{tint}_arm.png'
            arm_R = f'{self.assets_dir}/Arm_R/{tint}_arm.png'
            neck = f'{self.assets_dir}/Neck/{tint}_neck.png'
            hand_L = f'{self.assets_dir}/Hand_L/{tint}_hand.png'
            hand_R = f'{self.assets_dir}/Hand_R/{tint}_hand.png'
            leg_L = f'{self.assets_dir}/Leg_L/{tint}_leg.png'
            leg_R = f'{self.assets_dir}/Leg_R/{tint}_leg.png'
            
            for hair in tqdm(hairs, desc='Hairs :'):
                for face in faces:
                    for shoe_L in shoes:
                        shoe_R = f'{self.assets_dir}/Shoes_R/{os.path.basename(shoe_L)}'
                        
                        for pant_L in pants_L:
                            pant_R = f'{self.assets_dir}/Pants_R/{os.path.basename(pant_L)}'
                            pants_color = os.path.basename(pant_L).split('_')[0]
                            pants = [f'{self.assets_dir}/Pants/{f}' for f in os.listdir(f'{self.assets_dir}/Pants') if f.endswith('.png') and pants_color in f]
                            for pant in pants:
                                for shirt_L in shirts_L:
                                    shirt_R = f'{self.assets_dir}/Shirt_R/{os.path.basename(shirt_L)}'
                                    shirt_color = os.path.basename(shirt_L).split('_')[0][:-3]
                                    shirts = [f'{self.assets_dir}/Shirt/{f}' for f in os.listdir(f'{self.assets_dir}/Shirt') if f.endswith('.png') and shirt_color in f][0:1]
                                    for shirt in shirts:
                                        image_paths = [head, arm_L, arm_R, neck, leg_L, leg_R, hand_L, hand_R, pant, pant_L, pant_R, shirt, shirt_L, shirt_R, shoe_L, shoe_R, hair, face]
                                        image_paths = sort_paths_by_order(image_paths)
                                        characters.append(image_paths)
        if save and output_path:                          
            with open(output_path,mode='w') as f:
                json.dump(characters,f, indent=4)
            f.close()
        return characters, output_path
          