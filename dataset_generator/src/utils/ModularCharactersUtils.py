import os
import json
from PIL import Image
from enum import StrEnum
from typing import Tuple

MODULES_ORDER = [
    'Arm_L',
    'Arm_R',
    'Neck',
    'Head',
    'Hand_L',
    'Hand_R',
    'Shirt',
    'Shirt_L',
    'Shirt_R',
    'Leg_L',
    'Leg_R',
    'Shoes_L',
    'Shoes_R',
    'Pants_L',
    'Pants_R',
    'Pants',
    'Face',
    'Hair'
]

MODULES_OFFSETS = {
    "Hair": {
        "standard": [
            220,
            29
        ],
        "Man2": [
            220,
            22
        ],
        "Man8": [
            220,
            99
        ],
        "Woman3": [
            221,
            29
        ],
        "Woman4": [
            216,
            29
        ],
        "Woman5": [
            214,
            29
        ],
        "Woman6": [
            214,
            29
        ]
    },
    "Neck": {
        "standard": [
            250,
            189
        ]
    },
    "Head": {
        "standard": [
            211,
            44
        ]
    },
    "Shirt": {
        "standard": [
            222,
            197
        ],
        "4": [
            219,
            197
        ]
    },
    "Arm_R": {
        "standard": [
            341,
            197
        ]
    },
    "Arm_L": {
        "standard": [
            86,
            197
        ]
    },
    "Shirt_L": {
        "standard": [
            131,
            198
        ],
        "short": [
            131,
            198
        ],
        "long": [
            87,
            197
        ],
        "shorter": [
            182,
            198
        ]
    },
    "Shirt_R": {
        "standard": [
            348,
            198
        ],
        "short": [
            348,
            198
        ],
        "long": [
            339,
            197
        ],
        "shorter": [
            344,
            199
        ]
    },
    "Pants": {
        "standard": [
            222,
            364
        ]
    },
    "Hand_R": {
        "standard": [
            467,
            294
        ]
    },
    "Hand_L": {
        "standard": [
            72,
            294
        ]
    },
    "Leg_R": {
        "standard": [
            307,
            384
        ]
    },
    "Leg_L": {
        "standard": [
            198,
            384
        ]
    },
    "Pants_L": {
        "standard": [
            195,
            380
        ],
        "shorter": [
            198,
            380
        ],
        "long": [
            191,
            380
        ]
    },
    "Pants_R": {
        "standard": [
            296,
            380
        ]
    },
    "Face": {
        "standard": [
            250,
            89
        ],
        "1": [
            250,
            89
        ],
        "2": [
            244,
            94
        ],
        "3": [
            242,
            109
        ],
        "4": [
            242,
            104
        ]
    },
    "Shoes_L": {
        "standard": [
            162,
            529
        ]
    },
    "Shoes_R": {
        "standard": [
            342,
            529
        ]
    }
}

class CharModule(StrEnum):
    Arm_L = 'Arm_L'
    Arm_R = 'Arm_R'
    Face = 'Face'
    Hair = 'Hair'
    Hand_L = 'Hand_L'
    Hand_R = 'Hand_R'
    Head = 'Head'
    Leg_L = 'Leg_L'
    Leg_R = 'Leg_R'
    Neck = 'Neck'
    Pants = 'Pants'
    Pants_L = 'Pants_L'
    Pants_R = 'Pants_R'
    Shirt = 'Shirt'
    Shirt_L = 'Shirt_L'
    Shirt_R = 'Shirt_R'
    Shoes_L = 'Shoes_L'
    Shoes_R = 'Shoes_R'
    Unknown = 'Unknown'
    

def sort_paths_by_order(paths:list[str]) -> list[str]:
    """
    Sorts the list of paths based on the predefined order.
    """
    order_dict = {name: index for index, name in enumerate(MODULES_ORDER)}
    return sorted(paths, key=lambda path: order_dict.get(path.split('/')[-2], float('inf')))


def get_class_from_path(path:str)->str:
    """
    Get the class of a module thanks to its path. The stucture of the dataset should be :
    asset_dir
    |
    ├── class1_name
    |   ├── module1.png
    |   ├── module2.png
    ⁝           ⁝
    ├── class2_name
    ⁝
    
    """
    class_name = path.split('/')[-2]
    basename = os.path.basename(path).split('.')[0]
    if class_name == 'Arm_L':
        return f'{basename.split("_")[0].capitalize()} Left Arm'
    if class_name == 'Arm_R':
        return f'{basename.split("_")[0].capitalize()} Right Arm'
    if class_name == 'Head':
        return f'{basename.split("_")[0].capitalize()} Head'
    if class_name == 'Neck':
        return f'{basename.split("_")[0].capitalize()} Neck'
    if class_name == 'Leg_L':
        return f'{basename.split("_")[0].capitalize()} Left Leg'
    if class_name == 'Leg_R':
        return f'{basename.split("_")[0].capitalize()} Right Leg'
    if class_name == 'Hand_L':
        return f'{basename.split("_")[0].capitalize()} Left Hand'
    if class_name == 'Hand_R':
        return f'{basename.split("_")[0].capitalize()} Right Hand'
    if class_name == 'Shirt':
        return f'{basename.split("_")[0][:-5].capitalize()} Shirt'
    if class_name == 'Shirt_L':
        color, size = basename.split("_")
        return f'{color[:-3].capitalize()} {size.capitalize()} Left Shirt'
    if class_name == 'Shirt_R':
        color, size = basename.split("_")
        return f'{color[:-3].capitalize()} {size.capitalize()} Right Shirt'
    if class_name == 'Shoes_L':
        return f'{basename.split(".")[0][:-5].capitalize()} Left Shoe'
    if class_name == 'Shoes_R':
        return f'{basename.split(".")[0][:-5].capitalize()} Right Shoe'
    if class_name == 'Pants_L':
        color, size = basename.split("_")
        return f'{color[5:].capitalize()} {size.capitalize()} Left Pant'
    if class_name == 'Pants_R':
        color, size = basename.split("_")
        return f'{color[5:].capitalize()} {size.capitalize()} Right Pant'
    if class_name == 'Pants':
        return f'{basename[5:].capitalize()} Pants'
    if class_name == 'Face':
        return 'Face'
    if class_name == 'Hair':
        color, style = basename.split('_')
        return f'{color.capitalize()} {style.capitalize()} Hair'
    return basename


def merge_composents(modules_paths: list[str], output_path:str|None=None, save=False, output_size = None) -> Tuple[Image.Image, bool]:    
    save = save and bool(output_path)
    
    modules_paths = sort_paths_by_order(modules_paths)
    images = [Image.open(path) for path in modules_paths]
    # Calculate the width and height of the merged image
    total_width = 600
    total_height = 600

    # Create a new image with the appropriate size
    merged_image = Image.new('RGBA', (total_width, total_height))

    # Paste each image into the merged image
    for i, img in enumerate(images):
        class_name = modules_paths[i].split('/')[-2]
        special_type = os.path.basename(modules_paths[i]).split('.')[0].split('_')[-1]
        try:
            class_offsets = MODULES_OFFSETS.get(class_name, {'standard':(0, 0)})
            x_offset, y_offset = class_offsets.get(special_type, class_offsets['standard'])
        except Exception as e:
            print(f"Error with the offset dict (should contain a 'standard' value for each class) : {e}")
            x_offset, y_offset = 0, 0
            continue
        merged_image.alpha_composite(img, (x_offset, y_offset))
    # Save the merged image
    
    if output_size != None:
        merged_image = merged_image.resize((output_size,output_size))
    if save and output_path:
        try:
            merged_image.save(output_path)
        except Exception as e:
            save = False
        
    return merged_image, save


        
    