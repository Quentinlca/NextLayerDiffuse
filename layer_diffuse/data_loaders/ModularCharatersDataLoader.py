import torch
from datasets import load_dataset
from torchvision import transforms
import os

class ModularCharactersDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset_name:str, split:str, image_size:int, batch_size:int=16, shuffle:bool=True) :
        if not os.path.exists('cache/datasets'):
            os.makedirs('cache/datasets', exist_ok=True)
        dataset = load_dataset(dataset_name, split=split)

        vocab = dataset['prompt'] # type: ignore
        vocab = list(dict.fromkeys(dataset['prompt'])) # type: ignore
        vocab = sorted(vocab) # type: ignore
        self.vocab = dict(zip(vocab, range(len(vocab))))
        
        preprocess = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        
        def transform(rows:dict)->dict:
            images_input = [preprocess(image) for image in rows['input']]
            images_target = [preprocess(image) for image in rows['target']]
            class_labels = [torch.tensor(self.vocab.get(prompt,-1),dtype=torch.long).unsqueeze(0) for prompt in rows['prompt']]
            return {'input': images_input,
                    'target': images_target,
                    'label': class_labels}
        dataset.set_transform(transform) # type: ignore
        
        return super().__init__(dataset,  # type: ignore
                                batch_size=batch_size,
                                shuffle=shuffle)
        
def get_modular_char_dataloader(dataset_name:str, split:str, image_size:int, batch_size:int=16, shuffle:bool=True):
    return ModularCharactersDataLoader(dataset_name=dataset_name,
                                       split=split,
                                       image_size=image_size,
                                       batch_size=batch_size,
                                       shuffle=shuffle)