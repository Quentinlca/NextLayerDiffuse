from models import DDPMNextTokenV1
from data_loaders import ModularCharatersDataLoader
from PIL import Image
dataset_name = "QLeca/modular_charactersv2"
val_split = f"train[95%:100%]"
train_split = f"train[0%:95%]"


from safetensors import safe_open

# tensors = {}
# with safe_open("/Data/quentin.leca/training_output/diffusion_pytorch_model.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)
# print(tensors)

pipeline = DDPMNextTokenV1.DDPMNextTokenV1Pipeline()
train_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(dataset_name=dataset_name,
                                                                          split=train_split,
                                                                          image_size=pipeline.train_config.image_size,
                                                                          batch_size=pipeline.train_config.train_batch_size,
                                                                          shuffle=True)
# pipeline.load_config('/Data/quentin.leca/training_output/')
val_dataloader = ModularCharatersDataLoader.get_modular_char_dataloader(dataset_name=dataset_name,
                                                                          split=val_split,
                                                                          image_size=pipeline.train_config.image_size,
                                                                          batch_size=pipeline.train_config.eval_batch_size,
                                                                          shuffle=True)

pipeline.train(train_dataloader, val_dataloader)

# for batch in val_dataloader:
#     images = pipeline(batch['input'], batch['prompt'])
#     print(images.shape)
#     # Image.fromarray(images)
#     break