# This is a Next Layer Prediction using DDPM methods
# For training, we give it the previous layers and the target (noised) and expect the Unet to predict the noise

# For diffusion we give it the previous layers and random noise and want to get the next layer
import torch
from dataclasses import dataclass
import json

from diffusers.configuration_utils import ConfigMixin
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import notebook_launcher

 
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

import torch
import torchvision
import torch.nn.functional as F
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from typing import Tuple

IMAGE_SIZE = 128

@dataclass
class TrainingConfig:
    image_size = IMAGE_SIZE  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "training_output"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "QLeca/NextLayerModularCharacterModel"  # the name of the repository to create on the HF Hub
    hub_private_repo = None
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    repo_id = ''

class ModelConfig(dict):
    def __init__(self) -> None:
        self.config = {}
        self.config['sample_size'] = IMAGE_SIZE # The size of the input
        self.config['in_channels'] = 8 # 8 channels because 2 RGBA images concatenated
        self.config['out_channels'] = 4 
        self.config['down_block_types'] = (
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        )
        self.config['up_block_types'] = (
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
        self.config['block_out_channels'] = (128, 128, 256, 256, 512, 512)
        self.config['layers_per_block']=2
        # self.config['addition_embed_type'] = 'text'
        # self.config['class_embed_type'] = 'identity'
        # self.config['class_embeddings_concat'] = False
        
class SchedulerConfig(dict):
    def __init__(self) -> None:
        super().__init__()
        self.config = {}
        self.config['num_train_timesteps'] = 1000

class DDPMNextTokenV1Pipeline():
    def __init__(self):
        self.train_config = TrainingConfig()
        self.model_config = ModelConfig()
        self.scheduler_config = SchedulerConfig()
        self.unet = UNet2DConditionModel(**self.model_config.config)
        self.scheduler=DDPMScheduler(**self.scheduler_config.config)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @torch.no_grad()
    def __call__(self, input_images: torch.Tensor, prompts, num_inference_steps: int = 50):
        xt = torch.randn((input_images.shape[0],
                          self.model_config.config['out_channels'],
                          self.train_config.image_size,
                          self.train_config.image_size)
                         ).to(self.device)
        input_images = input_images.to(self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in tqdm(self.scheduler.timesteps.numpy()):
            # Get prediction of noise
            noisy_samples = torch.concat([input_images, xt], dim=1).to(self.device)
            noise_pred = self.unet(sample=noisy_samples, 
                                   timestep=t,
                                   encoder_hidden_states=torch.zeros([noisy_samples.shape[0],32,1280]).to(self.device)).sample
            
            # Use scheduler to get x0 and xt-1
            xt = self.scheduler.step(noise_pred, t, xt, return_dict=False)[0]
            # Save x0
        images = (xt / 2 + 0.5).clamp(0, 1)
        # images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
            
    @torch.no_grad()
    def save_training_samples(self, input_images, target_images, prompts, epoch:int, num_inference_steps: int = 50, ):
        xt = torch.randn((input_images.shape[0],
                          self.model_config.config['out_channels'],
                          self.train_config.image_size,
                          self.train_config.image_size)
                         ).to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        input_images = input_images.to(self.device)
        for t in tqdm(self.scheduler.timesteps.numpy()):
            # Get prediction of noise
            noisy_samples = torch.concat([input_images, xt], dim=1)
            noise_pred = self.unet(sample=noisy_samples, 
                                   timestep=t,
                                   encoder_hidden_states=torch.zeros([noisy_samples.shape[0],32,1280]).to(self.device)).sample
            
            # Use scheduler to get x0 and xt-1
            xt = self.scheduler.step(noise_pred, t, xt, return_dict=False)[0]
            # Save x0
            images = (xt / 2 + 0.5).clamp(0, 1)
            images = images.cpu()
            targets = target_images.cpu()
            concat = torch.concat([images, targets],dim=0)
            grid = make_grid(concat)
            img = torchvision.transforms.ToPILImage()(grid)
            if not os.path.exists(os.path.join(self.train_config.output_dir, 'samples_epoch_{}'.format(epoch))):
                os.mkdir(os.path.join(self.train_config.output_dir, 'samples_epoch_{}'.format(epoch)))
            img.save(os.path.join(self.train_config.output_dir, 'samples_epoch_{}'.format(epoch), 'x0_{}.png'.format(t)))
            img.close()

    def train_accelerate(self, train_dataloader, val_dataloader):


        args = (train_dataloader, val_dataloader)

        notebook_launcher(self.train, args, num_processes=1)

    def train(self, train_dataloader, val_dataloader):

        # Initialize accelerator and tensorboard logging
        
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.train_config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
                                                        optimizer=optimizer,
                                                        num_warmup_steps=self.train_config.lr_warmup_steps,
                                                        num_training_steps=(len(train_dataloader) * self.train_config.num_epochs),
                                                    )
        
        accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.train_config.output_dir, "logs")
        )
        
        if accelerator.is_main_process:
            if self.train_config.output_dir is not None:
                os.makedirs(self.train_config.output_dir, exist_ok=True)
            if self.train_config.push_to_hub:
                self.train_config.repo_id = create_repo(
                    repo_id=self.train_config.hub_model_id or Path(self.train_config.output_dir).name, exist_ok=True
                ).repo_id
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        self.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        # Now you train the model
        for epoch in range(self.train_config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                input_images = batch["input"].to(self.device)
                target_images = batch["target"].to(self.device)
                prompts = batch['prompt']
                # Sample noise to add to the images
                noise = torch.randn(target_images.shape, device=self.device)
                bs = target_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.scheduler.config['num_train_timesteps'], (bs,), device=self.device,
                    dtype=torch.int
                )
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_targets = self.scheduler.add_noise(target_images, noise, timesteps) # type: ignore

                with accelerator.accumulate(self.unet):
                    # Predict the noise residual
                    noisy_samples = torch.concat([input_images, noisy_targets], dim=1)
                    noise_pred = self.unet(sample=noisy_samples, 
                                           timestep=timesteps, 
                                           encoder_hidden_states=torch.zeros([noisy_samples.shape[0],32,1280]).to(self.device)).sample
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:

                if (epoch + 1) % self.train_config.save_image_epochs == 0 or epoch == self.train_config.num_epochs - 1:
                    for batch in val_dataloader:
                        self.save_training_samples(input_images=batch['input'].to(self.device),
                                                   target_images=batch['target'].to(self.device),
                                                   prompts=batch['prompt'],
                                                   epoch=epoch)
                        break
                
                if (epoch + 1) % self.train_config.save_model_epochs == 0 or epoch == self.train_config.num_epochs - 1:
                    if self.train_config.push_to_hub:
                        upload_folder(
                            repo_id=self.train_config.repo_id,
                            folder_path=self.train_config.output_dir,
                            commit_message=f"Epoch {epoch}",
                            ignore_patterns=["samples_*", "epoch_*"],
                        )
                    else:
                        self.unet.save_pretrained(self.train_config.output_dir)

    def load_config(self, model_dir):
        # config = json.loads(config_path)
        self.unet.from_pretrained(model_dir)
        self.unet.to(self.device) # type: ignore
        print('Model Loaded')
        # if type(model) == Tuple:
        #     self.unet = model[0]
        #     self.unet.from_pretrained(checkpoint_path)
        #     print('Model loaded')
        # if type(model) == UNet2DConditionModel:
        #     self.unet = model
        #     self.unet.from_pretrained(checkpoint_path)
        #     print('Model loaded')
            