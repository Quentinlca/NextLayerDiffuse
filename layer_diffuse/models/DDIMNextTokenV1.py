# This is a Next Layer Prediction using DDPM methods
# For training, we give it the previous layers and the target (noised) and expect the Unet to predict the noise

# For diffusion we give it the previous layers and random noise and want to get the next layer
import torch
from dataclasses import dataclass
import json

from diffusers.configuration_utils import ConfigMixin
from diffusers.optimization import get_cosine_schedule_with_warmup

from accelerate import notebook_launcher

from diffusers.models.unets.unet_2d import UNet2DModel
 
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
import huggingface_hub
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
import wandb
import wandb.util
import time
from torchmetrics.image.fid import FrechetInceptionDistance

IMAGE_SIZE = 128

@dataclass
class TrainingConfig:
    image_size = IMAGE_SIZE  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    sample_size = 8  # how many images to sample during training
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 100
    save_image_epochs = 1
    save_model_epochs = 3
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "training_outputs/DDIMNextTokenV1"  # the model name locally and on the HF Hub
    backup_output_dir = "training_outputs/DDIMNextTokenV1_backup"  # the model name locally and on the HF Hub if push_to_hub fails
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "QLeca/DDIMNextTokenV1"  # the name of the repository to create on the HF Hub
    wandb_project_name = "ddim-next-token-v1"  # the name of the project on Weights & Biases
    seed = 0

@dataclass
class InferenceConfig:
    num_inference_steps = 50  # number of denoising steps

class ModelConfig(dict):
    def __init__(self) -> None:
        self.config = {}
        self.config['sample_size'] = IMAGE_SIZE # The size of the input
        self.config['in_channels'] = 6 # 6 channels because 2 RGB images concatenated
        self.config['out_channels'] = 3 # 3 channels for the output image
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
        self.config['layers_per_block']= 2
        self.config['class_embed_type'] = 'identity'
        self.config['num_class_embeds'] = 0  # Number of class embeddings, set to 0 for no class conditioning
        
class SchedulerConfig(dict):
    def __init__(self) -> None:
        super().__init__()
        self.config = {}
        self.config['num_train_timesteps'] = 1000

class DDIMNextTokenV1Pipeline():
    def __init__(self):
        # Initialize the training and model configurations
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_id = ''
        self.model_version = ''
        self.train_config = TrainingConfig()
        self.model_config = ModelConfig()
        self.scheduler_config = SchedulerConfig()
        self.inference_config = InferenceConfig()
        self.unet = UNet2DModel(**self.model_config.config).to(self.device) # type: ignore
        self.scheduler=DDIMScheduler(**self.scheduler_config.config)
        
        wandb.login(key=os.environ.get("WANDB_API_KEY"))  # Ensure you have your WANDB_API_KEY set in your environment
        huggingface_hub.login(token=os.environ.get("HUGGINGFACE_HUB_TOKEN"), new_session=False)  # Ensure you have your HUGGINGFACE_HUB_TOKEN set in your environment
        
        if not os.path.exists(self.train_config.output_dir):
            os.makedirs(self.train_config.output_dir)
        if not huggingface_hub.repo_exists(self.train_config.hub_model_id):
            self.repo_URL = create_repo(repo_id=self.train_config.hub_model_id, exist_ok=True)
            print(f"Created repository {self.train_config.hub_model_id} on Hugging Face Hub.")
        self.repo = huggingface_hub.Repository(
            local_dir=self.train_config.output_dir,
            clone_from=self.train_config.hub_model_id,
            git_user="QLeca",
            git_email="quentin.leca@polytechnique.edu")
        self.repo.git_checkout(revision='main', create_branch_ok=True)  # Checkout the main branch
        self.repo.git_pull(rebase=True)  # Pull the latest changes from the hub
        
        print("Pipeline initialized on device : ", self.device)
        
    @torch.no_grad()
    def __call__(self, input_images: torch.Tensor, class_labels: torch.Tensor, num_inference_steps: int = 0):
        assert self.model_config.config['num_class_embeds'] != 0, "Class embeddings are not set. Please set the class vocabulary using set_class_vocabulary() method."
        if num_inference_steps == 0:
            num_inference_steps = self.inference_config.num_inference_steps
        
        self.unet.eval()

        xt = torch.randn((input_images.shape[0],
                          self.model_config.config['out_channels'],
                          self.train_config.image_size,
                          self.train_config.image_size)
                         ).to(self.device)
        input_images = input_images.to(self.device)
        class_labels = class_labels.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in tqdm(self.scheduler.timesteps.numpy(), desc="Inference", unit="step"):
            # Get prediction of noise
            noisy_samples = torch.concat([input_images, xt], dim=1).to(self.device)
            time_step = torch.as_tensor(t, device=self.device)
            
            noise_pred = self.unet.forward(sample=noisy_samples,
                                           timestep=time_step,
                                           class_labels=class_labels).sample # type: ignore
            
            # Use scheduler to get x0 and xt-1
            xt = self.scheduler.step(noise_pred, t, xt, return_dict=False)[0]
            # Save x0
        return xt
            
    @torch.no_grad()
    def save_training_samples(self, dataloader, epoch:int, generator: torch.Generator | None = None, num_inference_steps: int = 0) -> str:
        if num_inference_steps == 0:
            num_inference_steps = self.inference_config.num_inference_steps
        
        random_sample_idx = torch.randint(0, len(dataloader.dataset), (self.train_config.sample_size,), device=self.device, generator=generator) # type: ignore
        random_sample = dataloader.dataset[random_sample_idx]
        
        input_images = torch.stack(random_sample['input'])
        target_images = torch.stack(random_sample['target'])
        class_labels = torch.stack(random_sample['label'])
        
        output_images = self.__call__(input_images, class_labels, num_inference_steps)
        
        output_images = (output_images * 0.5 + 0.5).clamp(0, 1).cpu()
        
        input_images = (input_images * 0.5 + 0.5).clamp(0, 1).cpu()
        
        target_images = (target_images * 0.5  + 0.5).clamp(0, 1).cpu()
        
        concat = torch.concat([input_images, output_images, target_images])
        grid = make_grid(concat, nrow=input_images.shape[0])
        img = torchvision.transforms.ToPILImage()(grid)
        if self.train_config.push_to_hub:
            print(f"Saving sample images to {self.train_config.output_dir} ...")
            self.repo.git_checkout(revision=self.train_id, create_branch_ok=True)
            img.save(os.path.join(self.train_config.output_dir, 'result_epoch_{}.png'.format(epoch)))
            self.repo.push_to_hub(commit_message=f"Sample images for epoch {epoch}")
            img.close()
            return os.path.join(self.train_config.output_dir, 'result_epoch_{}.png'.format(epoch))
        else:
            save_dir = os.path.join(self.train_config.backup_output_dir, self.train_id)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f"Saving sample images to {save_dir} ...")
            img.save(os.path.join(save_dir, 'result_epoch_{}.png'.format(epoch)))
            img.close()
            return os.path.join(save_dir, 'result_epoch_{}.png'.format(epoch))

    def train_accelerate(self, train_dataloader, val_dataloader, train_size = 1000, val_size = 100):


        args = (train_dataloader, val_dataloader, train_size, val_size)

        notebook_launcher(self.train, args, num_processes=1)

    def train(self, train_dataloader, val_dataloader, train_size = 1000, val_size = 100):
        self.train_id = f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.set_num_class_embeds(len(train_dataloader.vocab))
        
        # Initialize the wandb run
        wandb.init(
            project=self.train_config.wandb_project_name,
            name=self.train_id,
            config={
                "image_size": self.train_config.image_size,
                "num_epochs": self.train_config.num_epochs,
                "train_batch_size": self.train_config.train_batch_size,
                "training_steps": train_size,
                "validation_steps": val_size,
                "eval_batch_size": self.train_config.eval_batch_size,
                "learning_rate": self.train_config.learning_rate,
                "lr_warmup_steps": self.train_config.lr_warmup_steps,
                "huggingface_repo_id": self.train_config.hub_model_id,
                "num_class_embeds": self.model_config.config['num_class_embeds'],
                "dataset_name": train_dataloader.dataset_name,
                "train_split": train_dataloader.split,
                "val_split": val_dataloader.split,
            }
        )

        # Create the optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.train_config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
                                                        optimizer=optimizer,
                                                        num_warmup_steps=self.train_config.lr_warmup_steps,
                                                        num_training_steps=(train_size * self.train_config.num_epochs // self.train_config.train_batch_size),
                                                    )
        
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.train_config.output_dir, "logs")
        )
        
        # Create the repo if push_to_hub is enabled
        if accelerator.is_main_process:
            if self.train_config.push_to_hub :
                print(f"Creating branch {self.train_id} in repository {self.train_config.hub_model_id} ...")
                # Create a new branch for this run with the name of the run
                self.repo.git_checkout(revision=self.train_id, create_branch_ok=True)
            

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        self.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            self.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        global_step = 0
        # Now you train the model
        random_generator = torch.Generator(device=self.device) # For the dataset sampling
        for epoch in range(self.train_config.num_epochs):
            # Take a random subset of the training dataset of size train_size
            if train_size is not None and train_size < len(train_dataloader.dataset): # type: ignore
                indices = torch.randperm(len(train_dataloader.dataset), # type: ignore
                                         generator=random_generator,
                                         device=self.device)[:train_size] 
                subset = torch.utils.data.Subset(train_dataloader.dataset, indices) # type: ignore
                train_dataloader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=self.train_config.train_batch_size,
                    shuffle=True,
                    num_workers=getattr(train_dataloader, 'num_workers', 0),
                    pin_memory=getattr(train_dataloader, 'pin_memory', False),
                    drop_last=getattr(train_dataloader, 'drop_last', False),
                )
            if val_size is not None and val_size < len(val_dataloader.dataset): # type: ignore
                indices = torch.randperm(len(val_dataloader.dataset), # type: ignore
                                         generator=random_generator,
                                         device=self.device)[:val_size] 
                subset = torch.utils.data.Subset(val_dataloader.dataset, indices) # type: ignore
                val_dataloader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=self.train_config.eval_batch_size,
                    shuffle=False,
                    num_workers=getattr(val_dataloader, 'num_workers', 0),
                    pin_memory=getattr(val_dataloader, 'pin_memory', False),
                    drop_last=getattr(val_dataloader, 'drop_last', False),
                )
            
            # Training loop
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, unit="batch")
            progress_bar.set_description(f"Epoch {epoch}")
            self.unet.train()
            for step, batch in enumerate(train_dataloader):
                input_images = batch["input"].to(self.device)
                target_images = batch["target"].to(self.device)
                class_labels = batch['label'].to(self.device)  
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
                    noise_pred = self.unet.forward(sample=noisy_samples,
                                           timestep=timesteps,
                                           class_labels=class_labels).sample
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step, 'epoch': epoch}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                wandb.log(logs)
            
            self.model_version = f"{self.train_id}_epoch_{epoch}"
            
            # Evaluation loop
            with torch.no_grad():   
                val_loss = 0.0
                self.unet.eval()
                for batch in tqdm(val_dataloader, desc="Evaluating", unit="batch"):
                    input_images = batch['input'].to(self.device)
                    target_images = batch['target'].to(self.device)
                    class_labels = batch['label'].to(self.device) 
                    
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
                    # Predict the noise residual
                    noisy_samples = torch.concat([input_images, noisy_targets], dim=1)
                    noise_pred = self.unet.forward(sample=noisy_samples,
                                           timestep=timesteps,
                                           class_labels=class_labels).sample
                    loss = F.mse_loss(noise_pred, noise)
                    val_loss += loss.item()
                val_loss /= len(val_dataloader)
                logs = {"val_loss": val_loss, "step": global_step, 'epoch': epoch}
                wandb.log(logs)
                accelerator.log(logs, step=global_step)
            
                        # After each epoch you optionally sample some demo images with evaluate() and save the model
            
            # Saving the model and images
            if accelerator.is_main_process:
                # Saving the images
                if (epoch + 1) % self.train_config.save_image_epochs == 0 or epoch == self.train_config.num_epochs - 1:
                    image_path = self.save_training_samples(dataloader=val_dataloader, epoch=epoch, generator=random_generator)
                    wandb.log({"sample_images": wandb.Image(image_path, caption=f"Epoch {epoch} samples"), 'epoch': epoch})
                # Saving the images
                if (epoch + 1) % self.train_config.save_model_epochs == 0 or epoch == self.train_config.num_epochs - 1:
                    self.save_model(revision=self.train_id, epoch=epoch)

    def save_model(self, revision: str = 'main', epoch: int=0):
        commit_message = f"Model saved at epoch {epoch}"
        if self.train_config.push_to_hub:
            print(f"Saving model to {self.train_config.output_dir} : {commit_message} ...")
            self.repo.git_checkout(revision=revision, create_branch_ok=True)
            self.unet.save_pretrained(self.train_config.output_dir)
            response = self.repo.push_to_hub(commit_message=commit_message)
            if response is not None:
                print(f"Model saved to {self.train_config.hub_model_id} : {commit_message}.")
            else:
                save_dir = os.path.join(self.train_config.backup_output_dir, self.train_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(f"Failed to push to hub, saving model to {save_dir} instead.")
                self.unet.save_pretrained(save_dir, variant=f'epoch_{epoch}')
                print(f"Model saved to {save_dir} : {commit_message}.")
        else:
            save_dir = os.path.join(self.train_config.backup_output_dir, self.train_id, f'epoch_{epoch}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f"Saving model to {save_dir} ...")
            self.unet.save_pretrained(save_dir)
            print(f"Model saved to {save_dir} : {commit_message}.")
        
    def load_model_from_local_dir(self, model_dir: str):
        self.unet = UNet2DModel.from_pretrained(self.train_config.output_dir)
        self.model_config.config = self.unet.config 
        self.unet.to(self.device) # type: ignore
        self.model_version = model_dir.split('/')[-1]  # Assuming the model_dir is structured as 'output_dir/model_version'
        print('Model Loaded with version: ', self.model_version)
    
    def load_model_from_hub(self, run: str, epoch: int) -> bool:
        revision = None
        try:
            commits = [{'title':commit.title, 'id':commit.commit_id} 
                       for commit in huggingface_hub.list_repo_commits(repo_id=self.train_config.hub_model_id,
                                                                       revision=run) 
                       if commit.title.startswith('Model saved at')]
        except Exception as e:
            print(f"Run {run} not found, check the run name and try again ...")
            return False
        for commit in commits:
            if commit['title'] == f'Model saved at epoch {epoch}':
                revision = commit['id']
                break
        if revision is None:
            print(f"Run {run} does not have a commit for epoch {epoch}, check the run name and try again ...")
            return False
        try:
            self.repo.git_checkout(revision=revision)
        except Exception as e:
            print(f"Failed to checkout revision {revision} for run {run}. Error: {e}")
            return False
        self.unet = UNet2DModel.from_pretrained(self.train_config.output_dir)
        self.model_config.config = self.unet.config 
        self.unet.to(self.device) # type: ignore
        self.model_version = f"{run}_epoch_{epoch}"
        print(f"Model loaded from version {self.model_version}.")
        self.repo.git_checkout(revision='main')
        return True
        
    def list_versions(self):
        branches = huggingface_hub.list_repo_refs(repo_id=self.train_config.hub_model_id).branches
        huggingface_hub.list_repo_commits(repo_id=self.train_config.hub_model_id)
        versions = []
        for branch in branches:
            if branch.name.startswith('run_'):
                commits = [commit.title for commit in huggingface_hub.list_repo_commits(repo_id=self.train_config.hub_model_id,
                                                                                        revision=branch.name) 
                           if commit.title.startswith('Model saved at')]
                versions.append({'name': branch.name, 'commits': commits})
        return versions

    def set_num_class_embeds(self, num_class_embeds: int):
        """
        Set the class vocabulary for the model.
        This is used to condition the model on specific classes.
        """
        self.unet.config['num_class_embeds'] = num_class_embeds
        self.model_config.config['num_class_embeds'] = num_class_embeds
        
    def get_FID_score(self, dataloader, num_inference_steps: int = 0):
        """
        Compute the FID score of the model on the given dataloader.
        """
        if num_inference_steps == 0:
            num_inference_steps = self.inference_config.num_inference_steps
        # Initialize FID metric
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

        # Loop through the dataloader and accumulate FID statistics
        for batch in tqdm(dataloader, desc="Calculating FID score", unit="batch"):
            input_images = batch['input']
            target_images = batch['target'].to(self.device)
            labels = batch['label']
            with torch.no_grad():
                outputs = self.__call__(input_images=input_images, 
                                        class_labels=labels,
                                        num_inference_steps=num_inference_steps)
                # Rescale to [0, 1]
                fake_images = (outputs * 0.5 + 0.5).clamp(0, 1)
                real_images = (target_images * 0.5 + 0.5).clamp(0, 1)
                
                # FID expects float32 and shape [N, 3, H, W]
                fid_metric.update(fake_images.float(), real=False)
                fid_metric.update(real_images.float(), real=True)

        return fid_metric.compute().item()
    
    def save_stats(self, stats: dict, run:str, epoch:int) -> bool:
        """
        Save the training statistics to a JSON file.
        """
        
        self.repo.git_checkout(revision='main')
        stats_path = os.path.join(self.train_config.output_dir, 'stats.json')
        existing_stats = []
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                existing_stats = json.load(f)
        existing_stats.append(stats)
        with open(stats_path, 'w') as f:
            json.dump(existing_stats, f, indent=4)
        
        commit_message='Saved training statistics for model version {run}_epoch_{epoch}..'
        response = self.repo.push_to_hub(commit_message=commit_message,)
        if response is not None:
            print(f"Model saved to {self.train_config.hub_model_id} : {commit_message}.")
            return True
        else:
            return False
            
        