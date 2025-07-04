# Base class for Next Layer Prediction using diffusion methods
# This class contains all the common functionality shared by DDPM and DDIM variants

import torch
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any
import huggingface_hub

from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import notebook_launcher, Accelerator
from diffusers.models.unets.unet_2d import UNet2DModel
from huggingface_hub import create_repo
import huggingface_hub
from tqdm.auto import tqdm
from pathlib import Path
import os

import torch
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
import time
from torchmetrics.image.fid import FrechetInceptionDistance

IMAGE_SIZE = 128


@dataclass
class BaseTrainingConfig:
    image_size = IMAGE_SIZE  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    sample_size = 8  # how many images to sample during training
    num_epochs = 50
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    lr_warmup_steps = 125
    save_image_epochs = 1
    save_model_epochs = 3
    seed = 0
    train_size = 128000
    val_size = 32000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = ""  # to be set by subclasses
    backup_output_dir = ""  # to be set by subclasses
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = ""  # to be set by subclasses
    wandb_project_name = ""  # to be set by subclasses
    FID_eval_steps = 5

    def get_dict(self):
        return {
            attribute: getattr(self, attribute)
            for attribute in dir(self)
            if not attribute.startswith("__") and not callable(getattr(self, attribute))
        }


@dataclass 
class BaseInferenceConfig:
    num_inference_steps = 50  # number of denoising steps

    def get_dict(self):
        return {
            attribute: getattr(self, attribute)
            for attribute in dir(self)
            if not attribute.startswith("__") and not callable(getattr(self, attribute))
        }


class BaseModelConfig(dict):
    def __init__(self) -> None:
        self.config = {}
        self.config["sample_size"] = IMAGE_SIZE  # The size of the input
        self.config["in_channels"] = 6  # Default: 6 channels (2 RGB images concatenated)
        self.config["out_channels"] = 3  # Default: 3 channels for the output image
        self.config["down_block_types"] = (
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        )
        self.config["up_block_types"] = (
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
        self.config["block_out_channels"] = (128, 128, 256, 256, 512, 512)
        self.config["layers_per_block"] = 2
        self.config["class_embed_type"] = "identity"
        self.config["num_class_embeds"] = 0  # Number of class embeddings, set to 0 for no class conditioning


class BaseSchedulerConfig(dict):
    def __init__(self) -> None:
        super().__init__()
        self.config = {}
        self.config["num_train_timesteps"] = 1000
        self.config["beta_start"] = 0.0001
        self.config["beta_end"] = 0.02
        self.config["beta_schedule"] = "squaredcos_cap_v2"


class BaseNextTokenPipeline(ABC):
    def __init__(
        self,
        train_config: Optional[BaseTrainingConfig] = None,
        model_config: Optional[BaseModelConfig] = None,
        scheduler_config: Optional[BaseSchedulerConfig] = None,
        inference_config: Optional[BaseInferenceConfig] = None,
    ):
        # Initialize the training and model configurations
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.train_id = ""
        self.model_version = ""

        # Set the configurations - to be customized by subclasses
        self.train_config = (
            train_config if train_config is not None else self._get_default_train_config()
        )
        self.model_config = model_config if model_config is not None else self._get_default_model_config()
        self.scheduler_config = (
            scheduler_config if scheduler_config is not None else self._get_default_scheduler_config()
        )
        self.inference_config = (
            inference_config if inference_config is not None else self._get_default_inference_config()
        )

        # Initialize model and scheduler - to be implemented by subclasses
        self.unet: UNet2DModel = self._create_unet()
        self.scheduler: Any = self._create_scheduler()
        
        # Move model to device
        self.unet = self.unet.to(self.device) # type: ignore

        # Initialize external services
        self._init_external_services()

        # Initialize repository
        self._init_repository()
        print("Pipeline initialized on device : ", self.device)

    @abstractmethod
    def _get_default_train_config(self) -> BaseTrainingConfig:
        """Return the default training configuration for this pipeline."""
        pass

    @abstractmethod
    def _get_default_model_config(self) -> BaseModelConfig:
        """Return the default model configuration for this pipeline."""
        pass

    @abstractmethod
    def _get_default_scheduler_config(self) -> BaseSchedulerConfig:
        """Return the default scheduler configuration for this pipeline."""
        pass

    @abstractmethod
    def _get_default_inference_config(self) -> BaseInferenceConfig:
        """Return the default inference configuration for this pipeline."""
        pass

    @abstractmethod
    def _create_unet(self) -> UNet2DModel:
        """Create and return the UNet model."""
        pass

    @abstractmethod
    def _create_scheduler(self) -> Any:
        """Create and return the scheduler."""
        pass

    @torch.no_grad()
    @abstractmethod
    def __call__(
        self,
        input_images: torch.Tensor,
        class_labels: torch.Tensor,
        num_inference_steps: int = 0,
    ) -> torch.Tensor:
        """Run inference to generate the next layer."""
        pass

    @torch.no_grad()
    def save_training_samples(
        self,
        dataloader,
        epoch: int,
        generator: torch.Generator | None = None,
        num_inference_steps: int = 0,
    ) -> str:
        """Save training samples for visualization."""
        if num_inference_steps == 0:
            num_inference_steps = self.inference_config.num_inference_steps

        random_sample_idx = torch.randint(
            0, len(dataloader.dataset), (self.train_config.sample_size,), device=self.device, generator=generator
        )  # type: ignore
        random_sample = dataloader.dataset[random_sample_idx]

        input_images = torch.stack(random_sample["input"])
        target_images = torch.stack(random_sample["target"])
        class_labels = torch.stack(random_sample["label"])

        output_images = self.__call__(input_images, class_labels, num_inference_steps)

        # Check for NaN values and replace them with zeros
        if torch.isnan(output_images).any():
            print(
                f"Warning: NaN values detected in output images at epoch {epoch}. Replacing with zeros."
            )
            output_images = torch.nan_to_num(output_images, nan=0.0)

        output_images = (output_images * 0.5 + 0.5).clamp(0, 1).cpu()
        input_images = (input_images * 0.5 + 0.5).clamp(0, 1).cpu()
        target_images = (target_images * 0.5 + 0.5).clamp(0, 1).cpu()

        # Additional check after normalization
        if (
            torch.isnan(output_images).any()
            or torch.isnan(input_images).any()
            or torch.isnan(target_images).any()
        ):
            print(
                f"Warning: NaN values detected after normalization at epoch {epoch}. Replacing with zeros."
            )
            output_images = torch.nan_to_num(output_images, nan=0.0)
            input_images = torch.nan_to_num(input_images, nan=0.0)
            target_images = torch.nan_to_num(target_images, nan=0.0)

        concat = torch.concat([input_images, output_images, target_images])
        grid = make_grid(concat, nrow=input_images.shape[0])
        img = torchvision.transforms.ToPILImage()(grid)
        
        if self.train_config.push_to_hub and self.repo is not None:
            print(f"Saving sample images to {self.train_config.output_dir} ...")
            self.repo.git_checkout(revision=self.train_id, create_branch_ok=True)
            img.save(
                os.path.join(
                    self.train_config.output_dir, "result_epoch_{}.png".format(epoch)
                )
            )
            self.repo.push_to_hub(commit_message=f"Sample images for epoch {epoch}")
            img.close()
            return os.path.join(
                self.train_config.output_dir, "result_epoch_{}.png".format(epoch)
            )
        else:
            save_dir = os.path.join(self.train_config.backup_output_dir, self.train_id)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f"Saving sample images to {save_dir} ...")
            img.save(os.path.join(save_dir, "result_epoch_{}.png".format(epoch)))
            img.close()
            return os.path.join(save_dir, "result_epoch_{}.png".format(epoch))

    def train_accelerate(
        self, train_dataloader, val_dataloader, train_size=1000, val_size=100, **params
    ):
        """Launch training with accelerate."""
        # Pass the parameters as keyword arguments to the train method
        def train_wrapper():
            return self.train(
                train_dataloader, val_dataloader, train_size, val_size, **params
            )

        notebook_launcher(train_wrapper, args=(), num_processes=1)

    def train(
        self, train_dataloader, val_dataloader, train_size=1000, val_size=100, **params
    ):
        """Main training loop."""
        self.train_id = f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.set_num_class_embeds(len(train_dataloader.vocab))
        self.train_config.train_size = train_size
        self.train_config.val_size = val_size

        num_cycles = params.get("num_cycles", 0.5)
        train_tags = params.get("train_tags", None)

        # Initialize the wandb run
        wandb.init(
            project=self.train_config.wandb_project_name,
            name=self.train_id,
            tags=train_tags,
            config={
                "run_id": self.train_id,
                "train_config": self.train_config.get_dict(),
                "dataset": {
                    "name": train_dataloader.dataset_name,
                    "train_split": train_dataloader.split,
                    "val_split": val_dataloader.split,
                },
                "model_config": self.model_config.config,
                "scheduler_config": self.scheduler_config.config,
                "inference_config": self.inference_config.get_dict(),
                "optimizer": "AdamW",
                "Lr_scheduler": "Cosine with warmup",
                "other_params": params,
            },
        )

        # Create the optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            self.unet.parameters(), lr=self.train_config.learning_rate
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.train_config.lr_warmup_steps,
            num_training_steps=(
                train_size
                * self.train_config.num_epochs
                // (
                    self.train_config.train_batch_size
                    * self.train_config.gradient_accumulation_steps
                )
            ),
            num_cycles=num_cycles,
        )

        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.train_config.output_dir, "logs"),
        )

        # Create the repo if push_to_hub is enabled
        if accelerator.is_main_process:
            if self.train_config.push_to_hub and self.repo is not None:
                print(
                    f"Creating branch {self.train_id} in repository {self.train_config.hub_model_id} ..."
                )
                # Create a new branch for this run with the name of the run
                self.repo.git_checkout(revision=self.train_id, create_branch_ok=True)

        # Prepare everything
        self.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
            accelerator.prepare(
                self.unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
        )

        global_step = 0
        nan_count = 0  # Counter for NaN occurrences
        max_nan_tolerance = 10  # Maximum number of NaN occurrences before stopping

        # Now you train the model
        random_generator = torch.Generator(
            device=self.device
        )  # For the dataset sampling
        random_generator.manual_seed(self.train_config.seed)
        for epoch in range(self.train_config.num_epochs):
            # Check for NaN values in model parameters at the start of each epoch
            if self.check_model_for_nan():
                print(
                    f"NaN values detected in model parameters at the start of epoch {epoch}. Attempting to reset..."
                )
                if self.reset_nan_parameters():
                    print("Model parameters reset. Continuing training...")
                else:
                    print("Failed to reset parameters. Training may be unstable.")

            # Take a random subset of the training dataset of size train_size
            # if train_size is not None and train_size < len(train_dataloader.dataset):  # type: ignore
            #     indices = torch.randperm(
            #         len(train_dataloader.dataset),  # type: ignore
            #         generator=random_generator,
            #         device=self.device,
            #     )[:train_size]
            #     subset = torch.utils.data.Subset(train_dataloader.dataset, indices)  # type: ignore
            #     train_dataloader = torch.utils.data.DataLoader(
            #         subset,
            #         batch_size=self.train_config.train_batch_size,
            #         shuffle=True,
            #         num_workers=getattr(train_dataloader, "num_workers", 0),
            #         pin_memory=getattr(train_dataloader, "pin_memory", False),
            #         drop_last=getattr(train_dataloader, "drop_last", False),
            #     )
            # if val_size is not None and val_size < len(val_dataloader.dataset):  # type: ignore
            #     indices = torch.randperm(
            #         len(val_dataloader.dataset),  # type: ignore
            #         generator=random_generator,
            #         device=self.device,
            #     )[:val_size]
            #     subset = torch.utils.data.Subset(val_dataloader.dataset, indices)  # type: ignore
            #     val_dataloader = torch.utils.data.DataLoader(
            #         subset,
            #         batch_size=self.train_config.eval_batch_size,
            #         shuffle=False,
            #         num_workers=getattr(val_dataloader, "num_workers", 0),
            #         pin_memory=getattr(val_dataloader, "pin_memory", False),
            #         drop_last=getattr(val_dataloader, "drop_last", False),
            #     )

            # Training loop
            progress_bar = tqdm(
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process,
                unit="batch",
            )
            progress_bar.set_description(f"Epoch {epoch}")
            self.unet.train()
            
            for step, batch in enumerate(train_dataloader):
                if step >= train_size:
                    break
                input_images = batch["input"].to(self.device)
                target_images = batch["target"].to(self.device)
                class_labels = batch["label"].to(self.device)

                # Check for NaN values in input data
                if torch.isnan(input_images).any() or torch.isnan(target_images).any():
                    print(
                        f"Warning: NaN values detected in input data at step {step}, epoch {epoch}. Skipping this batch."
                    )
                    nan_count += 1
                    if nan_count > max_nan_tolerance:
                        print(
                            f"Too many NaN occurrences ({nan_count}). Stopping training early."
                        )
                        return
                    continue

                # Sample noise to add to the images
                noise = torch.randn(target_images.shape, device=self.device)
                bs = target_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler.config["num_train_timesteps"],
                    (bs,),
                    device=self.device,
                    dtype=torch.int,
                )
                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_targets = self.scheduler.add_noise(target_images, noise, timesteps)  # type: ignore

                with accelerator.accumulate(self.unet):
                    # Predict the noise residual - this may be overridden by subclasses
                    noise_pred = self._forward_unet(input_images, noisy_targets, timesteps, class_labels)
                    loss = F.mse_loss(noise_pred, noise)

                    # Check for NaN loss and skip this batch if detected
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(
                            f"Warning: NaN or Inf loss detected at step {step}, epoch {epoch}. Skipping this batch."
                        )
                        nan_count += 1
                        if nan_count > max_nan_tolerance:
                            print(
                                f"Too many NaN occurrences ({nan_count}). Stopping training early."
                            )
                            return
                        continue

                    # Check for NaN values in noise prediction
                    if torch.isnan(noise_pred).any():
                        print(
                            f"Warning: NaN values detected in noise prediction at step {step}, epoch {epoch}. Skipping this batch."
                        )
                        nan_count += 1
                        if nan_count > max_nan_tolerance:
                            print(
                                f"Too many NaN occurrences ({nan_count}). Stopping training early."
                            )
                            return
                        continue

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        # Check for NaN gradients
                        has_nan_gradients = False
                        for name, param in self.unet.named_parameters():
                            if param.grad is not None and (
                                torch.isnan(param.grad).any()
                                or torch.isinf(param.grad).any()
                            ):
                                print(
                                    f"Warning: NaN or Inf gradient detected in parameter {name} at step {step}, epoch {epoch}"
                                )
                                has_nan_gradients = True
                                break

                        if has_nan_gradients:
                            optimizer.zero_grad()
                            nan_count += 1
                            if nan_count > max_nan_tolerance:
                                print(
                                    f"Too many NaN occurrences ({nan_count}). Stopping training early."
                                )
                                return
                        else:
                            accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                wandb.log(logs)

            self.model_version = f"{self.train_id}_epoch_{epoch}"

            # Evaluation loop
            with torch.no_grad():
                val_loss = 0.0
                self.unet.eval()
                for step,batch in tqdm(enumerate(val_dataloader), desc="Evaluating", unit="batch"):
                    if step >= val_size:
                        break
                    input_images = batch["input"].to(self.device)
                    target_images = batch["target"].to(self.device)
                    class_labels = batch["label"].to(self.device)

                    # Sample noise to add to the images
                    noise = torch.randn(target_images.shape, device=self.device)
                    bs = target_images.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.scheduler.config["num_train_timesteps"],
                        (bs,),
                        device=self.device,
                        dtype=torch.int,
                    )
                    # Add noise to the clean images according to the noise magnitude at each timestep
                    noisy_targets = self.scheduler.add_noise(target_images, noise, timesteps)  # type: ignore
                    # Predict the noise residual
                    noise_pred = self._forward_unet(input_images, noisy_targets, timesteps, class_labels)
                    loss = F.mse_loss(noise_pred, noise)
                    val_loss += loss.item()
                val_loss /= val_size
                logs = {"val_loss": val_loss, "step": global_step, "epoch": epoch}
                if self.train_config.FID_eval_steps > 0 and (epoch + 1) % self.train_config.FID_eval_steps == 0:
                    FID_score = self.get_FID_score(
                        dataloader=val_dataloader,
                        num_inference_steps=self.inference_config.num_inference_steps,
                    )
                    logs["FID_score"] = FID_score
                wandb.log(logs)
                accelerator.log(logs, step=global_step)
                

            # Saving the model and images
            if accelerator.is_main_process:
                # Saving the images
                if (
                    (epoch + 1) % self.train_config.save_image_epochs == 0
                    or epoch == self.train_config.num_epochs - 1
                ):
                    image_path = self.save_training_samples(
                        dataloader=val_dataloader,
                        epoch=epoch,
                        generator=random_generator,
                    )
                    wandb.log(
                        {
                            "sample_images": wandb.Image(
                                image_path, caption=f"Epoch {epoch} samples"
                            ),
                            "epoch": epoch,
                        }
                    )
                # Saving the model
                if (
                    (epoch + 1) % self.train_config.save_model_epochs == 0
                    or epoch == self.train_config.num_epochs - 1
                ):
                    self.save_model(revision=self.train_id, epoch=epoch)

    @abstractmethod
    def _forward_unet(self, input_images: torch.Tensor, noisy_targets: torch.Tensor, 
                     timesteps: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet. May be overridden by subclasses."""
        pass

    def save_model(self, revision: str = "main", epoch: int = 0):
        """Save the model to hub or local directory."""
        commit_message = f"Model saved at epoch {epoch}"
        if self.train_config.push_to_hub and self.repo is not None:
            print(
                f"Saving model to {self.train_config.output_dir} : {commit_message} ..."
            )
            self.repo.git_checkout(revision=revision, create_branch_ok=True)
            self.unet.save_pretrained(self.train_config.output_dir)
            response = self.repo.push_to_hub(commit_message=commit_message)
            if response is not None:
                print(
                    f"Model saved to {self.train_config.hub_model_id} : {commit_message}."
                )
            else:
                save_dir = os.path.join(
                    self.train_config.backup_output_dir, self.train_id
                )
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(f"Failed to push to hub, saving model to {save_dir} instead.")
                self.unet.save_pretrained(save_dir, variant=f"epoch_{epoch}")
                print(f"Model saved to {save_dir} : {commit_message}.")
        else:
            save_dir = os.path.join(
                self.train_config.backup_output_dir, self.train_id, f"epoch_{epoch}"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f"Saving model to {save_dir} ...")
            self.unet.save_pretrained(save_dir)
            print(f"Model saved to {save_dir} : {commit_message}.")

    def load_model_from_local_dir(self, model_dir: str):
        """Load model from local directory."""
        self.unet = self._create_unet_from_pretrained(self.train_config.output_dir)
        self.model_config.config = self.unet.config
        self.unet = self.unet.to(self.device)  # type: ignore
        self.model_version = model_dir.split("/")[
            -1
        ]  # Assuming the model_dir is structured as 'output_dir/model_version'
        print("Model Loaded with version: ", self.model_version)

    @abstractmethod
    def _create_unet_from_pretrained(self, model_dir: str) -> UNet2DModel:
        """Create UNet from pretrained model directory."""
        pass

    def load_model_from_hub(self, run: str, epoch: int) -> bool:
        """Load model from Hugging Face Hub."""
        assert self.repo is not None, "Repository is not initialized."
        revision = None
        try:
            commits = [
                {"title": commit.title, "id": commit.commit_id}
                for commit in huggingface_hub.list_repo_commits(
                    repo_id=self.train_config.hub_model_id, revision=run
                )
                if commit.title.startswith("Model saved at")
            ]
        except Exception as e:
            print(f"Run {run} not found, check the run name and try again ...")
            return False
        for commit in commits:
            if commit["title"] == f"Model saved at epoch {epoch}":
                revision = commit["id"]
                break
        if revision is None:
            print(
                f"Run {run} does not have a commit for epoch {epoch}, check the run name and try again ..."
            )
            return False
        try:
            self.repo.git_checkout(revision=revision)
        except Exception as e:
            print(f"Failed to checkout revision {revision} for run {run}. Error: {e}")
            return False
        self.unet = self._create_unet_from_pretrained(self.train_config.output_dir)
        self.model_config.config = self.unet.config
        self.unet.to(self.device)  # type: ignore
        self.model_version = f"{run}_epoch_{epoch}"
        print(f"Model loaded from version {self.model_version}.")
        self.repo.git_checkout(revision="main")
        return True

    def list_versions(self):
        """List available model versions."""
        branches = huggingface_hub.list_repo_refs(
            repo_id=self.train_config.hub_model_id
        ).branches
        versions = []
        for branch in branches:
            if branch.name.startswith("run_"):
                commits = [
                    commit.title
                    for commit in huggingface_hub.list_repo_commits(
                        repo_id=self.train_config.hub_model_id, revision=branch.name
                    )
                    if commit.title.startswith("Model saved at")
                ]
                versions.append({"name": branch.name, "commits": commits})
        return versions

    def set_num_class_embeds(self, num_class_embeds: int):
        """Set the class vocabulary for the model."""
        self.unet.config["num_class_embeds"] = num_class_embeds
        self.model_config.config["num_class_embeds"] = num_class_embeds

    def get_FID_score(self, dataloader, num_inference_steps: int = 0):
        """Compute the FID score of the model on the given dataloader."""
        if num_inference_steps == 0:
            num_inference_steps = self.inference_config.num_inference_steps
        # Initialize FID metric
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(
            self.device
        )

        # Loop through the dataloader and accumulate FID statistics
        for batch in tqdm(dataloader, desc="Calculating FID score", unit="batch"):
            input_images = batch["input"]
            target_images = batch["target"].to(self.device)
            labels = batch["label"]
            with torch.no_grad():
                outputs = self.__call__(
                    input_images=input_images,
                    class_labels=labels,
                    num_inference_steps=num_inference_steps,
                )
                # Rescale to [0, 1]
                fake_images = (outputs * 0.5 + 0.5).clamp(0, 1)
                real_images = (target_images * 0.5 + 0.5).clamp(0, 1)

                # FID expects float32 and shape [N, 3, H, W]
                fid_metric.update(fake_images.float(), real=False)
                fid_metric.update(real_images.float(), real=True)

        return fid_metric.compute().item()

    def save_stats(self, stats: dict, run: str, epoch: int) -> bool:
        """Save the training statistics to a JSON file."""
        assert self.repo is not None, "Repository is not initialized."
        self.repo.git_checkout(revision="main")
        stats_path = os.path.join(self.train_config.output_dir, "stats.json")
        existing_stats = []
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                existing_stats = json.load(f)
        existing_stats.append(stats)
        with open(stats_path, "w") as f:
            json.dump(existing_stats, f, indent=4)

        commit_message = f"Saved training statistics for model version {run}_epoch_{epoch}."
        response = self.repo.push_to_hub(
            commit_message=commit_message,
        )
        if response is not None:
            print(
                f"Stats saved to {self.train_config.hub_model_id} : {commit_message}."
            )
            return True
        else:
            return False

    def check_model_for_nan(self):
        """Check if any model parameters contain NaN values."""
        for name, param in self.unet.named_parameters():
            if torch.isnan(param).any():
                print(f"Warning: NaN values detected in parameter: {name}")
                return True
        return False

    def reset_nan_parameters(self):
        """Reset any NaN parameters to their initialization values."""
        nan_found = False
        for name, param in self.unet.named_parameters():
            if torch.isnan(param).any():
                print(f"Resetting NaN parameter: {name}")
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
                nan_found = True
        return nan_found

    def _init_external_services(self):
        """Initialize external services like wandb and huggingface hub"""
        try:
            import wandb
            wandb.login(
                key=os.environ.get("WANDB_API_KEY")
            )  # Ensure you have your WANDB_API_KEY set in your environment
        except Exception as e:
            print(
                "Failed to login to Weights & Biases. Please ensure you have your WANDB_API_KEY set in your environment."
            )
            raise e
        try:
            huggingface_hub.login(
                token=os.environ.get("HUGGINGFACE_HUB_TOKEN"), new_session=False
            )  # Ensure you have your HUGGINGFACE_HUB_TOKEN set in your environment
        except Exception as e:
            print(
                "Failed to login to Hugging Face Hub. Please ensure you have your HUGGINGFACE_HUB_TOKEN set in your environment."
            )
            raise e
    
    def _init_repository(self):
        """Initialize the repository for saving models"""
        # Set the hub model ID based on the username and model name
        try:
            username = huggingface_hub.whoami()['name']
            if not self.train_config.hub_model_id:
                self.train_config.hub_model_id = f"{username}/{self.__class__.__name__}"
        except:
            print("Warning: Could not get username from HuggingFace Hub")
        
        if not os.path.exists(self.train_config.output_dir):
            os.makedirs(self.train_config.output_dir)
        
        # Initialize repo only if pushing to hub is enabled
        if self.train_config.push_to_hub:
            if not huggingface_hub.repo_exists(self.train_config.hub_model_id):
                self.repo_URL = create_repo(
                    repo_id=self.train_config.hub_model_id, exist_ok=True
                )
                print(
                    f"Created repository {self.train_config.hub_model_id} on Hugging Face Hub."
                )
            self.repo = huggingface_hub.Repository(
                local_dir=self.train_config.output_dir,
                clone_from=self.train_config.hub_model_id,
                git_user="QLeca",
                git_email="quentin.leca@polytechnique.edu",
            )
            self.repo.git_checkout(
                revision="main", create_branch_ok=True
            )  # Checkout the main branch
            self.repo.git_pull(rebase=True)  # Pull the latest changes from the hub
        else:
            self.repo = None
