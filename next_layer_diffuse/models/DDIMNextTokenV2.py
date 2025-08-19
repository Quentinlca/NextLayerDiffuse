# Example of DDIMNextTokenV1 refactored to inherit from BaseNextTokenPipeline

import torch
from dataclasses import dataclass
from typing import Optional

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm
import time
import wandb
from accelerate import notebook_launcher, Accelerator
import os
import torch.nn.functional as F

from diffusers.optimization import get_cosine_schedule_with_warmup


from .BaseNextTokenPipeline import (
    BaseNextTokenPipeline, 
    BaseTrainingConfig, 
    BaseModelConfig, 
    BaseSchedulerConfig, 
    BaseInferenceConfig
)


@dataclass
class DDIMTrainingConfig(BaseTrainingConfig):
    output_dir = "training_outputs/DDIMNextTokenV2"
    backup_output_dir = "training_outputs/DDIMNextTokenV2_backup"
    hub_model_id = "QLeca/DDIMNextTokenV2"
    wandb_project_name = "ddim-next-token-v2"


@dataclass
class DDIMInferenceConfig(BaseInferenceConfig):
    num_inference_steps = 50


class DDIMModelConfig(BaseModelConfig):
    def __init__(self) -> None:
        super().__init__()
        # DDIM uses 6 channels (2 RGB images) and 3 output channels
        self.config["in_channels"] = 6
        self.config["out_channels"] = 4


class DDIMSchedulerConfig(BaseSchedulerConfig):
    def __init__(self) -> None:
        super().__init__()
        # DDIM specific scheduler configuration


class DDIMNextTokenV2Pipeline(BaseNextTokenPipeline):
    """Refactored DDIM pipeline inheriting from BaseNextTokenPipeline."""
    
    def __init__(
        self,
        train_config: Optional[DDIMTrainingConfig] = None,
        model_config: Optional[DDIMModelConfig] = None,
        scheduler_config: Optional[DDIMSchedulerConfig] = None,
        inference_config: Optional[DDIMInferenceConfig] = None,
    ):
        super().__init__(train_config, model_config, scheduler_config, inference_config)

    def _get_default_train_config(self) -> BaseTrainingConfig:
        return DDIMTrainingConfig()

    def _get_default_model_config(self) -> BaseModelConfig:
        return DDIMModelConfig()

    def _get_default_scheduler_config(self) -> BaseSchedulerConfig:
        return DDIMSchedulerConfig()

    def _get_default_inference_config(self) -> BaseInferenceConfig:
        return DDIMInferenceConfig()

    def _create_unet(self):
        return UNet2DModel(**self.model_config.config).to(self.device) # type: ignore

    def _create_scheduler(self):
        return DDIMScheduler(**self.scheduler_config.config)

    def _create_unet_from_pretrained(self, model_dir: str):
        return UNet2DModel.from_pretrained(model_dir).to(self.device)  # type: ignore

    def _forward_unet(self, input_images: torch.Tensor, 
                      noisy_targets: torch.Tensor, 
                      timesteps: torch.Tensor, 
                      class_labels: torch.Tensor) -> torch.Tensor:
        noisy_samples = torch.concat([input_images, noisy_targets], dim=1)
        return self.unet.forward(
            sample=noisy_samples,
            timestep=timesteps,
            class_labels=class_labels,
        ).sample # type: ignore
    
    @torch.no_grad()
    def __call__(
        self,
        input_images: torch.Tensor,
        class_labels: torch.Tensor,
        num_inference_steps: int = 0,
    ) -> torch.Tensor:
        """DDIM-specific inference implementation."""
        assert (
            self.model_config.config["num_class_embeds"] != 0
        ), "Class embeddings are not set. Please set the class vocabulary using set_class_vocabulary() method."
        
        if num_inference_steps == 0:
            num_inference_steps = self.inference_config.num_inference_steps

        self.unet.eval()

        xt = torch.randn(
            (
                input_images.shape[0],
                self.model_config.config["out_channels"],
                self.train_config.image_size,
                self.train_config.image_size,
            )
        ).to(self.device)
        
        input_images = input_images.to(self.device)
        class_labels = class_labels.to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps.numpy(), desc="Inference", unit="step"):
            # Get prediction of noise
            noisy_samples = torch.concat([input_images, xt], dim=1).to(self.device)
            time_step = torch.as_tensor(t, device=self.device)

            noise_pred = self.unet.forward(
                sample=noisy_samples, timestep=time_step, class_labels=class_labels
            ).sample # type: ignore

            # Check for NaN values in noise prediction
            if torch.isnan(noise_pred).any():
                print(
                    f"Warning: NaN values detected in noise prediction during inference at timestep {t}"
                )
                noise_pred = torch.nan_to_num(noise_pred, nan=0.0)

            # Use scheduler to get x0 and xt-1
            xt = self.scheduler.step(noise_pred, t, xt, return_dict=False)[0]

            # Check for NaN values in xt
            if torch.isnan(xt).any():
                print(
                    f"Warning: NaN values detected in xt during inference at timestep {t}"
                )
                xt = torch.nan_to_num(xt, nan=0.0)

        return xt

    
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

            
            # Training loop
            progress_bar = tqdm(
                total=train_size,
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
                    outputs = self.scheduler.step(noise_pred, timesteps, noisy_targets, return_dict=False)[1] 
                    outputs_RGB = outputs[:, :3, :, :]  # Extract RGB channels
                    outputs_alpha = outputs[:, 3:4, :, :]  # Extract alpha channel
                    # Combine input images and predicted outputs
                    target_pred = input_images * (1 - outputs_alpha) + outputs_RGB * outputs_alpha  
                    loss = F.mse_loss(target_pred, target_images)

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
                    outputs = self.scheduler.step(noise_pred, timesteps, noisy_targets, return_dict=False)[1] 
                    outputs_RGB = outputs[:, :3, :, :]  # Extract RGB channels
                    outputs_alpha = outputs[:, 3:4, :, :]  # Extract alpha channel
                    # Combine input images and predicted outputs
                    target_pred = input_images * (1 - outputs_alpha) + outputs_RGB * outputs_alpha  
                    loss = F.mse_loss(target_pred, target_images)
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
