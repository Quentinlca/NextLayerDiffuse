# Example of DDIMNextTokenV1 refactored to inherit from BaseNextTokenPipeline

import torch
from dataclasses import dataclass
from typing import Optional

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

from .BaseNextTokenPipeline import (
    BaseNextTokenPipeline, 
    BaseTrainingConfig, 
    BaseModelConfig, 
    BaseSchedulerConfig, 
    BaseInferenceConfig
)


@dataclass
class DDIMTrainingConfig(BaseTrainingConfig):
    output_dir = "training_outputs/DDIMNextTokenV1"
    backup_output_dir = "training_outputs/DDIMNextTokenV1_backup"
    hub_model_id = "QLeca/DDIMNextTokenV1"
    wandb_project_name = "ddim-next-token-v1"


@dataclass
class DDIMInferenceConfig(BaseInferenceConfig):
    num_inference_steps = 50


class DDIMModelConfig(BaseModelConfig):
    def __init__(self) -> None:
        super().__init__()
        # DDIM uses 6 channels (2 RGB images) and 3 output channels
        self.config["in_channels"] = 6
        self.config["out_channels"] = 3


class DDIMSchedulerConfig(BaseSchedulerConfig):
    def __init__(self) -> None:
        super().__init__()
        # DDIM specific scheduler configuration


class DDIMNextTokenV1PipelineRefactored(BaseNextTokenPipeline):
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
