# This is a Next Layer Prediction using DDPM methods
# For training, we give it the previous layers and the target (noised) and expect the Unet to predict the noise
# For diffusion we give it the previous layers and random noise and want to get the next layer

import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .BaseNextTokenPipeline import (
    BaseNextTokenPipeline, 
    BaseTrainingConfig, 
    BaseModelConfig, 
    BaseSchedulerConfig
)

IMAGE_SIZE = 128

@dataclass
class DDPMv3TrainingConfig(BaseTrainingConfig):
    """Training configuration specific to DDPMNextTokenV3"""
    output_dir: str = "training_outputs/DDPMNextTokenV3"
    backup_output_dir: str = "training_outputs/DDPMNextTokenV3_backup"
    hub_model_id: str = "QLeca/DDPMNextTokenV3"
    wandb_project_name: str = "ddpm-next-token-v3"
    num_epochs: int = 50
    learning_rate: float = 1e-5
    lr_warmup_steps: int = 100
    save_model_epochs: int = 3

class DDPMv3ModelConfig(BaseModelConfig):
    """Model configuration specific to DDPMNextTokenV3"""
    def __init__(self) -> None:
        super().__init__()
        self.config.update({
            'in_channels': 6,  # 6 channels because 2 RGB images concatenated
            'out_channels': 3,  # 3 channels for the output image
            'layers_per_block': 2,
            'class_embed_type': 'identity',
            'num_class_embeds': 0,  # Number of class embeddings, set to 0 for no class conditioning
        })

class DDPMv3SchedulerConfig(BaseSchedulerConfig):
    """Scheduler configuration specific to DDPMNextTokenV3"""
    def __init__(self) -> None:
        super().__init__()
        self.config.update({
            'num_train_timesteps': 1000
        })

class DDPMNextTokenV3Pipeline(BaseNextTokenPipeline):
    """DDPM-based pipeline for next token prediction (Version 3 - RGB)"""
    
    def __init__(self):
        # Initialize configurations
        train_config = DDPMv3TrainingConfig()
        model_config = DDPMv3ModelConfig()
        scheduler_config = DDPMv3SchedulerConfig()
        
        super().__init__(train_config, model_config, scheduler_config)
    
    def _create_unet(self) -> UNet2DModel:
        """Create the UNet model"""
        return UNet2DModel(**self.model_config.config)
    
    def _create_scheduler(self) -> Any:
        """Create the DDPM scheduler"""
        return DDPMScheduler(**self.scheduler_config.config)
    
    def _inference_step(self, model_input: torch.Tensor, timestep: torch.Tensor, 
                       encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform a single inference step"""
        # For DDPM, we predict noise
        noise_pred = self.unet(
            model_input, 
            timestep, 
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        return noise_pred
