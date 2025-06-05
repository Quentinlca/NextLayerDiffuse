import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# Import necessary libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Constants
from constants import *

def tensors_to_pil_images(output):
    # Convert a batch of tensors to PIL images.
    #get the batch dimension to a list
    # Remove the batch dimension to get a list of tensors
    tensor_batch = [img for img in output]
    tensor_batch_reshaped = []

    for i, img in enumerate(tensor_batch):
        img_reshaped = img.view(WINDOW_SIZE, 4, 64, 64)  # Reshape to [8, 4, 64, 64]
        tensor_batch_reshaped.append(img_reshaped)
        

    # Now you have a tensor where the first dimension represents each of the 8 images.
    # You can iterate through the first dimension to get each image and convert it to PIL.

    PIL_batch = []
    for img_tensor in tensor_batch_reshaped:
        pil_images = []
        for i in range(WINDOW_SIZE):
        # Get the tensor for the i-th image: [4, 64, 64]
            image_tensor = img_tensor[i]

            # Permute from [C, H, W] to [H, W, C]: [64, 64, 4]
            image_tensor_permuted = image_tensor.permute(1, 2, 0)

            # Convert to NumPy array and scale to [0, 255]
            image_np = (image_tensor_permuted.detach().cpu().numpy() * 255).astype(np.uint8)

            # Convert NumPy array to PIL Image
            image_pil = Image.fromarray(image_np, 'RGBA')
            pil_images.append(image_pil)
        PIL_batch.append(pil_images)
    return PIL_batch

def save_examples(batch_input_tensors, batch_target_tensors, batch_output_tensors, saving_dir):
    """
    Save input, target, and output tensors as images.
    
    Args:
        batch_input_tensors (torch.Tensor): Input tensors of shape [B, C, H, W].
        batch_target_tensors (torch.Tensor): Target tensors of shape [B, C, H, W].
        batch_output_tensors (torch.Tensor): Output tensors of shape [B, C, H, W].
        save_path (str): Path to save the images. If None, images will not be saved.
    """
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    # Convert tensors to PIL images
    input_images = tensors_to_pil_images(batch_input_tensors)
    target_images = tensors_to_pil_images(batch_target_tensors)
    output_images = tensors_to_pil_images(batch_output_tensors)

    # Display the images in a grid format
    
    for i in range(len(input_images)):
        fig, axes = get_single_image_grid(input_images[i], target_images[i], output_images[i])
        
        # Collect all figures as images, then concatenate vertically and save as one image
        fig.savefig(f"{saving_dir}/example_{i}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

def get_single_image_grid(input_images, target_images, output_images):
    """
    Create a grid of images for input, target, and output.
    
    Args:
        input_images (list): List of input images.
        target_images (list): List of target images.
        output_images (list): List of output images.
        
    Returns:
        fig, axes: Matplotlib figure and axes objects.
    """
    if len(input_images) != len(target_images) or len(input_images) != len(output_images):
        print("Warning: The number of input and output images do not match.")
        return

    num_images = len(input_images)
    # Create a figure and a set of subplots with 2 columns (input and output)
    fig, axes = plt.subplots(num_images, 3, figsize=(3, num_images))

    for i in range(num_images):
        # Display input image
        axes[i, 0].imshow(input_images[i])
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 0].axis('off') # Hide axes

        # Display target image
        axes[i, 1].imshow(target_images[i])
        axes[i, 1].set_title(f"Target {i+1}")
        axes[i, 1].axis('off') # Hide axes
        
        # Display target image
        axes[i, 2].imshow(output_images[i])
        axes[i, 2].set_title(f"Output {i+1}")
        axes[i, 2].axis('off') # Hide axes

    plt.tight_layout() # Adjust layout to prevent overlap
    return fig, axes


def display_single_image_grid(input_images, target_images, output_images):
    
    fig, axes = get_single_image_grid(input_images, target_images, output_images)
    # Display the grid of images
    plt.show()