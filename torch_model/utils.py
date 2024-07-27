import torch
import numpy as np
from PIL import Image
from typing import List


def decode_image_batch(
    image_tensor_batch: torch.Tensor, palette: np.ndarray
) -> List[Image.Image]:
    out_imgs = []
    for batch_idx in range(image_tensor_batch.shape[0]):
        img_tensor = image_tensor_batch[batch_idx].argmax(dim=-1).cpu()
        rgb_image = palette[img_tensor]
        pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8))
        out_imgs.append(pil_image)

    return out_imgs

def image_grid(image_list: List[List[Image.Image]]) -> Image.Image:
    num_rows, num_cols = len(image_list), len(image_list[0])
    image_width, image_height = image_list[0][0].size

    grid_width = num_cols * image_width
    grid_height = num_rows * image_height

    grid_image = Image.new("RGB", (grid_width, grid_height))

    for row in range(num_rows):
        for col in range(num_cols):
            x_offset = col * image_width
            y_offset = row * image_height
            grid_image.paste(image_list[row][col], (x_offset, y_offset))
    return grid_image
