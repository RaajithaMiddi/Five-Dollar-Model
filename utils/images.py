from typing import List

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans


def _oht_images(images, num_levels=16):
    """
    Converts quantized RGB images to OHT quantized ndarray of images.

    :param images: ndarray of n images by lxw size with value equal to a quantized level
    :param levels: the number of levels
    :return: an ndarray that adds an additional dimension that is the OHT vector of levels
    """
    n, l, w = images.shape
    # Initialize the one-hot encoded image
    oht_images = np.zeros((n, l, w, num_levels), dtype=np.uint8)

    # Populate the one-hot encoded image
    for i in range(n):
        for j in range(num_levels):
            oht_images[i, :, :, j] = (images[i, :, :] == j).astype(np.uint8)

    return oht_images


def quantize_images(images, n_colors=16, model=None, seed=0):
    """

    :param images: (n, w, h, 3) numpy array
    :param n_colors: number of RGB colors to quantize
    :param model: k-means model; overrides n_colors if present
    :return: OHT images (n, w, h) with value up to n_colors
             RGB values of encoding
             k-means model in case it needs to predict future
    """

    n, l, w, d = images.shape
    pixels = np.reshape(images, (n * l * w, d))

    if not model:
        model = KMeans(n_clusters=n_colors, random_state=seed).fit(pixels)

    rgb_colors = model.cluster_centers_.astype(int)

    quantized_pixels = model.predict(pixels)
    quantized_images = np.reshape(quantized_pixels, (n, l, w))

    return _oht_images(quantized_images), rgb_colors, model


def _oht_to_rgb_numpy(image, palette):
    active_level = np.argmax(image, axis=-1)
    return np.asarray(palette[active_level], dtype=np.uint8)


def _oht_to_rgb_torch(images, palette):
    img_tensor = images.argmax(dim=-1)
    return palette[img_tensor].cpu().numpy().astype(np.uint8)


def convert_images_to_rgb(images, palette, library='numpy'):
    """
    Converts a 16x16x16 OHT image to a 16x16x3 RGB image
    :param image: OHT 2d array
    :param centroids:
    :return:
    """
    if library == 'numpy':
        return np.asarray([_oht_to_rgb_numpy(i, palette) for i in images])

    return _oht_to_rgb_torch(images, palette)


def decode_image_batch(image_batches: torch.Tensor, palette: torch.Tensor) -> List[Image.Image]:
    out_imgs = []

    for batch_idx in range(image_batches.shape[0]):
        rgb_images = convert_images_to_rgb(image_batches[batch_idx], palette, library='torch')
        pil_image = Image.fromarray(rgb_images)
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