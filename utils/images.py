import numpy as np
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


def convert_images_to_rgb(images, centroids):
    """
    Converts a 16x16 OHT image to a 16x16x3 RGB image
    :param image: OHT 2d array
    :param centroids:
    :return:
    """
    return np.asarray([centroids[i] for i in images])