import numpy as np
from sklearn.cluster import KMeans


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

    return quantized_images, rgb_colors, model


def convert_images_to_rgb(images, centroids):
    """
    Converts a 16x16 OHT image to a 16x16x3 RGB image
    :param image: OHT 2d array
    :param centroids:
    :return:
    """
    return np.asarray([centroids[i] for i in images])