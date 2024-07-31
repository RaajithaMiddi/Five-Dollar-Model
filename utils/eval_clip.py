import images
import torch
from transformers import CLIPModel, CLIPProcessor


def get_clip_embeddings(images_oht, rgb_colors, labels):
    """
    Use CLIP to get embeddings for images and labels in order to compare with
    the five dollar model image results.

    :param images_oht: one-hot encoded images
    :param rgb_colors: RGB values of encoding
    :param labels: list of labels
    """

    # Load CLIP model
    clip_uri_name = "openai/clip-vit-base-patch32"
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained(clip_uri_name)
    CLIP_MODEL = CLIPModel.from_pretrained(clip_uri_name)

    rgb_images = images.convert_images_to_rgb(images_oht, rgb_colors)

    PRE_PROMPT = "a pixelated, pixel-art image of "
    annotated_labels = [PRE_PROMPT + l for l in labels]

    # Preprocess images and texts
    inputs = CLIP_PROCESSOR(
        text=annotated_labels, images=rgb_images, return_tensors="pt", padding=True
    )

    # Generate embeddings
    with torch.no_grad():
        outputs = CLIP_MODEL(**inputs)

    # Get image and text embeddings
    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds

    return image_embeddings, text_embeddings
