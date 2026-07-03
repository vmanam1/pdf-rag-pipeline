from functools import lru_cache

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


@lru_cache(maxsize=1)
def _get_clip_model_and_processor():
    return (
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    )

def vectorize_text(input_text):
    """
    Converts a text string into an embedding using the CLIP model.

    Args:
        input_text (str): Input text string to be vectorized.

    Returns:
        list: The embedding of the text as a list.
    """
    try:
        clip_processor, clip_model = _get_clip_model_and_processor()
        inputs = clip_processor(text=[input_text], return_tensors="pt", truncation=True)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs)
        return text_embedding.squeeze().numpy().tolist()
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

def vectorize_image(image_path):
    """
    Converts an image into an embedding using the CLIP model.

    Args:
        image_path (str): Path to the input image to be vectorized.

    Returns:
        list: The embedding of the image as a list.
    """
    try:
        clip_processor, clip_model = _get_clip_model_and_processor()
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)
        return image_embedding.squeeze().numpy().tolist()
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}")
        return None