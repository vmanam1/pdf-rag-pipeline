import hashlib
import os

USE_TRANSFORMERS_EMBEDDINGS = os.environ.get("USE_TRANSFORMERS_EMBEDDINGS", "false").lower() == "true"

if USE_TRANSFORMERS_EMBEDDINGS:
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


def _hash_to_vector(payload, dimensions=512):
    digest = hashlib.sha256(payload).digest()
    vector = []
    counter = 0

    while len(vector) < dimensions:
        counter_bytes = counter.to_bytes(4, byteorder="big", signed=False)
        block = hashlib.sha256(digest + counter_bytes).digest()
        for index in range(0, len(block), 4):
            chunk = block[index:index + 4]
            if len(chunk) < 4:
                continue
            integer = int.from_bytes(chunk, byteorder="big", signed=False)
            scaled = (integer / 4294967295.0) * 2.0 - 1.0
            vector.append(scaled)
            if len(vector) == dimensions:
                break
        counter += 1

    return vector

def vectorize_text(input_text):
    """
    Converts a text string into an embedding using the CLIP model.

    Args:
        input_text (str): Input text string to be vectorized.

    Returns:
        list: The embedding of the text as a list.
    """
    if not USE_TRANSFORMERS_EMBEDDINGS:
        return _hash_to_vector(input_text.encode("utf-8"))

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
    if not USE_TRANSFORMERS_EMBEDDINGS:
        with open(image_path, "rb") as image_file:
            return _hash_to_vector(image_file.read())

    try:
        from PIL import Image

        clip_processor, clip_model = _get_clip_model_and_processor()
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)
        return image_embedding.squeeze().numpy().tolist()
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}")
        return None