# ollama_captioner.py
# Captioner that uses Ollama's native Python SDK for vision models.
# More reliable than OpenAI-compatible API for Ollama vision models.

import base64
from io import BytesIO
from typing import List

import ollama
from PIL import Image

from remembr.captioners.captioner import Captioner

DEFAULT_PROMPT = (
    "Briefly describe the objects appearing in the video and their most prominent features, "
    "such as a black trash can, a fire hydrant, lift door and etc. Specifically focus on the objects, environmental "
    "features, events/activities, and other interesting details. "
    "Also briefly describe the entire scene. Keep the entire description as concise as possible."
)


class OllamaCaptioner(Captioner):
    """
    Captioner that uses Ollama's native Python SDK for vision models.

    This is more reliable than the OpenAI-compatible API for Ollama vision models
    like qwen2.5vl, qwen3-vl, llava, minicpm-v, etc.

    Args:
        model: Ollama vision model name (e.g., "qwen3-vl:8b", "minicpm-v:8b")
        host: Ollama server URL (default: "http://localhost:11434")
        prompt: Custom prompt for captioning (optional)
        options: Additional Ollama options (temperature, num_ctx, etc.)
    """

    def __init__(
        self,
        model: str = "qwen3-vl:4b",
        host: str = "http://localhost:11434",
        prompt: str = None,
        temperature: float = 0.2,
        num_ctx: int = 4096,
        **kwargs
    ):
        self.model = model
        self.host = host
        self.prompt = prompt or DEFAULT_PROMPT
        self.temperature = temperature
        self.num_ctx = num_ctx

        # Initialize Ollama client
        self.client = ollama.Client(host=host)

        print("Using OllamaCaptioner:")
        print(f"\tModel: {self.model}")
        print(f"\tHost: {self.host}")

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        # Convert RGBA to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def caption(self, images: List[Image.Image]) -> str:
        """
        Generate caption for multiple images using Ollama's native API.

        Args:
            images: List of PIL Images

        Returns:
            Caption text as string
        """
        if not images:
            return "No images provided"

        # Convert all images to base64
        image_data = [self._image_to_base64(img) for img in images]

        try:
            # Call Ollama chat API with images
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': self.prompt,
                        'images': image_data  # Pass base64 images
                    }
                ],
                options={
                    'temperature': self.temperature,
                    'num_ctx': self.num_ctx,
                },
                keep_alive="30m"  # Keep model loaded for 30 minutes
            )

            return response['message']['content'].strip()

        except Exception as e:
            raise RuntimeError(f"Ollama captioning failed: {e}") from e
