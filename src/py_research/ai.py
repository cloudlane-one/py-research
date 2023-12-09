"""Functions for working with AI models."""

from base64 import b64encode
from io import BytesIO

import openai
import PIL.Image as img  # noqa: N813
import requests


def _encode_image(image: img.Image) -> str:
    """Encode image for sending to API."""
    img_bytes = BytesIO()
    image.save(img_bytes, format="jpeg")
    return b64encode(img_bytes.getbuffer()).decode("utf-8")


def describe_image(img: img.Image) -> str | None:
    """Describe given image via AI model."""
    base64_image = _encode_image(img)

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is shown in the following image? "
                        "Use 200 words or less. "
                        "Also mention what kind of image it is "
                        "(photo, drawing, painting, digital art, etc.), "
                        "as well as its style, composition and meaning.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content


def generate_image(desc: str) -> img.Image | None:
    """Generate image from given description via AI model."""
    client = openai.OpenAI()

    response = client.images.generate(
        model="dall-e-3",
        prompt=desc,
        size="1792x1024",
        quality="hd",
        n=1,
    )

    url = response.data[0].url
    if url is None:
        return None

    img_response = requests.get(url)
    return img.open(BytesIO(img_response.content))
