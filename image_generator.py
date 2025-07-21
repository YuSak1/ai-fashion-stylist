from openai import OpenAI
import os
import requests
from PIL import Image
from io import BytesIO
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_outfit_image(prompt: str) -> Image.Image:
    print("Generating an image...")
    start_time = time.time()

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    image_url = response.data[0].url
    image_bytes = requests.get(image_url).content
    image = Image.open(BytesIO(image_bytes))

    end_time = time.time()
    duration = end_time - start_time
    print(f"Image is generated. Process time: {duration:.2f} seconds.")

    return image
