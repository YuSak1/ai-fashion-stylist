import gradio as gr
from openai import OpenAI
from PIL import Image
import base64
import io
import os
import requests
from bs4 import BeautifulSoup
from image_generator import generate_outfit_image
from datetime import datetime

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_image(query: str):
    # print("Searching for images...")
    url = f"https://www.glami.cz/?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if (
                "glami.cz" in src
                and src.endswith((".jpg", ".jpeg"))
                and "logo" not in src.lower()
                and "placeholder" not in src.lower()
                and "sprite" not in src.lower()
                and src.startswith("http")
            ):
                return src

    except Exception as e:
        print(f"Failed to fetch image for '{query}':", e)

    return None


def extract_keywords_with_gpt(suggestion_text: str) -> list[str]:
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Extract 2–4 short, searchable fashion product names from this suggestion:\n\n"
                            f"{suggestion_text}\n\n"
                            "Return a comma-separated list like: 'beige sweatshirt, black trousers'"
                        )
                    }
                ]
            }],
            max_tokens=100
        )
        raw_keywords = response.choices[0].message.content.strip()
        return [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]

    except Exception as e:
        print("Keyword extraction failed:", e)
        return []


def create_image_gallery(suggestion_text: str) -> tuple[list, list[str]]:
    keywords = extract_keywords_with_gpt(suggestion_text)
    gallery = []
    urls = []

    for keyword in keywords:
        query = keyword.replace(" ", "+")
        search_url = f"https://www.glami.cz/?q={query}"
        image_url = get_image(keyword)
        urls.append(search_url)
        if image_url:
            label = f"[{keyword}]({search_url})"
            gallery.append((image_url, label))

    return gallery, urls


def detect_and_suggest(image: Image.Image, gender: str):
    try:
        base64_image = encode_image(image)

        text_prompt = (
            f"You are a professional fashion stylist for {gender.lower()} fashion. "
            "First, describe the fashion item in the image in 1 sentence. "
            "Always mention the color and type of the item. Try to describe in details as much as possible. "
            "Even if the item is unclear or difficult to find, describe any fashion item visible in the image. "
            
            "Then, suggest 3 outfit items that would go well with it. Show them in a list. "
            "Use clear, stylish, modern language. "
            
            """
            Output will be in the following format:

            This is ...
            1. ...
            2. ...
            3. ...
            """
        )

        # print("text_prompt: \n", text_prompt)
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=250
        )

        full_response = response.choices[0].message.content.strip()

        if "\n" in full_response:
            lines = full_response.split("\n", 1)
            description = lines[0].strip()
            suggestion = lines[1].strip()
        else:
            description = full_response
            suggestion = ""

        prompt = (
            f"Image of 1 person with outfit for {gender.lower()}. "
            f"Description of the outfit items: {description} {suggestion} "
            "Make sure the details of the items can be seen in the image."
            "Do not use any other items that are not mentioned."
            "Super realistic photo with cinematic background."
        )

        print("=====================================================")
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(time_now)
        print("prompt: \n", prompt)
        outfit_image = generate_outfit_image(prompt)
        gallery, search_urls = create_image_gallery(suggestion)

        markdown_links = (
            "<h2>Search Links to GLAMI</h2>\n<ul style='font-size: 20px;'>"
            + "".join([
                f"<li><a href='{url}' target='_blank'>{url.split('=')[-1].replace('+', ' ').capitalize()}</a></li>"
                for url in search_urls
            ])
            + "</ul>"
        )

        # return description, suggestion, outfit_image, gallery, markdown_links, gr.update(value="Run Again")
        return description, suggestion, outfit_image, markdown_links, gr.update(value="Run Again (if you get errors or want different results)")

    except Exception as e:
        return f"Error: {e}", "", None, [], ""


example_list_path = [os.path.join("example_img", ex) for ex in os.listdir("example_img")]
with gr.Blocks().queue() as demo:
    gr.HTML("<center><h1>🧥 AI Fashion Stylist 👟</h1></center>")
    gr.HTML("<center><h3>1.Upload a fashion item. &ensp; 2.Select target gender. &ensp; 3.Click Run button to start.</p>"
            "<h3>The AI recommends matching fashion items and generates a realistic model image.✨</p></center>")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload a fashion item")
            gender_input = gr.Radio(choices=["Men", "Women", "Unisex"], label="Select Target Gender", value="Unisex")
            inputs = [image_input, gender_input]

            example = gr.Examples(
                inputs=image_input,
                examples_per_page=6,
                examples=example_list_path
            )

            try_button = gr.Button(value="Run", variant='primary')

        with gr.Column():
            description_output = gr.Textbox(label="Detected Item")
            suggestion_output = gr.Textbox(label="Suggested Outfit")
            image_output = gr.Image(label="AI-Generated Outfit", type="pil")
            # gallery_output = gr.Gallery(label="Matching Items from GLAMI", columns=3, height="auto")
            url_output = gr.Markdown(label="Search URLs")

    # Linking the button to the processing function
    try_button.click(
        fn=detect_and_suggest,
        inputs=inputs,
        outputs=[
            description_output,
            suggestion_output,
            image_output,
            # gallery_output,
            url_output,
            try_button
        ])


if __name__ == "__main__":
    demo.launch()
