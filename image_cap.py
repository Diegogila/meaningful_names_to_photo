from transformers import Blip2Processor, Blip2ForConditionalGeneration
import gradio as gr
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
import requests
from io import BytesIO

# Initialize the processor and model from pre-trained weights
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

def caption_image(image=None, text=None):
    """
    Generate a caption for an image or a website URL.
    
    Args:
        image (np.ndarray): The input image.
        text (str): The website URL.
    
    Returns:
        str: The generated caption.
    """
    if image is not None:
        return caption_by_image(image)
    elif text is not None:
        return caption_by_website(text)
    else:
        return "No input provided."

def caption_by_image(input_image: np.ndarray):
    """
    Generate a caption for an input image.
    
    Args:
        input_image (np.ndarray): The input image in numpy array format.
    
    Returns:
        str: The generated caption.
    """
    text = "the image of"
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the image and generate a caption
    inputs = processor(images=raw_image, text=text, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption.capitalize()

def caption_by_website(url):
    """
    Generate captions for images found on a website.
    
    Args:
        url (str): The website URL.
    
    Returns:
        str: The generated captions for all images on the website.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_elements = soup.find_all('img')
    caption_list = ""
    for img in img_elements:
        img_url = img.get('src')

        # Skip certain types of images
        if 'svg' in img_url or '1x1' in img_url:
            continue

        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue

        try:
            response = requests.get(img_url)
            raw_img = Image.open(BytesIO(response.content))

            # Skip small images
            if raw_img.size[0] * raw_img.size[1] < 400:
                continue

            raw_img = raw_img.convert('RGB')

            # Process the image and generate a caption
            inputs = processor(raw_img, return_tensors='pt')
            outputs = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            caption_list = caption_list + f"- {caption.capitalize()}\n"
            
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

    return caption_list

def update_interface(input_type):
    """
    Update the interface based on the selected input type.
    
    Args:
        input_type (str): The selected input type ("Image" or "Text").
    
    Returns:
        tuple: The updated visibility states for the image and text inputs.
    """
    if input_type == "Image":
        return gr.Image(visible=True), gr.Textbox(visible=False)
    elif input_type == "Text":
        return gr.Image(visible=False), gr.Textbox(visible=True)
    else:
        return gr.Image(visible=False), gr.Textbox(visible=False)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Captioning")
    gr.Markdown("Choose whether to input an image or text for captioning.")

    input_type = gr.Radio(
        choices=["Image", "Text"],
        label="Select input type",
        value="Image"
    )

    image_input = gr.Image(visible=True, label="Upload Image")
    text_input = gr.Textbox(visible=False, label="Enter a Website Url")

    generate_button = gr.Button("Generate Caption")

    output = gr.Textbox(label="Caption Output", max_lines=500)

    # Update the interface when the input type changes
    input_type.change(
        update_interface,
        inputs=input_type,
        outputs=[image_input, text_input]
    )

    # Generate the caption when the button is clicked
    generate_button.click(
        caption_image,
        inputs=[image_input, text_input],
        outputs=output
    )

# Launch the Gradio demo
demo.launch(share=True)

