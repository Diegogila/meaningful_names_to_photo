from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr
import numpy as np
from PIL import Image


processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    text = "the image of"
    raw_image = Image.fromarray(input_image).convert('RGB')

    inputs = processor(images=raw_image, text=text, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption


iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
    )

iface.launch()