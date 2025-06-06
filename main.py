from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import io
import os
import uuid

# Load model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Init FastAPI app
app = FastAPI()

# Setup templates folder
templates = Jinja2Templates(directory="templates")

# Create static/images folder if not exists
os.makedirs("static/images", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper function to generate caption
def generate_caption(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Use greedy search (no beam search for GPT2!)
    output_ids = model.generate(pixel_values, max_length=16)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# HTML form endpoint
@app.get("/", response_class=HTMLResponse)
async def form_post(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "caption": None, "image_path": None})

# Image upload endpoint
@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Save image to static/images folder
    image_id = str(uuid.uuid4())
    image_filename = f"static/images/{image_id}.png"
    with open(image_filename, "wb") as f:
        f.write(image_bytes)

    # Generate caption
    caption = generate_caption(image_bytes)

    # Return HTML with caption and image path
    return templates.TemplateResponse("index.html", {
        "request": request,
        "caption": caption,
        "image_path": "/" + image_filename
    })
