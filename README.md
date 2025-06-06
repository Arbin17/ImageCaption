# 🖼️ Image Captioning Demo using FastAPI + ViT-GPT2

A simple web app to generate captions for uploaded images using the [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) model.

✅ Upload image  
✅ Display image  
✅ Generate caption using Vision Transformer + GPT2  
✅ Built with FastAPI and HTML



## How it works

- Frontend: HTML upload form
- Backend: FastAPI + Transformers
- Model: ViT encoder + GPT2 decoder (`nlpconnect/vit-gpt2-image-captioning`)
- Displays both uploaded image and generated caption

---

## Setup Instructions

### 1️⃣ Clone this repo

```bash
git clone https://github.com/Arbin17/ImageCaption
cd ImageCaption
```
### 2️⃣ Create virtual environment
```
python -m venv venv
source venv/bin/activate  # Linux / Mac
# OR
venv\Scripts\activate     # Windows
```
3️⃣ Install dependencies
```
pip install -r requirements.txt
```
