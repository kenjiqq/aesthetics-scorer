import gradio as gr
import torch
from model import preprocess, load_model
from transformers import CLIPModel, CLIPProcessor

MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(MODEL)
vision_model = model.vision_model
vision_model.to(DEVICE)
del model
clip_processor = CLIPProcessor.from_pretrained(MODEL)

rating_model = load_model("aesthetics_scorer/models/aesthetics_scorer_rating_openclip_vit_h_14.pth").to(DEVICE)

def predict(img):
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_output = vision_model(**inputs)
    pooled_output = vision_output.pooler_output
    embedding = preprocess(pooled_output)
    with torch.no_grad():
        output = rating_model(embedding)
    return output.detach().cpu().item()

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs="number"
).launch()