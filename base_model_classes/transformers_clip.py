
from transformers import CLIPModel, CLIPProcessor
import torch

class TransformersClip(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        model = CLIPModel.from_pretrained(model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
        self.vision_model
        self.visual_projection
        del model

    def forward(self, inputs):
        vision_output = self.vision_model(**inputs)
        pooled_output = vision_output.pooler_output
        projected_embedding = self.visual_projection(pooled_output)
        return pooled_output.cpu().detach(), projected_embedding.cpu().detach()
        
    def process_images(self, pil_images):
        return self.processor(images=pil_images, return_tensors="pt")
