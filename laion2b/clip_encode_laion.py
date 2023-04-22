# Based on from https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/visulaize_100k_from_LAION400M.py

import webdataset as wds
from PIL import Image
import io
import json
import torch
from torchvision import datasets
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, ChainDataset
import json
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from laion2b.data import get_dataloader

CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
  model = CLIPModel.from_pretrained(CLIP_MODEL)
  vision_model = model.vision_model
  visual_projection = model.visual_projection
  vision_model.to(device)
  visual_projection.to(device)
  del model
  clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

  vision_model.eval()

  result_df = pd.DataFrame(columns=["image_url", "pooled_output", "projected_embedding"])

  data_loader = get_dataloader(512)

  batch_metadata = []
  batch_images = []
  for i, batch in enumerate(tqdm(data_loader)):
      batch_urls, pil_images = batch
    
      images = clip_processor(images=pil_images, return_tensors="pt").to(device)

      with torch.no_grad():
          vision_output = vision_model(**images)
      pooled_output = vision_output.pooler_output
      projected_embedding = visual_projection(pooled_output)
      result_df = pd.concat([result_df, pd.DataFrame({
          "image_url": batch_urls, 
          "pooled_output": list(pooled_output.cpu().detach().numpy()),
          "projected_embedding": list(projected_embedding.cpu().detach().numpy()),
      })], ignore_index=True)

  result_df.to_parquet("parquets/laion_embeddings.parquet")
