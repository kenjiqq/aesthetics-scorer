# Based on from https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/visulaize_100k_from_LAION400M.py

import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ChainDataset
import json

from PIL import Image, ImageFile

from aesthetics_scorer.model import load_model, preprocess
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
SCORER_MODEL = "aesthetics_scorer/aesthetics_scorer_openclip_vit_h_14.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(CLIP_MODEL)
vision_model = model.vision_model
vision_model.to(device)
del model
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

rating_model = load_model(SCORER_MODEL).to(device)

vision_model.eval()
rating_model.eval()

urls= []
predictions=[]

# this will run inference over 10 webdataset tar files from LAION 400M and sort them into 20 categories
# you can DL LAION 400M and convert it to wds tar files with img2dataset ( https://github.com/rom1504/img2dataset ) 

datasets = []
for j in range(100):
   if j<10:
     # change the path to the tar files accordingly
     datasets.append( wds.WebDataset("laion2b-en/dataset/0000"+str(j)+".tar"))
   else:
     datasets.append(wds.WebDataset("laion2b-en/dataset/000"+str(j)+".tar"))

dataset = ChainDataset(datasets)

def collate_fn(batch):
   for item in batch:
      item["url"] = json.loads(item['json'])["url"]
   return batch

data_loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_fn )
batch_metadata = []
batch_images = []
for i, batch in tqdm(enumerate(data_loader)):
   batch_urls = [item["url"] for item in batch]
   pil_images = [Image.open(io.BytesIO(item["jpg"])) for item in batch]
   
   images = clip_processor(images=pil_images, return_tensors="pt").to(device)

   with torch.no_grad():
      image_features = vision_model(**images)

   embedding = preprocess(image_features.pooler_output)
   with torch.no_grad():
      prediction = rating_model(embedding)
   urls.extend(batch_urls)
   predictions.extend(prediction.detach().cpu())


df = pd.DataFrame(list(zip(urls, predictions)),
               columns =['filepath', 'prediction'])


BUCKET_STEP = 0.25
buckets = [(i, i + BUCKET_STEP) for i in np.arange(0, 10+BUCKET_STEP, BUCKET_STEP)]


html= f"<h1>Aesthetic scores for {len(predictions)} LAION 5b samples</h1>"

for [a,b] in buckets:
    total_part = df[(  (df["prediction"] ) *1>= a) & (  (df["prediction"] ) *1 <= b)]
    count_part = len(total_part) / len(df) * 100
    estimated =int ( len(total_part) )
    part = total_part[:50]

    html+=f"<h2>In bucket {a} - {b} there is {count_part:.2f}% samples:{estimated:.2f} </h2> <div>"
    for filepath in part["filepath"]:
        html+='<img src="'+filepath +'" height="200" />'

    html+="</div>"
with open("visualize/laion5b-visualize.html", "w") as f:
    f.write(html)
    
