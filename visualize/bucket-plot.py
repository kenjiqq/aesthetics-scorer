# Inspired by https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/visulaize_100k_from_LAION400M.py

import pandas as pd
from tqdm import tqdm
import torch
import os
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from aesthetics_scorer.model import load_model as openclip_load_model, preprocess

validate_df = pd.read_parquet("parquets/validate_split.parquet")

def openclip_rater(model_name, embeddings_file):
    model = openclip_load_model(f"aesthetics_scorer/{model_name}.pth").to("cuda")
    embeddings_df = pd.read_parquet(embeddings_file)
    
    # Function to apply the model on a single row
    def predict(row):
        embedding = np.array(row["pooled_output"])
        embedding = preprocess(torch.from_numpy(embedding).unsqueeze(0)).to("cuda")
        with torch.no_grad():
            prediction = model(embedding)
        return prediction.item()
    
    return model_name, embeddings_df, predict

def get_image_data_as_base64(image_path, height=200):
    """Read image file, resize it, and return base64 encoded data."""
    with Image.open(image_path) as img:
        aspect_ratio = img.width / img.height
        new_width = int(height * aspect_ratio)
        img_resized = img.resize((new_width, height))
        
        buffer = BytesIO()
        img_resized.save(buffer, format="JPEG")
        img_data = buffer.getvalue()
        
    return base64.b64encode(img_data).decode('utf-8')

def generate_bucket_section(a, b, total_part):
    """Generate an HTML section for the given bucket range."""
    count_part = len(total_part) / len(df) * 100
    estimated = int(len(total_part))
    part = total_part[:50]
    html = f"<h2>In bucket {a} - {b} there is {count_part:.2f}% samples:{estimated:.2f} </h2> <div>"
    for image_name in part["image_name"]:
        image_path = os.path.join("dataset", image_name)
        b64_img = get_image_data_as_base64(image_path)
        html += f'<img src="data:image/jpeg;base64,{b64_img}" height="200" />'
    html += "</div>"
    return html

def make_html(name, df):
    buckets = [(i, i + 1) for i in range(20)]
    html = f'<h1>Rated images in buckets for {name}</h1>'

    for [a, b] in buckets:
        a = a / 2
        b = b / 2
        total_part = df[((df[name]) * 1 >= a) & ((df[name]) * 1 <= b)]
        html += generate_bucket_section(a, b, total_part)

    with open(f"./visualize/visualize-{name}.html", "w") as f:
        f.write(html)

for rater in [
            lambda: openclip_rater("aesthetics_scorer_openclip_vit_bigg_14", "parquets/openclip_vit_bigg_14.parquet"), 
            lambda: openclip_rater("aesthetics_scorer_openclip_vit_h_14", "parquets/openclip_vit_h_14.parquet"), 
            lambda: openclip_rater("aesthetics_scorer_openclip_vit_l_14", "parquets/openclip_vit_l_14.parquet"), 
        ]:
    (name, embeddings_df, predict) = rater()
    df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")
    
    # Apply the model on each row
    df[name] = df.apply(predict, axis=1)

    make_html(name, df)
