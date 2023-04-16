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

BUNDLE_IMAGES = False

validate_df = pd.read_parquet("parquets/validate_split.parquet")

def openclip_rater(model_name, embeddings_file, min=1, max=10):
    model = openclip_load_model(f"aesthetics_scorer/models/{model_name}.pth").to("cuda")
    embeddings_df = pd.read_parquet(embeddings_file)
    
    # Function to apply the model on a single row
    def predict(row):
        embedding = np.array(row["pooled_output"])
        embedding = preprocess(torch.from_numpy(embedding).unsqueeze(0)).to("cuda")
        with torch.no_grad():
            prediction = torch.clamp(model(embedding), min=min, max=max)
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
    part = total_part[:100]
    html = f"<h2>In bucket {a} - {b} there is {count_part:.2f}% samples:{estimated:.2f} </h2> <div>"
    for image_name in part["image_name"]:
        if BUNDLE_IMAGES:
            image_path = os.path.join("dataset", image_name)
            b64_img = get_image_data_as_base64(image_path)
            html += f'<img src="data:image/jpeg;base64,{b64_img}" height="200" />'
        else:
            html += f'<img src="https://s3cdn.stability.ai/diffusion-db/{image_name}" height="200" />'
    html += "</div>"
    return html

def make_html(name, df, min, max):
    BUCKET_STEP = 0.5
    buckets = [(i, i + BUCKET_STEP) for i in np.arange(min, max, BUCKET_STEP)]
    
    html = f'<h1>{name} scores on {len(df)} validation samples</h1>'
    for [a, b] in buckets:
        total_part = df[((df[name]) * 1 >= a) & ((df[name]) * 1 <= b)]
        html += generate_bucket_section(a, b, total_part)

    with open(f"./visualize/visualize-{name}.html", "w") as f:
        f.write(html)

for config in [
            (lambda min, max: openclip_rater("aesthetics_scorer_rating_openclip_vit_bigg_14", "parquets/openclip_vit_bigg_14.parquet", min, max), 1, 10), 
            (lambda min, max: openclip_rater("aesthetics_scorer_rating_openclip_vit_h_14", "parquets/openclip_vit_h_14.parquet", min, max), 1, 10),
            (lambda min, max: openclip_rater("aesthetics_scorer_rating_openclip_vit_l_14", "parquets/openclip_vit_l_14.parquet", min, max), 1, 10) ,
            (lambda min, max: openclip_rater("aesthetics_scorer_artifacts_openclip_vit_bigg_14", "parquets/openclip_vit_bigg_14.parquet", min, max), 0, 5) ,
            (lambda min, max: openclip_rater("aesthetics_scorer_artifacts_openclip_vit_h_14", "parquets/openclip_vit_h_14.parquet", min, max), 0, 5),
            (lambda min, max: openclip_rater("aesthetics_scorer_artifacts_openclip_vit_l_14", "parquets/openclip_vit_l_14.parquet", min, max), 0, 5),
        ]:
    (name, embeddings_df, predict) = config[0](config[1], config[2])
    df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")
    
    # Apply the model on each row
    df[name] = df.apply(predict, axis=1)

    make_html(name, df, min=config[1], max=config[2])
