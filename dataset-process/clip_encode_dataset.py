import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import os

from transformers import CLIPModel, CLIPProcessor

configs = [
    {
        "MODEL": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "FILE_NAME": "openclip_vit_bigg_14"
    },
    {
        "MODEL": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "FILE_NAME": "openclip_vit_h_14"
    },
    {
        "MODEL": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "FILE_NAME": "openclip_vit_l_14"
    }
]

BATCH_SIZE = 400

df = pd.read_parquet('parquets/prepared_hord_diffusion_dataset.parquet')

for config in configs:
    MODEL = config["MODEL"]
    FILE_NAME = config["FILE_NAME"]
    embedding_file_path = f"parquets/{FILE_NAME}.parquet"
    if os.path.exists(embedding_file_path):
        result_df = pd.read_parquet(embedding_file_path)
    else:
        result_df = pd.DataFrame(columns=["image_name", "pooled_output", "projected_embedding"])

    missing_image_names = df[~df["image_name"].isin(result_df["image_name"])]["image_name"].unique()

    print(f"Missing {len(missing_image_names)} embeddings...")

    model = CLIPModel.from_pretrained(MODEL)
    vision_model = model.vision_model
    visual_projection = model.visual_projection
    vision_model.to("cuda")
    visual_projection.to("cuda")
    del model
    processor = CLIPProcessor.from_pretrained(MODEL)

    for pos in tqdm(range(0, len(missing_image_names), BATCH_SIZE)):
        name_batch = missing_image_names[pos:pos+BATCH_SIZE]
        image_paths = [f"dataset/{id}" for id in name_batch]
        pil_images = [Image.open(image_path) for image_path in image_paths]
        inputs = processor(images=pil_images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            vision_output = vision_model(**inputs)
            pooled_output = vision_output.pooler_output
            projected_embedding = visual_projection(pooled_output)
            result_df = pd.concat([result_df, pd.DataFrame({
                "image_name": name_batch, 
                "pooled_output": list(pooled_output.cpu().detach().numpy()),
                "projected_embedding": list(projected_embedding.cpu().detach().numpy()),
            })], ignore_index=True)

    result_df.to_parquet(embedding_file_path)








