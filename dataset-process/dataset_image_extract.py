import requests
import pandas as pd
import os
import zipfile
import shutil
from PIL import Image

VERIFY_EXISTING_IMAGES = False # slow

DATASET_PATH = "W:/diffusiondb/"

df = pd.read_parquet('parquets/prepared_hord_diffusion_dataset.parquet')

image_names = df["image_name"].unique()

def fetch_file(zip_path, img_name, dataset_img_path):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(img_name) as zf:
            with open(dataset_img_path, "wb") as f:
                shutil.copyfileobj(zf, f)

for index, part_group in df.groupby("part_id"):
    print(f"Parsing part {index}")
    zip_path = os.path.join(DATASET_PATH, f"part-{str(index).zfill(6)}.zip")
    for img_name in part_group["image_name"]:
        dataset_img_path = os.path.join("dataset", img_name)
        if not os.path.exists(dataset_img_path):
            fetch_file(zip_path, img_name, dataset_img_path)
        elif VERIFY_EXISTING_IMAGES:
            try:
                Image.open(dataset_img_path).verify()
            except Exception:
                fetch_file(zip_path, img_name, dataset_img_path)
                            