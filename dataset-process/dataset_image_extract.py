import pandas as pd
import os
import zipfile
import shutil
from PIL import Image
import utils.download as download
import requests
from tqdm import tqdm

VERIFY_EXISTING_IMAGES = False # slow
DOWNLOAD_FROM_CDN = True

DATASET_PATH = "W:/diffusiondb/"
CDN_URL = "https://s3cdn.stability.ai/diffusion-db/"

df = pd.read_parquet('parquets/prepared_hord_diffusion_dataset.parquet')

image_names = df["image_name"].unique()

def download_zip(index):
    download.main(index, index+1, DATASET_PATH, large=True)

def fetch_file_from_zip(zip_path, img_name, dataset_img_path):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(img_name) as zf:
            with open(dataset_img_path, "wb") as f:
                shutil.copyfileobj(zf, f)

def download_image(url, dataset_img_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(dataset_img_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image, { url } Status code: {response.status_code}")

if DOWNLOAD_FROM_CDN:
    for img_name in tqdm(df["image_name"]):
            dataset_img_path = os.path.join("dataset", img_name)
            if not os.path.exists(dataset_img_path):
                download_url = f'{CDN_URL}{img_name}'
                download_image(download_url, dataset_img_path)
                
            elif VERIFY_EXISTING_IMAGES:
                try:
                    Image.open(dataset_img_path).verify()
                except Exception:
                    download_image(download_url, dataset_img_path)
else:
    for index, part_group in df.groupby("part_id"):
        print(f"Parsing part {index}, {len(part_group)} images...")
        zip_path = os.path.join(DATASET_PATH, f"part-{str(index).zfill(6)}.zip")
        if not os.path.exists(zip_path):
            print(f"missing part {index}, downloading...")
            download_zip(index)
        else:
            try:
                with zipfile.ZipFile(zip_path) as z:
                    if len(z.filelist) != 1001:
                        print(f"invalid filecount in part {index}, found {len(z.filelist)}, will download...")
                        download_zip(index)
            except zipfile.BadZipFile:
                print(f"invalid zip file in part {index}, will download...")
                download_zip(index)

        for img_name in part_group["image_name"]:
            dataset_img_path = os.path.join("dataset", img_name)
            if not os.path.exists(dataset_img_path):
                fetch_file_from_zip(zip_path, img_name, dataset_img_path)
            elif VERIFY_EXISTING_IMAGES:
                try:
                    Image.open(dataset_img_path).verify()
                except Exception:
                    fetch_file_from_zip(zip_path, img_name, dataset_img_path)
                                