import pandas as pd
import os
import zipfile
import shutil
from PIL import Image
import utils.download as download
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

VERIFY_EXISTING_IMAGES = False # slow
DOWNLOAD_FROM_CDN = True

DATASET_PATH = "W:/diffusiondb/"
DATASET_TARGET = "W:/dataset/"
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
    missing_images = []
    print("Verifying images...")
    for img_name in tqdm(df["image_name"]):
            dataset_img_path = os.path.join(DATASET_TARGET, img_name)
            if not os.path.exists(dataset_img_path):
                missing_images.append(img_name)
            elif VERIFY_EXISTING_IMAGES:
                try:
                    Image.open(dataset_img_path).verify()
                except Exception:
                    print(f"Image {img_name} is corrupted, will download...")
                    missing_images.append(img_name)
    print(f"Missing {len(missing_images)} images, downloading...")

    def download_and_save_image(img_name):
        dataset_img_path = os.path.join(DATASET_TARGET, img_name)
        download_url = f'{CDN_URL}{img_name}'
        download_image(download_url, dataset_img_path)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_img_name = {executor.submit(download_and_save_image, img_name): img_name for img_name in missing_images}
        for future in tqdm(as_completed(future_to_img_name), total=len(missing_images), desc="Downloading images"):
            img_name = future_to_img_name[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f'{img_name} generated an exception: {exc}')

# else:
#     for index, part_group in df.groupby("part_id"):
#         print(f"Parsing part {index}, {len(part_group)} images...")
#         zip_path = os.path.join(DATASET_PATH, f"part-{str(index).zfill(6)}.zip")
#         if not os.path.exists(zip_path):
#             print(f"missing part {index}, downloading...")
#             download_zip(index)
#         else:
#             try:
#                 with zipfile.ZipFile(zip_path) as z:
#                     if len(z.filelist) != 1001:
#                         print(f"invalid filecount in part {index}, found {len(z.filelist)}, will download...")
#                         download_zip(index)
#             except zipfile.BadZipFile:
#                 print(f"invalid zip file in part {index}, will download...")
#                 download_zip(index)

#         for img_name in part_group["image_name"]:
#             dataset_img_path = os.path.join(DATASET_TARGET, img_name)
#             if not os.path.exists(dataset_img_path):
#                 fetch_file_from_zip(zip_path, img_name, dataset_img_path)
#             elif VERIFY_EXISTING_IMAGES:
#                 try:
#                     Image.open(dataset_img_path).verify()
#                 except Exception:
#                     fetch_file_from_zip(zip_path, img_name, dataset_img_path)
                                