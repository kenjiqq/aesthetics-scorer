import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import os
from base_model_classes.openclip import OpenClip
from base_model_classes.transformers_clip import TransformersClip
from torch.utils.data import Dataset, DataLoader

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" # avoid annoying cublas warning

configs = [
    # {
    #     "MODEL": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    #     "FILE_NAME": "openclip_vit_bigg_14",
    #     "TYPE": "transformers"
    # },
    # {
    #     "MODEL": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    #     "FILE_NAME": "openclip_vit_h_14",
    #     "TYPE": "transformers"
    # },
    # {
    #     "MODEL": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    #     "FILE_NAME": "openclip_vit_l_14",
    #     "TYPE": "transformers"
    # },
    {
        "MODEL": "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
        "FILE_NAME": "dfn5b_vit_h_14",
        "TYPE": "openclip"
    }
]

BATCH_SIZE = 300
DATASET_PATH = "W:/dataset/"
SAVE_EVERY_N_BATCHES = 20

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(os.path.join(DATASET_PATH, image_path))
        if self.transform:
            image = self.transform(image).squeeze(0)
        return image_path, image

def main():
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

        if len(missing_image_names) == 0:
            print(f"All embeddings already computed for ${MODEL}")
            continue

        print(f"Missing {len(missing_image_names)} embeddings...")

        if config["TYPE"] == "transformers":
            model = TransformersClip(MODEL)
        elif config["TYPE"] == "openclip":
            model = OpenClip(MODEL)
        else:
            raise Exception("Unknown model type")

        model = model.to("cuda")

        dataset = CustomDataset(missing_image_names, transform=model.process_images)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=3, persistent_workers=True)

        batch_counter = 0
        for batch in tqdm(loader):
            batch_counter += 1
            name_batch, image_batch = batch
            image_batch = image_batch.to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast():
                pooled_output, projected_embedding = model(image_batch)
                if torch.all(pooled_output == 0) or torch.all(projected_embedding == 0):
                    raise ValueError("The output tensor is all zeros.")
                result_df = pd.concat([result_df, pd.DataFrame({
                    "image_name": name_batch, 
                    "pooled_output": list(pooled_output.cpu().detach().numpy().astype('float32')),
                    "projected_embedding": list(projected_embedding.cpu().detach().numpy().astype('float32')),
                })], ignore_index=True)
            if batch_counter % SAVE_EVERY_N_BATCHES == 0:
                result_df.to_parquet(embedding_file_path)        

        result_df.to_parquet(embedding_file_path)

if __name__ == "__main__":
    main()







