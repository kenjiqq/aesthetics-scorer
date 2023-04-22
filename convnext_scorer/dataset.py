
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from convnext_scorer.model import get_transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# custom dataset for loading images from a dataframe
class AestheticDataset(Dataset):
    def __init__(self, df, train=True, return_label=True):
        self.df = df
        self.return_label = return_label
        self._length = len(self.df)
        # add transforms
        self.transforms = get_transforms(train)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        # check what relative path resolves to

        # read image with PIL
        img = Image.open(os.path.join("dataset/", row["image_name"])).convert("RGB")
        img = self.transforms(img)
        if self.return_label:
            label = torch.Tensor([row["label"]])
            return img, label
        else:
            return img
        

def prepare_data_loader(path, type, batch_size, shuffle=False, train=True):
    df = pd.read_parquet(path)
    # filter out images that have NaN column values
    df = df[~df[type].isna()]
    df = df[["image_name", type]].rename(columns={type: "label"})
    lowest_score_count = min(df["label"].value_counts() )
    balanced_df = df.groupby("label").apply(lambda x: x.sample(n=lowest_score_count, random_state=42)).reset_index(drop=True)

    dataset = AestheticDataset(df, train=train)
    balanced_dataset = AestheticDataset(balanced_df, train=train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2)
    balanced_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2)
    return loader, balanced_loader