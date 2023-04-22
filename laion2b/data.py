import json
import webdataset as wds
from torch.utils.data import DataLoader, ChainDataset
import pandas as pd
from tqdm import tqdm

laion_info = pd.read_parquet("laion2b/parquet/dataset.parquet")

def filter(x):
    return laion_info.iloc[int(json.loads(x["json"])["key"])]["NSFW"] not in ["UNSURE", "NSFW"]


class MapItems():
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        json, image = x
        return json["url"], self.transforms(image) if self.transforms != None else image

def get_dataloader(batch_size, transforms=None):
    
    dataset = wds.WebDataset("laion2b/dataset/{00000..00099}.tar").select(filter).decode("pil").to_tuple("json", "jpg").map(MapItems(transforms)).batched(batch_size)
    data_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2)
    return data_loader

if __name__ == "__main__":
    dataloader = get_dataloader(2)
    count = 0
    for sample in enumerate(tqdm(dataloader)):
        continue