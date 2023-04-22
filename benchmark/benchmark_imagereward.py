# adapted from https://github.com/THUDM/ImageReward/blob/main/test.py
# Compare with table 2 in https://arxiv.org/pdf/2304.05977.pdf

import os
import json
import torch    
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from aesthetics_scorer.model import preprocess, load_model as clip_load_model
from convnext_scorer.model import load_model as convnext_load_model, get_transforms as convnext_get_transforms
import pandas as pd
from datasets import load_dataset

MODEL_CONFIG = [
    {
        "type": "clip",
        "weights": "aesthetics_scorer/models/aesthetics_scorer_rating_openclip_vit_bigg_14.pth",
        "clip_model": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    },
    {
       "type": "clip",
        "weights": "aesthetics_scorer/models/aesthetics_scorer_rating_openclip_vit_h_14.pth",
        "clip_model": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    },
    {
        "type": "clip",
        "weights": "aesthetics_scorer/models/aesthetics_scorer_rating_openclip_vit_l_14.pth",
        "clip_model": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    },
    {
        "type": "convnext",
        "weights": "convnext_scorer/models/aesthetics_rating_convnext_large_2e_b2e.safetensors",
    },
    {
        "type": "convnext",
        "weights": "convnext_scorer/models/aesthetics_rating_realfake_2e_b2e.safetensors",
    }
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClipScorer():
    def __init__(self, clip_model, weights):
        super().__init__()
        model = CLIPModel.from_pretrained(clip_model)
        self.vision_model = model.vision_model.to(device)
        del model
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.rating_model = clip_load_model(weights).to(device)
    
    def rank_images(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_output = self.vision_model(**inputs)
            pooled_output = vision_output.pooler_output
            embedding = preprocess(pooled_output)
            ratings = self.rating_model(embedding)
 
        return ratings.squeeze().detach().cpu().numpy().tolist()
    

class ConvnextScorer():
    def __init__(self, weights):
        super().__init__()
        self.model = convnext_load_model(weights).to("cuda")
        self.model.eval()
        self.transforms = convnext_get_transforms(False)
    
    def rank_images(self, images):
        inputs = torch.stack([self.transforms(image) for image in images]).to(device)
        with torch.no_grad():
            ratings = self.model(inputs)
        
        return ratings.detach().cpu().numpy().tolist()
    

def acc(samples, predictions):
    tol_cnt = 0.
    true_cnt = 0.
    for idx in range(len(samples)):
        item_base = samples[idx]
        item = predictions[idx]
        for i in range(len(item_base)):
            for j in range(i+1, len(item_base)):
                if item_base[i] > item_base[j]:
                    if item[i] >= item[j]:
                        tol_cnt += 1
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                elif item_base[i] < item_base[j]:
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                    elif item[i] <= item[j]:
                        tol_cnt += 1
    
    return true_cnt / tol_cnt
    

def flatten_nested_list(nested_list):
    flat_list = []
    lengths = []
    for sublist in nested_list:
        flat_list.extend(sublist)
        lengths.append(len(sublist))
    return flat_list, lengths

def restore_nested_list(flat_list, lengths):
    restored_list = []
    start = 0
    for length in lengths:
        restored_list.append(flat_list[start:start+length])
        start += length
    return restored_list

def collate(data):
    return { "generations": [element["generations"] for element in data], "ranking": [element["ranking"] for element in data]}

def test():
    dataset = load_dataset("kenjiqq/imagereward-evaluation", split='test',)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)
    result_df = pd.DataFrame(columns =["model", "accuracy"])

    for config in MODEL_CONFIG:
        total_predictions = []
        samples = []
        if config["type"] == "clip":
            model = ClipScorer(config["clip_model"], config["weights"])
        elif config["type"] == "convnext":
            model = ConvnextScorer(config["weights"])

        with torch.no_grad():
            for _, batch in enumerate(tqdm(loader)):
                flat_list, shape = flatten_nested_list(batch["generations"])
                img_list = [image.convert("RGB") for image in flat_list]
                predictions = model.rank_images(img_list)
                predictions = restore_nested_list(predictions, shape)
                samples.extend(batch["ranking"])
                total_predictions.extend(predictions)
        
        test_acc = acc(samples, total_predictions)
        print(f"Test Acc: {100 * test_acc:.2f}%")
        model_name = os.path.basename(config["weights"])
        result_df.loc[len(result_df)] = [model_name, test_acc * 100]
    
    with open(f"benchmark/results/imagereward.txt", "w") as f:
            f.write(result_df.to_markdown(index=False))

if __name__ == "__main__":
    test()