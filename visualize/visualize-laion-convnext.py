# Based on from https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/visulaize_100k_from_LAION400M.py

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from convnext_scorer.model import get_transforms, load_model
import os

from laion2b.data import get_dataloader

MODELS = [
    {"name": "aesthetics_artifacts_convnext_large_2e_b2e", "type": "artifacts"},
    {"name": "aesthetics_artifacts_realfake_2e_b2e", "type": "artifacts"},
    {"name": "aesthetics_rating_convnext_large_2e_b2e", "type": "rating"},
    {"name": "aesthetics_rating_realfake_2e_b2e", "type": "rating"},

]
GENERATE_FROM_CACHE = True

CONFIG = {
    "artifacts": (0, 5),
    "rating": (1, 10)
}

def predict(model, min, max):
    model = load_model(f"convnext_scorer/models/{model}.safetensors").to("cuda")
    model.eval()

    data_loader = get_dataloader(512, transforms=get_transforms(False))

    urls = []
    predictions = []
    for i, batch in enumerate(tqdm(data_loader)):
        batch_urls, batch_images = batch
        with torch.no_grad():
            prediction  = torch.clamp(model(batch_images.to("cuda")), min, max)
        urls.extend(batch_urls)
        predictions.extend([x.item() for x in prediction.detach().cpu()])
    return urls, predictions

if __name__ == "__main__":

    for model in MODELS:
        model_name = model["name"]
        model_type = model["type"]

        cache_file = f"visualize/cache/laion5b-{model_name}.csv"
        if not GENERATE_FROM_CACHE or not os.path.exists(cache_file):
            print(f"Generating predictions for {model_name}...")
            urls, predictions = predict(model_name, *CONFIG[model_type])
            df = pd.DataFrame(list(zip(urls, predictions)),
            columns =['filepath', 'prediction'])
            df.to_csv(f"visualize/cache/laion5b-{model_name}.csv", index=False)
        else:
            print(f"Loading predictions for {model_name} from cache...")
            df = pd.read_csv(f"visualize/cache/laion5b-{model_name}.csv")
        
        BUCKET_STEP = 0.5
        buckets = [(i, i + BUCKET_STEP) for i in np.arange(*CONFIG[model_type], BUCKET_STEP)]


        html= f"<h1>Aesthetic {model_type} scores for {len(df)} LAION 5b samples</h1>"

        for [a,b] in buckets:
            total_part = df[(  (df["prediction"] ) *1>= a) & (  (df["prediction"] ) *1 <= b)]
            count_part = len(total_part) / len(df) * 100
            estimated =int ( len(total_part) )
            part = total_part.sample(min(len(total_part), 50))

            html+=f"<h2>In bucket {a} - {b} there is {count_part:.2f}% samples:{estimated:.2f} </h2> <div>"
            for filepath in part["filepath"]:
                html+='<img src="'+filepath +'" height="200" />'

            html+="</div>"
        result_path = f"visualize/laion/visualize-laion5b-{model_type}-{model_name}.html"
        with open(result_path, "w") as f:
            f.write(html)
            print(f"Saved to {result_path}")
    
