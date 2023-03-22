import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from aesthetics_scorer.model import load_model, preprocess

MODELS = [
    {
        "name": "aesthetics_scorer_openclip_vit_bigg_14",
        "embeddings": "parquets/openclip_vit_bigg_14.parquet"
    },
    {
        "name": "aesthetics_scorer_openclip_vit_h_14",
        "embeddings": "parquets/openclip_vit_h_14.parquet"
    },
    {
        "name": "aesthetics_scorer_openclip_vit_l_14",
        "embeddings": "parquets/openclip_vit_l_14.parquet"
    }
]

results = []
for model_config in MODELS:
    print(f"Testing {model_config['name']}...")
    model = load_model(f"aesthetics_scorer/{model_config['name']}.pth").to("cuda")

    embeddings_df = pd.read_parquet(model_config["embeddings"])
    validate_df = pd.read_parquet("parquets/validate_split.parquet")
    df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")

    target = []
    predictions = []

    for index, row in tqdm(df.iterrows()):
        rating, embedding = (row["rating"], row["pooled_output"])
        embedding = preprocess(torch.from_numpy(np.array(embedding)).unsqueeze(0)).to("cuda")
        with torch.no_grad():
            prediction = model(embedding)
            target.append(torch.Tensor([rating]).squeeze())
            predictions.append(prediction.cpu().detach().squeeze())

    lossL1 = torch.nn.L1Loss()(torch.Tensor(predictions), torch.Tensor(target)).numpy()
    lossL2 = torch.nn.MSELoss()(torch.Tensor(predictions), torch.Tensor(target)).numpy()
    print("Loss L1: ", lossL1)
    print("Loss L2: ", lossL2)
    results.append({
         "name": model_config['name'],
         "l1": lossL1,
         "l2": lossL2
    })

    
with open(f"benchmark/aesthetic_scorer_openclip.txt", "w") as f:
    for res in results:
        f.write(res["name"] + "\n")
        f.write("Loss L1: " + str(res["l1"]) + "\n")
        f.write("Loss L2: " + str(res["l2"]) + "\n\n")