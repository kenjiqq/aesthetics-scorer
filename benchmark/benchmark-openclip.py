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

result_df = pd.DataFrame(columns=["name", ])
for model_config in MODELS:
    print(f"Testing {model_config['name']}...")
    model = load_model(f"aesthetics_scorer/{model_config['name']}.pth").to("cuda")

    embeddings_df = pd.read_parquet(model_config["embeddings"])
    validate_df = pd.read_parquet("parquets/validate_split.parquet")
    df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")

    result = {"name": [model_config['name']]}
    for range in [False, (0, 3), (3, 7), (7, 10)]:
        if range:
            val_df = df[(df["rating"] >= range[0]) & (df["rating"] <= range[1])]
        else:
            val_df = df
        target = []
        predictions = []

        for index, row in tqdm(val_df.iterrows()):
            rating, embedding = (row["rating"], row["pooled_output"])
            embedding = preprocess(torch.from_numpy(np.array(embedding)).unsqueeze(0)).to("cuda")
            with torch.no_grad():
                prediction = model(embedding)
                target.append(torch.Tensor([rating]).squeeze())
                predictions.append(prediction.cpu().detach().squeeze())

        lossL1 = torch.nn.L1Loss()(torch.Tensor(predictions), torch.Tensor(target)).item()
        lossL2 = torch.nn.MSELoss()(torch.Tensor(predictions), torch.Tensor(target)).item()
        if range:
            range_text = f"Rating {range[0]} - {range[1]}"
        else:
            range_text = "Full"
        print(range_text)
        print("Loss L1: ", lossL1)
        print("Loss L2: ", lossL2)
        result[f"{range_text}: L1"] = [lossL1]
        result[f"{range_text}: L2"] = [lossL2]
    result_df = pd.concat([result_df, pd.DataFrame(result)], ignore_index=True)
    
with open(f"benchmark/aesthetic_scorer_openclip.txt", "w") as f:
    f.write(result_df.to_markdown())