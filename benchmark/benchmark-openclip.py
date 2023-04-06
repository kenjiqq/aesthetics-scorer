import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from aesthetics_scorer.model import load_model, preprocess

MODELS = {
    "rating": [
        {
            "name": "aesthetics_scorer_rating_openclip_vit_bigg_14",
            "embeddings": "parquets/openclip_vit_bigg_14.parquet"
        },
        {
            "name": "aesthetics_scorer_rating_openclip_vit_h_14",
            "embeddings": "parquets/openclip_vit_h_14.parquet"
        },
        {
            "name": "aesthetics_scorer_rating_openclip_vit_l_14",
            "embeddings": "parquets/openclip_vit_l_14.parquet"
        }
    ],

    "artifacts": [
        {
            "name": "aesthetics_scorer_artifacts_openclip_vit_bigg_14",
            "embeddings": "parquets/openclip_vit_bigg_14.parquet"
        },
        {
            "name": "aesthetics_scorer_artifacts_openclip_vit_h_14",
            "embeddings": "parquets/openclip_vit_h_14.parquet"
        },
        {
            "name": "aesthetics_scorer_artifacts_openclip_vit_l_14",
            "embeddings": "parquets/openclip_vit_l_14.parquet"
        }
    ]
}

for type in ["rating", "artifacts"]:
    result_df = pd.DataFrame(columns=["name", ])
    for model_config in MODELS[type]:
        print(f"Testing {model_config['name']}...")
        model = load_model(f"aesthetics_scorer/models/{model_config['name']}.pth").to("cuda")

        embeddings_df = pd.read_parquet(model_config["embeddings"])
        validate_df = pd.read_parquet("parquets/validate_split.parquet")
        validate_df = validate_df[~validate_df[type].isna()]
        df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")

        result = {"name": [model_config['name']]}
        for range in [False, (0, 3), (3, 7), (7, 10)] if type == "rating" else [False]:
            if range:
                val_df = df[(df[type] >= range[0]) & (df[type] <= range[1])]
            else:
                val_df = df
            target = []
            predictions = []

            for index, row in tqdm(val_df.iterrows()):
                rating, embedding = (row[type], row["pooled_output"])
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
        
    with open(f"benchmark/aesthetic_scorer_{type}_openclip.txt", "w") as f:
        f.write(result_df.to_markdown())