import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from convnext_scorer.dataset import prepare_data_loader
from aesthetics_scorer.model import load_model as clip_load_model, preprocess as clip_preprocess
from convnext_scorer.model import load_model as convnext_load_model

MODELS = {
    "rating": [
        {
            "name": "aesthetics_scorer_rating_openclip_vit_bigg_14",
            "embeddings": "parquets/openclip_vit_bigg_14.parquet",
            "base": "openclip"
        },
        {
            "name": "aesthetics_scorer_rating_openclip_vit_h_14",
            "embeddings": "parquets/openclip_vit_h_14.parquet",
            "base": "openclip"
        },
        {
            "name": "aesthetics_scorer_rating_openclip_vit_l_14",
            "embeddings": "parquets/openclip_vit_l_14.parquet",
            "base": "openclip"
        },
        {
            "name": "aesthetics_rating_convnext_large_2e_b2e",
            "base": "convnext"
        },
        {
            "name": "aesthetics_rating_realfake_2e_b2e",
            "base": "convnext"
        }
    ],

    "artifacts": [
        {
            "name": "aesthetics_scorer_artifacts_openclip_vit_bigg_14",
            "embeddings": "parquets/openclip_vit_bigg_14.parquet",
            "base": "openclip"
        },
        {
            "name": "aesthetics_scorer_artifacts_openclip_vit_h_14",
            "embeddings": "parquets/openclip_vit_h_14.parquet",
            "base": "openclip"
        },
        {
            "name": "aesthetics_scorer_artifacts_openclip_vit_l_14",
            "embeddings": "parquets/openclip_vit_l_14.parquet",
            "base": "openclip"
        },
        {
            "name": "aesthetics_artifacts_convnext_large_2e_b2e",
            "base": "convnext",
        },
        {
            "name": "aesthetics_artifacts_realfake_2e_b2e",
            "base": "convnext"
        },
    ]
}

def benchmark():
    for type in ["rating", "artifacts"]:
        result_df = pd.DataFrame(columns=["name", ])
        for model_config in MODELS[type]:
            print(f"Testing {model_config['name']}...")
            if model_config["base"] == "openclip":
                model = clip_load_model(f"aesthetics_scorer/models/{model_config['name']}.pth").to("cuda")
                embeddings_df = pd.read_parquet(model_config["embeddings"])
                validate_df = pd.read_parquet("parquets/validate_split.parquet")
                validate_df = validate_df[~validate_df[type].isna()]
                merged_df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")
                embeddings_tensor = torch.stack(merged_df["pooled_output"].apply(lambda x: clip_preprocess(torch.from_numpy(np.array(x)))).tolist())
                labels_tensor = torch.Tensor(merged_df[type].values.tolist())
                loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(embeddings_tensor, labels_tensor), batch_size=512, shuffle=False)

            elif model_config["base"] == "convnext":
                model = convnext_load_model(f"convnext_scorer/models/{model_config['name']}.safetensors").to("cuda")
                loader, _ = prepare_data_loader("parquets/validate_split.parquet", type, 512, shuffle=False, train=False)

            model.eval()

            target = []
            predictions = []

            for _, input_data in enumerate(tqdm(loader)):
                x, labels = input_data
                with torch.no_grad():
                    prediction = model(x.to("cuda"))
                    target.extend(labels.squeeze())
                    predictions.extend(prediction.cpu().detach().squeeze())

            df = pd.DataFrame(list(zip(target, predictions)), columns =[type, 'prediction'])
            result = {"name": [model_config['name']]}
            for range in [False, (0, 3), (3, 7), (7, 10)] if type == "rating" else [False]:
                if range:
                    val_df = df[(df[type] >= range[0]) & (df[type] <= range[1])]
                else:
                    val_df = df
                
                lossL1 = torch.nn.L1Loss()(torch.Tensor(val_df["prediction"].values.tolist()), torch.Tensor(val_df[type].values.tolist())).item()
                lossL2 = torch.nn.MSELoss()(torch.Tensor(val_df["prediction"].values.tolist()), torch.Tensor(val_df[type].values.tolist())).item()
                if range:
                    range_text = f"Range {range[0]} - {range[1]}"
                else:
                    range_text = "Full"
                print(range_text)
                print("Loss L1: ", lossL1)
                print("Loss L2: ", lossL2)
                result[f"{range_text}: L1"] = [lossL1]
                result[f"{range_text}: L2"] = [lossL2]
            result_df = pd.concat([result_df, pd.DataFrame(result)], ignore_index=True)
            
        with open(f"benchmark/results/validation_{type}.txt", "w") as f:
            f.write(result_df.to_markdown())

if __name__ == "__main__":
    benchmark()