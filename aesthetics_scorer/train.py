import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import wandb
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule
import os

from model import AestheticScorer, preprocess

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" # avoid annoying cublas warning


EMBEDDING_FILE = "parquets/dfn5b_vit_h_14.parquet"
MODEL_NAME = "dfn5b_vit_h_14"
SCORE_TYPE = "rating" # "rating", "artifacts"

SWEEP = False
WANDB_MODE = "online" # "disabled", "online", "offline"


config =dict({
    "epochs": 10,
    "learning_rate": 1e-3,
    "balanced_learning_rate": 1e-4,
    "batch_size": 8192,
    "activation": False,
    "dropout": 0.0,
    "embedding_type": "pooled_output", # "pooled_output", "projected_embedding"
    "normalize": True,
    "scheduler": "cosine", # "cosine", "constant",
    "hidden_dim": 1024,
    "reduce_dims": False,
    "output_activation": None, # "sigmoid", None,
    "balanced_finetune": True,
    "berhu_threshold": False,
    "oversample": False,
})

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
save_name = f"aesthetics_scorer/models/aesthetics_scorer_{SCORE_TYPE}_{MODEL_NAME}.pth"

# load the training data 
embeddings_df = pd.read_parquet(EMBEDDING_FILE)
train_df = pd.read_parquet("parquets/train_split.parquet")
train_df = pd.merge(train_df, embeddings_df, left_on="image_name", right_on="image_name")
validate_df = pd.read_parquet("parquets/validate_split.parquet")
validate_df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")

# filter out images that have NaN column values
train_df = train_df[~train_df[SCORE_TYPE].isna()]
validate_df = validate_df[~validate_df[SCORE_TYPE].isna()]

if config["oversample"]:
    # oversample to minimum 20000 samples per score
    def oversample(df, score_type, min_count):
        for score in df[score_type].unique():
            count = df[score_type].value_counts()[score]
            if count < min_count:
                df = pd.concat([df, df[df[score_type] == score].sample(n=min_count-count, replace=True, random_state=42)], ignore_index=True)
        return df

    max_count = max(train_df[SCORE_TYPE].value_counts())
    train_df = oversample(train_df, SCORE_TYPE, max_count)

def main():
    # start a new wandb run to track this script
    with wandb.init(
        mode=WANDB_MODE,
        # set the wandb project where this run will be logged
        project="aesthetics_scorer",
        tags=[f"{MODEL_NAME}", SCORE_TYPE],
        config=config
    ):
        def prepare_data_loader(df, shuffle=False):
            embeddings = torch.tensor(np.array(df[wandb.config["embedding_type"]].to_list()))
            if wandb.config["normalize"]:
                embeddings = preprocess(embeddings)
            scores = torch.tensor(np.array(df[SCORE_TYPE].to_list()).astype("float32"))
            scores = scores.reshape(-1, 1)
            dataset = TensorDataset(embeddings, scores) 
            loader = DataLoader(dataset, batch_size=wandb.config["batch_size"], shuffle=shuffle)
            return loader
        
        # setup validation loaders
        val_loader = prepare_data_loader(validate_df, shuffle=False)
        lowest_score_count = min(validate_df[SCORE_TYPE].value_counts() )
        balanced_val_df = validate_df.groupby(SCORE_TYPE).apply(lambda x: x.sample(n=lowest_score_count, random_state=42)).reset_index(drop=True)
        balanced_val_loader = prepare_data_loader(balanced_val_df, shuffle=False)

        # initialize model
        model = AestheticScorer(train_df[wandb.config["embedding_type"]][0].size, use_activation=wandb.config["activation"], dropout=wandb.config["dropout"], hidden_dim=wandb.config["hidden_dim"], reduce_dims=wandb.config["reduce_dims"], output_activation=wandb.config["output_activation"]).to(device)

        def berhu_loss(y_pred, y_true, c):
            error = y_true - y_pred
            absolute_error = abs(error)
            
            mask = absolute_error <= c
            l1_loss = absolute_error * mask
            l2_loss = ((error**2 + c**2) / (2 * c)) * (~mask)
            
            return (l1_loss + l2_loss).mean()
        
        def train(train_loader, scheduler, lr):
            total_steps = wandb.config["epochs"] * len(train_loader)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
            if scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
            else:
                scheduler = get_constant_schedule(optimizer)

            criterion_mse = nn.MSELoss()
            criterion_mae = nn.L1Loss()

            model.train()

            for epoch in range(wandb.config["epochs"]):
                print(f"------------ Epoch {epoch} ------------")
                epoch_losses = []
                for _, input_data in enumerate(train_loader):
                    optimizer.zero_grad()
                    x, y = input_data
                    x = x.to(device).float()
                    y = y.to(device)
                
                    output = model(x)

                    if wandb.config["berhu_threshold"]:
                        loss = berhu_loss(output, y, wandb.config["berhu_threshold"])
                    else:
                        loss = criterion_mse(output, y)
                    loss.backward()
                    epoch_losses.append(loss.detach().item())

                    optimizer.step()
                    scheduler.step()

                    wandb.log({ "train_loss": loss.detach().item(), "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]})

                print('Train:          Loss %6.4f' % (sum(epoch_losses)/len(epoch_losses)))

                def validate(loader, include_quantiles=False):
                    val_losses_mse = []
                    val_losses_mae = []

                    lower_quantile_losses_mse = []
                    lower_quantile_losses_mae = []

                    upper_quantile_losses_mse = []
                    upper_quantile_losses_mae = []
                    
                    for _, input_data in enumerate(loader):
                        optimizer.zero_grad()
                        x, y = input_data
                        x = x.to(device).float()
                        y = y.to(device)

                        with torch.no_grad():
                            output = model(x)
                        loss_mse = criterion_mse(output, y)
                        loss_mae = criterion_mae(output, y)

                        val_losses_mse.append(loss_mse.detach().item())
                        val_losses_mae.append(loss_mae.detach().item())

                        avg_loss_mse = sum(val_losses_mse) / len(val_losses_mse)
                        avg_loss_mae = sum(val_losses_mae) / len(val_losses_mae)
                        
                        if not include_quantiles:
                            return avg_loss_mse, avg_loss_mae
                        
                        # Compute lower quantile losses
                        lower_mask = y <= 2
                        if lower_mask.any():
                            lower_output = output[lower_mask]
                            lower_y = y[lower_mask]
                            loss_mse_lower = criterion_mse(lower_output, lower_y)
                            loss_mae_lower = criterion_mae(lower_output, lower_y)
                            lower_quantile_losses_mse.append(loss_mse_lower.detach().item())
                            lower_quantile_losses_mae.append(loss_mae_lower.detach().item())

                        # Compute upper quantile losses
                        upper_mask = y >= 9
                        if upper_mask.any():
                            upper_output = output[upper_mask]
                            upper_y = y[upper_mask]
                            loss_mse_upper = criterion_mse(upper_output, upper_y)
                            loss_mae_upper = criterion_mae(upper_output, upper_y)
                            upper_quantile_losses_mse.append(loss_mse_upper.detach().item())
                            upper_quantile_losses_mae.append(loss_mae_upper.detach().item())

                    avg_loss_mse_lower = sum(lower_quantile_losses_mse) / len(lower_quantile_losses_mse) if lower_quantile_losses_mse else 0
                    avg_loss_mae_lower = sum(lower_quantile_losses_mae) / len(lower_quantile_losses_mae) if lower_quantile_losses_mae else 0
                    
                    avg_loss_mse_upper = sum(upper_quantile_losses_mse) / len(upper_quantile_losses_mse) if upper_quantile_losses_mse else 0
                    avg_loss_mae_upper = sum(upper_quantile_losses_mae) / len(upper_quantile_losses_mae) if upper_quantile_losses_mae else 0

                    return avg_loss_mse, avg_loss_mae, avg_loss_mse_lower, avg_loss_mae_lower, avg_loss_mse_upper, avg_loss_mae_upper
                
                val_loss_mse, val_loss_mae, val_loss_mse_lower, val_loss_mae_lower, val_loss_mse_upper, val_loss_mae_upper = validate(val_loader, include_quantiles=True)
                wandb.log({ "epoch": epoch, "val_mse_loss": val_loss_mse, "val_mae_loss":  val_loss_mae, "val_mse_loss_lower": val_loss_mse_lower, "val_mae_loss_lower": val_loss_mae_lower, "val_mse_loss_upper": val_loss_mse_upper, "val_mae_loss_upper": val_loss_mae_upper})
                print('Validation: MSE Loss %6.4f' % (val_loss_mse))
                print('Validation: MAE Loss %6.4f' % (val_loss_mae))
                print('Validation: MSE Loss Lower %6.4f' % (val_loss_mse_lower))
                print('Validation: MAE Loss Lower %6.4f' % (val_loss_mae_lower))
                print('Validation: MSE Loss Upper %6.4f' % (val_loss_mse_upper))
                print('Validation: MAE Loss Upper %6.4f' % (val_loss_mae_upper))
                
                balanced_val_loss_mse, balanced_val_loss_mae = validate(balanced_val_loader)
                wandb.log({ "balanced_val_mse_loss": balanced_val_loss_mse, "balanced_val_mae_loss":  balanced_val_loss_mae, "epoch": epoch })
                print('Balanced Validation: MSE Loss %6.4f' % (balanced_val_loss_mse))
                print('Balanced Validation: MAE Loss %6.4f' % (balanced_val_loss_mae))

        # Full training set
        train_loader = prepare_data_loader(train_df, shuffle=True)
        train(train_loader, wandb.config["scheduler"], wandb.config["learning_rate"])

        if wandb.config["balanced_finetune"]:
            # Finetune on balanced training set
            lowest_score_count = min(train_df[SCORE_TYPE].value_counts() )
            balanced_train_df = train_df.groupby(SCORE_TYPE).apply(lambda x: x.sample(n=lowest_score_count, random_state=42)).reset_index(drop=True)
            balanced_train_loader = prepare_data_loader(balanced_train_df, shuffle=True)
            train(balanced_train_loader, wandb.config["scheduler"], wandb.config["balanced_learning_rate"])

        model.save(save_name)
        print("Training done")

if SWEEP:
    import yaml
    with open('sweep.yml', 'r') as file:
        sweep_configuration = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="aesthetics_scorer")
    wandb.agent(sweep_id, function=main)
else:
    main()