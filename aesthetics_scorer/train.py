import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import wandb
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule

from model import AestheticScorer, preprocess

EMBEDDING_FILE = "parquets/openclip_vit_bigg_14.parquet"
MODEL_NAME = "openclip_vit_bigg_14"

SWEEP = False

config =dict({
    "epochs": 25,
    "learning_rate": 1e-3,
    "batch_size": 8192,
    "activation": False,
    "dropout": 0.0,
    "embedding_type": "pooled_output", # "pooled_output", "projected_embedding"
    "normalize": True,
    "scheduler": "cosine", # "cosine", "constant",
    "hidden_dim": 1024,
    "reduce_dims": False,
    "output_activation": None # "sigmoid", None
})

# load the training data 
embeddings_df = pd.read_parquet(EMBEDDING_FILE)
train_df = pd.read_parquet("parquets/train_split.parquet")
train_df = pd.merge(train_df, embeddings_df, left_on="image_name", right_on="image_name")
validate_df = pd.read_parquet("parquets/validate_split.parquet")
validate_df = pd.merge(validate_df, embeddings_df, left_on="image_name", right_on="image_name")

def train():
    # start a new wandb run to track this script
    with wandb.init(
        mode="disabled",
        # set the wandb project where this run will be logged
        project="aesthetics_scorer",
        tags=f"{MODEL_NAME}",
        config=config
    ):
        train_embeddings = torch.tensor(np.array(train_df[wandb.config["embedding_type"]].to_list()))
        if wandb.config["normalize"]:
            train_embeddings = preprocess(train_embeddings)
        train_ratings = torch.tensor(np.array(train_df["rating"].to_list()).astype("float32"))
        train_ratings = train_ratings.reshape(-1, 1)
        train_dataset = TensorDataset(train_embeddings, train_ratings) 
        train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=True) 

        validate_embeddings = torch.tensor(np.array(validate_df[wandb.config["embedding_type"]].to_list()))
        if wandb.config["normalize"]:
            validate_embeddings = preprocess(validate_embeddings)
        validate_ratings = torch.tensor(np.array(validate_df["rating"].to_list()).astype("float32"))
        validate_ratings = validate_ratings.reshape(-1, 1)
        val_dataset = TensorDataset(validate_embeddings, validate_ratings) 
        val_loader = DataLoader(val_dataset, batch_size=wandb.config["batch_size"]) 

        device = torch.device('cuda')

        model = AestheticScorer(train_embeddings.size(-1), use_activation=wandb.config["activation"], dropout=wandb.config["dropout"], hidden_dim=wandb.config["hidden_dim"], reduce_dims=wandb.config["reduce_dims"], output_activation=wandb.config["output_activation"]).to(device)

        total_steps = wandb.config["epochs"] * len(train_loader)
        optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config["learning_rate"]) 
        if wandb.config["scheduler"] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
        else:
            scheduler = get_constant_schedule(optimizer)

        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()

        model.train()
        
        best_loss = 9999
        save_name = f"aesthetics_scorer/aesthetics_scorer_{MODEL_NAME}.pth"

        for epoch in range(wandb.config["epochs"]):
            print(f"------------ Epoch {epoch} ------------")
            epoch_losses = []
            for batch_index, input_data in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)
            
                output = model(x)
                loss = criterion_mse(output, y)
                loss.backward()
                epoch_losses.append(loss.detach().item())

                optimizer.step()
                scheduler.step()

                wandb.log({ "train_loss": loss.detach().item(), "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]})

            print('Train:          Loss %6.4f' % (sum(epoch_losses)/len(epoch_losses)))
            val_losses_mse = []
            val_losses_mae = []
            
            for batch_index, input_data in enumerate(val_loader):
                optimizer.zero_grad()
                x, y = input_data
                x = x.to(device).float()
                y = y.to(device)

                with torch.no_grad():
                    output = model(x)
                loss = criterion_mse(output, y)
                lossMAE = criterion_mae(output, y)

                val_losses_mse.append(loss.detach().item())
                val_losses_mae.append(lossMAE.detach().item())

            wandb.log({ "val_mse_loss": sum(val_losses_mse)/len(val_losses_mse), "val_mae_loss":  sum(val_losses_mae)/len(val_losses_mae), "epoch": epoch })
            print('Validation: MSE Loss %6.4f' % (sum(val_losses_mse) / len(val_losses_mse)))
            print('Validation: MAE Loss %6.4f' % (sum(val_losses_mae) / len(val_losses_mae)))
            if sum(val_losses_mse)/len(val_losses_mse) < best_loss:
                print("Best MSE Val loss so far. Saving model")
                best_loss = sum(val_losses_mse)/len(val_losses_mse)
                print( f"New best loss {best_loss:6.4f}") 
                model.save(save_name)

        print("Training done")

if SWEEP:
    import yaml
    with open('sweep.yml', 'r') as file:
        sweep_configuration = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="aesthetics_scorer")
    wandb.agent(sweep_id, function=train)
else:
    train()