import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule
from torchvision.io import read_image, ImageReadMode
from torchinfo import summary
from timm.models.convnext import ConvNeXtStage
from convnext_scorer.dataset import prepare_data_loader
from model import ConvnextAestheticsScorer, load_model, get_transforms
from dataset import AestheticDataset

MODEL_NAME = "convnext_large" #convnext_large_fake_detector, convnext_large
SCORE_TYPE = "artifacts" # "rating", "artifacts"

SWEEP = False

config =dict({
    "epochs": 2,
    "balanced_epochs": 2,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "gradient_accumulation": 2,
    "scheduler": "cosine", # "cosine", "constant",
    "output_activation": None, # "sigmoid", None,
    "frozen_epochs": None,
    "unfreeze_layers": [""],
    "balanced_finetune": True,
    "balanced_learning_rate": 1e-4,
    "head_layer_count": 2,
    "resume": None, #"convnext_scorer/fake_detector_model.safetensors",
    #"resume_new_head": True,
})

if config["frozen_epochs"] == None:
    config["unfreeze_layers"] = None

torch.manual_seed(17)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
save_name = f"convnext_scorer/models/aesthetics_{SCORE_TYPE}_{MODEL_NAME}.safetensors"

def main():
    config["effective_batch_size"] = config["batch_size"] * config["gradient_accumulation"] if config["gradient_accumulation"] > 1 else config["batch_size"]
    # start a new wandb run to track this script
    with wandb.init(
        #mode="disabled",
        # set the wandb project where this run will be logged
        project="aesthetics_scorer",
        tags=["convnext", f"{MODEL_NAME}", SCORE_TYPE],
        config=config
    ):
        
        # setup validation loaders
        val_loader, balanced_val_loader = prepare_data_loader("parquets/validate_split.parquet", SCORE_TYPE, wandb.config["batch_size"], shuffle=False, train=False)

        # initialize model
        if wandb.config["resume"] != None:
            model = load_model(wandb.config["resume"], new_head=wandb.config["resume_new_head"]).to(device)
        else:
            model = ConvnextAestheticsScorer(model=MODEL_NAME, pretrained=True, head_layer_count=wandb.config["head_layer_count"]).to(device)

        global_epoch = 0
        
        def train(train_loader, scheduler, epochs, lr, unfreeze = True):
            nonlocal global_epoch

            for params in model.parameters():
                params.requires_grad = False
            for params in model.model.head.parameters():
                params.requires_grad = True

            if wandb.config["frozen_epochs"] == None or unfreeze == False:
                summary(model, input_size=(1, 3, 224, 224), depth=5, col_names=["input_size", "output_size", "num_params", "trainable"])
            gradient_accumulation_steps = wandb.config["gradient_accumulation"] if wandb.config["gradient_accumulation"] > 1 else 1
            total_steps = (epochs * len(train_loader)) // gradient_accumulation_steps
            # setup adamw optimizer with only params that require grad
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
            
            if scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
            else:
                scheduler = get_constant_schedule(optimizer)

            criterion_mse = nn.MSELoss()
            criterion_mae = nn.L1Loss()

            model.train()

            for epoch in range(global_epoch, global_epoch + epochs):
                print(f"------------ Epoch {epoch} ------------")

                
                # unfreeze model after frozen period
                if unfreeze == True and epoch == wandb.config["frozen_epochs"]:
                    print("Unfreezing model parameters")
                    unfree_layers = tuple(wandb.config["unfreeze_layers"])
                    for name, module in model.named_modules():
                        if wandb.config["unfreeze_layers"] != None and not name.startswith(unfree_layers):
                            continue
                        for params in module.parameters():
                            params.requires_grad = True
                    summary(model, input_size=(1, 3, 224, 224), depth=5, col_names=["input_size", "output_size", "num_params", "trainable"])

                epoch_losses = []
                
                
                optimizer.zero_grad()
                for index, input_data in enumerate(tqdm(train_loader)):
                    x, y = input_data
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)
                    loss = criterion_mse(output, y)
                    loss_scaled = loss / gradient_accumulation_steps
                    loss_scaled.backward()
                    if (index + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    epoch_losses.append(loss.detach().item())
                    wandb.log({ "train_loss": loss.detach().item(), "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]})

                print('Train:          Loss %6.4f' % (sum(epoch_losses)/len(epoch_losses)))

                def validate(loader):
                    val_losses_mse = []
                    val_losses_mae = []
                    
                    for _, input_data in enumerate(tqdm(loader)):
                        x, y = input_data
                        x = x.to(device).float()
                        y = y.to(device)

                        with torch.no_grad():
                            output = model(x)
                        loss = criterion_mse(output, y)
                        lossMAE = criterion_mae(output, y)

                        val_losses_mse.append(loss.detach().item())
                        val_losses_mae.append(lossMAE.detach().item())
                    return sum(val_losses_mse) / len(val_losses_mse), sum(val_losses_mae) / len(val_losses_mae)
                
                val_loss_mse, val_loss_mae = validate(val_loader)
                wandb.log({ "val_mse_loss": val_loss_mse, "val_mae_loss":  val_loss_mae, "epoch": epoch })
                print('Validation: MSE Loss %6.4f' % (val_loss_mse))
                print('Validation: MAE Loss %6.4f' % (val_loss_mae))
                
                balanced_val_loss_mse, balanced_val_loss_mae = validate(balanced_val_loader)
                wandb.log({ "balanced_val_mse_loss": balanced_val_loss_mse, "balanced_val_mae_loss":  balanced_val_loss_mae, "epoch": epoch })
                print('Balanced Validation: MSE Loss %6.4f' % (balanced_val_loss_mse))
                print('Balanced Validation: MAE Loss %6.4f' % (balanced_val_loss_mae))
            
                global_epoch += 1

        # Full training set
        train_loader, balanced_train_loader = prepare_data_loader("parquets/train_split.parquet", SCORE_TYPE, wandb.config["batch_size"], shuffle=True, train=True)
        train(train_loader, wandb.config["scheduler"], wandb.config["epochs"], wandb.config["learning_rate"])

        if wandb.config["balanced_finetune"]:
            # Finetune on balanced training set
            print("----------------------------------------")
            print("Finetuning on balanced training set")
            print("----------------------------------------")
            train(balanced_train_loader, wandb.config["scheduler"], wandb.config["balanced_epochs"], wandb.config["balanced_learning_rate"], unfreeze=False)

        model.save(save_name)
        print("Training done")

if __name__ == '__main__':
    if SWEEP:
        import yaml
        with open('sweep.yml', 'r') as file:
            sweep_configuration = yaml.safe_load(file)
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="aesthetics_scorer")
        wandb.agent(sweep_id, function=main)
    else:
        main()