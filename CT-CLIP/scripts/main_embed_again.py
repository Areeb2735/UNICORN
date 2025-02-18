import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torchinfo import summary

from utils import make_time_bins
from utils import encode_survival, mtlr_neg_log_likelihood, make_optimizer
from utils import mtlr_survival, mtlr_risk, roc_auc_at_times, brier_score_at_times
from prognosis_model import *
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from lifelines.utils import concordance_index
from data_inference_hector import Hector_Dataset_emb

seed = 42
torch.manual_seed(seed) 
generator = torch.Generator().manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import wandb

sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'average_best_ci',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'values': [1.00000000e-05, 1.17210230e-05, 1.37382380e-05, 1.61026203e-05, 1.88739182e-05, 2.21221629e-05, 2.59294380e-05, 3.03919538e-05, 3.56224789e-05, 4.17531894e-05, 4.89390092e-05, 5.73615251e-05, 6.72335754e-05, 7.88046282e-05, 9.23670857e-05, 1.08263673e-04, 1.26896100e-04, 1.48735211e-04, 1.74332882e-04, 2.04335972e-04, 2.39502662e-04, 2.80721620e-04, 3.29034456e-04, 3.85662042e-04, 4.52035366e-04, 5.29831691e-04, 6.21016942e-04, 7.27895384e-04, 8.53167852e-04, 1.00000000e-03]
        },
        'C1': {
            'values': [1, 10, 100]
        },
        'num_time_bins': {
            'values': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'model': {
            'values': ['embd_model_linear_with_adapter', 'embd_model_linear_without_adapter', 'embd_model_conv_without_adapter', 'embd_model_conv_with_adapter']
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="ct_prognosis_hyperparameter_tuning_2")

def exp(config=None) -> None:
    with wandb.init(config=config):
        config = wandb.config

        parser = argparse.ArgumentParser(description='Training Script')
        parser.add_argument('--name', type=str, default='dummy', help='Name of the experiment')

        args = parser.parse_args()

        exp_name = args.name

        hect_dataset = Hector_Dataset_emb(emd_path = 'docs/embeddings/spatial.npy',  
                        csv_file ="docs/TNM_hector_prompts.csv")

        best_ci_list = []
        for fold in range(5):
            print(f"Fold: {fold}")

            train_dataset, test_dataset = hect_dataset.train_val_split(fold=fold)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            df = pd.read_csv("docs/TNM_hector_prompts.csv")
            # time_bins = make_time_bins(df['RFS'].values, event = df['Relapse'].values)
            # num_time_bins = len(time_bins)

            num_time_bins = config.num_time_bins
            time_bins = make_time_bins(df['RFS'].values, event=df['Relapse'].values, num_bins=num_time_bins)

            model_map = {
            'embd_model_linear_with_adapter': embd_model_linear_with_adapter,
            'embd_model_linear_without_adapter': embd_model_linear_without_adapter,
            'embd_model_conv_without_adapter': embd_model_conv_without_adapter,
            'embd_model_conv_with_adapter': embd_model_conv_with_adapter
            }
            model = model_map[config.model](num_time_bins)

            model.to(device)

            img_emb, text_emb, relapse, RFS, _, _ = next(iter(train_loader))

            img_emb = img_emb.to(device)
            text_emb = text_emb.to(device)
            relapse = relapse.to(device)
            RFS = RFS.to(device)
            summary(model, input_data=[img_emb, text_emb ], depth=8, col_names=["input_size", "output_size", "num_params"],)

            for name, param in model.named_parameters():
                print(name, param.requires_grad)

            model.train()
            num_epochs = 20
            verbose=True
            optimizer = make_optimizer(AdamW, model, lr=config.lr, weight_decay=0.00001)
            scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
            pbar =  trange(num_epochs, disable=not verbose)
            best_ci = 0
            best_epoch = 0
            for i in pbar:
                model.train()
                for j, (img_emb, text_emb, relapse, RFS, _, _) in enumerate(train_loader):
                    img_emb = img_emb.to(device)
                    text_emb = text_emb.to(device)
                    y = encode_survival(RFS, relapse, time_bins).to(device)
                    relapse = relapse.to(device)
                    RFS = RFS.to(device)
                    y_pred = model(img_emb, text_emb)
                    loss = mtlr_neg_log_likelihood(y_pred, y, model, C1=config.C1, average=True)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.set_description(f"[epoch {i+1: 4}/{num_epochs}]")
                pbar.set_postfix_str(f"loss = {loss.item():.4f}")
                # if i % 1 == 0:
                    # torch.save(model.state_dict(), f'/home/Mohammad.Qazi@mbzuai.ac.ae/project/ct_rate/model_{i}.pth')
                ci = validation(model, test_loader, device)
                if i > -1:
                    if ci > best_ci:
                        best_ci = ci
                        best_epoch = i+1
                        save_path = os.path.join(f"docs/weights/{exp_name}/{fold}", 'best_model.pth')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save(model.state_dict(), save_path)
                        print(f"Model saved at epoch {best_epoch}")
                
                scheduler.step()

            best_ci = round(best_ci, 4)
            best_ci_list.append(best_ci)

            os.makedirs(os.path.dirname(f'docs/weights/{exp_name}/{fold}/result.txt'), exist_ok=True)
            with open(f'docs/weights/{exp_name}/{fold}/result.txt', 'w') as f:
                f.write(f"Best CI: {best_ci:.4f}\n")
                f.write(f"Best epoch: {best_epoch}\n")
            print(f"Best CI: {best_ci:.4f}")
            print(f"Best epoch: {best_epoch}")
        
        # **Compute and Print Average Best CI**
        average_best_ci = round(sum(best_ci_list) / len(best_ci_list), 4)
        print(f"\nAverage Best CI over {len(best_ci_list)} folds: {average_best_ci:.4f}")

        # Save overall results
        with open(f'docs/weights/{exp_name}/final_result.txt', 'w') as f:
            f.write(f"Average Best CI: {average_best_ci:.4f}\n")

        wandb.log({'Average Best CI': average_best_ci})

def validation(model, test_loader, device):
    print("Validation of the model")
    model.eval()
    pred_risk_all = []
    relapse_all = []
    RFS_all = []
    pred_survival_all = []
    with torch.no_grad():
        for img_emb, text_emb, relapse, RFS, _, _ in test_loader:
            img_emb = img_emb.to(device)
            text_emb = text_emb.to(device)

            y_pred = model(img_emb, text_emb)
            pred_survival = mtlr_survival(y_pred).cpu().numpy()
            pred_risk = mtlr_risk(y_pred).cpu().numpy()

            pred_risk_all.append(pred_risk.item()) 
            relapse_all.append(relapse.item())
            RFS_all.append(RFS.item())
            pred_survival_all.append(list(pred_survival[0]))
    
    for i in range(len(pred_survival_all)):
        for j in range(len(pred_survival_all[i])):
            if pred_survival_all[i][j] > 1:
                pred_survival_all[i][j] = 1

    ci = concordance_index(RFS_all, -np.array(pred_risk_all), event_observed=relapse_all)
    print(f"Concordance Index: {ci:.4f}")

    eval_times = np.quantile(np.array([RFS_all[i] for i in range(len(RFS_all)) if relapse_all[i] == 1]), [.25, .5, .75]).astype(int)
    bs = brier_score_at_times(np.array(RFS_all), np.array(pred_survival_all), np.array(relapse_all), eval_times)
    auc = roc_auc_at_times(np.array(RFS_all), np.array(pred_survival_all), np.array(relapse_all), eval_times)
    metrics = []

    metrics.append({
        "model": "mtlr",
        **{f"bs_{t}": bs[i] for i, t in enumerate(eval_times)},
        **{f"auc_{t}": auc[i] for i, t in enumerate(eval_times)}
    })

    print(pd.DataFrame(metrics).round(3))
    return ci

# if __name__ == "__main__":
wandb.agent(sweep_id, function=exp, count = 500)
