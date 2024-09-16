"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import utils

from tqdm import tqdm
import numpy as np
import csv

from tabulate import tabulate
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset, RandomSampler
from utils import create_k_folds
import matplotlib.pyplot as plt



logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
class CPUSampler(RandomSampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return (i for i in torch.randperm(len(self.data_source), generator=torch.Generator()).tolist())

    def __len__(self):
        return len(self.data_source)
    
def plot_learning_curves(fold_losses, avg_train_losses, avg_val_losses):
    num_folds = len(fold_losses)
    fig, axes = plt.subplots(num_folds + 3, 1, figsize=(10, 6*(num_folds + 3)), sharex=True)
    fig.suptitle('Learning Curves for K-Fold Cross Validation', fontsize=16)

    # Individual fold plots
    for i, losses in enumerate(fold_losses):
        ax = axes[i]
        ax.plot(losses['train'], label='Train')
        ax.plot(losses['val'], label='Validation')
        ax.set_title(f'Fold {i+1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

    # Overlaid learning curves
    ax_overlay_train = axes[-3]
    ax_overlay_val = axes[-2]
    
    for i, losses in enumerate(fold_losses):
        ax_overlay_train.plot(losses['train'], label=f'Fold {i+1}')
        ax_overlay_val.plot(losses['val'], label=f'Fold {i+1}')
    
    ax_overlay_train.set_title('Overlaid Training Curves')
    ax_overlay_train.set_xlabel('Epoch')
    ax_overlay_train.set_ylabel('Loss')
    ax_overlay_train.legend()

    ax_overlay_val.set_title('Overlaid Validation Curves')
    ax_overlay_val.set_xlabel('Epoch')
    ax_overlay_val.set_ylabel('Loss')
    ax_overlay_val.legend()

    # Average losses
    ax_avg = axes[-1]
    x = range(num_folds)
    width = 0.35
    ax_avg.bar([i - width/2 for i in x], avg_train_losses, width, label='Avg Train Loss')
    ax_avg.bar([i + width/2 for i in x], avg_val_losses, width, label='Avg Val Loss')
    ax_avg.set_title('Average Losses per Fold')
    ax_avg.set_xlabel('Fold')
    ax_avg.set_ylabel('Average Loss')
    ax_avg.set_xticks(x)
    ax_avg.set_xticklabels([f'Fold {i+1}' for i in x])
    ax_avg.legend()

    plt.tight_layout()
    
    # Display the plot
    plt.show()

    # Save individual plots
    for i, losses in enumerate(fold_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses['train'], label='Train')
        plt.plot(losses['val'], label='Validation')
        plt.title(f'Learning Curves for Fold {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'learning_curves_fold_{i+1}.png')
        plt.close()

    # Save the overlaid plots
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(fold_losses):
        plt.plot(losses['train'], label=f'Fold {i+1} Train')
        plt.plot(losses['val'], label=f'Fold {i+1} Val')
    plt.title('Overlaid Learning Curves for All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('learning_curves_overlaid.png')
    plt.close()

    # Save the combined plot
    fig.savefig('learning_curves_all_folds.png')
    plt.close(fig)

def create_loss_table(train_losses, val_losses, overall_train_loss, overall_val_loss):
        headers = ["Fold", "Avg Train Loss", "Avg Validation Loss"]
        table_data = []
    
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            table_data.append([f"Fold {i}", f"{train_loss:.5f}", f"{val_loss:.5f}"])
    
        table_data.append(["Overall", f"{overall_train_loss:.5f}", f"{overall_val_loss:.5f}"])
    
    # Print the table
        print("\nCross-Validation Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save the table as a CSV file
        with open('cross_validation_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(table_data)
    
        print("\nCross-validation results have been saved to 'cross_validation_results.csv'")

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, best=None, device='gpu', n_splits=5):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device 
        self.n_splits = n_splits
        
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print('We are using the gpu now! device={}'.format(self.device))
        
        self.best_loss = best

    

    def cross_validate(self, num_folds=5, seed=42):
        print(f"Starting cross-validation with {num_folds} folds")
    
        total_size = len(self.train_dataset)
        print(f"Total dataset size: {total_size}")

        folds = create_k_folds(self.train_dataset, num_folds=num_folds)
        print(f"Number of folds created: {len(folds)}")

        best_fold_loss = float('inf')
        best_fold_model = None
    
        fold_losses = []
        all_fold_avg_train_losses = []
        all_fold_avg_val_losses = []

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            print(f"\nStarting fold {fold_idx + 1}/{num_folds}")
            print(f"Number of train indices: {len(train_indices)}")
            print(f"Number of val indices: {len(val_indices)}")

            if max(train_indices) >= total_size or max(val_indices) >= total_size:
                print(f"WARNING: Index out of bounds in fold {fold_idx + 1}")
                continue

            train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
            val_subset = torch.utils.data.Subset(self.train_dataset, val_indices)

            print(f"Train subset size: {len(train_subset)}")
            print(f"Val subset size: {len(val_subset)}")

            train_loader = DataLoader(train_subset, pin_memory=True,
                                sampler=CPUSampler(train_subset),
                                batch_size=self.config.batch_size,
                                num_workers=self.config.num_workers)
            val_loader = DataLoader(val_subset, pin_memory=True,
                                sampler=CPUSampler(val_subset),
                                batch_size=self.config.batch_size,
                                num_workers=self.config.num_workers)

            # Reset the model
            self.model.apply(self.model.module._init_weights)
    
            # Initialize optimizer
            optimizer = self.model.module.configure_optimizers(self.config)
    
            # Run training and validation for the current fold
            try:
                fold_loss, fold_train_losses, fold_val_losses = self.run_fold(train_loader, val_loader, optimizer)
                fold_losses.append({'train': fold_train_losses, 'val': fold_val_losses})
                
                avg_train_loss = np.mean(fold_train_losses)
                avg_val_loss = np.mean(fold_val_losses)
                all_fold_avg_train_losses.append(avg_train_loss)
                all_fold_avg_val_losses.append(avg_val_loss)
                
                print(f"Fold {fold_idx + 1} average train loss: {avg_train_loss:.5f}")
                print(f"Fold {fold_idx + 1} average validation loss: {avg_val_loss:.5f}")

                if fold_loss < best_fold_loss:
                    best_fold_loss = fold_loss
                    best_fold_model = self.model.state_dict()
            except Exception as e:
                print(f"Error in fold {fold_idx + 1}: {str(e)}")
                continue

        # Calculate overall average losses
        overall_avg_train_loss = np.mean(all_fold_avg_train_losses)
        overall_avg_val_loss = np.mean(all_fold_avg_val_losses)
        print(f"\nOverall average train loss: {overall_avg_train_loss:.5f}")
        print(f"Overall average validation loss: {overall_avg_val_loss:.5f}")

        # Create and save the table of losses
        create_loss_table(all_fold_avg_train_losses, all_fold_avg_val_losses, overall_avg_train_loss, overall_avg_val_loss)

        plot_learning_curves(fold_losses, all_fold_avg_train_losses, all_fold_avg_val_losses)

        # Load the best model from cross-validation
        if best_fold_model is not None:
            self.model.load_state_dict(best_fold_model)
            self.save_checkpoint()
            print(f"Best fold validation loss: {best_fold_loss}")
        else:
            print("Warning: No best model found. Check if all folds failed.")
    

    def run_fold(self, train_loader, val_loader, optimizer):
        best_val_loss = float('inf')
        fold_train_losses = []
        fold_val_losses = []
        for epoch in range(self.config.max_epochs):
            train_loss = self.run_epoch(train_loader, optimizer, is_train=True)
            val_loss = self.run_epoch(val_loader, optimizer, is_train=False)
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return best_val_loss, fold_train_losses, fold_val_losses

    def run_epoch(self, loader, optimizer, is_train=True):
        model = self.model
        model.train(is_train)
        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y, p, v) in pbar:
            x, y, p, v = x.to(self.device), y.to(self.device), p.to(self.device), v.to(self.device)
            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y, p, v, tokenizer=self.train_dataset.itos)
                loss = loss.mean()
                losses.append(loss.item())

            if is_train:
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                optimizer.step()

            pbar.set_description(f"{'train' if is_train else 'val'} loss {loss.item():.5f}")

        return float(np.mean(losses))

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)